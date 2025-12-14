"""
뉴스 수집기 모듈
Google News RSS를 통해 타겟팅된 뉴스를 수집하고 정제합니다.
"""
import time
import logging
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote, urlparse
import feedparser
import trafilatura
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from config import get_config
    from database import DatabaseManager
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from config import get_config
    from database import DatabaseManager

logger = logging.getLogger(__name__)


class NewsCollector:
    """뉴스 수집기 클래스"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Args:
            db_manager: DatabaseManager 인스턴스 (None이면 새로 생성)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.config = get_config()
        # 카테고리 기반 구조로 변경 (하위 호환성을 위해 search_groups도 지원)
        self.categories = self.config.get('categories', [])
        if not self.categories:
            # 기존 search_groups가 있으면 카테고리로 변환
            self.categories = self._convert_search_groups_to_categories()
        self.publisher_weights = self.config.get('publisher_weights', {})
        self.rate_limit_delay = self.config.get('rate_limit', {}).get('google_rss_delay', 2.0)
        self.repost_sites = [site.lower() for site in self.config.get('repost_sites', [])]  # 소문자로 변환하여 비교
        
        # 스니펫 추출 활성화 여부 (일시적으로 비활성화 가능)
        self.enable_snippet_extraction = False  # True로 변경하면 스니펫 추출 활성화
        
        # 언론사 가중치 딕셔너리 생성
        self.source_weights = self._build_source_weights()
    
    def _convert_search_groups_to_categories(self) -> List[Dict[str, Any]]:
        """기존 search_groups를 카테고리 형식으로 변환 (하위 호환성)"""
        search_groups = self.config.get('search_groups', [])
        categories = []
        for group in search_groups:
            region = group.get('region', 'US').lower()
            if region == 'us':
                language = 'en'
                hl = 'en-US'
                gl = 'US'
                ceid = 'US:en'
            elif region == 'kr':
                language = 'ko'
                hl = 'ko'
                gl = 'KR'
                ceid = 'KR:ko'
            else:
                language = 'en'
                hl = 'en-US'
                gl = 'US'
                ceid = 'US:en'
            
            category = {
                'name': group.get('name', 'Unknown'),
                'max_articles': 100,
                'rss_url_template': "https://news.google.com/rss/search?q={keyword}&hl={hl}&gl={gl}&ceid={ceid}",
                'region': region,
                'language': language,
                'keywords': group.get('keywords', [])
            }
            categories.append(category)
        return categories
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그를 제거하고 텍스트만 추출합니다"""
        if not text:
            return ""
        
        # 여러 번 반복하여 모든 HTML 태그 제거
        old_text = ""
        while old_text != text:
            old_text = text
            text = re.sub(r'<[^>]+>', '', text)
        
        # HTML 엔티티 디코딩 (더 많은 엔티티 처리)
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '...',
            '&nbsp': ' ',
        }
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # 숫자 엔티티 디코딩 (&#123; 형태)
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
        
        # URL 제거 (http://, https://로 시작하는 링크)
        text = re.sub(r'https?://[^\s<>"]+', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _build_source_weights(self) -> Dict[str, float]:
        """언론사 가중치 딕셔너리를 생성합니다"""
        weights = {}
        
        # Tier 1 언론사 처리
        tier1_config = self.publisher_weights.get('tier1', {})
        if isinstance(tier1_config, dict):
            # 딕셔너리 형태: {sources: [...], weight: 1.5}
            tier1_sources = tier1_config.get('sources', [])
            tier1_weight = tier1_config.get('weight', 1.5)
        else:
            tier1_sources = []
            tier1_weight = 1.5
        
        for source in tier1_sources:
            if isinstance(source, str):
                weights[source.lower()] = tier1_weight
        
        # Tier 2 언론사 처리
        tier2_config = self.publisher_weights.get('tier2', {})
        if isinstance(tier2_config, dict):
            tier2_sources = tier2_config.get('sources', [])
            tier2_weight = tier2_config.get('weight', 1.2)
        else:
            tier2_sources = []
            tier2_weight = 1.2
        
        for source in tier2_sources:
            if isinstance(source, str):
                weights[source.lower()] = tier2_weight
        
        # 기본 가중치
        default_config = self.publisher_weights.get('default', {})
        if isinstance(default_config, dict):
            default_weight = default_config.get('weight', 1.0)
        else:
            default_weight = 1.0
        
        tier1_count = len([w for w in weights.values() if w == tier1_weight])
        tier2_count = len([w for w in weights.values() if w == tier2_weight])
        
        logger.info(f"언론사 가중치 설정 완료: Tier1={tier1_count}개(weight={tier1_weight}), "
                   f"Tier2={tier2_count}개(weight={tier2_weight}), Default={default_weight}")
        
        return weights
    
    def _get_source_weight(self, source: str) -> float:
        """언론사 이름으로 가중치를 가져옵니다"""
        if not source:
            return 1.0
        
        source_lower = source.lower()
        return self.source_weights.get(source_lower, self.publisher_weights.get('default', {}).get('weight', 1.0))
    
    def _get_rank_bonus(self, rank: int) -> float:
        """
        검색 노출 순위에 따른 가점을 계산합니다.
        
        Args:
            rank: 검색 노출 순위 (1부터 시작)
        
        Returns:
            순위 가점
            - 1-5위: 0.2
            - 5-10위: 0.15
            - 10-20위: 0.1
            - 20-50위: 0.05
            - 50-100위: 0
        """
        if rank is None or rank <= 0:
            return 0.0
        
        if rank <= 5:
            return 0.2
        elif rank <= 10:
            return 0.15
        elif rank <= 20:
            return 0.1
        elif rank <= 50:
            return 0.05
        else:
            return 0.0
    
    def _build_google_news_url(self, keyword: str, category: Dict[str, Any]) -> str:
        """
        Google News RSS URL을 생성합니다 (카테고리 기반).
        
        Args:
            keyword: 검색 키워드
            category: 카테고리 딕셔너리 (region, language, rss_url_template 포함)
        
        Returns:
            Google News RSS URL
        """
        # Google News RSS는 site: 구문을 지원하지 않으므로 제거
        # site: 구문이 포함된 경우 경고 로그 출력
        original_keyword = keyword
        if 'site:' in keyword.lower():
            logger.warning(f"키워드에 'site:' 구문이 포함되어 있습니다. Google News RSS는 site: 구문을 지원하지 않으므로 제거합니다.")
            logger.warning(f"원본 키워드: {keyword[:200]}")
            # site: 구문 제거 (site:xxx OR site:yyy 패턴 제거)
            # site:xxx OR site:yyy 패턴 제거
            keyword = re.sub(r'\s*\(?\s*site:[^)]+\)?\s*', '', keyword, flags=re.IGNORECASE)
            # 남은 괄호 정리
            keyword = re.sub(r'\(\s*\)', '', keyword)  # 빈 괄호 제거
            keyword = keyword.strip()
            logger.warning(f"수정된 키워드: {keyword[:200]}")
        
        # URL 인코딩 (안전한 문자만 인코딩)
        # Google News RSS는 + 기호를 공백으로 인식하므로 quote_plus 대신 quote 사용
        encoded_keyword = quote(keyword, safe='')
        
        # 카테고리에서 지역 및 언어 정보 가져오기
        region = category.get('region', 'us').lower()
        language = category.get('language', 'en').lower()
        
        # Region별 언어 및 지역 코드 설정
        if region == 'us':
            hl = 'en-US'  # 언어
            gl = 'US'     # 지역
            ceid = 'US:en'
        elif region == 'kr':
            hl = 'ko'     # 언어
            gl = 'KR'     # 지역
            ceid = 'KR:ko'
        else:
            # 기본값 (US)
            hl = 'en-US'
            gl = 'US'
            ceid = 'US:en'
        
        # RSS URL 템플릿 사용 (템플릿이 있으면 사용, 없으면 기본 형식)
        rss_template = category.get('rss_url_template', 
            "https://news.google.com/rss/search?q={keyword}&hl={hl}&gl={gl}&ceid={ceid}")
        
        # 템플릿에서 {query}와 {keyword} 모두 지원 (하위 호환성)
        # {query}가 있으면 {keyword}로 변환
        if '{query}' in rss_template:
            rss_template = rss_template.replace('{query}', '{keyword}')
            logger.debug("템플릿의 {query}를 {keyword}로 변환했습니다.")
        
        try:
            url = rss_template.format(
                keyword=encoded_keyword,
                query=encoded_keyword,  # 하위 호환성을 위해 query도 제공
                hl=hl,
                gl=gl,
                ceid=ceid
            )
        except KeyError as e:
            logger.error(f"URL 템플릿 포맷 오류: {e}, 템플릿: {rss_template}, 키워드: {encoded_keyword[:100]}")
            raise
        
        logger.debug(f"생성된 RSS URL: {url[:200]}...")
        return url
    
    def _get_final_url(self, url: str) -> str:
        """
        Google News 리다이렉트 URL을 따라가서 최종 목적지(언론사 원문 URL)를 반환합니다.
        
        Args:
            url: Google News 리다이렉트 URL 또는 일반 URL
        
        Returns:
            최종 목적지 URL (실패 시 원본 URL 반환)
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            response = requests.get(
                url, 
                headers=headers, 
                allow_redirects=True, 
                timeout=10,
                stream=True  # 메모리 효율을 위해 스트림 모드 사용
            )
            final_url = response.url
            logger.debug(f"URL 리다이렉트 추적: {url[:80]}... -> {final_url[:80]}...")
            return final_url
        except requests.exceptions.Timeout:
            logger.warning(f"URL 리다이렉트 추적 타임아웃: {url[:80]}...")
            return url
        except requests.exceptions.RequestException as e:
            logger.warning(f"URL 리다이렉트 추적 실패: {e} (URL: {url[:80]}...)")
            return url
        except Exception as e:
            logger.warning(f"URL 리다이렉트 추적 중 예상치 못한 오류: {e} (URL: {url[:80]}...)")
            return url
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_rss_feed(self, url: str) -> feedparser.FeedParserDict:
        """
        RSS 피드를 가져옵니다 (재시도 로직 포함).
        
        Args:
            url: RSS 피드 URL
        
        Returns:
            feedparser 결과
        """
        try:
            feed = feedparser.parse(url)
            
            if feed.bozo and feed.bozo_exception:
                error_msg = str(feed.bozo_exception)
                error_type = type(feed.bozo_exception).__name__
                logger.error(f"RSS 파싱 오류 (URL: {url[:100]}...): {error_type}: {error_msg}")
                raise Exception(f"RSS 파싱 오류: {error_type}: {error_msg}")
            
            return feed
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"RSS 피드 가져오기 실패 (URL: {url[:100]}...): {error_type}: {error_msg}", exc_info=True)
            raise
    
    def _extract_article_content(self, url: str, title: str) -> str:
        """
        trafilatura를 사용하여 기사 본문을 추출합니다.
        Google News 리다이렉트 URL을 먼저 해결한 후 원문을 추출합니다.
        
        Args:
            url: 기사 URL (Google News 리다이렉트일 수 있음)
            title: 기사 제목 (fallback용)
        
        Returns:
            추출된 본문 또는 빈 문자열 (실패 시)
        """
        try:
            # Rate limiting을 위한 짧은 대기
            time.sleep(0.5)
            
            # 1단계: Google News 리다이렉트 URL을 최종 URL로 변환
            final_url = self._get_final_url(url)
            
            if not final_url or final_url == url:
                # 리다이렉트가 없거나 실패한 경우 원본 URL 사용
                logger.debug(f"리다이렉트 없음 또는 실패, 원본 URL 사용: {url[:80]}...")
                final_url = url
            
            # 2단계: requests로 HTML 내용 가져오기 (User-Agent 헤더 포함)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            try:
                response = requests.get(final_url, headers=headers, timeout=10)
                response.raise_for_status()
                html_content = response.text
            except requests.exceptions.RequestException as e:
                logger.debug(f"HTML 다운로드 실패 (URL: {final_url[:80]}...): {e}")
                # trafilatura.fetch_url로 폴백 시도
                html_content = trafilatura.fetch_url(final_url, timeout=10)
                if not html_content:
                    logger.debug(f"trafilatura.fetch_url도 실패: {final_url[:80]}...")
                    return ""
            
            # 3단계: trafilatura로 본문 추출
            if html_content:
                content = trafilatura.extract(
                    html_content, 
                    include_comments=False, 
                    include_tables=False,
                    include_images=False,
                    include_links=False
                )
                if content and len(content) > 100:  # 최소 길이 체크
                    # 너무 긴 경우 앞부분만 사용 (임베딩 효율을 위해)
                    max_length = 2000
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    logger.debug(f"본문 추출 성공: {len(content)}자 (최종 URL: {final_url[:80]}...)")
                    return content
                else:
                    logger.debug(f"본문 추출 결과가 너무 짧음: {len(content) if content else 0}자")
            else:
                logger.debug(f"HTML 내용이 비어있음: {final_url[:80]}...")
                
        except Exception as e:
            logger.debug(f"본문 추출 실패 (URL: {url}): {e}")
        
        # Fallback: 빈 문자열 반환 (호출자가 처리)
        return ""
    
    def _parse_rss_entry(self, entry: feedparser.FeedParserDict, keyword: str, category: Dict[str, Any], rank: int = None) -> Optional[Dict[str, Any]]:
        """
        RSS 엔트리를 파싱하여 뉴스 딕셔너리로 변환합니다.
        
        Args:
            entry: feedparser 엔트리
            keyword: 검색 키워드
            category: 카테고리 딕셔너리
            rank: 검색 노출 순위 (1부터 시작)
        
        Returns:
            뉴스 딕셔너리 또는 None (파싱 실패 시)
        """
        try:
            title = entry.get('title', '').strip()
            link = entry.get('link', '').strip()
            
            if not title or not link:
                logger.warning(f"제목 또는 링크가 없는 엔트리 건너뜀: {entry}")
                return None
            
            # source 추출 (link에서 도메인 추출 또는 source 필드)
            source = entry.get('source', {}).get('title', '') if hasattr(entry, 'source') else ''
            if not source:
                # URL에서 도메인 추출
                parsed_url = urlparse(link)
                source = parsed_url.netloc.replace('www.', '')
            
            # 제목에서 소스 제거 (예: "제목 - 소스명" 형식)
            # " - " 패턴을 찾아서 그 뒤의 텍스트를 제거
            if ' - ' in title:
                # " - " 뒤의 텍스트가 소스명일 가능성이 높음
                title_parts = title.split(' - ', 1)
                if len(title_parts) == 2:
                    # 소스명과 일치하는지 확인 (대소문자 무시)
                    potential_source = title_parts[1].strip()
                    if potential_source.lower() == source.lower() or source.lower() in potential_source.lower():
                        title = title_parts[0].strip()
                        logger.debug(f"제목에서 소스 제거: '{title_parts[0]} - {potential_source}' -> '{title}'")
                    else:
                        # 소스명과 일치하지 않으면 마지막 " - " 패턴만 제거
                        title = title.rsplit(' - ', 1)[0].strip()
                        logger.debug(f"제목에서 마지막 ' - ' 패턴 제거: '{entry.get('title', '')}' -> '{title}'")
            
            # published_at 파싱
            published_at = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6])
                except Exception as e:
                    logger.debug(f"날짜 파싱 실패: {e}")
            
            # 스니펫 수집 비활성화: 빈 문자열로 설정
            snippet = ""
            
            # importance_score 계산 (언론사 가점 + 순위 가점)
            source_weight = self._get_source_weight(source)
            rank_bonus = self._get_rank_bonus(rank) if rank else 0.0
            importance_score = source_weight + rank_bonus
            
            # 카테고리 정보
            category_name = category.get('name', 'Unknown')
            
            return {
                'title': title,
                'link': link,
                'snippet': snippet,
                'source': source,
                'published_at': published_at,
                'importance_score': importance_score,
                'category_name': category_name,
                'search_keyword': keyword,
                'search_rank': rank
            }
        
        except Exception as e:
            logger.error(f"RSS 엔트리 파싱 실패: {e}, Entry: {entry}")
            return None
    
    def collect_from_category(self, category: Dict[str, Any], max_articles: Optional[int] = None) -> Dict[str, int]:
        """
        카테고리에서 뉴스를 수집합니다.
        
        Args:
            category: 카테고리 딕셔너리 (name, region, language, keywords, max_articles, rss_url_template)
            max_articles: 최대 수집 기사 수 (None이면 카테고리의 max_articles 사용)
        
        Returns:
            수집 결과 딕셔너리 (collected, skipped, errors)
        """
        category_name = category.get('name', 'Unknown')
        region = category.get('region', 'us')
        language = category.get('language', 'en')
        keywords = category.get('keywords', [])
        category_max = category.get('max_articles', 100)
        
        # max_articles가 지정되지 않았으면 카테고리의 max_articles 사용
        if max_articles is None:
            max_articles = category_max
        else:
            # 둘 중 작은 값 사용
            max_articles = min(max_articles, category_max)
        
        logger.info(f"카테고리 '{category_name}' 수집 시작 (Region: {region}, Language: {language}, Keywords: {len(keywords)}개, 최대: {max_articles}개)")
        
        collected = 0
        skipped = 0
        errors = 0
        
        for keyword in keywords:
            # 최대 수집 수 체크
            if max_articles and collected >= max_articles:
                logger.info(f"최대 수집 수({max_articles})에 도달하여 중단합니다.")
                break
            try:
                # Rate Limiting
                time.sleep(self.rate_limit_delay)
                
                # RSS URL 생성
                rss_url = self._build_google_news_url(keyword, category)
                logger.debug(f"RSS 피드 가져오기: {keyword} -> {rss_url}")
                
                # RSS 피드 가져오기
                feed = self._fetch_rss_feed(rss_url)
                
                if not feed.entries:
                    logger.warning(f"키워드 '{keyword}'에 대한 결과가 없습니다")
                    continue
                
                logger.info(f"키워드 '{keyword}': {len(feed.entries)}개 엔트리 발견")
                
                # 각 엔트리 처리 (순위는 인덱스 + 1)
                for idx, entry in enumerate(feed.entries):
                    # 최대 수집 수 체크
                    if max_articles and collected >= max_articles:
                        break
                    
                    try:
                        # 검색 노출 순위 (1부터 시작)
                        search_rank = idx + 1
                        
                        # 엔트리 파싱
                        news_data = self._parse_rss_entry(entry, keyword, category, rank=search_rank)
                        
                        if not news_data:
                            skipped += 1
                            continue
                        
                        # URL 중복 체크
                        if self.db_manager.news_exists(news_data['link']):
                            logger.debug(f"중복 기사 건너뜀 (URL): {news_data['link']}")
                            skipped += 1
                            continue
                        
                        # 제목 기준 중복 체크 및 재배포 사이트 처리
                        existing_news = self.db_manager.get_news_by_title(news_data['title'])
                        if existing_news:
                            # 제목이 동일한 기사가 이미 존재
                            existing_source = (existing_news.get('source') or '').lower()
                            new_source = (news_data.get('source') or '').lower()
                            
                            # 재배포 사이트 확인
                            existing_is_repost = any(repost in existing_source for repost in self.repost_sites)
                            new_is_repost = any(repost in new_source for repost in self.repost_sites)
                            
                            if new_is_repost:
                                # 새 기사가 재배포 사이트면 건너뛰기
                                logger.debug(f"재배포 사이트 기사 건너뜀: {news_data['title'][:50]}... (Source: {news_data.get('source')})")
                                skipped += 1
                                continue
                            elif existing_is_repost:
                                # 기존 기사가 재배포 사이트면 기존 기사 삭제 후 새 기사 저장
                                logger.info(f"재배포 사이트 기사 교체: {news_data['title'][:50]}... (기존: {existing_news.get('source')} → 새: {news_data.get('source')})")
                                self.db_manager.delete_news(existing_news['id'])
                                # 새 기사 저장 (아래 코드로 계속 진행)
                            else:
                                # 둘 다 재배포 사이트가 아니면 기존 기사 유지
                                logger.debug(f"중복 기사 건너뜀 (제목 동일): {news_data['title'][:50]}... (기존 Source: {existing_news.get('source')})")
                                skipped += 1
                                continue
                        
                        # DB에 저장
                        news_id = self.db_manager.insert_news(
                            title=news_data['title'],
                            link=news_data['link'],
                            snippet=news_data['snippet'],
                            source=news_data['source'],
                            published_at=news_data['published_at'],
                            importance_score=news_data['importance_score'],
                            category_name=news_data.get('category_name'),
                            search_keyword=news_data.get('search_keyword'),
                            search_rank=news_data.get('search_rank')
                        )
                        
                        if news_id:
                            collected += 1
                            logger.info(f"✅ 뉴스 저장: {news_data['title'][:50]}...")
                        else:
                            skipped += 1
                    
                    except Exception as e:
                        logger.error(f"엔트리 처리 중 오류: {e}")
                        errors += 1
                        continue
                
                # 최대 수집 수 체크
                if max_articles and collected >= max_articles:
                    break
            
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"키워드 '{keyword[:100]}...' 수집 중 오류: {error_type}: {error_msg}", exc_info=True)
                errors += 1
                continue
        
        result = {
            'collected': collected,
            'skipped': skipped,
            'errors': errors
        }
        
        logger.info(f"카테고리 '{category_name}' 수집 완료: {result}")
        return result
    
    def collect_all(self, max_articles: Optional[int] = None) -> Dict[str, Any]:
        """
        모든 카테고리에서 뉴스를 수집합니다.
        
        Args:
            max_articles: 전체 최대 수집 기사 수 (None이면 각 카테고리의 max_articles 사용)
        
        Returns:
            전체 수집 결과
        """
        logger.info("=" * 50)
        logger.info("전체 뉴스 수집 시작")
        if max_articles:
            logger.info(f"전체 최대 수집 수: {max_articles}개")
        logger.info("=" * 50)
        
        total_collected = 0
        total_skipped = 0
        total_errors = 0
        category_results = []
        
        for category in self.categories:
            # 남은 수집 수 계산
            if max_articles:
                remaining = max_articles - total_collected
                if remaining <= 0:
                    logger.info(f"전체 최대 수집 수({max_articles})에 도달하여 중단합니다.")
                    break
            else:
                remaining = None
            
            result = self.collect_from_category(category, max_articles=remaining)
            category_results.append({
                'category': category.get('name', 'Unknown'),
                **result
            })
            
            total_collected += result['collected']
            total_skipped += result['skipped']
            total_errors += result['errors']
        
        summary = {
            'total_collected': total_collected,
            'total_skipped': total_skipped,
            'total_errors': total_errors,
            'category_results': category_results
        }
        
        logger.info("=" * 50)
        logger.info(f"전체 수집 완료: 총 {total_collected}개 수집, {total_skipped}개 건너뜀, {total_errors}개 오류")
        logger.info("=" * 50)
        
        # 구글 스프레드시트에 수집 결과 기록
        try:
            from modules.google_sheets import GoogleSheetsExporter
            sheets_exporter = GoogleSheetsExporter()
            if sheets_exporter.spreadsheet:
                # 수집된 뉴스 데이터 가져오기
                with self.db_manager.get_connection() as conn:
                    query = """
                        SELECT 
                            id, category_name, search_keyword, title, link, snippet,
                            source, search_rank, published_at, created_at,
                            importance_score, analyzed
                        FROM news
                        ORDER BY category_name, search_rank ASC, created_at DESC
                        LIMIT 10000
                    """
                    import pandas as pd
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        news_list = df.to_dict('records')
                        sheets_exporter.export_news_collection(news_list)
                        logger.info("구글 스프레드시트에 뉴스 수집 데이터 기록 완료")
        except Exception as e:
            logger.warning(f"구글 스프레드시트 기록 실패 (계속 진행): {e}")
        
        return summary


def main():
    """테스트용 메인 함수"""
    import sys
    from pathlib import Path
    
    # 프로젝트 루트를 Python 경로에 추가
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 수집기 생성 및 실행
    collector = NewsCollector()
    result = collector.collect_all()
    
    print("\n수집 결과:")
    print(f"  총 수집: {result['total_collected']}개")
    print(f"  건너뜀: {result['total_skipped']}개")
    print(f"  오류: {result['total_errors']}개")
    
    for category_result in result['category_results']:
        print(f"\n  [{category_result['category']}]")
        print(f"    수집: {category_result['collected']}개")
        print(f"    건너뜀: {category_result['skipped']}개")
        print(f"    오류: {category_result['errors']}개")


if __name__ == "__main__":
    main()

