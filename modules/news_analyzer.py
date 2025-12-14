"""
뉴스 분석기 모듈
ChromaDB + DBSCAN + MMR + LLM을 활용하여 뉴스를 클러스터링하고 분석합니다.
"""
import os
import time
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config
from database import DatabaseManager
from modules.feedback_loader import FeedbackLoader
from modules.feedback_analyzer import FeedbackAnalyzer

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """뉴스 분석기 클래스"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Args:
            db_manager: DatabaseManager 인스턴스 (None이면 새로 생성)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.config = get_config()
        
        # 설정 로드
        self.clustering_config = self.config.get('clustering', {})
        self.llm_config = self.config.get('llm', {})
        self.analysis_config = self.config.get('analysis', {})
        self.meta_tags = self.config.get('meta_tags', [])
        self.rate_limit_delay = self.config.get('rate_limit', {}).get('openai_delay', 0.1)
        
        # LLM Provider 확인
        self.llm_provider = self.llm_config.get('provider', 'openai')
        self.embedding_provider = self.llm_config.get('embedding_provider', 'openai')
        
        # OpenAI 클라이언트 초기화 (임베딩용)
        if self.embedding_provider == 'openai':
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "sk-your-api-key-here":
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            self.openai_client = OpenAI(api_key=api_key)
            self.embedding_model = self.llm_config.get('embedding_model', 'text-embedding-3-small')
        
        # Gemini 클라이언트 초기화 (LLM용)
        if self.llm_provider == 'gemini':
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            genai.configure(api_key=gemini_key)
            model_name = self.llm_config.get('model', 'gemini-2.0-flash-exp')
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini 모델 초기화: {model_name}")
        else:
            # OpenAI LLM 사용 시
            self.gemini_model = None
        
        # ChromaDB 초기화
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="news_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("뉴스 분석기 초기화 완료")
    
    def _generate_vector_id(self, url: str) -> str:
        """URL에서 벡터 ID를 생성합니다 (해시값)"""
        return hashlib.md5(url.encode()).hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_embedding(self, text: str) -> List[float]:
        """
        OpenAI API를 사용하여 텍스트의 임베딩을 생성합니다.
        
        Args:
            text: 임베딩할 텍스트
        
        Returns:
            임베딩 벡터
        """
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def _store_embedding(self, vector_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """ChromaDB에 임베딩을 저장합니다"""
        try:
            self.collection.add(
                ids=[vector_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"임베딩 저장 실패 (vector_id: {vector_id}): {e}")
            raise
    
    def _get_embedding_from_chroma(self, vector_id: str) -> Optional[List[float]]:
        """ChromaDB에서 임베딩을 가져옵니다"""
        try:
            results = self.collection.get(ids=[vector_id], include=['embeddings'])
            if results['embeddings']:
                return results['embeddings'][0]
            return None
        except Exception as e:
            logger.debug(f"임베딩 조회 실패 (vector_id: {vector_id}): {e}")
            return None
    
    def _prepare_text_for_embedding(self, news: Dict[str, Any]) -> str:
        """뉴스 딕셔너리에서 임베딩용 텍스트를 준비합니다"""
        title = news.get('title', '')
        source = news.get('source', '')
        
        # 제목만 사용 (스니펫은 수집하지 않음)
        text = title
        
        if source:
            text += f"\n\n출처: {source}"
        
        return text
    
    def _cluster_news(self, embeddings: np.ndarray, category: str = None) -> np.ndarray:
        """
        DBSCAN을 사용하여 뉴스를 클러스터링합니다.
        
        Args:
            embeddings: 임베딩 벡터 배열
            category: 카테고리 이름 (카테고리별 파라미터 적용용)
        
        Returns:
            클러스터 라벨 배열 (-1은 노이즈)
        """
        # 카테고리별 eps 값 적용
        base_eps = self.clustering_config.get('eps', 0.30)
        if category == 'Korea Market':
            eps = 0.45  # 한국 증시 뉴스는 반복 표현이 많아 완화된 기준 적용
            logger.info(f"[{category}] 카테고리별 eps 적용: {base_eps} -> {eps}")
        else:
            eps = base_eps
        
        min_samples = self.clustering_config.get('min_samples', 3)
        
        logger.info(f"DBSCAN 클러스터링 시작 (eps={eps}, min_samples={min_samples}, 샘플 수={len(embeddings)})")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)
        
        unique_labels = set(labels)
        noise_count = list(labels).count(-1)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        logger.info(f"클러스터링 완료: {cluster_count}개 클러스터, {noise_count}개 노이즈")
        
        return labels
    
    def _split_large_cluster(self, news_list: List[Dict[str, Any]], 
                            embeddings: np.ndarray, max_size: int) -> List[List[int]]:
        """
        큰 클러스터를 Recursive DBSCAN으로 분할합니다.
        밀도 기반 분할로 의미적 경계를 고려합니다.
        
        Args:
            news_list: 뉴스 리스트
            embeddings: 임베딩 배열
            max_size: 최대 클러스터 크기
        
        Returns:
            분할된 클러스터 인덱스 리스트
        """
        if len(news_list) <= max_size:
            return [list(range(len(news_list)))]
        
        # 기존 eps에 0.75를 곱하여 더 엄격한 기준 적용
        base_eps = self.clustering_config.get('eps', 0.30)
        stricter_eps = base_eps * 0.75
        min_samples = 2  # 소수 이슈도 포착
        
        logger.info(f"Recursive DBSCAN 분할 시작: {len(news_list)}개 기사, eps={stricter_eps:.3f}, min_samples={min_samples}")
        
        # DBSCAN으로 재클러스터링
        sub_dbscan = DBSCAN(eps=stricter_eps, min_samples=min_samples, metric='cosine')
        sub_labels = sub_dbscan.fit_predict(embeddings)
        
        # 서브 클러스터 수집
        unique_sub_labels = set(sub_labels)
        if -1 in unique_sub_labels:
            unique_sub_labels.remove(-1)
        
        clusters = []
        for sub_cluster_id in unique_sub_labels:
            cluster_indices = [idx for idx, label in enumerate(sub_labels) if label == sub_cluster_id]
            if cluster_indices:
                clusters.append(cluster_indices)
        
        # 노이즈 처리: "기타 관련 뉴스 (Misc)" 서브 클러스터로 묶기
        noise_indices = [idx for idx, label in enumerate(sub_labels) if label == -1]
        if len(noise_indices) >= 2:  # 최소 2개 이상이면 묶기
            clusters.append(noise_indices)
            logger.info(f"노이즈 기사 {len(noise_indices)}개를 '기타 관련 뉴스' 서브 클러스터로 묶음")
        elif len(noise_indices) == 1:
            # 1개만 있으면 가장 가까운 클러스터에 할당
            noise_idx = noise_indices[0]
            if clusters:
                # 가장 가까운 클러스터 찾기
                noise_embedding = embeddings[noise_idx:noise_idx+1]
                min_distance = float('inf')
                closest_cluster_idx = 0
                for i, cluster_indices in enumerate(clusters):
                    cluster_embeddings = embeddings[cluster_indices]
                    similarities = cosine_similarity(noise_embedding, cluster_embeddings)[0]
                    max_similarity = similarities.max()
                    distance = 1.0 - max_similarity
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster_idx = i
                clusters[closest_cluster_idx].append(noise_idx)
                logger.debug(f"단일 노이즈 기사를 가장 가까운 클러스터에 할당")
        
        logger.info(f"Recursive DBSCAN 분할 완료: {len(news_list)}개 -> {len(clusters)}개 서브클러스터")
        
        return clusters
    
    def _mmr_selection(self, news_list: List[Dict[str, Any]], 
                      embeddings: np.ndarray, 
                      max_count: int) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) 알고리즘으로 대표 기사를 선택합니다.
        importance_score가 높은 기사를 우선적으로 선택합니다.
        
        Args:
            news_list: 뉴스 리스트
            embeddings: 임베딩 배열
            max_count: 선택할 최대 기사 수
        
        Returns:
            선택된 기사의 인덱스 리스트
        """
        if len(news_list) <= max_count:
            return list(range(len(news_list)))
        
        # 중요도 점수로 먼저 정렬 (높은 순서대로)
        sorted_indices = sorted(
            range(len(news_list)),
            key=lambda i: news_list[i].get('importance_score', 1.0),
            reverse=True
        )
        
        # 정렬된 뉴스와 임베딩 재구성
        sorted_news = [news_list[i] for i in sorted_indices]
        sorted_embeddings = embeddings[sorted_indices]
        
        mmr_diversity = self.clustering_config.get('mmr_diversity', 0.7)
        relevance_weight = 1.0 - mmr_diversity
        
        selected_indices = []
        remaining_indices = set(range(len(sorted_news)))
        
        # 첫 번째 기사: importance_score가 가장 높은 것 (이미 정렬되어 있음)
        first_idx = 0
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 나머지 기사 선택
        while len(selected_indices) < max_count and remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            selected_embeddings = sorted_embeddings[selected_indices]
            
            for idx in remaining_indices:
                # Relevance: importance_score (이미 정렬되어 있으므로 인덱스가 낮을수록 높은 점수)
                relevance = sorted_news[idx].get('importance_score', 1.0)
                
                # Diversity: 선택된 기사들과의 최소 유사도
                current_embedding = sorted_embeddings[idx:idx+1]
                similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                diversity = 1.0 - similarities.max()  # 최대 유사도가 낮을수록 다양성 높음
                
                # MMR 점수
                mmr_score = relevance_weight * relevance + mmr_diversity * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # 원래 인덱스로 변환
        original_selected_indices = [sorted_indices[i] for i in selected_indices]
        
        logger.info(f"MMR 선택 완료: {len(original_selected_indices)}개 기사 선택")
        logger.debug(f"선택된 기사 중요도 점수: {[news_list[i].get('importance_score', 1.0) for i in original_selected_indices]}")
        
        return original_selected_indices
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_summarizer(self, category: str, news_list: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        LLM을 사용하여 뉴스 클러스터의 대표 타이틀과 Executive Summary를 생성합니다.
        
        Args:
            category: 카테고리 이름 (예: Global Macro, Korea Market)
            news_list: 뉴스 리스트 (최대 5개, importance_score 기준 상위 기사)
        
        Returns:
            {generated_title: str, executive_summary: str}
        """
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # 뉴스 리스트 포맷팅
            news_context = ""
            for idx, news in enumerate(news_list, 1):
                title = news.get('title', 'Unknown Title')
                source = news.get('source', 'Unknown Source')
                news_context += f"{idx}. 제목: {title} / 출처: {source}\n"
            
            # System Role 프롬프트
            system_prompt = """당신은 금융 시장의 복잡한 뉴스를 분석하여 통찰력 있는 보고서를 작성하는 수석 애널리스트입니다. 파편화된 뉴스 헤드라인들을 읽고, 전체 맥락을 관통하는 하나의 핵심 주제와 요약 리포트를 작성해야 합니다."""
            
            # User Prompt
            user_prompt = f"""[지시사항]
아래 제공된 {len(news_list)}개의 뉴스 기사들은 동일한 이슈(Cluster)로 묶인 기사들입니다. 
이 기사들을 종합적으로 분석하여 다음 두 가지를 출력하시오.

1. **New Issue Title**: 개별 기사 제목을 그대로 쓰지 말고, {len(news_list)}개 기사를 모두 아우르는 **통찰력 있고 간결한 한 문장의 대표 제목**을 새로 작성하시오. (수치나 핵심 키워드는 반드시 포함할 것)
2. **Executive Summary**: 해당 이슈가 발생한 배경, 주요 수치(금액, 지수 등), 그리고 시장에 미치는 영향을 포함하여 3~5문장으로 요약하시오.

[입력 데이터: 뉴스 리스트]
{news_context}

[출력 형식 (JSON)]
{{
  "generated_title": "생성된 대표 타이틀",
  "executive_summary": "작성된 요약 내용..."
}}

카테고리: {category}"""
            
            # LLM 호출
            if self.llm_provider == 'gemini':
                # Gemini 사용
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': self.llm_config.get('temperature', 0.3),
                        'max_output_tokens': self.llm_config.get('max_tokens', 2000),
                    }
                )
                response_text = response.text.strip()
            else:
                # OpenAI 사용 (추후 구현 가능)
                raise ValueError(f"LLM Provider '{self.llm_provider}'는 아직 지원되지 않습니다.")
            
            # JSON 파싱
            # JSON 블록 추출 (```json ... ``` 또는 { ... } 형식)
            json_text = response_text
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()
            elif '{' in response_text and '}' in response_text:
                # JSON 객체만 추출
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_text = response_text[start_idx:end_idx]
            
            # JSON 파싱
            try:
                result = json.loads(json_text)
                generated_title = result.get('generated_title', '').strip()
                executive_summary = result.get('executive_summary', '').strip()
                
                # 검증: 빈 값이면 fallback
                if not generated_title or not executive_summary:
                    logger.warning(f"LLM 응답에서 빈 값 발견, fallback 사용")
                    return self._fallback_synthesize(news_list)
                
                logger.debug(f"LLM 요약 생성 성공: {generated_title[:50]}...")
                return {
                    'generated_title': generated_title[:200],  # 최대 200자
                    'executive_summary': executive_summary[:1000]  # 최대 1000자
                }
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 파싱 실패: {e}, 원본 응답: {response_text[:200]}...")
                return self._fallback_synthesize(news_list)
        
        except Exception as e:
            logger.error(f"LLM 요약 생성 실패: {e}")
            return self._fallback_synthesize(news_list)
    
    def _fallback_synthesize(self, selected_news: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        LLM 호출 실패 시 fallback으로 기본 제목과 요약 생성
        
        Args:
            selected_news: 선택된 뉴스 리스트
        
        Returns:
            {generated_title: str, executive_summary: str}
        """
        if len(selected_news) == 1:
            title = selected_news[0].get('title', 'Unknown Issue')[:200]
            summary = f"관련 뉴스: {title}"
        else:
            # 여러 기사인 경우 첫 번째 기사 제목 사용
            title = selected_news[0].get('title', 'Unknown Issue')[:200]
            # 요약은 여러 기사 제목 결합
            titles = [news.get('title', '') for news in selected_news[:3]]
            summary = f"관련 뉴스: {' | '.join(titles)}"[:1000]
        
        return {
            'generated_title': title,
            'executive_summary': summary
        }
    
    def _synthesize_and_tag(self, selected_news: List[Dict[str, Any]], category: str = 'Unknown') -> Dict[str, Any]:
        """
        LLM을 사용하여 기사들을 종합하여 이슈 정보를 생성합니다.
        
        Args:
            selected_news: 선택된 뉴스 리스트 (importance_score 기준 상위 5개)
            category: 카테고리 이름
        
        Returns:
            {title, summary, primary_tag, secondary_tags}
        """
        # importance_score 기준으로 정렬하여 상위 5개만 선택
        sorted_news = sorted(selected_news, key=lambda x: x.get('importance_score', 1.0), reverse=True)
        top_5_news = sorted_news[:5]
        
        # LLM으로 타이틀과 요약 생성
        llm_result = self._call_llm_summarizer(category, top_5_news)
        
        # 기본 태그 설정 (태그 생성 기능 제거)
        primary_tag = 'Unknown'
        secondary_tags = []
        
        return {
            'title': llm_result['generated_title'],
            'summary': llm_result['executive_summary'],
            'primary_tag': primary_tag,
            'secondary_tags': secondary_tags
        }
    
    def analyze_news(self, batch_size: int = 100, max_iterations: int = None) -> Dict[str, Any]:
        """
        분석되지 않은 뉴스를 배치 단위로 분석합니다. (레거시 메서드)
        카테고리별 순차 처리를 위해 analyze_news_by_category() 사용을 권장합니다.
        
        Args:
            batch_size: 한 번에 처리할 뉴스 수
            max_iterations: 최대 반복 횟수 (None이면 모든 뉴스 처리)
        
        Returns:
            분석 결과 통계
        """
        logger.info("=" * 50)
        logger.info("뉴스 분석 시작 (레거시 모드: 모든 카테고리 혼합 처리)")
        logger.info("=" * 50)
        
        total_stats = {
            'total_news': 0,
            'clusters_created': 0,
            'issues_created': 0,
            'noise_count': 0,
            'iterations': 0
        }
        
        iteration = 0
        
        while True:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                logger.info(f"최대 반복 횟수({max_iterations})에 도달하여 중단합니다.")
                break
            
            # 분석되지 않은 뉴스 가져오기
            unanalyzed_news = self.db_manager.get_unanalyzed_news(limit=batch_size)
            
            if not unanalyzed_news:
                logger.info("분석할 뉴스가 없습니다.")
                break
            
            logger.info(f"[반복 {iteration}] 분석 대상: {len(unanalyzed_news)}개 뉴스")
            
            # 배치 처리
            batch_result = self._process_batch(unanalyzed_news)
            
            # 통계 누적
            total_stats['total_news'] += batch_result['total_news']
            total_stats['clusters_created'] += batch_result['clusters_created']
            total_stats['issues_created'] += batch_result['issues_created']
            total_stats['noise_count'] += batch_result['noise_count']
            total_stats['iterations'] = iteration
            
            logger.info(f"[반복 {iteration}] 완료: {batch_result['issues_created']}개 이슈 생성")
        
        logger.info("=" * 50)
        logger.info(f"전체 분석 완료: {total_stats}")
        logger.info("=" * 50)
        
        return total_stats
    
    def analyze_news_by_category(self, category_name: str, batch_size: int = 100, max_iterations: int = None) -> Dict[str, Any]:
        """
        특정 카테고리의 분석되지 않은 뉴스를 배치 단위로 분석합니다.
        
        Args:
            category_name: 카테고리 이름
            batch_size: 한 번에 처리할 뉴스 수
            max_iterations: 최대 반복 횟수 (None이면 모든 뉴스 처리)
        
        Returns:
            분석 결과 통계
        """
        logger.info("=" * 50)
        logger.info(f"카테고리 '{category_name}' 분석 시작")
        logger.info("=" * 50)
        
        total_stats = {
            'total_news': 0,
            'clusters_created': 0,
            'issues_created': 0,
            'noise_count': 0,
            'iterations': 0
        }
        
        iteration = 0
        
        while True:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                logger.info(f"[{category_name}] 최대 반복 횟수({max_iterations})에 도달하여 중단합니다.")
                break
            
            # 특정 카테고리의 분석되지 않은 뉴스 가져오기
            unanalyzed_news = self.db_manager.get_unanalyzed_news_by_category(category_name, limit=batch_size)
            
            if not unanalyzed_news:
                logger.info(f"[{category_name}] 분석할 뉴스가 없습니다.")
                break
            
            logger.info(f"[{category_name}] [반복 {iteration}] 분석 대상: {len(unanalyzed_news)}개 뉴스")
            
            # 배치 처리
            batch_result = self._process_batch(unanalyzed_news)
            
            # 통계 누적
            total_stats['total_news'] += batch_result['total_news']
            total_stats['clusters_created'] += batch_result['clusters_created']
            total_stats['issues_created'] += batch_result['issues_created']
            total_stats['noise_count'] += batch_result['noise_count']
            total_stats['iterations'] = iteration
            
            logger.info(f"[{category_name}] [반복 {iteration}] 완료: {batch_result['issues_created']}개 이슈 생성")
        
        logger.info("=" * 50)
        logger.info(f"카테고리 '{category_name}' 분석 완료: {total_stats}")
        logger.info("=" * 50)
        
        return total_stats
    
    def _process_batch(self, unanalyzed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        배치 단위로 뉴스를 처리합니다.
        
        Args:
            unanalyzed_news: 분석할 뉴스 리스트
        
        Returns:
            배치 처리 결과 통계
        """
        
        # 피드백 로드 및 가중치 적용
        try:
            feedback_loader = FeedbackLoader()
            feedbacks = feedback_loader.get_all()
            
            if feedbacks:
                logger.info(f"사용자 피드백 {len(feedbacks)}개 로드 완료, 가중치 적용 중...")
                feedback_analyzer = FeedbackAnalyzer()
            else:
                feedback_analyzer = None
                logger.debug("사용자 피드백이 없습니다.")
        except Exception as e:
            logger.warning(f"피드백 로드 실패 (계속 진행): {e}")
            feedback_analyzer = None
            feedbacks = []
        
        # 임베딩 생성 및 저장
        embeddings_list = []
        news_with_embeddings = []
        
        for news in unanalyzed_news:
            try:
                vector_id = self._generate_vector_id(news['link'])
                
                # ChromaDB에서 기존 임베딩 확인
                existing_embedding = self._get_embedding_from_chroma(vector_id)
                
                if existing_embedding:
                    embedding = existing_embedding
                    logger.debug(f"기존 임베딩 사용: {news['title'][:50]}")
                else:
                    # 새 임베딩 생성
                    text = self._prepare_text_for_embedding(news)
                    embedding = self._get_embedding(text)
                    
                    # ChromaDB에 저장
                    self._store_embedding(
                        vector_id=vector_id,
                        embedding=embedding,
                        metadata={
                            'news_id': news['id'],
                            'title': news['title'][:200],
                            'url': news['link']
                        }
                    )
                    
                    # DB에 vector_id 업데이트
                    self.db_manager.update_news_vector_id(news['id'], vector_id)
                
                embeddings_list.append(embedding)
                news_with_embeddings.append(news)
            
            except Exception as e:
                logger.error(f"임베딩 생성/저장 실패 (뉴스 ID: {news.get('id')}): {e}")
                continue
        
        if not embeddings_list:
            logger.warning("임베딩이 생성되지 않았습니다")
            return {
                'total_news': len(unanalyzed_news),
                'clusters_created': 0,
                'issues_created': 0,
                'noise_count': 0
            }
        
        embeddings_array = np.array(embeddings_list)
        
        # 피드백 기반 점수 적용
        if feedback_analyzer and feedbacks:
            try:
                logger.info("피드백 기반 점수 적용 중...")
                news_with_embeddings = feedback_analyzer.apply_feedback_scores(
                    news_with_embeddings,
                    feedbacks,
                    news_embeddings=embeddings_array,
                    similarity_threshold=0.7,
                    max_importance_boost=0.15  # 최대 15% 상승
                )
                # 피드백 점수를 DB에 업데이트
                for news in news_with_embeddings:
                    if 'user_feedback_score' in news:
                        with self.db_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            # user_feedback_score와 feedback_applied_to_importance 업데이트
                            cursor.execute("""
                                UPDATE news 
                                SET user_feedback_score = ?,
                                    feedback_applied_to_importance = ?
                                WHERE id = ?
                            """, (
                                news.get('user_feedback_score', 0.0),
                                1 if news.get('feedback_applied_to_importance', False) else 0,
                                news['id']
                            ))
                            # apply_to_importance=True인 경우만 importance_score 업데이트
                            if news.get('feedback_applied_to_importance', False):
                                cursor.execute("""
                                    UPDATE news 
                                    SET importance_score = ? 
                                    WHERE id = ?
                                """, (news['importance_score'], news['id']))
                            conn.commit()
                logger.info("피드백 기반 점수 적용 완료")
            except Exception as e:
                logger.warning(f"피드백 점수 적용 실패 (계속 진행): {e}")
        
        # 카테고리별로 분리하여 클러스터링
        categories_dict = {}
        for i, news in enumerate(news_with_embeddings):
            category = news.get('category_name', 'Unknown')
            if category not in categories_dict:
                categories_dict[category] = {'indices': [], 'news': [], 'embeddings': []}
            categories_dict[category]['indices'].append(i)
            categories_dict[category]['news'].append(news)
            categories_dict[category]['embeddings'].append(embeddings_list[i])
        
        # 전체 결과 집계용
        clusters_created = 0
        issues_created = 0
        noise_count = 0
        
        # SheetExporter 초기화 (카테고리별 중간 저장용)
        from modules.sheet_exporter import SheetExporter
        sheet_exporter = SheetExporter()
        is_first_category = True  # 첫 번째 카테고리 여부 추적
        
        # 카테고리별로 클러스터링 및 이슈 생성 수행
        for category, category_data in categories_dict.items():
            logger.info(f"카테고리 '{category}' 처리 시작: {len(category_data['news'])}개 뉴스")
            
            category_embeddings = np.array(category_data['embeddings'])
            category_labels = self._cluster_news(category_embeddings, category=category)
            
            # 카테고리별 클러스터 수 계산
            unique_category_labels = set(category_labels)
            if -1 in unique_category_labels:
                unique_category_labels.remove(-1)
            category_cluster_count = len(unique_category_labels)
            category_noise_count = list(category_labels).count(-1)
            
            clusters_created += category_cluster_count
            noise_count += category_noise_count
            
            logger.info(f"카테고리 '{category}' 클러스터링 완료: {category_cluster_count}개 클러스터, {category_noise_count}개 노이즈")
            
            # 카테고리별 이슈 생성 및 처리
            category_issues_created, category_skipped_news_ids = self._process_category_clusters(
                category_data['news'],
                category_embeddings,
                category_labels,
                category
            )
            
            issues_created += category_issues_created
            
            # 카테고리별 매핑 데이터를 스프레드시트에 즉시 출력 (중간 저장)
            if sheet_exporter.spreadsheet:
                try:
                    self._export_category_mapping_to_sheet(sheet_exporter, category, is_first_category)
                    logger.info(f"카테고리 '{category}' 매핑 데이터를 스프레드시트에 저장했습니다.")
                except Exception as e:
                    logger.error(f"카테고리 '{category}' 매핑 데이터 저장 실패: {e}", exc_info=True)
            
            # 첫 번째 카테고리 처리 완료
            is_first_category = False
            
            # 메모리 최적화: 카테고리 처리 후 변수 정리
            del category_embeddings, category_labels, category_data
            import gc
            gc.collect()
            
            logger.info(f"카테고리 '{category}' 처리 완료: {category_issues_created}개 이슈 생성")
        
        # 전체 결과를 위한 labels는 더 이상 필요 없음 (카테고리별로 처리 완료)
        
        result = {
            'total_news': len(unanalyzed_news),
            'clusters_created': clusters_created,
            'issues_created': issues_created,
            'noise_count': noise_count
        }
        
        logger.info("=" * 50)
        logger.info(f"뉴스 분석 완료: {result}")
        logger.info("=" * 50)
        
        return result
    
    def _process_category_clusters(self, category_news: List[Dict[str, Any]], 
                                   category_embeddings: np.ndarray,
                                   category_labels: np.ndarray,
                                   category: str) -> Tuple[int, List[int]]:
        """
        카테고리별 클러스터를 처리하고 이슈를 생성합니다.
        
        Args:
            category_news: 카테고리별 뉴스 리스트
            category_embeddings: 카테고리별 임베딩 배열
            category_labels: 카테고리별 클러스터 라벨 배열
            category: 카테고리 이름
        
        Returns:
            (생성된 이슈 수, 건너뛴 뉴스 ID 리스트)
        """
        min_cluster_size = self.analysis_config.get('min_cluster_size', 3)
        max_cluster_size = self.analysis_config.get('max_cluster_size', 50)
        max_articles_per_cluster = self.analysis_config.get('max_articles_per_cluster', 5)
        
        issues_created = 0
        skipped_news_ids = []
        
        # 노이즈 처리: importance_score가 높은 단독 기사도 처리
        noise_indices = [i for i, label in enumerate(category_labels) if label == -1]
        high_importance_noise = []
        for idx in noise_indices:
            news = category_news[idx]
            importance = news.get('importance_score', 1.0)
            if importance >= 1.3:
                high_importance_noise.append(idx)
                logger.info(f"[{category}] 높은 중요도 노이즈 처리: {news['title'][:50]}... (점수: {importance:.2f})")
        
        # 처리되지 않은 노이즈(낮은 중요도)도 분석 완료로 표시
        remaining_noise_indices = [idx for idx in noise_indices if idx not in high_importance_noise]
        if remaining_noise_indices:
            remaining_noise_ids = [category_news[idx]['id'] for idx in remaining_noise_indices]
            skipped_news_ids.extend(remaining_noise_ids)
            logger.debug(f"[{category}] 낮은 중요도 노이즈 {len(remaining_noise_ids)}개를 분석 완료로 표시")
        
        # 높은 중요도 노이즈를 단독 이슈로 처리
        for idx in high_importance_noise:
            news = category_news[idx]
            try:
                issue_data = self._synthesize_and_tag([news], category=category)
                issue_id = self.db_manager.insert_issue(
                    title=issue_data['title'],
                    summary=issue_data['summary'],
                    primary_tag=issue_data['primary_tag'],
                    secondary_tags=issue_data['secondary_tags'],
                    cluster_id=-1
                )
                self.db_manager.link_issue_to_news(issue_id, [news['id']])
                self.db_manager.mark_news_analyzed([news['id']])
                issues_created += 1
                logger.info(f"[{category}] 단독 이슈 생성: {issue_data['title'][:50]}...")
            except Exception as e:
                logger.error(f"[{category}] 단독 이슈 생성 실패 (뉴스 ID: {news.get('id')}): {e}")
                skipped_news_ids.append(news['id'])
                continue
        
        # 클러스터별 처리
        unique_labels = set(category_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in unique_labels:
            cluster_indices = [i for i, label in enumerate(category_labels) if label == cluster_id]
            cluster_news = [category_news[i] for i in cluster_indices]
            cluster_embeddings = category_embeddings[cluster_indices]
            
            # 최소 크기 체크
            if len(cluster_news) < min_cluster_size:
                logger.debug(f"[{category}] 클러스터 {cluster_id}: 크기가 작아 건너뜀 ({len(cluster_news)}개)")
                skipped_news_ids.extend([news['id'] for news in cluster_news])
                continue
            
            # 큰 클러스터 분할
            if len(cluster_news) > max_cluster_size:
                sub_clusters = self._split_large_cluster(cluster_news, cluster_embeddings, max_cluster_size)
            else:
                sub_clusters = [list(range(len(cluster_news)))]
            
            # 각 서브클러스터 처리
            for sub_cluster_indices in sub_clusters:
                sub_cluster_news = [cluster_news[i] for i in sub_cluster_indices]
                sub_cluster_embeddings = cluster_embeddings[sub_cluster_indices]
                
                # MMR로 대표 기사 선택
                selected_indices = self._mmr_selection(
                    sub_cluster_news,
                    sub_cluster_embeddings,
                    max_articles_per_cluster
                )
                selected_news = [sub_cluster_news[i] for i in selected_indices]
                
                # importance_score 기준으로 정렬하여 상위 5개만 선택
                sorted_sub_cluster_news = sorted(sub_cluster_news, key=lambda x: x.get('importance_score', 1.0), reverse=True)
                top_5_news = sorted_sub_cluster_news[:5]
                
                # LLM으로 종합 및 태깅
                try:
                    issue_data = self._synthesize_and_tag(top_5_news, category=category)
                    issue_id = self.db_manager.insert_issue(
                        title=issue_data['title'],
                        summary=issue_data['summary'],
                        primary_tag=issue_data['primary_tag'],
                        secondary_tags=issue_data['secondary_tags'],
                        cluster_id=cluster_id
                    )
                    news_ids = [news['id'] for news in sub_cluster_news]
                    self.db_manager.link_issue_to_news(issue_id, news_ids)
                    self.db_manager.mark_news_analyzed(news_ids)
                    issues_created += 1
                    logger.info(f"[{category}] 이슈 생성: {issue_data['title'][:50]}...")
                except Exception as e:
                    logger.error(f"[{category}] 이슈 생성 실패 (클러스터 {cluster_id}): {e}")
                    failed_news_ids = [news['id'] for news in sub_cluster_news]
                    skipped_news_ids.extend(failed_news_ids)
                    continue
        
        # 처리되지 않은 뉴스들을 분석 완료로 표시
        if skipped_news_ids:
            self.db_manager.mark_news_analyzed(skipped_news_ids)
            logger.debug(f"[{category}] 처리되지 않은 뉴스 {len(skipped_news_ids)}개를 분석 완료로 표시")
        
        return issues_created, skipped_news_ids
    
    def _export_category_mapping_to_sheet(self, sheet_exporter, category: str, clear_existing: bool = False):
        """
        카테고리별 이슈-뉴스 매핑을 스프레드시트에 출력합니다.
        
        Args:
            sheet_exporter: SheetExporter 인스턴스
            category: 카테고리 이름
            clear_existing: 기존 데이터를 지울지 여부 (첫 번째 카테고리일 때만 True)
        """
        import pandas as pd
        import math
        
        with self.db_manager.get_connection() as conn:
            # 해당 카테고리의 이슈-뉴스 매핑 조회 (Source 포함)
            mapping_query = """
                SELECT 
                    n.category_name,
                    i.id as issue_id,
                    i.title as issue_title,
                    COUNT(*) OVER (PARTITION BY i.id) as number_of_news,
                    n.source,
                    n.title as news_title,
                    n.link
                FROM issue_news_mapping m
                JOIN issues i ON m.issue_id = i.id
                JOIN news n ON m.news_id = n.id
                WHERE n.category_name = ?
                ORDER BY i.id, n.importance_score DESC
            """
            df_mapping = pd.read_sql_query(mapping_query, conn, params=(category,))
            
            if df_mapping.empty:
                logger.debug(f"[{category}] 매핑 데이터가 없습니다.")
                return
            
            # 이슈별 점수 계산
            issue_scores = {}
            for issue_id in df_mapping['issue_id'].unique():
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        AVG(n.importance_score) as avg_importance,
                        COUNT(m.news_id) as news_count
                    FROM issues i
                    LEFT JOIN issue_news_mapping m ON i.id = m.issue_id
                    LEFT JOIN news n ON m.news_id = n.id
                    WHERE i.id = ?
                    GROUP BY i.id
                """, (int(issue_id),))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    avg_importance = result[0] or 1.0
                    news_count = result[1] or 0
                    score = (avg_importance ** 3) * math.log2(news_count + 1)
                    issue_scores[int(issue_id)] = score
                else:
                    issue_scores[int(issue_id)] = 0.0
            
            # 점수 컬럼 추가
            df_mapping['score'] = df_mapping['issue_id'].map(issue_scores)
            
            # 필요한 컬럼만 선택 (category_name, issue_title, number_of_news, score, source, news_title, link)
            df_mapping_final = df_mapping[['category_name', 'issue_title', 'number_of_news', 'score', 'source', 'news_title', 'link']].copy()
            df_mapping_final['number_of_news'] = df_mapping_final['number_of_news'].astype(int)
            
            # 카테고리별, 스코어별 정렬
            df_mapping_final = df_mapping_final.sort_values(
                by=['category_name', 'score'],
                ascending=[True, False],
                na_position='last'
            )
            
            # 스프레드시트에 출력 (첫 번째 카테고리일 때만 기존 데이터 지우기)
            sheet_name = '이슈-뉴스 매핑'
            success = sheet_exporter.export_to_sheet(
                df_mapping_final, 
                sheet_name=sheet_name,
                clear_existing=clear_existing,  # 첫 번째 카테고리일 때만 True
                append_mode=not clear_existing  # 첫 번째가 아니면 추가 모드
            )
            
            if success:
                logger.info(f"[{category}] 매핑 데이터 {len(df_mapping_final)}행을 스프레드시트에 추가했습니다.")
            else:
                logger.error(f"[{category}] 매핑 데이터 저장 실패")
    
    def _merge_similar_issues(self, issues_list: List[Dict[str, Any]], conn) -> List[Dict[str, Any]]:
        """
        유사한 이슈들을 병합합니다.
        이슈 제목 간 유사도가 0.85 이상인 경우, 점수가 더 높은 쪽으로 병합합니다.
        
        Args:
            issues_list: 이슈 리스트
            conn: 데이터베이스 연결
        
        Returns:
            병합된 이슈 리스트
        """
        if len(issues_list) <= 1:
            return issues_list
        
        logger.info(f"이슈 병합 시작: {len(issues_list)}개 이슈")
        
        # 이슈 제목 임베딩 생성
        issue_titles = [issue.get('title', '') for issue in issues_list]
        issue_embeddings = []
        
        for title in issue_titles:
            try:
                embedding = self._get_embedding(title)
                issue_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"이슈 제목 임베딩 생성 실패: {e}")
                # 실패 시 빈 임베딩 추가 (병합에서 제외됨)
                issue_embeddings.append(np.zeros(1536))  # text-embedding-3-small 차원
        
        if not issue_embeddings:
            return issues_list
        
        issue_embeddings = np.array(issue_embeddings)
        
        # 유사도 계산
        similarity_matrix = cosine_similarity(issue_embeddings)
        
        # 병합할 이슈 그룹 찾기
        merged_groups = []
        merged_indices = set()
        merge_threshold = 0.85
        
        for i in range(len(issues_list)):
            if i in merged_indices:
                continue
            
            group = [i]
            for j in range(i + 1, len(issues_list)):
                if j in merged_indices:
                    continue
                
                similarity = similarity_matrix[i][j]
                if similarity >= merge_threshold:
                    group.append(j)
                    merged_indices.add(j)
            
            if len(group) > 1:
                merged_indices.add(i)
                merged_groups.append(group)
        
        if not merged_groups:
            logger.info("병합할 유사 이슈가 없습니다.")
            return issues_list
        
        logger.info(f"{len(merged_groups)}개 이슈 그룹을 병합합니다.")
        
        # 병합 실행
        cursor = conn.cursor()
        merged_count = 0
        
        for group in merged_groups:
            # 그룹 내 이슈들을 점수 기준으로 정렬 (높은 점수가 메인)
            group_issues = [issues_list[i] for i in group]
            group_issues.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            main_issue = group_issues[0]
            merge_issues = group_issues[1:]
            
            # 병합할 이슈들의 뉴스 ID 수집
            merge_news_ids = []
            for merge_issue in merge_issues:
                cursor.execute("""
                    SELECT news_id FROM issue_news_mapping WHERE issue_id = ?
                """, (merge_issue['id'],))
                news_ids = [row['news_id'] for row in cursor.fetchall()]
                merge_news_ids.extend(news_ids)
            
            # 메인 이슈에 뉴스 연결
            if merge_news_ids:
                # 중복 제거
                merge_news_ids = list(set(merge_news_ids))
                
                # 메인 이슈에 이미 연결된 뉴스 확인
                cursor.execute("""
                    SELECT news_id FROM issue_news_mapping WHERE issue_id = ?
                """, (main_issue['id'],))
                existing_news_ids = set(row['news_id'] for row in cursor.fetchall())
                
                # 새로운 뉴스만 추가
                new_news_ids = [nid for nid in merge_news_ids if nid not in existing_news_ids]
                
                if new_news_ids:
                    self.db_manager.link_issue_to_news(main_issue['id'], new_news_ids)
                    logger.info(f"이슈 병합: '{main_issue['title'][:50]}...'에 {len(new_news_ids)}개 뉴스 추가")
            
            # 병합할 이슈들 삭제
            for merge_issue in merge_issues:
                # 이슈-뉴스 매핑 삭제
                cursor.execute("DELETE FROM issue_news_mapping WHERE issue_id = ?", (merge_issue['id'],))
                # 이슈 삭제
                cursor.execute("DELETE FROM issues WHERE id = ?", (merge_issue['id'],))
                merged_count += 1
            
            conn.commit()
        
        logger.info(f"이슈 병합 완료: {merged_count}개 이슈 병합됨")
        
        # 병합 후 이슈 목록 다시 조회
        import pandas as pd
        merged_issues_query = """
            SELECT 
                i.id,
                COALESCE(MAX(n.category_name), 'Unknown') as category_name,
                i.title,
                i.summary,
                i.created_at,
                i.cluster_id,
                AVG(n.importance_score) as avg_importance,
                COUNT(m.news_id) as news_count
            FROM issues i
            LEFT JOIN issue_news_mapping m ON i.id = m.issue_id
            LEFT JOIN news n ON m.news_id = n.id
            GROUP BY i.id, i.title, i.summary, i.created_at, i.cluster_id
            LIMIT 1000
        """
        df_merged = pd.read_sql_query(merged_issues_query, conn)
        
        if df_merged.empty:
            return issues_list
        
        # 스코어 계산
        import math
        df_merged['score'] = df_merged.apply(
            lambda row: (row['avg_importance'] ** 3) * math.log2(row['news_count'] + 1) 
            if row['avg_importance'] is not None and row['news_count'] is not None else 0.0,
            axis=1
        )
        # 카테고리별, 스코어별 정렬
        df_merged = df_merged.sort_values(
            by=['category_name', 'score'],
            ascending=[True, False],
            na_position='last'
        )
        
        merged_issues = df_merged.to_dict('records')
        return merged_issues


def main():
    """테스트용 메인 함수"""
    import sys
    import io
    from pathlib import Path
    
    # Windows 콘솔 인코딩 설정
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # 프로젝트 루트를 Python 경로에 추가
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 분석기 생성 및 실행
    try:
        analyzer = NewsAnalyzer()
        result = analyzer.analyze_news(batch_size=100)
        
        print("\n" + "=" * 50)
        print("분석 결과")
        print("=" * 50)
        print(f"  총 뉴스: {result['total_news']}개")
        print(f"  클러스터: {result['clusters_created']}개")
        print(f"  이슈 생성: {result['issues_created']}개")
        print(f"  노이즈: {result['noise_count']}개")
        print("=" * 50)
    except Exception as e:
        logging.error(f"분석 실패: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

