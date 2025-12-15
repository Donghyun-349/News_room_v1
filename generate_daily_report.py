"""
Daily Market Executive Report Generator
금융 데이터 분석 및 자동화 리포팅 파이프라인
"""
import os
import sys
import io
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from config import get_config
from database import DatabaseManager
from modules.prompt_loader import PromptLoader
from modules.feedback_loader import FeedbackLoader
from modules.feedback_analyzer import FeedbackAnalyzer
import logging

logger = logging.getLogger(__name__)

# [수정] 템플릿 포맷팅 시 키가 없어도 에러가 나지 않도록 하는 안전한 딕셔너리
class SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # 데이터가 없으면 {키이름} 그대로 문자로 출력


class DailyReportGenerator:
    """일일 시장 리포트 생성기"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = DatabaseManager()
        self.llm_config = self.config.get('llm', {})
        self.llm_provider = self.llm_config.get('provider', 'openai')
        self.prompt_loader = PromptLoader()  # 프롬프트 로더 초기화
        
        # LLM 클라이언트 초기화
        if self.llm_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
            else:
                self.llm_client = None
                print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다. Mock 모드로 실행됩니다.")
        elif self.llm_provider == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel(self.llm_config.get('model', 'gemini-2.0-flash'))
            else:
                self.llm_client = None
                print("⚠️  GEMINI_API_KEY가 설정되지 않았습니다. Mock 모드로 실행됩니다.")
        else:
            self.llm_client = None
            print("⚠️  LLM 클라이언트를 초기화할 수 없습니다. Mock 모드로 실행됩니다.")
    
    def load_data(self, category_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Step 1: 데이터 전처리
        이슈-뉴스 매핑 데이터를 로드하고 전처리합니다.
        
        Args:
            category_filter: 필터링할 카테고리 리스트 (None이면 모든 카테고리)
        
        Returns:
            필터링된 데이터프레임
        """
        print("=" * 80)
        print("Step 1: 데이터 로드 및 전처리")
        if category_filter:
            print(f"카테고리 필터: {', '.join(category_filter)}")
        print("=" * 80)
        
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT 
                    n.category_name,
                    i.id as issue_id,
                    i.title as issue_title,
                    m.news_id,
                    n.title as news_title,
                    n.source,
                    n.link,
                    n.importance_score,
                    n.user_feedback_score,
                    n.feedback_applied_to_importance,
                    n.published_at,
                    n.created_at
                FROM issue_news_mapping m
                JOIN issues i ON m.issue_id = i.id
                JOIN news n ON m.news_id = n.id
            """
            
            params = []
            if category_filter:
                placeholders = ','.join(['?' for _ in category_filter])
                query += f" WHERE n.category_name IN ({placeholders})"
                params = category_filter
            
            query += " ORDER BY i.id, n.importance_score DESC"
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            if category_filter:
                raise ValueError(f"필터링된 카테고리({', '.join(category_filter)})에 대한 분석할 데이터가 없습니다.")
            else:
                raise ValueError("분석할 데이터가 없습니다.")
        
        print(f"✅ 총 {len(df)}개 매핑 데이터 로드 완료")
        print(f"   - 카테고리: {df['category_name'].unique().tolist()}")
        print(f"   - 이슈 수: {df['issue_id'].nunique()}개")
        print()
        
        return df
    
    def preprocess_clusters(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Step 1: 클러스터별 상위 5개 뉴스 추출
        - 클러스터 내 기사 선택: importance_score 기준
        - 섹터 분석 상위 클러스터 선정: 클러스터링 스코어 기준 (avg_importance³ × log₂(news_count + 1))
        """
        print("클러스터별 상위 5개 뉴스 추출 중...")
        
        clusters = {}
        
        # category_name과 issue_title 기준으로 그룹핑
        for (category, issue_title), group in df.groupby(['category_name', 'issue_title']):
            cluster_key = f"{category}::{issue_title}"
            issue_id = group.iloc[0]['issue_id']
            
            # 클러스터 내 기사 선택: importance_score 기준 내림차순 정렬
            group_sorted = group.sort_values('importance_score', ascending=False)
            
            # 상위 5개 추출 (importance_score 기준)
            top_5 = group_sorted.head(5)
            
            # 클러스터 데이터 구성
            cluster_data = {
                'category_name': category,
                'issue_title': issue_title,
                'issue_id': issue_id,
                'top_5_news': []
            }
            
            # 클러스터링 스코어 계산: (avg_importance³ × log₂(news_count + 1))
            # 전체 클러스터 뉴스 기준으로 계산 (상위 5개가 아닌 전체)
            avg_importance = float(group['importance_score'].mean())
            news_count = len(group)
            cluster_data['score'] = (avg_importance ** 3) * math.log2(news_count + 1)
            
            # 상위 5개 뉴스 정보 (importance_score 기준)
            for _, row in top_5.iterrows():
                news_item = {
                    'title': row['news_title'],
                    'source': row['source'],
                    'link': row['link'],
                    'importance_score': float(row['importance_score']),
                    'user_feedback_score': float(row.get('user_feedback_score', 0.0)) if pd.notna(row.get('user_feedback_score')) else 0.0,
                    'feedback_applied_to_importance': bool(row.get('feedback_applied_to_importance', False)) if pd.notna(row.get('feedback_applied_to_importance')) else False,
                    'published_at': row['published_at'] if pd.notna(row['published_at']) else None,
                    'created_at': row['created_at'] if pd.notna(row['created_at']) else None
                }
                cluster_data['top_5_news'].append(news_item)
            
            clusters[cluster_key] = cluster_data
        
        print(f"✅ {len(clusters)}개 클러스터 생성 완료")
        print()
        
        return clusters
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze_cluster(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Micro-Analysis (Cluster 단위 분석)
        각 클러스터에 대해 LLM을 호출하여 분석합니다.
        """
        top_5_news = cluster_data['top_5_news']
        category = cluster_data['category_name']
        
        # 뉴스 정보 포맷팅
        news_text = ""
        for i, news in enumerate(top_5_news, 1):
            date_str = "25.12.14"  # 기본값
            if news.get('published_at'):
                try:
                    if isinstance(news['published_at'], str):
                        dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                    else:
                        dt = news['published_at']
                    date_str = dt.strftime("%y.%m.%d")
                except:
                    pass
            
            news_text += f"{i}. 제목: {news['title']}\n"
            news_text += f"   출처: {news['source']}\n"
            news_text += f"   링크: {news['link']}\n"
            news_text += f"   날짜: {date_str}\n"
            news_text += f"   중요도: {news['importance_score']:.2f}\n\n"
        
        # 프롬프트 로더에서 프롬프트 가져오기
        default_prompt = """당신은 전문 금융 분석가입니다. 아래 5개의 뉴스 기사를 분석하여 JSON 형식으로 결과를 제공해주세요.

카테고리: {category}

뉴스 기사:
{news_text}

다음 JSON 구조로 응답해주세요:
{{
    "new_title": "5개 기사를 아우르는 통찰력 있는 대표 제목 (수치 포함, 건조한 분석가 톤)",
    "fact_check_analyst_view": "이슈의 발생 배경, 주요 수치(금액, 지수 등), 시장 영향력을 포함한 3~4문장의 핵심 요약",
    "selected_links": [
        "상위 3~4개 기사를 선정하여 아래 포맷으로 변환",
        "영어 기사: [yy.mm.dd] <한글 번역 제목> - [<원문 제목>](<링크>)",
        "한국어 기사: [yy.mm.dd] <기사 제목> - [링크](<링크>)"
    ]
}}

응답은 반드시 유효한 JSON 형식이어야 하며, 다른 설명 없이 JSON만 반환해주세요."""
        
        default_system = "당신은 전문 금융 분석가입니다. 객관적이고 건조한 톤으로 분석 결과를 제공합니다."
        
        prompt_data = self.prompt_loader.get_prompt(
            'micro_analysis',
            default_prompt=default_prompt,
            default_system=default_system,
            category=category,
            news_text=news_text
        )
        
        prompt = prompt_data['prompt']
        system_prompt = prompt_data.get('system_prompt') or default_system

        # LLM 호출
        if self.llm_client is None:
            # Mock 응답
            return {
                'new_title': f"{cluster_data['issue_title']} (분석 필요)",
                'fact_check_analyst_view': "LLM API 키가 설정되지 않아 분석을 수행할 수 없습니다. 실제 API 키를 설정해주세요.",
                'selected_links': [
                    f"[25.12.14] {news['title']} - [{news['title']}]({news['link']})"
                    for news in top_5_news[:3]
                ]
            }
        
        try:
            if self.llm_provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=self.llm_config.get('model', 'gpt-4'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.get('temperature', 0.3),
                    max_tokens=self.llm_config.get('max_tokens', 2000),
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
            elif self.llm_provider == 'gemini':
                response = self.llm_client.generate_content(
                    f"{prompt}\n\n응답은 반드시 유효한 JSON 형식이어야 합니다.",
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.llm_config.get('temperature', 0.3),
                        max_output_tokens=self.llm_config.get('max_tokens', 2000),
                    )
                )
                result_text = response.text
            
            # JSON 파싱
            # JSON 코드 블록 제거
            result_text = result_text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            # selected_links 포맷팅
            formatted_links = []
            for news in top_5_news[:4]:  # 상위 4개
                date_str = "25.12.14"
                if news.get('published_at'):
                    try:
                        if isinstance(news['published_at'], str):
                            dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                        else:
                            dt = news['published_at']
                        date_str = dt.strftime("%y.%m.%d")
                    except:
                        pass
                
                # 언어 감지 (간단한 휴리스틱)
                is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in news['title'])
                
                if is_korean:
                    formatted_links.append(f"[{date_str}] {news['title']} - [링크]({news['link']})")
                else:
                    formatted_links.append(f"[{date_str}] {news['title']} - [{news['title']}]({news['link']})")
            
            result['selected_links'] = formatted_links
            
            return result
            
        except Exception as e:
            print(f"⚠️  LLM 호출 실패: {e}")
            # Fallback
            return {
                'new_title': cluster_data['issue_title'],
                'fact_check_analyst_view': "LLM 분석 중 오류가 발생했습니다.",
                'selected_links': [
                    f"[25.12.14] {news['title']} - [{news['title']}]({news['link']})"
                    for news in top_5_news[:3]
                ]
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def consolidate_themes(self, analyzed_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Step 2: Theme Consolidation (핵심 단계)
        파편화된 이슈들을 하나의 거대한 테마로 병합합니다.
        """
        print("=" * 80)
        print("Step 2: Theme Consolidation (테마 통합)")
        print("=" * 80)
        
        # 분석 결과를 카테고리별로 그룹핑
        categories = {}
        for result in analyzed_results:
            category = result['category_name']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # 카테고리별로 점수 기준 정렬
        for category in categories:
            categories[category].sort(key=lambda x: x['score'], reverse=True)
        
        # 카테고리별 테마 통합
        consolidated_themes = {}
        
        for category, results in categories.items():
            print(f"\n[{category}] 테마 통합 중...")
            
            # LLM 프롬프트 구성
            analysis_list = ""
            for i, result in enumerate(results, 1):
                analysis_list += f"{i}. 제목: {result['new_title']}\n"
                analysis_list += f"   요약: {result['fact_check_analyst_view']}\n"
                analysis_list += f"   점수: {result['score']:.2f}\n\n"
            
            # 프롬프트 로더에서 프롬프트 가져오기
            default_prompt = """당신은 전문 금융 분석가입니다. 아래 {category} 카테고리의 분석 결과들을 검토하여 테마를 통합해주세요.

[지시사항]
1. **Grouping:** 내용이 유사한 하위 이슈들을 하나의 메인 테마(Main Theme)로 묶어주세요.
   - 예) "연준 금리 인하", "파월 발언", "매파적 인하", "3회 연속 인하" -> [Theme 1: 미 연준 금리 인하와 향후 정책 경로]로 통합
   - 예) "코스피 기관 매수", "네 마녀의 날", "4160선 회복" -> [Theme 2: 네 마녀의 날 수급 공방과 기관의 방어]로 통합

2. **Filtering:** 가장 중요한 Top 2~3개의 메인 테마만 선별하세요. (점수가 낮거나 자잘한 이슈는 과감히 제외하거나 메인 테마의 근거로 편입)

3. **No Repetition:** 절대 같은 사건(예: 금리 인하)을 두 개의 섹션으로 나누어 쓰지 마세요. 무조건 하나로 합치세요.

[분석 결과]
{analysis_list}

다음 JSON 구조로 응답해주세요:
{{
    "themes": [
        {{
            "theme_title": "메인 테마 제목",
            "related_issue_indices": [1, 3, 5],
            "deep_dive": "통합된 내용을 바탕으로 한 상세 분석 (배경, 수치, 전망 포함 5~6문장)"
        }}
    ]
}}

응답은 반드시 유효한 JSON 형식이어야 하며, 다른 설명 없이 JSON만 반환해주세요."""
            
            default_system = "당신은 전문 금융 분석가입니다. 객관적이고 건조한 톤으로 분석 결과를 제공합니다."
            
            prompt_data = self.prompt_loader.get_prompt(
                'theme_consolidation',
                default_prompt=default_prompt,
                default_system=default_system,
                category=category,
                analysis_list=analysis_list
            )
            
            prompt = prompt_data['prompt']
            system_prompt = prompt_data.get('system_prompt') or default_system
            
            # LLM 호출
            if self.llm_client is None:
                # Mock: 상위 2개만 선택
                top_2 = results[:2]
                themes = []
                for result in top_2:
                    themes.append({
                        'theme_title': result['new_title'],
                        'related_issue_indices': [results.index(result)],
                        'deep_dive': result['fact_check_analyst_view']
                    })
                consolidated_themes[category] = themes
            else:
                try:
                    if self.llm_provider == 'openai':
                        response = self.llm_client.chat.completions.create(
                            model=self.llm_config.get('model', 'gpt-4'),
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=self.llm_config.get('temperature', 0.3),
                            max_tokens=self.llm_config.get('max_tokens', 2000),
                            response_format={"type": "json_object"}
                        )
                        result_text = response.choices[0].message.content
                    elif self.llm_provider == 'gemini':
                        response = self.llm_client.generate_content(
                            f"{prompt}\n\n응답은 반드시 유효한 JSON 형식이어야 합니다.",
                            generation_config=genai.types.GenerationConfig(
                                temperature=self.llm_config.get('temperature', 0.3),
                                max_output_tokens=self.llm_config.get('max_tokens', 2000),
                            )
                        )
                        result_text = response.text
                    
                    # JSON 파싱
                    result_text = result_text.strip()
                    if result_text.startswith("```json"):
                        result_text = result_text[7:]
                    if result_text.startswith("```"):
                        result_text = result_text[3:]
                    if result_text.endswith("```"):
                        result_text = result_text[:-3]
                    result_text = result_text.strip()
                    
                    consolidation_result = json.loads(result_text)
                    themes = consolidation_result.get('themes', [])
                    
                    # 인덱스를 실제 결과로 변환
                    consolidated_list = []
                    for theme in themes[:3]:  # 최대 3개
                        related_results = []
                        for idx in theme.get('related_issue_indices', []):
                            if 0 <= idx - 1 < len(results):  # 1-based to 0-based
                                related_results.append(results[idx - 1])
                        
                        if related_results:
                            # 가장 높은 점수의 결과를 대표로 사용
                            main_result = max(related_results, key=lambda x: x['score'])
                            consolidated_list.append({
                                'theme_title': theme.get('theme_title', main_result['new_title']),
                                'deep_dive': theme.get('deep_dive', main_result['fact_check_analyst_view']),
                                'related_results': related_results,
                                'score': main_result['score']
                            })
                    
                    consolidated_themes[category] = consolidated_list
                    print(f"   ✅ {len(consolidated_list)}개 메인 테마 생성")
                    
                except Exception as e:
                    print(f"⚠️  테마 통합 실패: {e}, 상위 2개만 선택합니다.")
                    # Fallback: 상위 2개만 선택
                    top_2 = results[:2]
                    consolidated_list = []
                    for result in top_2:
                        consolidated_list.append({
                            'theme_title': result['new_title'],
                            'deep_dive': result['fact_check_analyst_view'],
                            'related_results': [result],
                            'score': result['score']
                        })
                    consolidated_themes[category] = consolidated_list
        
        print()
        return consolidated_themes
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_final_report(self, consolidated_themes: Dict[str, List[Dict[str, Any]]], 
                             sections: Optional[List[str]] = None) -> str:
        """
        Step 3: Final Report Generation
        통합된 테마를 바탕으로 최종 보고서를 생성합니다.
        
        Args:
            consolidated_themes: 통합된 테마 딕셔너리
            sections: 사용할 섹션 ID 리스트 (None이면 모든 섹션 사용)
        """
        print("=" * 80)
        print("Step 3: 최종 보고서 생성")
        print("=" * 80)
        
        # 전체 테마 요약 텍스트 생성
        themes_summary = ""
        for category, themes in consolidated_themes.items():
            themes_summary += f"\n## {category}\n\n"
            for theme in themes:
                themes_summary += f"**{theme['theme_title']}**\n"
                themes_summary += f"{theme['deep_dive']}\n\n"
        
        # 프롬프트 로더에서 프롬프트 가져오기
        default_prompt = """당신은 전문 금융 분석가입니다. 아래 통합된 테마 분석 결과를 종합하여 Executive Summary와 Investor Note를 작성해주세요.

통합된 테마 분석:
{themes_summary}

다음 JSON 구조로 응답해주세요:
{{
    "executive_summary": {{
        "global": "글로벌 시장 관점에서 핵심 1줄 요약",
        "korea": "한국 시장 관점에서 핵심 1줄 요약",
        "key_indicator": "주요 지표 관점에서 핵심 1줄 요약"
    }},
    "investor_note": {{
        "caution": "경계해야 할 리스크 요약 (2~3문장)",
        "action": "대응 전략 제언 (2~3문장)"
    }}
}}

응답은 반드시 유효한 JSON 형식이어야 하며, 다른 설명 없이 JSON만 반환해주세요."""
        
        default_system = "당신은 전문 금융 분석가입니다. 객관적이고 건조한 톤으로 분석 결과를 제공합니다."
        
        prompt_data = self.prompt_loader.get_prompt(
            'final_report',
            default_prompt=default_prompt,
            default_system=default_system,
            themes_summary=themes_summary
        )
        
        prompt = prompt_data['prompt']
        system_prompt = prompt_data.get('system_prompt') or default_system

        # LLM 호출
        if self.llm_client is None:
            # Mock 응답
            executive_summary = {
                'global': '글로벌 시장 분석 필요',
                'korea': '한국 시장 분석 필요',
                'key_indicator': '주요 지표 분석 필요'
            }
            investor_note = {
                'caution': 'LLM API 키가 설정되지 않아 상세 분석을 수행할 수 없습니다.',
                'action': '실제 API 키를 설정하여 리포트를 생성해주세요.'
            }
        else:
            try:
                if self.llm_provider == 'openai':
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_config.get('model', 'gpt-4'),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.llm_config.get('temperature', 0.3),
                        max_tokens=self.llm_config.get('max_tokens', 2000),
                        response_format={"type": "json_object"}
                    )
                    result_text = response.choices[0].message.content
                elif self.llm_provider == 'gemini':
                    response = self.llm_client.generate_content(
                        f"{prompt}\n\n응답은 반드시 유효한 JSON 형식이어야 합니다.",
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.llm_config.get('temperature', 0.3),
                            max_output_tokens=self.llm_config.get('max_tokens', 2000),
                        )
                    )
                    result_text = response.text
                
                # JSON 파싱
                result_text = result_text.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()
                
                result = json.loads(result_text)
                executive_summary = result.get('executive_summary', {})
                investor_note = result.get('investor_note', {})
                
            except Exception as e:
                print(f"⚠️  최종 리포트 생성 실패: {e}")
                executive_summary = {
                    'global': '분석 중 오류 발생',
                    'korea': '분석 중 오류 발생',
                    'key_indicator': '분석 중 오류 발생'
                }
                investor_note = {
                    'caution': 'LLM 분석 중 오류가 발생했습니다.',
                    'action': '데이터를 확인하고 다시 시도해주세요.'
                }
        
        # 스프레드시트에서 보고서 양식 로드 시도
        from modules.settings_loader import SettingsLoader
        settings_loader = SettingsLoader()
        report_template = settings_loader.get_report_template()
        
        # 스프레드시트 양식이 있고 sections가 지정된 경우
        if report_template and sections:
            return self._generate_report_with_template(
                report_template, sections, consolidated_themes, 
                executive_summary, investor_note
            )
        elif report_template:
            # sections가 없으면 모든 섹션 사용
            all_sections = sorted(report_template.keys(), 
                                 key=lambda x: report_template[x].get('order', 999))
            return self._generate_report_with_template(
                report_template, all_sections, consolidated_themes,
                executive_summary, investor_note
            )
        else:
            # 기본 템플릿 사용 (기존 코드)
            return self._generate_report_with_default_template(
                consolidated_themes, executive_summary, investor_note
            )
    
    def _generate_report_with_template(self, template: Dict[str, Dict[str, Any]], 
                                      sections: List[str],
                                      consolidated_themes: Dict[str, List[Dict[str, Any]]],
                                      executive_summary: Dict[str, str],
                                      investor_note: Dict[str, str]) -> str:
        """스프레드시트 템플릿을 사용하여 보고서 생성 (SafeDict 적용)"""
        report_parts = []
        date_short = datetime.now().strftime("%Y.%m.%d")
        generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # sections를 order 기준으로 정렬
        sorted_sections = sorted(
            [s for s in sections if s in template],
            key=lambda x: template[x].get('order', 999)
        )
        
        i = 0
        while i < len(sorted_sections):
            section_id = sorted_sections[i]
            section_data = template[section_id]
            template_text = section_data['template']
            
            # [수정] 각 섹션별로 필요한 데이터를 딕셔너리로 만들고 SafeDict로 감싸서 format_map 사용
            if section_id == 'a':  # header
                context = {'date_short': date_short}
                report_parts.append(template_text.format_map(SafeDict(context)) + "\n\n")
                
            elif section_id == 'b':  # executive_summary
                context = {
                    'executive_summary_global': executive_summary.get('global', 'N/A'),
                    'executive_summary_korea': executive_summary.get('korea', 'N/A'),
                    'executive_summary_key_indicator': executive_summary.get('key_indicator', 'N/A')
                }
                report_parts.append(template_text.format_map(SafeDict(context)) + "\n\n")
                
            elif section_id == 'c':  # sector_analysis_header
                report_parts.append(template_text)
                
            elif section_id == 'd':  # category_header
                # 카테고리별로 반복
                for category, themes in consolidated_themes.items():
                    context = {'category': category}
                    report_parts.append(template_text.format_map(SafeDict(context)) + "\n")
                    # 다음 섹션들도 처리 (e, f, g)
                    category_parts, i = self._process_category_sections(
                        template, sorted_sections, themes, 
                        i, date_short
                    )
                    report_parts.extend(category_parts)
                continue  # continue로 다음 루프로
                
            elif section_id == 'h':  # investor_note
                context = {
                    'investor_note_caution': investor_note.get('caution', 'N/A'),
                    'investor_note_action': investor_note.get('action', 'N/A')
                }
                report_parts.append(template_text.format_map(SafeDict(context)) + "\n\n")
                
            elif section_id == 'i':  # footer
                context = {'generated_time': generated_time}
                report_parts.append(template_text.format_map(SafeDict(context)))
            
            i += 1
        
        return "".join(report_parts)
    
    def _process_category_sections(self, template: Dict, 
                                   sections: List[str], themes: List[Dict],
                                   current_idx: int, date_short: str) -> tuple:
        """카테고리 내부 섹션 처리 (e, f, g) - (생성된 부분 리스트, 다음 인덱스) 반환"""
        category_sections = ['e', 'f', 'g']
        idx = current_idx + 1
        parts = []
        
        for theme in themes:
            # e, f, g 섹션 처리
            for section_id in category_sections:
                if idx >= len(sections) or sections[idx] != section_id:
                    continue
                    
                if section_id == 'e':  # theme_section
                    context = {
                        'theme_title': theme['theme_title'],
                        'deep_dive': theme['deep_dive']
                    }
                    parts.append(template[section_id]['template'].format_map(SafeDict(context)) + "\n")
                    idx += 1
                    
                elif section_id == 'f':  # key_news
                    key_news_list = self._format_key_news(theme, date_short)
                    if key_news_list:
                        context = {'key_news_list': key_news_list}
                        parts.append(template[section_id]['template'].format_map(SafeDict(context)) + "\n")
                    idx += 1
                    
                elif section_id == 'g':  # feedback_section
                    feedback_list = self._format_feedback_news(theme, date_short)
                    if feedback_list:
                        context = {'feedback_news_list': feedback_list}
                        parts.append(template[section_id]['template'].format_map(SafeDict(context)) + "\n")
                    idx += 1
        
        return parts, idx
    
    def _format_key_news(self, theme: Dict, date_short: str) -> str:
        """Key News 포맷팅"""
        all_news = []
        for result in theme['related_results']:
            if 'top_5_news' in result:
                for news in result.get('top_5_news', []):
                    all_news.append(news)
        
        if not all_news:
            return ""
        
        all_news.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        top_news = all_news[:4]
        
        news_list = []
        for news in top_news:
            date_str = "25.12.14"
            if news.get('published_at'):
                try:
                    if isinstance(news['published_at'], str):
                        dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                    else:
                        dt = news['published_at']
                    date_str = dt.strftime("%y.%m.%d")
                except:
                    pass
            
            is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in news['title'])
            
            if is_korean:
                news_list.append(f"- [{date_str}] {news['title']} - [링크]({news['link']})")
            else:
                korean_title = news['title']  # 실제로는 LLM 번역 필요
                news_list.append(f"- [{date_str}] {korean_title} - [{news['title']}]({news['link']})")
        
        return "\n".join(news_list)
    
    def _format_feedback_news(self, theme: Dict, date_short: str) -> str:
        """Feedback News 포맷팅"""
        all_news = []
        for result in theme['related_results']:
            if 'top_5_news' in result:
                for news in result.get('top_5_news', []):
                    all_news.append(news)
        
        if not all_news:
            return ""
        
        all_news.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        top_news = all_news[:4]
        
        # 피드백 점수가 있는 뉴스 별도 수집
        feedback_news = []
        top_news_ids = {news.get('link', '') for news in top_news}
        for news in all_news:
            if news.get('user_feedback_score', 0.0) > 0.0:
                if news.get('link', '') not in top_news_ids:
                    feedback_news.append(news)
        
        if not feedback_news:
            return ""
        
        feedback_news.sort(key=lambda x: x.get('user_feedback_score', 0.0), reverse=True)
        feedback_news = feedback_news[:2]
        
        # 피드백 로드
        feedbacks = []
        try:
            feedback_loader = FeedbackLoader()
            feedbacks = feedback_loader.get_all()
        except Exception as e:
            logger.warning(f"피드백 로드 실패: {e}")
        
        if not feedbacks:
            return ""
        
        feedback_list = []
        feedback_analyzer = FeedbackAnalyzer()
        
        for news in feedback_news:
            date_str = "25.12.14"
            if news.get('published_at'):
                try:
                    if isinstance(news['published_at'], str):
                        dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                    else:
                        dt = news['published_at']
                    date_str = dt.strftime("%y.%m.%d")
                except:
                    pass
            
            is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in news['title'])
            
            if is_korean:
                news_line = f"- [{date_str}] {news['title']} - [링크]({news['link']})"
            else:
                korean_title = news['title']
                news_line = f"- [{date_str}] {korean_title} - [{news['title']}]({news['link']})"
            
            # 매칭된 피드백 코멘트
            news_text = f"{news.get('title', '')} {news.get('snippet', '')}"
            news_embedding = feedback_analyzer.generate_embedding(news_text)
            
            matched_comments = []
            if news_embedding is not None:
                for feedback in feedbacks:
                    feedback_text = f"{feedback.get('news_title', '')} {feedback.get('user_comment', '')}"
                    feedback_embedding = feedback_analyzer.generate_embedding(feedback_text)
                    if feedback_embedding is not None:
                        similarity = feedback_analyzer.calculate_similarity(
                            feedback_embedding, 
                            np.array([news_embedding])
                        )[0]
                        if similarity >= 0.7:
                            matched_comments.append(feedback.get('user_comment', ''))
            
            feedback_list.append(news_line)
            if matched_comments:
                feedback_list.append("  \n  **💬 사용자 피드백:**")
                for comment in matched_comments:
                    feedback_list.append(f"  - {comment}")
        
        return "\n".join(feedback_list)
    
    def _generate_report_with_default_template(self, consolidated_themes: Dict[str, List[Dict[str, Any]]],
                                               executive_summary: Dict[str, str],
                                               investor_note: Dict[str, str]) -> str:
        """기본 템플릿 사용 (현재 코드 그대로)"""
        # Markdown 리포트 생성
        today = datetime.now().strftime("%Y년 %m월 %d일")
        date_short = datetime.now().strftime("%Y.%m.%d")
        report = f"""# 📅 Daily Market Executive Report

Date: {date_short}

## Executive Summary

- **Global:** {executive_summary.get('global', 'N/A')}
- **Korea:** {executive_summary.get('korea', 'N/A')}
- **Key Indicator:** {executive_summary.get('key_indicator', 'N/A')}

---

## Sector Analysis

"""
        
        # 카테고리별 섹션 작성
        for category, themes in consolidated_themes.items():
            report += f"### {category}\n\n"
            
            for theme in themes:
                # Main Theme Title
                report += f"#### {theme['theme_title']}\n\n"
                
                # Deep Dive
                report += f"**Deep Dive:**\n{theme['deep_dive']}\n\n"
                
                # Key News: importance_score 기준 상위 3~4개만 선택
                all_news = []
                for result in theme['related_results']:
                    # 원본 뉴스 데이터 수집
                    if 'top_5_news' in result:
                        for news in result.get('top_5_news', []):
                            all_news.append(news)
                
                # importance_score 기준 정렬 및 상위 3~4개 선택
                if all_news:
                    all_news.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
                    top_news = all_news[:4]  # 상위 4개
                    
                    # 피드백 점수가 있는 뉴스 별도 수집 (중복 제거)
                    feedback_news = []
                    top_news_ids = {news.get('link', '') for news in top_news}
                    for news in all_news:
                        if news.get('user_feedback_score', 0.0) > 0.0:
                            # Key News에 포함되지 않은 피드백 뉴스만 추가
                            if news.get('link', '') not in top_news_ids:
                                feedback_news.append(news)
                    
                    # 피드백 뉴스를 user_feedback_score 기준으로 정렬하고 상위 2개만 선택
                    feedback_news.sort(key=lambda x: x.get('user_feedback_score', 0.0), reverse=True)
                    feedback_news = feedback_news[:2]
                    
                    # 피드백 로드
                    feedback_loader = None
                    feedbacks = []
                    try:
                        feedback_loader = FeedbackLoader()
                        feedbacks = feedback_loader.get_all()
                    except Exception as e:
                        logger.warning(f"피드백 로드 실패: {e}")
                    
                    report += "**📰 Key News:**\n"
                    for news in top_news:
                        date_str = "25.12.14"
                        if news.get('published_at'):
                            try:
                                if isinstance(news['published_at'], str):
                                    dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                                else:
                                    dt = news['published_at']
                                date_str = dt.strftime("%y.%m.%d")
                            except:
                                pass
                        
                        # 언어 감지
                        is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in news['title'])
                        
                        if is_korean:
                            # 한국어 기사: [날짜] <기사 제목> - [링크](<링크>)
                            report += f"- [{date_str}] {news['title']} - [링크]({news['link']})\n"
                        else:
                            # 영어 기사: [날짜] <한글 번역 제목> - [<원문 제목>](<링크>)
                            korean_title = news['title']  # 실제로는 LLM 번역 필요
                            report += f"- [{date_str}] {korean_title} - [{news['title']}]({news['link']})\n"
                    
                    # 피드백 뉴스가 있으면 별도 섹션 추가
                    if feedback_news and feedbacks:
                        report += "\n**🔍 추가 관점 (사용자 피드백 반영):**\n"
                        feedback_analyzer = FeedbackAnalyzer()
                        
                        for news in feedback_news:
                            date_str = "25.12.14"
                            if news.get('published_at'):
                                try:
                                    if isinstance(news['published_at'], str):
                                        dt = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                                    else:
                                        dt = news['published_at']
                                    date_str = dt.strftime("%y.%m.%d")
                                except:
                                    pass
                            
                            # 언어 감지
                            is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in news['title'])
                            
                            if is_korean:
                                report += f"- [{date_str}] {news['title']} - [링크]({news['link']})\n"
                            else:
                                korean_title = news['title']
                                report += f"- [{date_str}] {korean_title} - [{news['title']}]({news['link']})\n"
                            
                            # 매칭된 피드백 코멘트 표시
                            # 뉴스 텍스트로 임베딩 생성
                            news_text = f"{news.get('title', '')} {news.get('snippet', '')}"
                            news_embedding = feedback_analyzer.generate_embedding(news_text)
                            
                            if news_embedding is not None:
                                matched_comments = []
                                for feedback in feedbacks:
                                    feedback_text = f"{feedback.get('news_title', '')} {feedback.get('user_comment', '')}"
                                    feedback_embedding = feedback_analyzer.generate_embedding(feedback_text)
                                    if feedback_embedding is not None:
                                        similarity = feedback_analyzer.calculate_similarity(
                                            feedback_embedding, 
                                            np.array([news_embedding])
                                        )[0]
                                        if similarity >= 0.7:
                                            matched_comments.append(feedback.get('user_comment', ''))
                                
                                if matched_comments:
                                    report += "  \n  **💬 사용자 피드백:**\n"
                                    for comment in matched_comments:
                                        report += f"  - {comment}\n"
                else:
                    # Fallback: selected_links 사용
                    report += "**📰 Key News:**\n"
                    if theme['related_results']:
                        for link in theme['related_results'][0].get('selected_links', [])[:4]:
                            report += f"- {link}\n"
                
                report += "\n"
        
        report += f"""---

## Investor Note

### Caution
{investor_note.get('caution', 'N/A')}

### Action
{investor_note.get('action', 'N/A')}

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report
    
    def save_report(self, report: str, filename: str = "daily_market_report.md"):
        """
        Step 4: 리포트를 파일로 저장
        """
        print("=" * 80)
        print("Step 4: 리포트 저장")
        print("=" * 80)
        
        output_path = Path(project_root) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 리포트 저장 완료: {output_path}")
        print()
    
    def run(self, category_filter: Optional[List[str]] = None, 
            report_name: Optional[str] = None, sections: Optional[List[str]] = None):
        """
        전체 파이프라인 실행
        
        Args:
            category_filter: 필터링할 카테고리 리스트 (None이면 모든 카테고리)
            report_name: 보고서 이름 (파일명에 사용)
            sections: 사용할 섹션 ID 리스트 (None이면 모든 섹션)
        """
        print("=" * 80)
        print("Daily Market Executive Report Generator")
        if category_filter:
            print(f"보고서 그룹: {', '.join(category_filter)}")
        print("=" * 80)
        print()
        
        try:
            # Step 1: 데이터 로드 및 전처리
            df = self.load_data(category_filter=category_filter)
            clusters = self.preprocess_clusters(df)
            
            if not clusters:
                print("⚠️  분석할 클러스터가 없습니다.")
                return None
            
            # Step 2: Micro-Analysis
            print("=" * 80)
            print("Step 2: Micro-Analysis (Cluster 단위 분석)")
            print("=" * 80)
            
            analyzed_results = []
            for i, (cluster_key, cluster_data) in enumerate(clusters.items(), 1):
                print(f"[{i}/{len(clusters)}] 분석 중: {cluster_data['issue_title']}")
                result = self.analyze_cluster(cluster_data)
                result.update({
                    'category_name': cluster_data['category_name'],
                    'issue_title': cluster_data['issue_title'],
                    'score': cluster_data['score'],
                    'top_5_news': cluster_data['top_5_news']  # 원본 뉴스 데이터 보존
                })
                analyzed_results.append(result)
                print(f"   ✅ 완료: {result['new_title']}")
            
            print()
            
            # Step 2-2: Theme Consolidation
            consolidated_themes = self.consolidate_themes(analyzed_results)
            
            # Step 3: Final Report Generation
            report = self.generate_final_report(consolidated_themes, sections=sections)
            
            # Step 4: 저장
            if report_name:
                filename = f"{report_name}.md"
            else:
                filename = "daily_market_report.md"
            self.save_report(report, filename=filename)
            
            print("=" * 80)
            print("✅ 리포트 생성 완료!")
            print("=" * 80)
            
            return report
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_by_groups(self):
        """
        보고서 그룹별로 보고서 생성
        설정 파일의 report_groups에 따라 여러 보고서를 생성합니다.
        """
        report_groups = self.config.get('report_groups', [])
        
        if not report_groups:
            # 보고서 그룹이 설정되지 않았으면 기존 방식으로 실행
            print("보고서 그룹이 설정되지 않았습니다. 모든 카테고리를 하나의 보고서로 생성합니다.")
            print()
            return self.run()
        
        print("=" * 80)
        print(f"보고서 그룹별 생성: {len(report_groups)}개 그룹")
        print("=" * 80)
        print()
        
        reports = {}
        for i, group in enumerate(report_groups, 1):
            group_name = group.get('name', f'Group {i}')
            categories = group.get('categories', [])
            output_file = group.get('output_file', f"{group_name.lower().replace(' ', '_')}.md")
            sections_str = group.get('sections', '')  # 새로 추가
            
            # sections 파싱 (쉼표로 구분된 문자열을 리스트로)
            sections = None
            if sections_str:
                sections = [s.strip() for s in sections_str.split(',') if s.strip()]
            
            if not categories:
                print(f"⚠️  [{group_name}] 카테고리가 지정되지 않았습니다. 건너뜁니다.")
                continue
            
            print(f"[{i}/{len(report_groups)}] {group_name} 생성 중...")
            print(f"   카테고리: {', '.join(categories)}")
            if sections:
                print(f"   섹션: {', '.join(sections)}")
            print()
            
            try:
                report = self.run(category_filter=categories, 
                                report_name=group_name, 
                                sections=sections)  # sections 전달
                if report:
                    reports[group_name] = {
                        'content': report,
                        'output_file': output_file
                    }
                    print(f"✅ [{group_name}] 보고서 생성 완료: {output_file}")
                else:
                    print(f"⚠️  [{group_name}] 보고서 생성 실패 (데이터 없음)")
            except Exception as e:
                print(f"❌ [{group_name}] 보고서 생성 실패: {e}")
                logger.error(f"보고서 그룹 '{group_name}' 생성 실패: {e}", exc_info=True)
            
            print()
        
        print("=" * 80)
        print(f"보고서 그룹별 생성 완료: {len(reports)}/{len(report_groups)}개 성공")
        print("=" * 80)
        
        return reports


if __name__ == "__main__":
    generator = DailyReportGenerator()
    
    # 보고서 그룹이 설정되어 있으면 그룹별로 생성, 없으면 기존 방식
    report_groups = generator.config.get('report_groups', [])
    if report_groups:
        generator.run_by_groups()
    else:
        generator.run()
