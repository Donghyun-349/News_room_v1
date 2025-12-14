"""
피드백 분석 및 유사도 계산 모듈
피드백과 뉴스 간의 유사도를 계산하여 가중치를 적용합니다.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """피드백을 분석하고 뉴스와의 유사도를 계산하는 클래스"""
    
    def __init__(self):
        """피드백 분석기 초기화"""
        from config import get_config
        self.config = get_config()
        self.llm_config = self.config.get('llm', {})
        self.embedding_provider = self.llm_config.get('embedding_provider', 'openai')
        self.embedding_model = self.llm_config.get('embedding_model', 'text-embedding-3-small')
        
        # 임베딩 클라이언트 초기화
        if self.embedding_provider == 'openai':
            import os
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.embedding_client = OpenAI(api_key=api_key)
            else:
                self.embedding_client = None
                logger.warning("OPENAI_API_KEY가 설정되지 않아 피드백 임베딩을 생성할 수 없습니다.")
        else:
            self.embedding_client = None
            logger.warning(f"지원하지 않는 임베딩 프로바이더: {self.embedding_provider}")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        텍스트의 임베딩을 생성합니다.
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터 또는 None
        """
        if not self.embedding_client or not text:
            return None
        
        try:
            if self.embedding_provider == 'openai':
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embedding = response.data[0].embedding
                return np.array(embedding)
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def calculate_similarity(self, feedback_embedding: np.ndarray, 
                           news_embeddings: np.ndarray) -> np.ndarray:
        """
        피드백 임베딩과 뉴스 임베딩 간의 유사도를 계산합니다.
        
        Args:
            feedback_embedding: 피드백 임베딩 벡터
            news_embeddings: 뉴스 임베딩 벡터 배열
            
        Returns:
            유사도 점수 배열 (0~1)
        """
        if feedback_embedding is None or news_embeddings is None or len(news_embeddings) == 0:
            return np.array([])
        
        # 2D 배열로 변환
        if feedback_embedding.ndim == 1:
            feedback_embedding = feedback_embedding.reshape(1, -1)
        if news_embeddings.ndim == 1:
            news_embeddings = news_embeddings.reshape(1, -1)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(feedback_embedding, news_embeddings)[0]
        
        # 음수 값을 0으로 클리핑 (유사도는 0~1 범위)
        similarities = np.clip(similarities, 0, 1)
        
        return similarities
    
    def calculate_user_feedback_score(self, matched_feedbacks: List[Dict[str, Any]]) -> float:
        """
        여러 피드백을 기반으로 user_feedback_score 계산 (가중 평균)
        
        Args:
            matched_feedbacks: 매칭된 피드백 리스트 (각 항목은 {'similarity': float, 'feedback': dict})
            
        Returns:
            user_feedback_score (0.0 ~ 1.0)
        """
        if not matched_feedbacks:
            return 0.0
        
        # 가중 평균 계산: similarity^2를 가중치로 사용
        total_weight = sum(fb['similarity'] for fb in matched_feedbacks)
        weighted_sum = sum(fb['similarity'] * fb['similarity'] for fb in matched_feedbacks)
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def apply_feedback_scores(self, news_list: List[Dict[str, Any]], 
                               feedbacks: List[Dict[str, Any]],
                               news_embeddings: Optional[np.ndarray] = None,
                               similarity_threshold: float = 0.7,
                               max_importance_boost: float = 0.15) -> List[Dict[str, Any]]:
        """
        피드백을 기반으로 뉴스에 별도 점수를 적용합니다.
        - user_feedback_score: 별도 점수 (가중 평균)
        - importance_score: apply_to_importance=True인 경우만 최대 15% 상승
        
        Args:
            news_list: 뉴스 리스트
            feedbacks: 피드백 리스트
            news_embeddings: 뉴스 임베딩 배열 (None이면 생성)
            similarity_threshold: 유사도 임계값 (이 이상이면 피드백 적용)
            max_importance_boost: importance_score 최대 상승률 (기본값: 0.15 = 15%)
            
        Returns:
            피드백 점수가 적용된 뉴스 리스트
        """
        if not feedbacks or not news_list:
            return news_list
        
        # 뉴스 임베딩이 없으면 생성
        if news_embeddings is None:
            logger.info("뉴스 임베딩 생성 중...")
            news_embeddings = []
            for news in news_list:
                # 제목과 스니펫을 결합하여 임베딩 생성
                text = f"{news.get('title', '')} {news.get('snippet', '')}"
                embedding = self.generate_embedding(text)
                if embedding is not None:
                    news_embeddings.append(embedding)
                else:
                    # 임베딩 생성 실패 시 0 벡터 사용
                    news_embeddings.append(np.zeros(1536))  # text-embedding-3-small의 차원
            
            news_embeddings = np.array(news_embeddings)
        
        # 피드백별로 매칭 계산
        news_feedbacks = {}  # {news_index: [matched_feedback_dicts]}
        for feedback in feedbacks:
            # 피드백 텍스트 생성 (뉴스 제목 + 사용자 코멘트)
            feedback_text = f"{feedback.get('news_title', '')} {feedback.get('user_comment', '')}"
            feedback_embedding = self.generate_embedding(feedback_text)
            
            if feedback_embedding is None:
                continue
            
            # 유사도 계산
            similarities = self.calculate_similarity(feedback_embedding, news_embeddings)
            
            # 유사도가 임계값 이상인 뉴스에 피드백 매칭
            for i, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    if i not in news_feedbacks:
                        news_feedbacks[i] = []
                    news_feedbacks[i].append({
                        'similarity': similarity,
                        'feedback': feedback
                    })
        
        # 각 뉴스에 피드백 점수 적용
        for i, news in enumerate(news_list):
            if i in news_feedbacks:
                matched_feedbacks = news_feedbacks[i]
                
                # user_feedback_score 계산 (가중 평균)
                user_feedback_score = self.calculate_user_feedback_score(matched_feedbacks)
                news['user_feedback_score'] = user_feedback_score
                news['matched_feedbacks'] = matched_feedbacks
                
                # apply_to_importance=True인 피드백만 importance_score에 반영
                apply_feedbacks = [
                    fb for fb in matched_feedbacks 
                    if fb['feedback'].get('apply_to_importance', False)
                ]
                
                if apply_feedbacks:
                    # 최대 유사도에 비례하여 최대 15% 상승
                    max_similarity = max(fb['similarity'] for fb in apply_feedbacks)
                    boost = 1.0 + (max_similarity - similarity_threshold) / (1.0 - similarity_threshold) * max_importance_boost
                    original_score = news.get('importance_score', 1.0)
                    news['importance_score'] = original_score * boost
                    news['feedback_applied_to_importance'] = True
                    logger.debug(f"뉴스 '{news.get('title', '')[:50]}...' importance_score 상승: {original_score:.2f} -> {news['importance_score']:.2f} (user_feedback_score: {user_feedback_score:.2f})")
                else:
                    news['feedback_applied_to_importance'] = False
                    logger.debug(f"뉴스 '{news.get('title', '')[:50]}...' user_feedback_score: {user_feedback_score:.2f} (일반 점수 미반영)")
            else:
                # 피드백이 없는 경우
                news['user_feedback_score'] = 0.0
                news['matched_feedbacks'] = []
                news['feedback_applied_to_importance'] = False
        
        return news_list
    
    def apply_feedback_weights(self, news_list: List[Dict[str, Any]], 
                               feedbacks: List[Dict[str, Any]],
                               news_embeddings: Optional[np.ndarray] = None,
                               similarity_threshold: float = 0.7,
                               max_boost: float = 1.5) -> List[Dict[str, Any]]:
        """
        [레거시 메서드] 피드백을 기반으로 뉴스의 중요도 점수에 가중치를 적용합니다.
        apply_feedback_scores()를 사용하는 것을 권장합니다.
        """
        return self.apply_feedback_scores(
            news_list, feedbacks, news_embeddings, 
            similarity_threshold, max_boost - 1.0
        )
    
    def find_matching_news(self, feedback: Dict[str, Any], 
                          news_list: List[Dict[str, Any]],
                          news_embeddings: Optional[np.ndarray] = None,
                          similarity_threshold: float = 0.7) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        피드백과 매칭되는 뉴스를 찾습니다.
        
        Args:
            feedback: 피드백 딕셔너리
            news_list: 뉴스 리스트
            news_embeddings: 뉴스 임베딩 배열
            similarity_threshold: 유사도 임계값
            
        Returns:
            (뉴스 인덱스, 유사도, 뉴스 딕셔너리) 튜플 리스트
        """
        if news_embeddings is None:
            return []
        
        feedback_text = f"{feedback.get('news_title', '')} {feedback.get('user_comment', '')}"
        feedback_embedding = self.generate_embedding(feedback_text)
        
        if feedback_embedding is None:
            return []
        
        similarities = self.calculate_similarity(feedback_embedding, news_embeddings)
        
        matches = []
        for i, similarity in enumerate(similarities):
            if similarity >= similarity_threshold:
                matches.append((i, similarity, news_list[i]))
        
        # 유사도 순으로 정렬
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches

