"""
데이터베이스 관리자 모듈
Connection Pool 및 CRUD 함수를 제공합니다.
"""
import os
import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 설정 파일 로드
try:
    from config import get_config
except ImportError:
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from config import get_config


class DatabaseManager:
    """데이터베이스 연결 및 CRUD 작업을 관리하는 클래스"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: 데이터베이스 파일 경로 (None이면 .env에서 가져옴)
        """
        if db_path is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./investment.db")
            if db_url.startswith("sqlite:///"):
                db_path = db_url.replace("sqlite:///", "")
            else:
                db_path = db_url
        
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 스키마가 없으면 생성/업데이트
        # (schema.sql은 CREATE TABLE IF NOT EXISTS 를 사용하므로 여러 번 실행해도 안전)
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _initialize_database(self):
        """
        database/schema.sql 파일을 사용해 데이터베이스 스키마를 초기화/보장합니다.
        
        - news, issues, issue_news_mapping 테이블과 관련 인덱스를 생성합니다.
        - schema.sql 은 IF NOT EXISTS 를 사용하므로 여러 번 실행해도 안전합니다.
        """
        try:
            schema_path = Path(__file__).parent / "schema.sql"
            if not schema_path.exists():
                logger.warning(f"데이터베이스 스키마 파일을 찾을 수 없습니다: {schema_path}")
                return
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                cursor.executescript(schema_sql)
                logger.info("데이터베이스 스키마 초기화/업데이트 완료 (schema.sql 적용)")
        except Exception as e:
            # 스키마 초기화 실패는 이후 로직에 영향을 줄 수 있으므로 로그만 남기고 예외는 다시 던집니다.
            logger.error(f"데이터베이스 스키마 초기화 실패: {e}", exc_info=True)
            raise
    
    # ========== News 관련 함수 ==========
    
    def insert_news(self, title: str, link: str, snippet: str = None, 
                   source: str = None, published_at: datetime = None,
                   vector_id: str = None, importance_score: float = 1.0,
                   category_name: str = None, search_keyword: str = None,
                   search_rank: int = None) -> int:
        """
        뉴스를 삽입합니다.
        
        Args:
            title: 기사 제목
            link: 기사 링크
            snippet: 기사 스니펫
            source: 언론사
            published_at: 발행일
            vector_id: ChromaDB 벡터 ID
            importance_score: 중요도 점수
            category_name: 카테고리 명
            search_keyword: 검색 키워드
            search_rank: 검색 노출 순위
        
        Returns:
            삽입된 뉴스의 ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO news 
                (title, link, snippet, source, published_at, vector_id, importance_score,
                 category_name, search_keyword, search_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, link, snippet, source, published_at, vector_id, importance_score,
                  category_name, search_keyword, search_rank))
            
            # 삽입된 ID 가져오기
            if cursor.lastrowid == 0:
                # 이미 존재하는 경우 ID 조회
                cursor.execute("SELECT id FROM news WHERE link = ?", (link,))
                result = cursor.fetchone()
                return result['id'] if result else None
            
            return cursor.lastrowid
    
    def get_unanalyzed_news(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        분석되지 않은 뉴스를 가져옵니다.
        설정된 연령 제한(analysis.max_article_age_hours)이 있으면 해당 시간 이내 발행된 기사만 반환합니다.
        null이거나 0이면 연령 제한 없이 모든 분석되지 않은 뉴스를 반환합니다.
        
        Args:
            limit: 최대 반환 개수
        
        Returns:
            분석되지 않은 뉴스 리스트
        """
        # 설정에서 최대 기사 연령 가져오기
        config = get_config()
        max_age_hours = config.get('analysis', {}).get('max_article_age_hours', 48)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # max_age_hours가 None이거나 0이면 연령 제한 없음
            if max_age_hours is None or max_age_hours == 0:
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE analyzed = 0 
                      AND published_at IS NOT NULL
                    ORDER BY published_at DESC, importance_score DESC
                    LIMIT ?
                """, (limit,))
            else:
                # 현재 시간에서 max_age_hours 시간 전 계산
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE analyzed = 0 
                      AND published_at IS NOT NULL
                      AND published_at >= ?
                    ORDER BY published_at DESC, importance_score DESC
                    LIMIT ?
                """, (cutoff_time, limit))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # 필터링된 기사 수 로깅
            if results:
                oldest_article = min(r.get('published_at') for r in results if r.get('published_at'))
                if isinstance(oldest_article, str):
                    oldest_article = datetime.fromisoformat(oldest_article.replace('Z', '+00:00'))
                age_hours = (datetime.now() - oldest_article).total_seconds() / 3600
                age_limit_text = f"{max_age_hours}시간" if max_age_hours else "제한 없음"
                logger.info(f"분석 대상 뉴스 {len(results)}개 선택 (최대 연령: {age_limit_text}, 가장 오래된 기사: {age_hours:.1f}시간 전)")
            
            return results
    
    def get_unanalyzed_news_by_category(self, category_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        특정 카테고리의 분석되지 않은 뉴스를 가져옵니다.
        설정된 연령 제한(analysis.max_article_age_hours)이 있으면 해당 시간 이내 발행된 기사만 반환합니다.
        null이거나 0이면 연령 제한 없이 모든 분석되지 않은 뉴스를 반환합니다.
        
        Args:
            category_name: 카테고리 이름
            limit: 최대 반환 개수
        
        Returns:
            분석되지 않은 뉴스 리스트
        """
        # 설정에서 최대 기사 연령 가져오기
        config = get_config()
        max_age_hours = config.get('analysis', {}).get('max_article_age_hours', 48)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # max_age_hours가 None이거나 0이면 연령 제한 없음
            if max_age_hours is None or max_age_hours == 0:
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE analyzed = 0 
                      AND published_at IS NOT NULL
                      AND category_name = ?
                    ORDER BY published_at DESC, importance_score DESC
                    LIMIT ?
                """, (category_name, limit))
            else:
                # 현재 시간에서 max_age_hours 시간 전 계산
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                cursor.execute("""
                    SELECT * FROM news 
                    WHERE analyzed = 0 
                      AND published_at IS NOT NULL
                      AND published_at >= ?
                      AND category_name = ?
                    ORDER BY published_at DESC, importance_score DESC
                    LIMIT ?
                """, (cutoff_time, category_name, limit))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # 필터링된 기사 수 로깅
            if results:
                oldest_article = min(r.get('published_at') for r in results if r.get('published_at'))
                if isinstance(oldest_article, str):
                    oldest_article = datetime.fromisoformat(oldest_article.replace('Z', '+00:00'))
                age_hours = (datetime.now() - oldest_article).total_seconds() / 3600
                age_limit_text = f"{max_age_hours}시간" if max_age_hours else "제한 없음"
                logger.info(f"[{category_name}] 분석 대상 뉴스 {len(results)}개 선택 (최대 연령: {age_limit_text}, 가장 오래된 기사: {age_hours:.1f}시간 전)")
            else:
                logger.info(f"[{category_name}] 분석할 뉴스가 없습니다.")
            
            return results
    
    def mark_news_analyzed(self, news_ids: List[int]):
        """뉴스를 분석 완료로 표시합니다"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(news_ids))
            cursor.execute(f"""
                UPDATE news 
                SET analyzed = 1 
                WHERE id IN ({placeholders})
            """, news_ids)
    
    def update_news_vector_id(self, news_id: int, vector_id: str):
        """뉴스의 vector_id를 업데이트합니다"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE news 
                SET vector_id = ? 
                WHERE id = ?
            """, (vector_id, news_id))
    
    def news_exists(self, link: str) -> bool:
        """링크로 뉴스가 이미 존재하는지 확인합니다"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM news WHERE link = ?", (link,))
            result = cursor.fetchone()
            return result['count'] > 0
    
    def get_news_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """제목으로 뉴스를 조회합니다 (정확히 일치하는 경우만)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, link, source, snippet, published_at, importance_score,
                       category_name, search_keyword, search_rank
                FROM news 
                WHERE title = ?
                LIMIT 1
            """, (title,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def delete_news(self, news_id: int) -> bool:
        """뉴스를 삭제합니다"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM news WHERE id = ?", (news_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # ========== Issues 관련 함수 ==========
    
    def insert_issue(self, title: str, summary: str, primary_tag: str,
                    secondary_tags: List[str] = None, cluster_id: int = None) -> int:
        """
        이슈를 삽입합니다.
        
        Returns:
            삽입된 이슈의 ID
        """
        secondary_tags_json = json.dumps(secondary_tags) if secondary_tags else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO issues (title, summary, primary_tag, secondary_tags, cluster_id)
                VALUES (?, ?, ?, ?, ?)
            """, (title, summary, primary_tag, secondary_tags_json, cluster_id))
            
            return cursor.lastrowid
    
    def link_issue_to_news(self, issue_id: int, news_ids: List[int]):
        """이슈와 뉴스를 연결합니다"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for news_id in news_ids:
                cursor.execute("""
                    INSERT OR IGNORE INTO issue_news_mapping (issue_id, news_id)
                    VALUES (?, ?)
                """, (issue_id, news_id))
    
    def calculate_issue_score(self, issue_id: int) -> float:
        """
        이슈의 랭킹 점수를 계산합니다.
        공식: (avg_importance ** 3) * log2(count + 1)
        
        Args:
            issue_id: 이슈 ID
        
        Returns:
            이슈 점수
        """
        import math
        with self.get_connection() as conn:
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
            """, (issue_id,))
            
            result = cursor.fetchone()
            if not result or result[0] is None:
                return 0.0
            
            avg_importance = result[0] or 1.0
            news_count = result[1] or 0
            
            # 세제곱 공식 적용
            score = (avg_importance ** 3) * math.log2(news_count + 1)
            return score
    
    def get_today_issues(self, date: datetime = None) -> List[Dict[str, Any]]:
        """
        오늘 생성된 이슈를 가져옵니다 (랭킹 점수 포함).
        
        Args:
            date: 조회할 날짜 (None이면 오늘)
        
        Returns:
            이슈 리스트 (랭킹 점수 포함, 점수 높은 순으로 정렬)
        """
        if date is None:
            date = datetime.now()
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM issues
                WHERE created_at >= ? AND created_at < ?
                ORDER BY created_at DESC
            """, (start_date, end_date))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # secondary_tags를 JSON에서 파싱하고 랭킹 점수 계산
            for result in results:
                if result.get('secondary_tags'):
                    result['secondary_tags'] = json.loads(result['secondary_tags'])
                else:
                    result['secondary_tags'] = []
                
                # 랭킹 점수 계산 및 추가
                result['ranking_score'] = self.calculate_issue_score(result['id'])
            
            # 랭킹 점수로 정렬 (높은 순)
            results.sort(key=lambda x: x.get('ranking_score', 0.0), reverse=True)
            
            return results
    
    def get_issue_details(self, issue_id: int) -> Optional[Dict[str, Any]]:
        """이슈 상세 정보를 가져옵니다 (연결된 뉴스 포함)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 이슈 정보
            cursor.execute("SELECT * FROM issues WHERE id = ?", (issue_id,))
            issue = cursor.fetchone()
            
            if not issue:
                return None
            
            issue_dict = dict(issue)
            if issue_dict.get('secondary_tags'):
                issue_dict['secondary_tags'] = json.loads(issue_dict['secondary_tags'])
            else:
                issue_dict['secondary_tags'] = []
            
            # 연결된 뉴스
            cursor.execute("""
                SELECT n.* FROM news n
                INNER JOIN issue_news_mapping m ON n.id = m.news_id
                WHERE m.issue_id = ?
                ORDER BY n.importance_score DESC, n.published_at DESC
            """, (issue_id,))
            
            issue_dict['news'] = [dict(row) for row in cursor.fetchall()]
            
            return issue_dict
    
    # ========== Trend 관련 함수 ==========
    
    def get_tag_trends(self, days: int = 30) -> pd.DataFrame:
        """
        태그별 트렌드 데이터를 가져옵니다.
        
        Args:
            days: 조회할 일수
        
        Returns:
            DataFrame (컬럼: date, tag, count)
        """
        start_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            query = """
                SELECT 
                    DATE(created_at) as date,
                    primary_tag as tag,
                    COUNT(*) as count
                FROM issues
                WHERE created_at >= ?
                GROUP BY DATE(created_at), primary_tag
                ORDER BY date, tag
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date,))
            
            # date를 datetime으로 변환
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
    
    # ========== Database Reset 함수 ==========
    
    def reset_database(self):
        """
        데이터베이스의 모든 데이터를 삭제합니다 (테이블 구조는 유지).
        주의: 이 함수는 모든 뉴스, 이슈, 매핑 데이터를 삭제합니다.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 외래 키 제약 조건을 일시적으로 비활성화
            cursor.execute("PRAGMA foreign_keys = OFF")
            
            try:
                # 매핑 테이블 먼저 삭제 (외래 키 제약 때문)
                cursor.execute("DELETE FROM issue_news_mapping")
                
                # 이슈 테이블 삭제
                cursor.execute("DELETE FROM issues")
                
                # 뉴스 테이블 삭제
                cursor.execute("DELETE FROM news")
                
                # AUTOINCREMENT 카운터 리셋 (선택사항)
                cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('news', 'issues', 'issue_news_mapping')")
                
                # 외래 키 제약 조건 다시 활성화
                cursor.execute("PRAGMA foreign_keys = ON")
                
                logger.info("데이터베이스 리셋 완료: 모든 뉴스, 이슈, 매핑 데이터가 삭제되었습니다.")
                
            except Exception as e:
                cursor.execute("PRAGMA foreign_keys = ON")
                raise

