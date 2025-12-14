"""
구글 스프레드시트 연동 모듈
뉴스 수집 및 분석 결과를 구글 스프레드시트에 자동으로 기록합니다.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread가 설치되지 않았습니다. 구글 스프레드시트 연동이 비활성화됩니다.")

logger = logging.getLogger(__name__)


class GoogleSheetsExporter:
    """구글 스프레드시트에 데이터를 내보내는 클래스"""
    
    def __init__(self, spreadsheet_id: Optional[str] = None):
        """
        구글 스프레드시트 클라이언트 초기화
        
        Args:
            spreadsheet_id: 사용할 스프레드시트 ID (None이면 설정 파일의 spreadsheet_id 사용)
        """
        # 순환 참조 방지를 위해 직접 YAML 파일 읽기
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        self.sheets_config = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.sheets_config = config.get('google_sheets', {})
            except Exception as e:
                logger.warning(f"YAML 설정 파일 읽기 실패: {e}")
        
        if not self.sheets_config.get('enabled', False):
            logger.info("구글 스프레드시트 연동이 비활성화되어 있습니다.")
            self.client = None
            self.spreadsheet = None
            return
        
        if not GSPREAD_AVAILABLE:
            logger.warning("gspread가 설치되지 않아 구글 스프레드시트 연동을 사용할 수 없습니다.")
            self.client = None
            self.spreadsheet = None
            return
        
        try:
            # 서비스 계정 인증
            service_account_json = self.sheets_config.get('service_account_json', '')
            if not service_account_json:
                # 환경변수에서 경로 가져오기
                service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', '')
            
            if service_account_json and Path(service_account_json).exists():
                # JSON 파일 경로로 인증
                scope = ['https://spreadsheets.google.com/feeds',
                        'https://www.googleapis.com/auth/drive']
                creds = Credentials.from_service_account_file(service_account_json, scopes=scope)
                self.client = gspread.authorize(creds)
            else:
                # 환경변수에서 직접 JSON 내용 가져오기
                import json
                service_account_info = os.getenv('GOOGLE_SERVICE_ACCOUNT_INFO', '')
                if service_account_info:
                    account_info = json.loads(service_account_info)
                    scope = ['https://spreadsheets.google.com/feeds',
                            'https://www.googleapis.com/auth/drive']
                    creds = Credentials.from_service_account_info(account_info, scopes=scope)
                    self.client = gspread.authorize(creds)
                else:
                    logger.warning("구글 서비스 계정 인증 정보를 찾을 수 없습니다.")
                    self.client = None
                    self.spreadsheet = None
                    return
            
            # 스프레드시트 열기
            # 우선순위: 파라미터 > 환경변수 > 설정 파일
            if spreadsheet_id:
                target_spreadsheet_id = spreadsheet_id
            else:
                # 환경변수에서 먼저 확인
                target_spreadsheet_id = os.getenv('GOOGLE_SPREADSHEET_ID', '')
                if not target_spreadsheet_id:
                    # 환경변수가 없으면 설정 파일에서 가져오기
                    target_spreadsheet_id = self.sheets_config.get('spreadsheet_id', '')
            
            if target_spreadsheet_id:
                self.spreadsheet = self.client.open_by_key(target_spreadsheet_id)
                logger.info(f"구글 스프레드시트 연결 성공: {self.spreadsheet.title}")
            else:
                logger.warning("스프레드시트 ID가 설정되지 않았습니다.")
                self.spreadsheet = None
                
        except Exception as e:
            logger.error(f"구글 스프레드시트 초기화 실패: {e}", exc_info=True)
            self.client = None
            self.spreadsheet = None
    
    def _get_or_create_worksheet(self, tab_name: str, headers: List[str]) -> Optional[gspread.Worksheet]:
        """
        워크시트를 가져오거나 생성합니다.
        
        Args:
            tab_name: 탭 이름
            headers: 헤더 리스트
        
        Returns:
            워크시트 객체 또는 None
        """
        if not self.spreadsheet:
            return None
        
        try:
            # 기존 워크시트 찾기
            try:
                worksheet = self.spreadsheet.worksheet(tab_name)
                # 헤더 확인 및 업데이트
                existing_headers = worksheet.row_values(1)
                if existing_headers != headers:
                    # 헤더가 다르면 업데이트
                    worksheet.clear()
                    worksheet.append_row(headers)
                return worksheet
            except gspread.exceptions.WorksheetNotFound:
                # 워크시트가 없으면 생성
                worksheet = self.spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)
                worksheet.append_row(headers)
                logger.info(f"새 워크시트 생성: {tab_name}")
                return worksheet
        except Exception as e:
            logger.error(f"워크시트 가져오기/생성 실패 ({tab_name}): {e}", exc_info=True)
            return None
    
    def export_news_collection(self, news_data: List[Dict[str, Any]]):
        """
        뉴스 수집 데이터를 스프레드시트에 기록합니다.
        
        Args:
            news_data: 뉴스 데이터 리스트
        """
        if not self.spreadsheet or not news_data:
            return
        
        tab_name = self.sheets_config.get('tabs', {}).get('news_collection', '뉴스 수집')
        
        # 데이터프레임으로 변환
        df = pd.DataFrame(news_data)
        
        # 정렬: 카테고리별, 검색 순위 오름차순 (1번이 상위)
        if 'category_name' in df.columns and 'search_rank' in df.columns:
            df = df.sort_values(
                by=['category_name', 'search_rank'],
                ascending=[True, True],
                na_position='last'
            )
        
        # 헤더 정의
        headers = ['ID', '카테고리', '검색 키워드', '제목', '링크', '스니펫', '출처', 
                  '검색 순위', '발행일', '수집일', '중요도 점수', '분석 상태']
        
        # 데이터 정리
        export_data = []
        for _, row in df.iterrows():
            export_data.append([
                row.get('id', ''),
                row.get('category_name', ''),
                row.get('search_keyword', ''),
                row.get('title', ''),
                row.get('link', ''),
                str(row.get('snippet', ''))[:500],  # 스니펫은 500자로 제한
                row.get('source', ''),
                row.get('search_rank', ''),
                row.get('published_at', ''),
                row.get('created_at', ''),
                row.get('importance_score', ''),
                '완료' if row.get('analyzed', 0) else '대기중'
            ])
        
        try:
            worksheet = self._get_or_create_worksheet(tab_name, headers)
            if worksheet:
                # 기존 데이터 지우기 (헤더 제외)
                if len(worksheet.get_all_values()) > 1:
                    worksheet.delete_rows(2, len(worksheet.get_all_values()))
                
                # 새 데이터 추가
                if export_data:
                    worksheet.append_rows(export_data)
                logger.info(f"뉴스 수집 데이터 {len(export_data)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.error(f"뉴스 수집 데이터 기록 실패: {e}", exc_info=True)
    
    def export_clustering_results(self, cluster_data: List[Dict[str, Any]]):
        """
        클러스터링 결과를 스프레드시트에 기록합니다.
        
        Args:
            cluster_data: 클러스터 데이터 리스트
        """
        if not self.spreadsheet or not cluster_data:
            return
        
        tab_name = self.sheets_config.get('tabs', {}).get('clustering', '클러스터링')
        
        # 헤더 정의
        headers = ['클러스터 ID', '뉴스 ID', '제목', '출처', '중요도 점수', '카테고리']
        
        # 데이터 정리
        export_data = []
        for cluster in cluster_data:
            cluster_id = cluster.get('cluster_id', -1)
            # numpy int64를 일반 int로 변환
            if hasattr(cluster_id, 'item'):
                cluster_id = int(cluster_id.item())
            else:
                cluster_id = int(cluster_id)
            
            for news in cluster.get('news_items', []):
                news_id = news.get('id', '')
                importance_score = news.get('importance_score', '')
                
                # numpy 타입 변환
                if hasattr(news_id, 'item'):
                    news_id = int(news_id.item())
                elif news_id is not None:
                    news_id = int(news_id)
                
                if hasattr(importance_score, 'item'):
                    importance_score = float(importance_score.item())
                elif importance_score is not None:
                    importance_score = float(importance_score)
                
                export_data.append([
                    cluster_id,
                    news_id,
                    news.get('title', ''),
                    news.get('source', ''),
                    importance_score,
                    news.get('category_name', '')
                ])
        
        try:
            worksheet = self._get_or_create_worksheet(tab_name, headers)
            if worksheet:
                # 기존 데이터 지우기 (헤더 제외)
                if len(worksheet.get_all_values()) > 1:
                    worksheet.delete_rows(2, len(worksheet.get_all_values()))
                
                # 새 데이터 추가
                if export_data:
                    worksheet.append_rows(export_data)
                logger.info(f"클러스터링 결과 {len(export_data)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.error(f"클러스터링 결과 기록 실패: {e}", exc_info=True)
    
    def export_issues(self, issues_data: List[Dict[str, Any]]):
        """
        이슈 목록을 스프레드시트에 기록합니다.
        
        Args:
            issues_data: 이슈 데이터 리스트
        """
        if not self.spreadsheet or not issues_data:
            return
        
        tab_name = self.sheets_config.get('tabs', {}).get('issues', '이슈 목록')
        
        # 헤더 정의 (태그 제거, 카테고리와 스코어 추가)
        headers = ['카테고리', '제목', '스코어', '요약', '생성일', '클러스터 ID']
        
        # 데이터 정리
        export_data = []
        for issue in issues_data:
            # bytes 타입을 문자열로 변환
            cluster_id = issue.get('cluster_id', '')
            if isinstance(cluster_id, bytes):
                cluster_id = ''
            elif cluster_id is not None:
                cluster_id = str(cluster_id)
            
            category_name = issue.get('category_name', '')
            if isinstance(category_name, bytes):
                category_name = category_name.decode('utf-8', errors='replace')
            
            title = issue.get('title', '')
            if isinstance(title, bytes):
                title = title.decode('utf-8', errors='replace')
            
            summary = issue.get('summary', '')
            if isinstance(summary, bytes):
                summary = summary.decode('utf-8', errors='replace')
            
            created_at = issue.get('created_at', '')
            if isinstance(created_at, bytes):
                created_at = created_at.decode('utf-8', errors='replace')
            
            export_data.append([
                category_name,
                title,
                round(issue.get('score', 0.0), 2) if issue.get('score') is not None else 0.0,
                summary,
                created_at,
                cluster_id
            ])
        
        try:
            worksheet = self._get_or_create_worksheet(tab_name, headers)
            if worksheet:
                # 기존 데이터 지우기 (헤더 제외)
                if len(worksheet.get_all_values()) > 1:
                    worksheet.delete_rows(2, len(worksheet.get_all_values()))
                
                # 새 데이터 추가
                if export_data:
                    worksheet.append_rows(export_data)
                logger.info(f"이슈 목록 {len(export_data)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.error(f"이슈 목록 기록 실패: {e}", exc_info=True)
    
    def export_mapping(self, mapping_data: List[Dict[str, Any]]):
        """
        이슈-뉴스 매핑을 스프레드시트에 기록합니다.
        
        Args:
            mapping_data: 매핑 데이터 리스트
        """
        if not self.spreadsheet or not mapping_data:
            return
        
        tab_name = self.sheets_config.get('tabs', {}).get('mapping', '이슈-뉴스 매핑')
        
        # 헤더 정의
        headers = ['매핑 ID', '이슈 ID', '이슈 제목', '뉴스 ID', '뉴스 제목', '뉴스 링크', '생성일']
        
        # 데이터 정리
        export_data = []
        for mapping in mapping_data:
            export_data.append([
                mapping.get('mapping_id', ''),
                mapping.get('issue_id', ''),
                mapping.get('issue_title', ''),
                mapping.get('news_id', ''),
                mapping.get('news_title', ''),
                mapping.get('news_link', ''),
                mapping.get('created_at', '')
            ])
        
        try:
            worksheet = self._get_or_create_worksheet(tab_name, headers)
            if worksheet:
                # 기존 데이터 지우기 (헤더 제외)
                if len(worksheet.get_all_values()) > 1:
                    worksheet.delete_rows(2, len(worksheet.get_all_values()))
                
                # 새 데이터 추가
                if export_data:
                    worksheet.append_rows(export_data)
                logger.info(f"이슈-뉴스 매핑 {len(export_data)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.error(f"이슈-뉴스 매핑 기록 실패: {e}", exc_info=True)
    
    def export_cluster_stats(self, stats_data: List[Dict[str, Any]]):
        """
        클러스터 통계를 스프레드시트에 기록합니다.
        
        Args:
            stats_data: 통계 데이터 리스트
        """
        if not self.spreadsheet or not stats_data:
            return
        
        tab_name = self.sheets_config.get('tabs', {}).get('cluster_stats', '클러스터 통계')
        
        # 헤더 정의
        headers = ['클러스터 ID', '크기', '평균 중요도', '태그', '대표 뉴스 제목']
        
        # 데이터 정리
        export_data = []
        for stat in stats_data:
            # bytes 타입을 문자열로 변환
            cluster_id = stat.get('cluster_id', '')
            if isinstance(cluster_id, bytes):
                cluster_id = ''
            elif cluster_id is not None:
                cluster_id = str(cluster_id)
            
            size = stat.get('size', '')
            if isinstance(size, bytes):
                size = ''
            
            avg_importance = stat.get('avg_importance', '')
            if isinstance(avg_importance, bytes):
                avg_importance = ''
            
            tag = stat.get('tag', '')
            if isinstance(tag, bytes):
                tag = tag.decode('utf-8', errors='replace')
            
            representative_title = stat.get('representative_title', '')
            if isinstance(representative_title, bytes):
                representative_title = representative_title.decode('utf-8', errors='replace')
            
            export_data.append([
                cluster_id,
                size,
                avg_importance,
                tag,
                representative_title
            ])
        
        try:
            worksheet = self._get_or_create_worksheet(tab_name, headers)
            if worksheet:
                # 기존 데이터 지우기 (헤더 제외)
                if len(worksheet.get_all_values()) > 1:
                    worksheet.delete_rows(2, len(worksheet.get_all_values()))
                
                # 새 데이터 추가
                if export_data:
                    worksheet.append_rows(export_data)
                logger.info(f"클러스터 통계 {len(export_data)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.error(f"클러스터 통계 기록 실패: {e}", exc_info=True)

