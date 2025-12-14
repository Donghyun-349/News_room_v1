"""
사용자 피드백 로더 모듈
스프레드시트에서 사용자 피드백을 읽어오는 모듈
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from modules.google_sheets import GoogleSheetsExporter

logger = logging.getLogger(__name__)


class FeedbackLoader:
    """스프레드시트에서 사용자 피드백을 로드하는 클래스"""
    
    def __init__(self):
        """피드백 로더 초기화"""
        import os
        from config import get_config
        config = get_config()
        sheets_config = config.get('google_sheets', {})
        # 설정 스프레드시트 ID 사용 (환경변수 우선, 없으면 설정 파일)
        settings_spreadsheet_id = os.getenv('GOOGLE_SETTINGS_SPREADSHEET_ID', '')
        if not settings_spreadsheet_id:
            settings_spreadsheet_id = sheets_config.get('settings_spreadsheet_id', '')
        self.sheets_exporter = GoogleSheetsExporter(spreadsheet_id=settings_spreadsheet_id)
        self.feedback_cache: List[Dict[str, Any]] = []
        self._load_feedback()
    
    def _load_feedback(self):
        """스프레드시트에서 피드백을 로드합니다."""
        if not self.sheets_exporter.spreadsheet:
            logger.warning("구글 스프레드시트가 연결되지 않아 피드백을 사용할 수 없습니다.")
            return
        
        try:
            # 설정 스프레드시트의 "feedback" 탭에서 읽기
            from config import get_config
            config = get_config()
            sheets_config = config.get('google_sheets', {})
            tabs_config = sheets_config.get('tabs', {})
            tab_name = tabs_config.get('feedback', 'feedback')  # 기본값: "feedback"
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            records = worksheet.get_all_records()
            
            feedbacks = []
            for record in records:
                # 필드 값을 문자열로 변환 (정수형 등 다른 타입 대응)
                def safe_str(value, default=''):
                    if value is None:
                        return default
                    return str(value).strip()
                
                # 필수 필드 확인
                news_title = safe_str(record.get('news_title', ''))
                user_comment = safe_str(record.get('user_comment', ''))
                
                if not news_title or not user_comment:
                    continue
                
                # 날짜 파싱
                date_str = safe_str(record.get('date', ''))
                date_obj = None
                if date_str:
                    try:
                        # 다양한 날짜 형식 지원
                        for fmt in ['%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                break
                            except:
                                continue
                    except:
                        pass
                
                # apply_to_importance 파싱
                apply_to_importance = False
                apply_str = safe_str(record.get('apply_to_importance', '')).lower()
                if apply_str in ['true', '1', 'yes', 'on', 'y']:
                    apply_to_importance = True
                
                feedback = {
                    'no': safe_str(record.get('no', '')),  # 번호 (선택사항)
                    'date': date_obj,
                    'date_str': date_str,
                    'news_title': news_title,
                    'source': safe_str(record.get('source', '')),
                    'user_comment': user_comment,
                    'apply_to_importance': apply_to_importance,
                    'raw_record': record  # 원본 데이터 보존
                }
                feedbacks.append(feedback)
            
            if feedbacks:
                self.feedback_cache = feedbacks
                logger.info(f"사용자 피드백 {len(feedbacks)}개를 로드했습니다.")
            else:
                logger.info("사용자 피드백이 없습니다.")
                
        except Exception as e:
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
            self.feedback_cache = []
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        모든 피드백을 가져옵니다.
        
        Returns:
            피드백 리스트
        """
        return self.feedback_cache
    
    def get_by_date(self, date: datetime) -> List[Dict[str, Any]]:
        """
        특정 날짜의 피드백을 가져옵니다.
        
        Args:
            date: 날짜
            
        Returns:
            해당 날짜의 피드백 리스트
        """
        return [
            fb for fb in self.feedback_cache
            if fb.get('date') and fb['date'].date() == date.date()
        ]
    
    def reload(self):
        """피드백을 다시 로드합니다."""
        self.feedback_cache.clear()
        self._load_feedback()

