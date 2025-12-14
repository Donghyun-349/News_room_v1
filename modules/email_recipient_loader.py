"""
이메일 수신자 로더 모듈
스프레드시트에서 이메일 수신자 설정을 읽어오는 모듈
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from modules.google_sheets import GoogleSheetsExporter

logger = logging.getLogger(__name__)


class EmailRecipientLoader:
    """스프레드시트에서 이메일 수신자 설정을 로드하는 클래스"""
    
    def __init__(self):
        """이메일 수신자 로더 초기화"""
        # 순환 참조 방지를 위해 직접 YAML 파일 읽기
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        sheets_config = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    sheets_config = config.get('google_sheets', {})
            except Exception as e:
                logger.warning(f"YAML 설정 파일 읽기 실패: {e}")
        
        # 설정 스프레드시트 ID 사용 (환경변수 우선, 없으면 설정 파일)
        settings_spreadsheet_id = os.getenv('GOOGLE_SETTINGS_SPREADSHEET_ID', '')
        if not settings_spreadsheet_id:
            settings_spreadsheet_id = sheets_config.get('settings_spreadsheet_id', '')
        self.sheets_exporter = GoogleSheetsExporter(spreadsheet_id=settings_spreadsheet_id)
        self.recipients_cache: List[Dict[str, Any]] = []
        self._load_recipients()
    
    def _load_recipients(self):
        """스프레드시트에서 이메일 수신자 설정을 로드합니다."""
        if not self.sheets_exporter.spreadsheet:
            logger.warning("구글 스프레드시트가 연결되지 않아 이메일 수신자 설정을 사용할 수 없습니다.")
            return
        
        try:
            tab_name = "이메일 수신자 설정"
            worksheet = None
            
            try:
                worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            except:
                logger.warning(f"'{tab_name}' 탭을 찾을 수 없습니다.")
                return
            
            records = worksheet.get_all_records()
            
            recipients = []
            for record in records:
                email = record.get('email', '').strip()
                name = record.get('name', '').strip()
                report_groups_str = record.get('report_groups', '').strip()
                enabled = record.get('enabled', True)
                
                # enabled가 False면 건너뛰기
                if isinstance(enabled, str):
                    enabled = enabled.upper() == 'TRUE'
                
                if not enabled:
                    continue
                
                if not email:
                    continue
                
                # report_groups 파싱 (쉼표로 구분)
                report_groups = []
                if report_groups_str:
                    report_groups = [g.strip() for g in report_groups_str.split(',') if g.strip()]
                
                recipients.append({
                    'email': email,
                    'name': name or email.split('@')[0],  # 이름이 없으면 이메일 앞부분 사용
                    'report_groups': report_groups
                })
            
            self.recipients_cache = recipients
            logger.info(f"이메일 수신자 {len(recipients)}명을 로드했습니다.")
            
        except Exception as e:
            logger.error(f"이메일 수신자 설정 로드 실패: {e}", exc_info=True)
            self.recipients_cache = []
    
    def get_all(self) -> List[Dict[str, Any]]:
        """모든 수신자 목록을 가져옵니다."""
        return self.recipients_cache
    
    def get_by_report_group(self, report_group_name: str) -> List[Dict[str, Any]]:
        """특정 보고서 그룹을 받는 수신자 목록을 가져옵니다."""
        recipients = []
        for recipient in self.recipients_cache:
            if report_group_name in recipient.get('report_groups', []):
                recipients.append(recipient)
        return recipients
    
    def reload(self):
        """수신자 설정을 다시 로드합니다."""
        self.recipients_cache.clear()
        self._load_recipients()


