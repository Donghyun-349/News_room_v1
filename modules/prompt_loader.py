"""
프롬프트 로더 모듈
구글 스프레드시트에서 프롬프트 지침을 읽어오는 모듈
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional
from modules.google_sheets import GoogleSheetsExporter

logger = logging.getLogger(__name__)


class PromptLoader:
    """스프레드시트에서 프롬프트를 로드하는 클래스"""
    
    def __init__(self):
        """프롬프트 로더 초기화"""
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
        self.prompts_cache: Dict[str, Dict[str, str]] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """스프레드시트에서 프롬프트를 로드합니다."""
        if not self.sheets_exporter.spreadsheet:
            logger.warning("구글 스프레드시트가 연결되지 않아 기본 프롬프트를 사용합니다.")
            return
        
        try:
            # "프롬프트 설정" 탭 찾기
            tab_name = "프롬프트 설정"
            worksheet = None
            
            try:
                worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            except:
                logger.warning(f"'{tab_name}' 탭을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
                return
            
            # 데이터 읽기
            records = worksheet.get_all_records()
            
            for record in records:
                prompt_type = record.get('prompt_type', '').strip()
                prompt_text = record.get('prompt_text', '').strip()
                system_prompt = record.get('system_prompt', '').strip()
                
                if prompt_type and prompt_text:
                    self.prompts_cache[prompt_type] = {
                        'prompt_text': prompt_text,
                        'system_prompt': system_prompt if system_prompt else None
                    }
                    logger.info(f"프롬프트 로드 완료: {prompt_type}")
            
            logger.info(f"총 {len(self.prompts_cache)}개 프롬프트를 로드했습니다.")
            
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {e}", exc_info=True)
    
    def get_prompt(self, prompt_type: str, default_prompt: str = None, 
                   default_system: str = None, **kwargs) -> Dict[str, str]:
        """
        프롬프트를 가져옵니다.
        
        Args:
            prompt_type: 프롬프트 타입 (micro_analysis, theme_consolidation, final_report)
            default_prompt: 기본 프롬프트 (스프레드시트에 없을 경우)
            default_system: 기본 시스템 프롬프트
            **kwargs: 프롬프트 템플릿에 전달할 변수들
        
        Returns:
            {'prompt': str, 'system_prompt': str} 딕셔너리
        """
        if prompt_type in self.prompts_cache:
            prompt_template = self.prompts_cache[prompt_type]['prompt_text']
            system_template = self.prompts_cache[prompt_type].get('system_prompt')
            
            # 템플릿 변수 치환
            try:
                prompt = prompt_template.format(**kwargs)
                system_prompt = system_template.format(**kwargs) if system_template else None
            except KeyError as e:
                logger.warning(f"프롬프트 템플릿 변수 오류: {e}, 기본 프롬프트 사용")
                prompt = default_prompt.format(**kwargs) if default_prompt else default_prompt
                system_prompt = default_system
            except Exception as e:
                logger.warning(f"프롬프트 포맷팅 오류: {e}, 원본 사용")
                prompt = prompt_template
                system_prompt = system_template
        else:
            # 기본 프롬프트 사용
            if default_prompt:
                prompt = default_prompt.format(**kwargs) if kwargs else default_prompt
            else:
                prompt = default_prompt
            system_prompt = default_system
        
        return {
            'prompt': prompt,
            'system_prompt': system_prompt
        }
    
    def reload(self):
        """프롬프트를 다시 로드합니다."""
        self.prompts_cache.clear()
        self._load_prompts()

