"""
설정 로더 모듈 (Singleton 패턴)
스프레드시트 우선, 없으면 YAML 파일을 읽어서 딕셔너리로 반환
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Singleton 패턴으로 설정을 로드하는 클래스"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """스프레드시트 우선, 없으면 settings.yaml 파일을 로드"""
        # 먼저 YAML 파일 로드 (기본 설정)
        config_path = Path(__file__).parent / "settings.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {config_path}"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 스프레드시트에서 설정 로드 시도 (우선 적용)
        spreadsheet_loaded = False
        try:
            from modules.settings_loader import SettingsLoader
            settings_loader = SettingsLoader()
            
            # 스프레드시트 연결 상태 확인
            if not settings_loader.sheets_exporter.spreadsheet:
                logger.warning("⚠️  구글 스프레드시트가 연결되지 않아 YAML 설정을 사용합니다.")
                logger.warning("   스프레드시트 설정을 사용하려면 구글 스프레드시트 연결을 확인하세요.")
            else:
                # 스프레드시트에서 로드한 설정으로 덮어쓰기
                spreadsheet_categories = settings_loader.get('categories')
                if spreadsheet_categories and len(spreadsheet_categories) > 0:
                    self._config['categories'] = spreadsheet_categories
                    logger.info(f"✅ 스프레드시트에서 카테고리 설정 {len(spreadsheet_categories)}개를 로드했습니다.")
                    spreadsheet_loaded = True
                else:
                    logger.warning("⚠️  스프레드시트에서 카테고리를 로드하지 못했습니다. YAML 설정을 사용합니다.")
                
                spreadsheet_meta_tags = settings_loader.get('meta_tags')
                if spreadsheet_meta_tags and len(spreadsheet_meta_tags) > 0:
                    self._config['meta_tags'] = spreadsheet_meta_tags
                    logger.info(f"✅ 스프레드시트에서 분류 태그 {len(spreadsheet_meta_tags)}개를 로드했습니다.")
                else:
                    logger.debug("스프레드시트에서 분류 태그를 로드하지 못했습니다. YAML 설정을 사용합니다.")
                
                spreadsheet_publisher_weights = settings_loader.get('publisher_weights')
                if spreadsheet_publisher_weights:
                    self._config['publisher_weights'] = spreadsheet_publisher_weights
                    logger.info("✅ 스프레드시트에서 언론사 가중치를 로드했습니다.")
                else:
                    logger.debug("스프레드시트에서 언론사 가중치를 로드하지 못했습니다. YAML 설정을 사용합니다.")
                
                spreadsheet_report_groups = settings_loader.get('report_groups')
                if spreadsheet_report_groups and len(spreadsheet_report_groups) > 0:
                    self._config['report_groups'] = spreadsheet_report_groups
                    logger.info(f"✅ 스프레드시트에서 보고서 그룹 설정 {len(spreadsheet_report_groups)}개를 로드했습니다.")
                else:
                    logger.debug("스프레드시트에서 보고서 그룹 설정을 로드하지 못했습니다. YAML 설정을 사용합니다.")
                
        except Exception as e:
            logger.warning(f"⚠️  스프레드시트 설정 로드 실패 (YAML 사용): {e}")
            logger.warning("   스프레드시트 설정을 사용하려면 오류를 확인하세요.")
        
        # 최종 확인: 카테고리가 로드되었는지 확인
        if not spreadsheet_loaded and self._config.get('categories'):
            logger.warning(f"⚠️  YAML 파일의 카테고리 {len(self._config['categories'])}개를 사용합니다.")
            logger.warning("   스프레드시트에서 카테고리를 로드하려면 구글 스프레드시트 연결을 확인하세요.")
    
    def get(self, key: str = None, default: Any = None) -> Any:
        """
        설정 값을 가져옵니다.
        
        Args:
            key: 점(.)으로 구분된 키 경로 (예: 'clustering.eps')
            default: 키가 없을 때 반환할 기본값
        
        Returns:
            설정 값 또는 전체 설정 딕셔너리
        """
        if self._config is None:
            self._load_config()
        
        if key is None:
            return self._config
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        """설정을 다시 로드합니다"""
        self._config = None
        self._load_config()


# Singleton 인스턴스 생성
config = ConfigLoader()


def get_config(key: str = None, default: Any = None) -> Any:
    """
    편의 함수: 설정 값을 가져옵니다.
    
    Usage:
        from config import get_config
        eps = get_config('clustering.eps')
        search_groups = get_config('search_groups')
    """
    return config.get(key, default)









