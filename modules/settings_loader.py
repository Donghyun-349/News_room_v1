"""
스프레드시트 설정 로더 모듈
구글 스프레드시트에서 설정을 읽어오는 모듈
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from modules.google_sheets import GoogleSheetsExporter

logger = logging.getLogger(__name__)


class SettingsLoader:
    """스프레드시트에서 설정을 로드하는 클래스"""
    
    def __init__(self):
        """설정 로더 초기화"""
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
        self.settings_cache: Dict[str, Any] = {}
        self._load_settings()
    
    def _load_settings(self):
        """스프레드시트에서 설정을 로드합니다."""
        if not self.sheets_exporter.spreadsheet:
            logger.warning("구글 스프레드시트가 연결되지 않아 스프레드시트 설정을 사용할 수 없습니다.")
            return
        
        try:
            # 1. 카테고리 설정 로드
            self._load_categories()
            
            # 2. 분류 태그 로드
            self._load_meta_tags()
            
            # 3. 언론사 가중치 로드
            self._load_publisher_weights()
            
            # 4. 시스템 설정 로드
            self._load_system_settings()
            
            # 5. 보고서 그룹 설정 로드
            self._load_report_groups()
            
            # 6. 보고서 양식 설정 로드
            self._load_report_template()
            
            logger.info("스프레드시트에서 설정을 로드했습니다.")
            
        except Exception as e:
            logger.error(f"스프레드시트 설정 로드 실패: {e}", exc_info=True)
    
    def _load_categories(self):
        """카테고리 설정을 로드합니다.
        
        스프레드시트 컬럼 순서:
        A: 리포트 그룹 (report_group)
        B: 카테고리 이름 (name)
        C: max 기사 (max_articles)
        D: RSS url (rss_url_template)
        E: 지역 (region)
        F: 언어 (language)
        G: 키워드 (keywords)
        """
        # #region agent log
        import json
        import time
        log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"settings_loader.py:67","message":"_load_categories entry","data":{"tab_name":"카테고리 설정"},"timestamp":int(time.time()*1000)})+'\n')
        except Exception as ex:
            logger.debug(f"Debug log write failed: {ex}")
        # #endregion
        
        tab_name = "카테고리 설정"
        
        try:
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            records = worksheet.get_all_records()
            
            # #region agent log
            try:
                import time
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"settings_loader.py:80","message":"Records loaded from sheet","data":{"total_records":len(records),"first_record_keys":list(records[0].keys()) if records else []},"timestamp":int(time.time()*1000)})+'\n')
            except Exception as ex:
                logger.debug(f"Debug log write failed: {ex}")
            # #endregion
            
            categories = []
            skipped_count = 0
            for idx, record in enumerate(records):
                # #region agent log
                try:
                    import time
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"settings_loader.py:91","message":"Processing record","data":{"record_index":idx,"record_keys":list(record.keys()),"name_value":record.get('name'),"name_type":type(record.get('name')).__name__},"timestamp":int(time.time()*1000)})+'\n')
                except Exception as ex:
                    logger.debug(f"Debug log write failed: {ex}")
                # #endregion
                
                if not record.get('name'):
                    skipped_count += 1
                    # #region agent log
                    try:
                        import time
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"settings_loader.py:99","message":"Skipped record - no name","data":{"record_index":idx},"timestamp":int(time.time()*1000)})+'\n')
                    except Exception as ex:
                        logger.debug(f"Debug log write failed: {ex}")
                    # #endregion
                    continue
                
                # keywords를 쉼표로 구분된 문자열에서 리스트로 변환
                keywords_str = record.get('keywords', '')
                if keywords_str is not None:
                    keywords_str = str(keywords_str).strip()
                else:
                    keywords_str = ''
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
                
                # report_group 필드 읽기 (대소문자 구분 없이, 타입 안전)
                def safe_str(value, default=''):
                    if value is None:
                        return default
                    return str(value).strip()
                
                report_group = safe_str(record.get('report_group', ''))
                if not report_group:
                    report_group = safe_str(record.get('Report Group', ''))  # 기존 컬럼명 지원
                if not report_group:
                    report_group = safe_str(record.get('report group', ''))  # 소문자 공백 지원
                
                # 필드 값을 안전하게 문자열로 변환
                def safe_str(value, default=''):
                    if value is None:
                        return default
                    return str(value).strip()
                
                category = {
                    'name': safe_str(record.get('name', '')),
                    'report_group': report_group,  # 보고서 그룹 추가
                    'max_articles': int(record.get('max_articles', 100)) if record.get('max_articles') else 100,
                    'rss_url_template': safe_str(record.get('rss_url_template', 'https://news.google.com/rss/search?q={keyword}&hl={hl}&gl={gl}&ceid={ceid}')),
                    'region': safe_str(record.get('region', 'us')),
                    'language': safe_str(record.get('language', 'en')),
                    'keywords': keywords
                }
                
                # #region agent log
                try:
                    import time
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B,C,D,E","location":"settings_loader.py:143","message":"Category parsed","data":{"name":category['name'],"report_group":category['report_group'],"max_articles":category['max_articles'],"rss_url":category['rss_url_template'],"region":category['region'],"language":category['language'],"keywords_count":len(category['keywords']),"keywords":category['keywords']},"timestamp":int(time.time()*1000)})+'\n')
                except Exception as ex:
                    logger.debug(f"Debug log write failed: {ex}")
                # #endregion
                
                categories.append(category)
            
            # #region agent log
            try:
                import time
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"settings_loader.py:154","message":"_load_categories exit","data":{"total_records":len(records),"categories_loaded":len(categories),"skipped":skipped_count},"timestamp":int(time.time()*1000)})+'\n')
            except Exception as ex:
                logger.debug(f"Debug log write failed: {ex}")
            # #endregion
            
            if categories:
                self.settings_cache['categories'] = categories
                logger.info(f"카테고리 {len(categories)}개를 로드했습니다.")
        except Exception as e:
            # #region agent log
            try:
                import time
                log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"settings_loader.py:174","message":"Exception in _load_categories","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)})+'\n')
            except Exception as ex:
                logger.debug(f"Debug log write failed: {ex}")
            # #endregion
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
    
    def _load_meta_tags(self):
        """분류 태그를 로드합니다."""
        tab_name = "분류 태그"
        
        try:
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            records = worksheet.get_all_records()
            
            meta_tags = []
            for record in records:
                tag = record.get('tag_name', '').strip()
                if tag:
                    meta_tags.append(tag)
            
            if meta_tags:
                self.settings_cache['meta_tags'] = meta_tags
                logger.info(f"분류 태그 {len(meta_tags)}개를 로드했습니다.")
        except Exception as e:
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
    
    def _load_publisher_weights(self):
        """언론사 가중치를 로드합니다."""
        tab_name = "언론사 가중치"
        
        try:
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            records = worksheet.get_all_records()
            
            publisher_weights = {
                'tier1': {'sources': [], 'weight': 1.35},
                'tier2': {'sources': [], 'weight': 1.2},
                'default': {'weight': 1.0}
            }
            
            for record in records:
                tier = record.get('tier', '').strip().lower()
                source = record.get('source', '').strip()
                weight = record.get('weight', '')
                
                if not source:
                    continue
                
                # weight가 있으면 float로 변환, 없으면 tier 기본값 사용
                if weight:
                    try:
                        weight_value = float(weight)
                    except:
                        weight_value = publisher_weights.get(tier, {}).get('weight', 1.0)
                else:
                    weight_value = publisher_weights.get(tier, {}).get('weight', 1.0)
                
                if tier in ['tier1', 'tier2']:
                    if tier not in publisher_weights:
                        publisher_weights[tier] = {'sources': [], 'weight': weight_value}
                    publisher_weights[tier]['sources'].append(source)
                    publisher_weights[tier]['weight'] = weight_value
                elif tier == 'default':
                    publisher_weights['default']['weight'] = weight_value
            
            # tier별로 weight가 설정되지 않았으면 기본값 사용
            if 'tier1' in publisher_weights and not publisher_weights['tier1'].get('weight'):
                publisher_weights['tier1']['weight'] = 1.35
            if 'tier2' in publisher_weights and not publisher_weights['tier2'].get('weight'):
                publisher_weights['tier2']['weight'] = 1.2
            
            self.settings_cache['publisher_weights'] = publisher_weights
            logger.info("언론사 가중치를 로드했습니다.")
        except Exception as e:
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
    
    def _load_system_settings(self):
        """
        시스템 설정을 로드합니다.

        현재는 단일 플래그(reset_database)만 사용하며,
        '시스템 설정' 시트의 B2 셀 값을 읽어 True/False 로 해석합니다.
        """
        tab_name = "시스템 설정"
        
        try:
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            
            # B2 셀 값 읽기 (reset_database 플래그)
            raw = worksheet.acell("B2").value
            raw_str = str(raw).strip().lower() if raw is not None else ""
            
            if not raw_str:
                reset = False
            elif raw_str in ['true', '1', 'yes', 'on']:
                reset = True
            elif raw_str in ['false', '0', 'no', 'off']:
                reset = False
            else:
                logger.warning(f"'시스템 설정'!B2 값을 해석할 수 없습니다: {raw!r}, 기본값 False 를 사용합니다.")
                reset = False
            
            system_settings = {'reset_database': reset}
            self.settings_cache['system_settings'] = system_settings
            logger.info(f"시스템 설정 로드 완료: reset_database={reset}")
        except Exception as e:
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
            # 기본값 설정
            self.settings_cache['system_settings'] = {'reset_database': False}
    
    def _load_report_groups(self):
        """보고서 그룹 설정을 카테고리에서 자동 생성합니다."""
        # 카테고리에서 report_group 추출하여 자동 생성
        categories = self.settings_cache.get('categories', [])
        
        if not categories:
            logger.warning("카테고리가 없어 보고서 그룹을 생성할 수 없습니다.")
            return
        
        # report_group별로 카테고리 그룹핑
        report_groups_dict = {}
        for category in categories:
            report_group = category.get('report_group', '').strip()
            category_name = category.get('name', '').strip()
            
            if not report_group:
                # report_group이 없으면 기본 그룹으로 처리
                report_group = 'default'
            
            if report_group not in report_groups_dict:
                report_groups_dict[report_group] = {
                    'name': report_group,
                    'categories': [],
                    'output_file': f"{report_group.lower().replace(' ', '_')}.md"
                }
            
            report_groups_dict[report_group]['categories'].append(category_name)
        
        # 딕셔너리를 리스트로 변환
        report_groups = list(report_groups_dict.values())
        
        if report_groups:
            self.settings_cache['report_groups'] = report_groups
            logger.info(f"보고서 그룹 {len(report_groups)}개를 카테고리에서 자동 생성했습니다.")
            for group in report_groups:
                logger.info(f"  - {group['name']}: {', '.join(group['categories'])}")
        else:
            logger.warning("보고서 그룹을 생성할 수 없습니다.")
    
    def _load_report_template(self):
        """보고서 양식 설정을 로드합니다."""
        tab_name = "보고서 양식 설정"
        
        try:
            worksheet = self.sheets_exporter.spreadsheet.worksheet(tab_name)
            records = worksheet.get_all_records()
            
            template_sections = {}
            for record in records:
                if not record.get('enabled', True):  # enabled가 False면 건너뛰기
                    continue
                    
                section_id = record.get('section_id', '').strip()
                section_name = record.get('section_name', '').strip()
                template_text = record.get('template_text', '').strip()
                section_order = record.get('section_order', '')
                
                if section_id and template_text:
                    try:
                        order = int(section_order) if section_order else 999
                    except:
                        order = 999
                    
                    template_sections[section_id] = {
                        'section_name': section_name,
                        'template': template_text,
                        'order': order
                    }
            
            if template_sections:
                self.settings_cache['report_template'] = template_sections
                logger.info(f"보고서 양식 설정 {len(template_sections)}개 섹션을 로드했습니다.")
            else:
                self.settings_cache['report_template'] = None
                logger.warning("보고서 양식 설정이 비어있습니다.")
        
        except Exception as e:
            logger.warning(f"'{tab_name}' 탭을 찾을 수 없거나 로드 실패: {e}")
            self.settings_cache['report_template'] = None
    
    def get_report_template(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """보고서 양식 템플릿을 가져옵니다."""
        return self.settings_cache.get('report_template')
    
    def get(self, key: str = None, default: Any = None) -> Any:
        """
        설정 값을 가져옵니다.
        
        Args:
            key: 설정 키 (categories, meta_tags, publisher_weights)
            default: 기본값
        
        Returns:
            설정 값
        """
        if key is None:
            return self.settings_cache
        
        return self.settings_cache.get(key, default)
    
    def reload(self):
        """설정을 다시 로드합니다."""
        self.settings_cache.clear()
        self._load_settings()

