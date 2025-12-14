"""
구글 스프레드시트에 Pandas DataFrame을 내보내는 모듈
"""
import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread가 설치되지 않았습니다. 구글 스프레드시트 연동이 비활성화됩니다.")

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SheetExporter:
    """구글 스프레드시트에 DataFrame을 내보내는 클래스"""
    
    def __init__(self):
        """구글 스프레드시트 클라이언트 초기화"""
        self.client = None
        self.spreadsheet = None
        
        if not GSPREAD_AVAILABLE:
            logger.warning("gspread가 설치되지 않아 구글 스프레드시트 연동을 사용할 수 없습니다.")
            return
        
        # 환경 변수에서 설정 읽기
        json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")
        
        if not json_path or not spreadsheet_id:
            logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON 또는 GOOGLE_SHEET_ID 환경변수가 설정되지 않았습니다.")
            return
        
        try:
            json_path = Path(json_path)
            if not json_path.exists():
                logger.error(f"서비스 계정 JSON 파일을 찾을 수 없습니다: {json_path}")
                return
            
            self.client = gspread.service_account(filename=str(json_path))
            self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            logger.info(f"구글 스프레드시트 연결 성공: {self.spreadsheet.title}")
        except Exception as e:
            logger.error(f"구글 스프레드시트 연결 실패: {e}")
            self.client = None
            self.spreadsheet = None
    
    def export_to_sheet(self, dataframe: pd.DataFrame, sheet_name: str = 'Sheet1', 
                       clear_existing: bool = True, append_mode: bool = False) -> bool:
        """
        Pandas DataFrame을 구글 스프레드시트의 지정된 시트에 내보냅니다.
        
        Args:
            dataframe: 내보낼 DataFrame
            sheet_name: 시트 이름 (기본값: 'Sheet1')
            clear_existing: 기존 데이터를 지울지 여부 (기본값: True)
            append_mode: 추가 모드 (True면 기존 데이터 뒤에 추가, False면 덮어쓰기)
        
        Returns:
            성공 여부 (bool)
        """
        if not self.spreadsheet:
            logger.error("스프레드시트가 초기화되지 않았습니다.")
            return False
        
        if dataframe.empty:
            logger.warning("내보낼 데이터가 없습니다.")
            return False
        
        try:
            # 시트 가져오기 또는 생성
            try:
                worksheet = self.spreadsheet.worksheet(sheet_name)
                logger.info(f"기존 시트 사용: {sheet_name}")
            except gspread.exceptions.WorksheetNotFound:
                # 시트가 없으면 생성
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows=max(1000, len(dataframe) + 100),
                    cols=max(20, len(dataframe.columns))
                )
                logger.info(f"새 시트 생성: {sheet_name}")
            
            # 추가 모드가 아닐 때만 기존 데이터 지우기
            if not append_mode:
                if clear_existing:
                    worksheet.clear()
                    logger.debug(f"기존 데이터 삭제 완료: {sheet_name}")
                
                # 헤더 추가
                headers = dataframe.columns.tolist()
                worksheet.append_row(headers)
                logger.debug(f"헤더 추가 완료: {headers}")
            
            # 데이터를 리스트로 변환 (NaN을 빈 문자열로 변환)
            # bytes 타입을 문자열로 변환
            df_clean = dataframe.fillna('').copy()
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # bytes 타입을 문자열로 변환
                    df_clean[col] = df_clean[col].apply(
                        lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) if x is not None else ''
                    )
            
            values = df_clean.values.tolist()
            
            # 배치 업데이트로 효율적으로 처리
            # Google Sheets API는 한 번에 최대 10,000개 셀까지 업데이트 가능
            # 한 행당 컬럼 수를 고려하여 배치 크기 조정
            batch_size = 500  # 행 단위 배치 크기
            total_rows = len(values)
            
            if total_rows > 0:
                # 추가 모드인 경우 현재 행 수 확인
                if append_mode:
                    existing_rows = len(worksheet.get_all_values())
                    start_row_offset = existing_rows
                else:
                    start_row_offset = 1  # 헤더 다음 행
                
                # 배치 단위로 나누어 업데이트
                for i in range(0, total_rows, batch_size):
                    batch = values[i:i + batch_size]
                    # 시작 행: 헤더 다음 행 또는 기존 데이터 다음 행
                    start_row = i + start_row_offset + 1
                    end_row = start_row + len(batch) - 1
                    
                    # 범위 지정 (워크시트 객체이므로 시트 이름 제외)
                    # 컬럼 문자 계산 (A=65, B=66, ...)
                    end_col = chr(64 + len(dataframe.columns)) if len(dataframe.columns) <= 26 else 'Z'
                    range_name = f"A{start_row}:{end_col}{end_row}"
                    
                    # 배치 업데이트
                    worksheet.update(range_name, batch, value_input_option='USER_ENTERED')
                    logger.debug(f"배치 업데이트 완료: {i+1}~{min(i+batch_size, total_rows)}행")
                
                logger.info(f"데이터 내보내기 완료: {total_rows}행, {len(dataframe.columns)}열 → {sheet_name}")
            else:
                logger.info(f"헤더만 추가됨: {sheet_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"스프레드시트 내보내기 실패 ({sheet_name}): {e}", exc_info=True)
            return False
