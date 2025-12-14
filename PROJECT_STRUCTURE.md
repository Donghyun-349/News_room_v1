# 프로젝트 구조

## 핵심 파일

### 실행 파일
- `main.py` - 메인 실행 파일 (뉴스 수집 + 분석)
- `generate_daily_report.py` - 일일 보고서 생성
- `export_to_google_sheets.py` - 스프레드시트 내보내기
- `web/app.py` - Streamlit 웹 앱

### 설정 파일
- `config/settings.yaml` - 기본 설정 파일
- `config/__init__.py` - 설정 로더 (스프레드시트 우선)
- `.env` - 환경 변수 (로컬만, gitignore됨)
- `.streamlit/config.toml` - Streamlit 설정

### 데이터베이스
- `database/schema.sql` - DB 스키마
- `database/db_manager.py` - DB 관리 모듈
- `database/__init__.py` - DB 모듈 초기화

### 핵심 모듈
- `modules/news_collector.py` - 뉴스 수집기
- `modules/news_analyzer.py` - 뉴스 분석기 (클러스터링, 이슈 생성)
- `modules/feedback_loader.py` - 사용자 피드백 로더
- `modules/feedback_analyzer.py` - 피드백 분석기
- `modules/google_sheets.py` - 구글 스프레드시트 연동
- `modules/settings_loader.py` - 스프레드시트 설정 로더
- `modules/prompt_loader.py` - 프롬프트 로더
- `modules/trend_calculator.py` - 트렌드 계산기
- `modules/sheet_exporter.py` - 스프레드시트 내보내기 (레거시)

### 웹 UI
- `web/app.py` - Streamlit 메인 앱
- `web/view_components.py` - UI 컴포넌트

## 데이터 흐름

```
뉴스 수집 (main.py)
  ↓
카테고리별 분석 (카테고리별 순차 처리)
  ↓
DB 저장
  ↓
스프레드시트 내보내기
  ↓
보고서 생성 (generate_daily_report.py)
  ↓
Streamlit 앱 (web/app.py)
```

## 설정 관리

### 스프레드시트 설정 (우선)
- 카테고리 설정 탭: `report_group`, `name`, `max_articles`, ...
- 분류 태그 탭
- 언론사 가중치 탭
- 시스템 설정 탭
- 프롬프트 탭
- 사용자 피드백 탭 (실행 결과 스프레드시트)

### YAML 설정 (기본값)
- `config/settings.yaml` - 스프레드시트에서 로드 실패 시 사용



