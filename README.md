# 🏗️ Smart Investment Assistant (SIA)

뉴스를 수집, 분석, 분류하여 투자 인사이트를 제공하는 지능형 시스템입니다.

## 주요 기능

- **타겟팅 수집**: 설정 가능한 키워드와 지역별 뉴스 수집
- **동적 분석**: DBSCAN 클러스터링과 LLM을 활용한 이슈 발견
- **하이브리드 태깅**: 동적 클러스터링 + 정적 태그 분류
- **트렌드 추적**: 태그별 시계열 데이터 시각화
- **모듈화 설계**: Config-Driven, Logic-UI 분리

## 기술 스택

- **Vector DB**: ChromaDB
- **Embedding**: OpenAI text-embedding-3-small
- **Clustering**: Scikit-learn DBSCAN
- **LLM**: OpenAI GPT-4o-mini
- **UI**: Streamlit
- **Database**: SQLite

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`env.example` 파일을 참고하여 `.env` 파일을 생성하세요:

```bash
cp env.example .env
```

`.env` 파일에 필요한 환경변수를 설정하세요:

```
# OpenAI API Key (임베딩용)
OPENAI_API_KEY=sk-your-api-key-here

# Gemini API Key (LLM용, 선택사항)
GEMINI_API_KEY=your-gemini-api-key-here

# Database URL (SQLite)
DATABASE_URL=sqlite:///./investment.db

# ChromaDB 저장 경로
CHROMA_DB_PATH=./chroma_db

# Google Sheets 스프레드시트 ID (선택사항)
GOOGLE_SPREADSHEET_ID=your-spreadsheet-id-here
GOOGLE_SETTINGS_SPREADSHEET_ID=your-settings-spreadsheet-id-here

# Google Service Account JSON 파일 경로 (선택사항)
GOOGLE_SERVICE_ACCOUNT_JSON=credentials/service_account.json
```

### 3. 데이터베이스 초기화

```bash
python database/setup_db.py
```

### 4. 설정 파일 확인

`config/settings.yaml` 파일에서 검색 그룹, 태그, 클러스터링 파라미터를 설정할 수 있습니다.

## 실행 방법

### 메인 프로세스 (스케줄러)

```bash
python main.py
```

### 웹 UI

```bash
streamlit run web/app.py
```

### 자동 실행 (GitHub Actions)

매일 오전 7시 30분에 자동으로 실행되며, 보고서가 이메일로 발송됩니다.

- **워크플로우 파일**: `.github/workflows/daily_report.yml`
- **스케줄**: 매일 오전 7시 30분 (KST)
- **수동 실행**: GitHub Actions 탭에서 수동 실행 가능

자세한 설정 방법은 [EMAIL_SETUP_GUIDE.md](EMAIL_SETUP_GUIDE.md)를 참고하세요.

## 프로젝트 구조

```
/smart-investment-assistant
├── config/              # 설정 파일
├── database/            # DB 스키마 및 관리
├── modules/             # 핵심 로직 (수집, 분석, 트렌드)
├── web/                 # Streamlit UI
├── tests/               # 단위 테스트
└── logs/                # 로그 파일
```

## 개발 상태

- ✅ Phase 1: 기반 공사 (완료)
- ⏳ Phase 2: 수집기 구현 (진행 예정)
- ⏳ Phase 3: 분석기 구현 (진행 예정)
- ⏳ Phase 4: 트렌드 분석기 (진행 예정)
- ⏳ Phase 5: UI 구현 (진행 예정)

## 라이선스

MIT










