# Streamlit 배포 체크리스트

## ✅ 프로젝트 정리 완료

### 삭제된 파일들
- ✅ 설정 스크립트 (setup_*.py) - 7개
- ✅ 마이그레이션 스크립트 (migrate_*.py) - 2개
- ✅ 중복 export 스크립트 - 5개
- ✅ 개별 실행 스크립트 (run_*.py) - 2개
- ✅ 개별 뷰 스크립트 (view_*.py) - 2개
- ✅ 통계 스크립트 (collection_stats.py) - 1개
- ✅ 불필요한 문서 파일 - 3개
- ✅ 생성된 보고서 파일 - 1개

**총 23개 파일 삭제 완료**

### 유지된 핵심 파일들
- ✅ `main.py` - 메인 실행 파일
- ✅ `generate_daily_report.py` - 보고서 생성
- ✅ `export_to_google_sheets.py` - 스프레드시트 내보내기
- ✅ `web/app.py` - Streamlit 앱
- ✅ `modules/` - 핵심 모듈들
- ✅ `config/` - 설정 파일
- ✅ `database/` - DB 관리

### 최적화된 requirements.txt
- ✅ 사용하지 않는 패키지 제거 (langchain, apscheduler, oauth2client)
- ✅ pytest는 주석 처리 (개발 환경에서만 필요)

## Streamlit 배포 준비

### 생성된 파일
- ✅ `.streamlit/config.toml` - Streamlit 설정
- ✅ `Procfile` - 배포 플랫폼용 (Heroku, Railway 등)

### 환경 변수 설정 필요
`.env` 파일에 다음 변수들이 설정되어 있어야 합니다:
- `OPENAI_API_KEY` - 필수
- `GEMINI_API_KEY` - 선택사항 (LLM provider가 gemini인 경우)
- `DATABASE_URL` - 기본값: `sqlite:///./investment.db`
- `CHROMA_DB_PATH` - 기본값: `./chroma_db`
- `GOOGLE_SERVICE_ACCOUNT_JSON` - 구글 스프레드시트 연동 시 필요

### 배포 전 확인 사항
1. ✅ `.env` 파일이 `.gitignore`에 포함되어 있는지 확인
2. ✅ `credentials/` 폴더가 `.gitignore`에 포함되어 있는지 확인
3. ✅ 데이터베이스 파일(`*.db`)이 `.gitignore`에 포함되어 있는지 확인
4. ✅ `chroma_db/` 폴더가 `.gitignore`에 포함되어 있는지 확인

## 배포 방법

### Streamlit Cloud
1. GitHub에 프로젝트 푸시
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 앱 생성
3. 환경 변수 설정
4. `web/app.py`를 메인 파일로 설정

### Heroku
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

### Railway
1. Railway에 프로젝트 연결
2. 환경 변수 설정
3. 자동 배포

## 실행 방법

### 로컬 개발
```bash
# 메인 프로세스
python main.py

# 보고서 생성
python generate_daily_report.py

# Streamlit 앱
streamlit run web/app.py
```

### 배포 환경
```bash
streamlit run web/app.py --server.port=$PORT --server.address=0.0.0.0
```



