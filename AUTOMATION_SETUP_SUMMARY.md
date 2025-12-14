# 자동화 설정 완료 요약

## ✅ 구현 완료 사항

### 1. 코드 수정

#### 새로 생성된 파일
- `modules/email_recipient_loader.py`: 스프레드시트에서 이메일 수신자 설정 로드
- `modules/email_sender.py`: 보고서를 이메일로 발송
- `.github/workflows/daily_report.yml`: GitHub Actions 워크플로우
- `EMAIL_SETUP_GUIDE.md`: 이메일 설정 가이드

#### 수정된 파일
- `main.py`: 보고서 생성 후 이메일 발송 로직 추가
- `env.example`: 이메일 설정 환경변수 추가
- `README.md`: 자동 실행 기능 설명 추가

### 2. 스프레드시트 설정

**"이메일 수신자 설정" 탭** 생성 필요:
- 컬럼: `email`, `name`, `report_groups`, `enabled`
- 실행: `python setup_email_recipients.py`

### 3. GitHub Actions 설정

**워크플로우 파일**: `.github/workflows/daily_report.yml`
- 스케줄: 매일 오전 7시 30분 (KST) = UTC 22:30
- 수동 실행 가능

## 📋 다음 단계 (사용자 작업)

### 1. GitHub Secrets 설정

GitHub 저장소의 **Settings > Secrets and variables > Actions**에서 다음 Secrets 추가:

**필수:**
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (Gemini 사용 시)
- `GOOGLE_SPREADSHEET_ID`
- `GOOGLE_SETTINGS_SPREADSHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON`

**이메일:**
- `SMTP_SERVER` (예: `smtp.gmail.com`)
- `SMTP_PORT` (예: `587`)
- `SMTP_USER` (이메일 주소)
- `SMTP_PASSWORD` (앱 비밀번호)
- `EMAIL_FROM` (발신자 이메일)

### 2. 스프레드시트 설정

1. **이메일 수신자 설정 탭 생성**
   ```bash
   python setup_email_recipients.py
   ```

2. **수신자 정보 입력**
   - 스프레드시트에서 "이메일 수신자 설정" 탭 열기
   - 예시 데이터를 실제 수신자 정보로 수정
   - `report_groups`에 받을 보고서 그룹 이름 입력
     - 예: `Macro` (하나만)
     - 예: `Macro, Korea 부동산` (여러 개)

### 3. Gmail 앱 비밀번호 생성 (Gmail 사용 시)

1. Google 계정 > 보안 > 2단계 인증 활성화
2. 앱 비밀번호 생성
3. 생성된 비밀번호를 `SMTP_PASSWORD`에 설정

## 🔄 실행 플로우

### 매일 오전 7시 30분 (자동)

```
GitHub Actions 트리거
  ↓
Python 환경 설정
  ↓
의존성 설치
  ↓
main.py 실행
  ├─ 뉴스 수집
  ├─ 뉴스 분석
  ├─ 보고서 생성 (그룹별)
  └─ 이메일 발송
      ├─ 스프레드시트에서 수신자 로드
      ├─ 수신자별 보고서 그룹 확인
      └─ 수신자별로 해당 보고서만 발송
```

### 수동 실행

GitHub Actions 탭에서 "Run workflow" 버튼으로 수동 실행 가능

## 📧 이메일 발송 로직

1. **수신자 로드**: 스프레드시트 "이메일 수신자 설정" 탭에서 로드
2. **보고서 매칭**: 각 수신자의 `report_groups`와 생성된 보고서 매칭
3. **이메일 발송**: 
   - 본문: 보고서 내용 (Markdown → HTML 변환)
   - 첨부: 보고서 파일 (.md)
   - 제목: `Daily Market Report - [보고서명] - YYYY-MM-DD`

## ⚙️ 설정 예시

### 스프레드시트 "이메일 수신자 설정" 탭

| email | name | report_groups | enabled |
|-------|------|--------------|---------|
| user1@example.com | 홍길동 | Macro | TRUE |
| user2@example.com | 김철수 | Macro, Korea 부동산 | TRUE |
| user3@example.com | 이영희 | Korea 부동산 | TRUE |

**결과:**
- `user1@example.com`: `Macro.md` 보고서만 받음
- `user2@example.com`: `Macro.md`, `Korea 부동산.md` 두 보고서 모두 받음
- `user3@example.com`: `Korea 부동산.md` 보고서만 받음

## 🔍 확인 사항

### GitHub Actions가 정상 작동하는지 확인

1. GitHub 저장소의 **Actions** 탭 확인
2. "Daily Report Generation" 워크플로우 실행 여부 확인
3. 로그에서 에러 메시지 확인

### 이메일 발송 확인

1. 수신자 이메일함 확인
2. 스팸함도 확인
3. GitHub Actions 로그에서 발송 결과 확인

## 📝 참고 문서

- [EMAIL_SETUP_GUIDE.md](EMAIL_SETUP_GUIDE.md): 상세 설정 가이드
- [REPORT_TEMPLATE_SETUP.md](REPORT_TEMPLATE_SETUP.md): 보고서 양식 설정 가이드


