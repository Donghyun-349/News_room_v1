# 이메일 자동 발송 설정 가이드

## 개요

매일 아침 7시 30분에 자동으로 보고서를 생성하고, 스프레드시트 설정에 따라 수신자별로 이메일을 발송합니다.

## 전체 플로우

### 1. GitHub Actions 자동 실행
- 매일 오전 7시 30분 (KST) 자동 실행
- `main.py` 실행 → 뉴스 수집 → 분석 → 보고서 생성 → 이메일 발송

### 2. 스프레드시트에서 수신자 및 보고서 유형 설정
- "이메일 수신자 설정" 탭에서 수신자별 보고서 유형 지정
- 각 수신자는 자신이 받을 보고서 그룹을 지정

### 3. 이메일 발송
- 생성된 보고서 내용을 본문에 포함하여 발송
- 수신자별로 지정된 보고서만 발송

## 설정 방법

### 1. 스프레드시트 설정

#### "이메일 수신자 설정" 탭 생성

스크립트 실행:
```bash
python setup_email_recipients.py
```

또는 수동으로 스프레드시트에 다음 탭을 생성:

**탭 이름**: `이메일 수신자 설정`

**컬럼 구조**:
| email | name | report_groups | enabled |
|-------|------|--------------|---------|
| user1@example.com | 홍길동 | Macro | TRUE |
| user2@example.com | 김철수 | Macro, Korea 부동산 | TRUE |
| user3@example.com | 이영희 | Korea 부동산 | TRUE |

**컬럼 설명**:
- `email`: 수신자 이메일 주소
- `name`: 수신자 이름 (선택사항)
- `report_groups`: 받을 보고서 그룹 이름 (쉼표로 구분)
  - 예: `Macro` (하나만)
  - 예: `Macro, Korea 부동산` (여러 개)
- `enabled`: 활성화 여부 (`TRUE` 또는 `FALSE`)

### 2. 환경변수 설정

#### 로컬 테스트용 (.env 파일)

`.env` 파일에 다음을 추가:

```env
# Email Settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com
```

#### GitHub Actions용 (Secrets)

GitHub 저장소의 **Settings > Secrets and variables > Actions**에서 다음 Secrets 추가:

**필수 Secrets:**
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (Gemini 사용 시)
- `GOOGLE_SPREADSHEET_ID`
- `GOOGLE_SETTINGS_SPREADSHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON` (또는 `GOOGLE_SERVICE_ACCOUNT_INFO`)

**이메일 관련 Secrets:**
- `SMTP_SERVER` (예: `smtp.gmail.com`)
- `SMTP_PORT` (예: `587`)
- `SMTP_USER` (이메일 주소)
- `SMTP_PASSWORD` (앱 비밀번호)
- `EMAIL_FROM` (발신자 이메일, 보통 `SMTP_USER`와 동일)

### 3. Gmail 사용 시 설정

1. **2단계 인증 활성화**
   - Google 계정 > 보안 > 2단계 인증

2. **앱 비밀번호 생성**
   - Google 계정 > 보안 > 2단계 인증 > 앱 비밀번호
   - 앱 선택: "메일"
   - 기기 선택: "기타(맞춤 이름)" → "GitHub Actions" 입력
   - 생성된 16자리 비밀번호를 `SMTP_PASSWORD`에 설정

### 4. GitHub Actions 워크플로우 확인

`.github/workflows/daily_report.yml` 파일이 생성되어 있는지 확인:

- 스케줄: 매일 오전 7시 30분 (KST) = UTC 22:30
- 수동 실행: GitHub Actions 탭에서 `workflow_dispatch`로 수동 실행 가능

## 동작 방식

### 1. 보고서 생성
- `main.py` 실행
- 뉴스 수집 → 분석 → 보고서 생성
- 보고서 그룹별로 파일 저장 (예: `Macro.md`, `Korea 부동산.md`)

### 2. 이메일 발송
- 스프레드시트에서 수신자 목록 로드
- 각 수신자의 `report_groups` 확인
- 수신자별로 해당 보고서만 발송
  - 예: `report_groups: "Macro"` → `Macro.md`만 발송
  - 예: `report_groups: "Macro, Korea 부동산"` → 두 보고서 모두 발송

### 3. 이메일 형식
- **제목**: `Daily Market Report - [보고서명] - YYYY-MM-DD`
- **본문**: 보고서 내용 (Markdown → HTML 변환)
- **첨부**: 보고서 파일 (.md)

## 테스트 방법

### 로컬 테스트

1. 환경변수 설정 (`.env` 파일)
2. 스프레드시트에 테스트 수신자 추가
3. 수동 실행:
   ```bash
   python main.py
   ```

### GitHub Actions 테스트

1. GitHub 저장소의 **Actions** 탭으로 이동
2. "Daily Report Generation" 워크플로우 선택
3. "Run workflow" 버튼 클릭
4. 실행 결과 확인

## 문제 해결

### 이메일이 발송되지 않는 경우

1. **환경변수 확인**
   - `SMTP_USER`, `SMTP_PASSWORD`가 올바른지 확인
   - Gmail 사용 시 앱 비밀번호 사용 확인

2. **스프레드시트 확인**
   - "이메일 수신자 설정" 탭이 존재하는지 확인
   - `enabled`가 `TRUE`인지 확인
   - `report_groups`에 올바른 보고서 그룹 이름이 있는지 확인

3. **로그 확인**
   - GitHub Actions 로그에서 에러 메시지 확인
   - 로컬 실행 시 콘솔 출력 확인

### 보고서가 발송되지 않는 경우

1. **보고서 그룹 이름 확인**
   - 스프레드시트의 `report_groups`와 실제 보고서 그룹 이름이 일치하는지 확인
   - 대소문자 구분 확인

2. **보고서 파일 확인**
   - 보고서가 실제로 생성되었는지 확인
   - 파일명이 올바른지 확인

## 주의사항

1. **Gmail 앱 비밀번호**: 일반 비밀번호가 아닌 앱 비밀번호를 사용해야 합니다.
2. **스프레드시트 권한**: 서비스 계정이 "이메일 수신자 설정" 탭을 읽을 수 있는 권한이 있어야 합니다.
3. **보고서 그룹 이름**: 스프레드시트의 `report_groups`와 실제 보고서 그룹 이름이 정확히 일치해야 합니다.


