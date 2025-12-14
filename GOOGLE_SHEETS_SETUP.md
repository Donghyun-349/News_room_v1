# 구글 스프레드시트 연동 설정 가이드

## 개요
뉴스 수집 및 분석 결과를 구글 스프레드시트에 자동으로 기록하는 기능입니다.

## 설정 방법

### 1. Google Cloud Console에서 서비스 계정 생성

1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. **API 및 서비스** > **라이브러리**에서 **Google Sheets API** 활성화
4. **API 및 서비스** > **사용자 인증 정보**로 이동
5. **사용자 인증 정보 만들기** > **서비스 계정** 선택
6. 서비스 계정 이름 입력 후 **만들기**
7. 역할은 **편집자** 또는 **소유자** 선택 (또는 역할 없이 진행)
8. **완료** 클릭

### 2. 서비스 계정 키 다운로드

1. 생성된 서비스 계정 클릭
2. **키** 탭으로 이동
3. **키 추가** > **새 키 만들기** 선택
4. 키 유형: **JSON** 선택
5. **만들기** 클릭하여 JSON 파일 다운로드

### 3. 스프레드시트 공유 설정

1. 구글 스프레드시트 열기
2. **공유** 버튼 클릭
3. 서비스 계정 이메일 주소 입력 (JSON 파일의 `client_email` 필드)
4. 권한: **편집자** 선택
5. **완료** 클릭

### 4. 환경 변수 설정

#### 방법 1: JSON 파일 경로 사용 (권장)

`.env` 파일에 추가:
```env
GOOGLE_SERVICE_ACCOUNT_JSON=./credentials/service_account.json
```

JSON 파일을 `credentials/` 폴더에 저장:
```
News_dev/
  credentials/
    service_account.json
```

#### 방법 2: JSON 내용 직접 설정

`.env` 파일에 추가:
```env
GOOGLE_SERVICE_ACCOUNT_INFO={"type":"service_account","project_id":"your-project",...}
```

### 5. 설정 파일 확인

`config/settings.yaml`에서 스프레드시트 ID 확인:
```yaml
google_sheets:
  enabled: true
  spreadsheet_id: "1fYl4NUIlYo67rqgGenH3ZEXaNrFPv-p_U6zwyJxQJvU"
```

## 탭 구성

### 설정 스프레드시트 탭

설정 스프레드시트(`settings_spreadsheet_id`)에는 다음 탭들이 필요합니다:

1. **카테고리 설정**: 뉴스 수집 카테고리 설정
2. **분류 태그**: 메타 태그 목록
3. **언론사 가중치**: 언론사 신뢰도 가중치
4. **시스템 설정**: 시스템 동작 설정

#### 시스템 설정 탭 구성

**시스템 설정** 탭은 다음과 같이 구성합니다:

| setting_name | value | 설명 |
|-------------|-------|------|
| reset_database | true/false | 실행 시 데이터베이스 리셋 여부 (기본값: false) |

예시:
- `reset_database` = `false`: 데이터베이스 리셋 안 함 (기존 데이터 유지)
- `reset_database` = `true`: 실행 시마다 데이터베이스 리셋

### 실행 결과 스프레드시트 탭

실행 결과 스프레드시트(`spreadsheet_id`)에는 다음 탭들이 자동으로 생성/업데이트됩니다:

1. **뉴스 수집**: 수집된 뉴스 목록
2. **이슈 목록**: 생성된 이슈 목록
3. **이슈-뉴스 매핑**: 이슈와 뉴스의 연결 관계

## 문제 해결

### 인증 오류
- 서비스 계정 JSON 파일 경로 확인
- 스프레드시트 공유 설정 확인 (서비스 계정 이메일 주소)
- Google Sheets API 활성화 확인

### 권한 오류
- 서비스 계정에 **편집자** 권한 부여 확인
- 스프레드시트 ID 확인

### 패키지 설치 오류
```bash
pip install gspread google-auth
```






