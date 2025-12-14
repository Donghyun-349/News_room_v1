# GitHub 업로드 전 보안 점검 체크리스트

## ✅ 확인 완료 항목

### 1. .gitignore 설정
- ✅ `.env` 파일 무시
- ✅ `credentials/` 폴더 무시
- ✅ `*.db`, `*.sqlite`, `*.sqlite3` 파일 무시
- ✅ `chroma_db/` 폴더 무시
- ✅ `logs/` 폴더 무시
- ✅ `__pycache__/` 폴더 무시
- ✅ 생성된 보고서 파일들 무시

### 2. 민감한 정보 확인
- ✅ API 키는 `.env` 파일에만 저장 (코드에 하드코딩 없음)
- ✅ 서비스 계정 JSON은 `credentials/` 폴더에만 저장
- ✅ 데이터베이스 파일은 무시됨

## ⚠️ 확인 필요 항목

### 1. `config/settings.yaml`의 스프레드시트 ID
현재 `settings.yaml` 파일에 다음 스프레드시트 ID가 하드코딩되어 있습니다:

```yaml
google_sheets:
  spreadsheet_id: "1fYl4NUIlYo67rqgGenH3ZEXaNrFPv-p_U6zwyJxQJvU"
  settings_spreadsheet_id: "1l4C0aoLfHHqQ7d74EkdmTiP_SB6T3bTLNgq2jystBQo"
```

**권장 사항:**
- 스프레드시트 ID는 일반적으로 공개되어도 문제가 없습니다 (접근 권한이 없으면 사용 불가)
- 하지만 민감한 데이터가 포함된 스프레드시트라면 제거하는 것을 권장합니다
- 제거 방법:
  1. `settings.yaml`에서 스프레드시트 ID 제거
  2. 환경변수나 `.env` 파일로 이동
  3. 또는 예시 값으로 변경

### 2. `export_issues_only.py` 파일
- 현재 `.gitignore`에 `export_issues_only.py`가 포함되어 있지만 파일이 존재합니다
- 이 파일이 필요하다면 `.gitignore`에서 제거하거나, 불필요하다면 파일을 삭제하세요

## 📝 추가 권장 사항

### 1. 환경변수로 이동 가능한 설정
다음 설정들을 환경변수로 이동하는 것을 고려하세요:
- 스프레드시트 ID
- 기타 민감할 수 있는 설정값

### 2. README 업데이트
- `.env.example` 파일 사용 방법 명시
- 스프레드시트 설정 방법 안내
- 보안 주의사항 추가

### 3. .gitignore 추가 고려사항
현재 `.gitignore`에 다음이 포함되어 있습니다:
- ✅ 모든 필수 항목 포함
- ✅ 생성된 보고서 파일들 무시
- ✅ 임시 파일들 무시

## 🔒 최종 확인 사항

업로드 전에 다음을 확인하세요:

1. [ ] `.env` 파일이 존재하지 않거나 무시되는지 확인
2. [ ] `credentials/` 폴더가 무시되는지 확인
3. [ ] 데이터베이스 파일(`*.db`)이 무시되는지 확인
4. [ ] 로그 파일이 무시되는지 확인
5. [ ] `settings.yaml`의 스프레드시트 ID가 공개해도 되는지 확인
6. [ ] 하드코딩된 API 키나 비밀번호가 없는지 확인
7. [ ] 생성된 보고서 파일들이 무시되는지 확인

## 🚀 업로드 준비 완료

위 항목들을 확인한 후 GitHub에 업로드하세요.


