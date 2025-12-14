# 설정 파일 관리 및 업데이트 가이드

## 🔄 설정 변경 반영 방식

### 현재 구조

1. **`config/settings.yaml`** (GitHub에 포함됨)
   - 검색 그룹, 키워드, 태그 등 공개 가능한 설정
   - 팀원들과 공유 가능
   - 코드와 함께 버전 관리

2. **`.env`** (GitHub에 제외됨)
   - API 키, 데이터베이스 경로 등 민감한 정보
   - 개인별로 다를 수 있음
   - `.gitignore`에 포함

### Streamlit에서 설정 변경 반영

**현재 상태: 자동 반영 안 됨** ❌

Streamlit은 설정 파일을 앱 시작 시 한 번만 읽습니다. 변경사항을 반영하려면:
1. Streamlit 서버 재시작 필요
2. 또는 설정 변경 UI 구현 필요

## 💡 해결 방안

### 방안 1: Streamlit 서버 재시작 (현재 방식)

```bash
# Streamlit 실행 중
# Ctrl + C로 중지 후

# 설정 파일 수정
# config/settings.yaml 편집

# Streamlit 재시작
streamlit run web/app.py
```

**장점:**
- 간단함
- 코드 수정 불필요

**단점:**
- 서버 재시작 필요
- 사용 중이면 불편

### 방안 2: 설정 관리 UI 추가 (권장) ⭐

Streamlit 앱에 설정 변경 페이지를 추가합니다.

**구현 예시:**
```python
# web/settings_page.py
import streamlit as st
import yaml
from pathlib import Path

def settings_page():
    st.header("⚙️ 설정 관리")
    
    # 현재 설정 로드
    config_path = Path("config/settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 키워드 추가 UI
    st.subheader("검색 키워드 추가")
    new_keyword = st.text_input("새 키워드")
    if st.button("추가"):
        # 설정 업데이트
        # ...
        st.success("설정이 업데이트되었습니다. 앱을 재시작하세요.")
    
    # 설정 파일 다운로드/업로드
    st.subheader("설정 파일 관리")
    st.download_button("설정 다운로드", config_path.read_text())
    uploaded_file = st.file_uploader("설정 파일 업로드")
    if uploaded_file:
        # 설정 파일 저장
        # ...
        st.success("설정이 업데이트되었습니다.")
```

### 방안 3: Hot Reload 구현

설정 파일 변경을 감지하여 자동으로 재로드하는 기능을 추가합니다.

```python
# config/__init__.py에 추가
import time
from pathlib import Path

class ConfigLoader:
    _last_modified = None
    
    def get(self, key=None, default=None):
        # 설정 파일이 변경되었는지 확인
        config_path = Path(__file__).parent / "settings.yaml"
        current_modified = config_path.stat().st_mtime
        
        if self._last_modified != current_modified:
            self._load_config()
            self._last_modified = current_modified
        
        # ... 기존 로직
```

**주의:** Streamlit의 캐싱 때문에 완전한 자동 반영은 어려울 수 있습니다.

## 🔧 GitHub 관리 전략

### 1. 설정 파일 분리

```
config/
├── settings.yaml          # 기본 설정 (GitHub 포함)
├── settings.local.yaml     # 로컬 오버라이드 (GitHub 제외)
└── settings.example.yaml  # 예시 파일 (GitHub 포함)
```

**구현:**
```python
# config/__init__.py
def _load_config(self):
    # 기본 설정 로드
    base_config = yaml.safe_load(open("config/settings.yaml"))
    
    # 로컬 설정이 있으면 병합
    local_path = Path("config/settings.local.yaml")
    if local_path.exists():
        local_config = yaml.safe_load(open(local_path))
        base_config = {**base_config, **local_config}  # 병합
    
    self._config = base_config
```

### 2. 환경별 설정

```yaml
# config/settings.dev.yaml (개발 환경)
search_groups:
  - name: "Test"
    region: "US"
    keywords:
      - "test keyword"

# config/settings.prod.yaml (운영 환경)
# 실제 운영 키워드들
```

### 3. .gitignore 업데이트

```gitignore
# 설정 파일
config/settings.local.yaml
config/settings.*.local.yaml

# 환경 변수
.env
.env.local
```

## 📋 실전 워크플로우

### 시나리오 1: 로컬에서 키워드 추가

1. `config/settings.yaml` 수정
2. 테스트: `python modules/news_collector.py`
3. Git에 커밋 (팀원과 공유)
4. Streamlit 재시작

### 시나리오 2: 팀원과 설정 공유

1. `config/settings.yaml` 수정
2. Git commit & push
3. 팀원들이 pull 받음
4. 각자 Streamlit 재시작

### 시나리오 3: 운영 환경 배포

1. `config/settings.yaml` 수정
2. Git commit & push
3. 서버에서 pull
4. Streamlit 서비스 재시작

## 🎯 권장 사항

### 단기 (현재)
- ✅ `config/settings.yaml` 직접 수정
- ✅ 변경 후 Streamlit 재시작
- ✅ Git으로 버전 관리

### 중기 (개선)
- ✅ 설정 관리 UI 페이지 추가
- ✅ 설정 파일 다운로드/업로드 기능
- ✅ 변경사항 미리보기

### 장기 (고급)
- ✅ Hot Reload 구현
- ✅ 환경별 설정 분리
- ✅ 설정 변경 히스토리

## 💻 빠른 해결책: 설정 관리 페이지 추가

Streamlit 앱에 설정 관리 페이지를 추가하는 것이 가장 실용적입니다.

**장점:**
- 코드 수정 없이 UI에서 변경
- 변경사항 즉시 확인
- 설정 백업/복원 가능

**구현 필요:**
- `web/settings_page.py` 생성
- `web/app.py`에 탭 추가
- 설정 파일 읽기/쓰기 로직

이 기능을 구현해드릴까요?










