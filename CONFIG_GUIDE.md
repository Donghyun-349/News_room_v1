# 설정 변경 가이드

이 프로젝트는 **Config-Driven** 설계로, 코드 수정 없이 `config/settings.yaml` 파일만 수정하면 모든 설정을 변경할 수 있습니다.

## 📝 설정 파일 위치

```
config/settings.yaml
```

## 🔧 변경 가능한 항목

### 1. 검색 그룹 추가/수정/삭제

**현재 구조:**
```yaml
search_groups:
  - name: "KR_Macro"
    region: "KR"
    keywords:
      - "한국은행 기준금리"
      - "물가 상승률"
      - "경제 성장률"
      - "환율"
```

**변경 예시:**

#### ✅ 키워드 추가
```yaml
search_groups:
  - name: "KR_Macro"
    region: "KR"
    keywords:
      - "한국은행 기준금리"
      - "물가 상승률"
      - "경제 성장률"
      - "환율"
      - "부동산 가격"  # 새로 추가
      - "고용률"        # 새로 추가
```

#### ✅ 새로운 검색 그룹 추가
```yaml
search_groups:
  - name: "KR_Macro"
    region: "KR"
    keywords:
      - "한국은행 기준금리"
      # ... 기존 키워드들
  
  - name: "KR_Bio"      # 새 그룹 추가
    region: "KR"
    keywords:
      - "바이오 신약"
      - "제약 회사"
      - "FDA 승인"
  
  - name: "US_Energy"   # 새 그룹 추가
    region: "US"
    keywords:
      - "oil price"
      - "energy crisis"
      - "renewable energy"
```

#### ✅ 검색 그룹 삭제
```yaml
search_groups:
  # - name: "KR_Macro"  # 주석 처리하거나 삭제
  #   region: "KR"
  #   keywords: [...]
  
  - name: "US_Semiconductor"  # 이 그룹만 유지
    region: "US"
    keywords: [...]
```

### 2. 메타 태그 추가/수정

**현재 구조:**
```yaml
meta_tags:
  - "Macro"
  - "Semiconductor"
  - "Bio"
  # ...
```

**변경 예시:**
```yaml
meta_tags:
  - "Macro"
  - "Semiconductor"
  - "Bio"
  - "Energy"        # 기존
  - "Finance"       # 기존
  - "Real_Estate"   # 새로 추가
  - "Automotive"    # 새로 추가
  - "Retail"        # 새로 추가
```

### 3. 언론사 가중치 변경

**현재 구조:**
```yaml
publisher_weights:
  tier1:
    sources:
      - "Reuters"
      - "Bloomberg"
      # ...
    weight: 1.5
  
  tier2:
    sources:
      - "CNBC"
      # ...
    weight: 1.2
```

**변경 예시:**

#### ✅ 언론사 추가
```yaml
publisher_weights:
  tier1:
    sources:
      - "Reuters"
      - "Bloomberg"
      - "Financial Times"
      - "한국경제"      # 기존
      - "매일경제신문"  # 새로 추가
    weight: 1.5
```

#### ✅ 가중치 변경
```yaml
publisher_weights:
  tier1:
    sources: [...]
    weight: 1.6  # 1.5에서 1.6으로 변경
```

### 4. 클러스터링 파라미터 조정

**현재 구조:**
```yaml
clustering:
  eps: 0.5              # 거리 임계값 (작을수록 엄격)
  min_samples: 3        # 최소 기사 수
  mmr_diversity: 0.7    # 다양성 가중치
```

**변경 예시:**
```yaml
clustering:
  eps: 0.4              # 더 엄격한 클러스터링
  min_samples: 5        # 더 큰 클러스터만 허용
  mmr_diversity: 0.8    # 더 다양한 기사 선택
```

### 5. Rate Limiting 조정

**현재 구조:**
```yaml
rate_limit:
  google_rss_delay: 2.0  # 초 단위
  openai_delay: 0.1
```

**변경 예시:**
```yaml
rate_limit:
  google_rss_delay: 1.0  # 더 빠른 수집 (IP 차단 위험 증가)
  openai_delay: 0.2      # 더 안전한 API 호출
```

### 6. LLM 모델 변경

**현재 구조:**
```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 2000
  embedding_model: "text-embedding-3-small"
```

**변경 예시:**
```yaml
llm:
  model: "gpt-4o"                    # 더 강력한 모델
  temperature: 0.2                   # 더 일관된 응답
  max_tokens: 3000                    # 더 긴 요약
  embedding_model: "text-embedding-3-large"  # 더 정확한 임베딩
```

## 🔄 변경 후 적용 방법

### 1. 설정 파일 수정
```bash
# config/settings.yaml 파일을 편집기로 열어서 수정
```

### 2. 변경사항 확인
```bash
# 설정이 제대로 로드되는지 테스트
python -c "from config import get_config; print(get_config('search_groups'))"
```

### 3. 뉴스 수집 재실행
```bash
# 새로운 키워드로 수집
python modules/news_collector.py
```

## 📋 변경 시 주의사항

### ✅ 안전한 변경
- 키워드 추가/삭제
- 검색 그룹 추가
- 메타 태그 추가
- 언론사 추가
- Rate Limiting 조정

### ⚠️ 신중하게 변경
- 클러스터링 파라미터 (분석 결과에 영향)
- LLM 모델 (비용 및 성능 영향)
- 가중치 값 (중요도 점수에 영향)

### ❌ 변경 후 재시작 필요
- Python 프로세스를 재시작해야 설정 변경이 적용됩니다
- Streamlit 앱이 실행 중이면 재시작 필요

## 💡 실전 예시

### 예시 1: 바이오 테마 추가
```yaml
search_groups:
  # ... 기존 그룹들 ...
  
  - name: "US_Bio"
    region: "US"
    keywords:
      - "FDA approval"
      - "biotech IPO"
      - "drug trial"
      - "pharmaceutical earnings"
  
  - name: "KR_Bio"
    region: "KR"
    keywords:
      - "신약 개발"
      - "바이오 벤처"
      - "식약처 승인"
```

그리고 메타 태그에 추가:
```yaml
meta_tags:
  # ... 기존 태그들 ...
  - "Bio"  # 이미 있음
```

### 예시 2: 더 많은 한국 언론사 추가
```yaml
publisher_weights:
  tier1:
    sources:
      - "Reuters"
      - "Bloomberg"
      - "조선일보"
      - "중앙일보"
      - "한국경제"
      - "매일경제신문"  # 새로 추가
      - "동아일보"      # 새로 추가
    weight: 1.5
```

### 예시 3: 더 빠른 수집 (주의 필요)
```yaml
rate_limit:
  google_rss_delay: 1.0  # 2.0에서 1.0으로 감소
```

## 🎯 요약

1. **코드 수정 불필요**: `config/settings.yaml`만 수정
2. **즉시 적용**: 다음 실행 시 자동 반영
3. **유연한 확장**: 검색 그룹, 키워드, 태그 자유롭게 추가
4. **안전한 조정**: 파라미터 값만 변경해도 동작

이렇게 Config-Driven 설계 덕분에 코드 수정 없이 모든 설정을 변경할 수 있습니다! 🎉










