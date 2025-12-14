# 보고서 양식 설정 가이드

## 개요

보고서 양식을 스프레드시트에서 관리하고, 보고서 그룹별로 다른 섹션 조합을 사용할 수 있습니다.

## 스프레드시트 설정

### 1. "보고서 양식 설정" 탭 생성

스프레드시트에 **"보고서 양식 설정"** 탭을 생성하고 다음 컬럼을 설정하세요:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| section_id | 섹션 식별자 (a~i) | a |
| section_name | 섹션 이름 | header |
| section_order | 섹션 순서 (숫자) | 1 |
| template_text | 템플릿 내용 (변수 포함) | # 📅 Daily Market Executive Report\n\nDate: {date_short} |
| enabled | 활성화 여부 (TRUE/FALSE) | TRUE |
| description | 설명 (선택) | 보고서 헤더 |

### 2. 기본 섹션 데이터

다음 데이터를 "보고서 양식 설정" 탭에 입력하세요:

#### 섹션 a: 헤더
```
section_id: a
section_name: header
section_order: 1
template_text: # 📅 Daily Market Executive Report

Date: {date_short}
enabled: TRUE
description: 보고서 헤더
```

#### 섹션 b: Executive Summary
```
section_id: b
section_name: executive_summary
section_order: 2
template_text: ## Executive Summary

- **Global:** {executive_summary_global}
- **Korea:** {executive_summary_korea}
- **Key Indicator:** {executive_summary_key_indicator}
enabled: TRUE
description: 요약 섹션
```

#### 섹션 c: Sector Analysis 헤더
```
section_id: c
section_name: sector_analysis_header
section_order: 3
template_text: ## Sector Analysis

enabled: TRUE
description: 섹터 분석 헤더
```

#### 섹션 d: 카테고리 헤더
```
section_id: d
section_name: category_header
section_order: 4
template_text: ### {category}

enabled: TRUE
description: 카테고리 헤더
```

#### 섹션 e: 테마 섹션
```
section_id: e
section_name: theme_section
section_order: 5
template_text: #### {theme_title}

**Deep Dive:**
{deep_dive}

enabled: TRUE
description: 테마 섹션
```

#### 섹션 f: 주요 뉴스
```
section_id: f
section_name: key_news
section_order: 6
template_text: **📰 Key News:**
{key_news_list}
enabled: TRUE
description: 주요 뉴스
```

#### 섹션 g: 피드백 섹션
```
section_id: g
section_name: feedback_section
section_order: 7
template_text: 

**🔍 추가 관점 (사용자 피드백 반영):**
{feedback_news_list}
enabled: TRUE
description: 피드백 섹션
```

#### 섹션 h: Investor Note
```
section_id: h
section_name: investor_note
section_order: 8
template_text: ---

## Investor Note

### Caution
{investor_note_caution}

### Action
{investor_note_action}
enabled: TRUE
description: 투자자 노트
```

#### 섹션 i: 푸터
```
section_id: i
section_name: footer
section_order: 9
template_text: ---

*Report generated on {generated_time}*
enabled: TRUE
description: 푸터
```

### 3. "보고서 그룹 설정" 탭 수정

기존 "보고서 그룹 설정" 탭에 `sections` 컬럼을 추가하세요:

| name | categories | output_file | sections | enabled |
|------|-----------|-------------|----------|---------|
| Global Macro Report | Global Macro | global_macro.md | a,b,c,d,e,f,h,i | TRUE |
| Korea 부동산 | Korea Market | korea_realestate.md | a,b,d,e,f,g,i | TRUE |

**sections 컬럼 설명:**
- 포함할 섹션 ID를 쉼표로 구분하여 입력
- 예: `a,b,c,d,e,f,h,i` (모든 섹션)
- 예: `a,b,d,e,f,g,i` (c, h 제외)

## 사용 예시

### 예시 1: Global Macro Report
- **sections**: `a,b,c,d,e,f,h,i`
- **포함 섹션**: 헤더, Executive Summary, Sector Analysis 헤더, 카테고리, 테마, Key News, Investor Note, 푸터
- **제외 섹션**: 피드백 섹션(g)

### 예시 2: Korea 부동산
- **sections**: `a,b,d,e,f,g,i`
- **포함 섹션**: 헤더, Executive Summary, 카테고리, 테마, Key News, 피드백 섹션, 푸터
- **제외 섹션**: Sector Analysis 헤더(c), Investor Note(h)

## 템플릿 변수

템플릿에서 사용 가능한 변수:

| 변수명 | 설명 | 사용 섹션 |
|--------|------|-----------|
| {date_short} | 날짜 (YY.MM.DD) | a |
| {executive_summary_global} | Global 요약 | b |
| {executive_summary_korea} | Korea 요약 | b |
| {executive_summary_key_indicator} | Key Indicator 요약 | b |
| {category} | 카테고리명 | d |
| {theme_title} | 테마 제목 | e |
| {deep_dive} | Deep Dive 내용 | e |
| {key_news_list} | 주요 뉴스 리스트 (포맷팅됨) | f |
| {feedback_news_list} | 피드백 뉴스 리스트 (포맷팅됨) | g |
| {investor_note_caution} | 주의사항 | h |
| {investor_note_action} | 행동 지침 | h |
| {generated_time} | 생성 시간 | i |

## 주의사항

1. **섹션 순서**: `section_order`는 반드시 숫자여야 합니다.
2. **섹션 ID**: a~i까지 사용 가능하며, 중복되지 않아야 합니다.
3. **템플릿 변수**: 변수명은 정확히 일치해야 합니다 (대소문자 구분).
4. **enabled**: FALSE로 설정하면 해당 섹션은 사용되지 않습니다.
5. **sections 필드**: 보고서 그룹에서 지정하지 않으면 모든 섹션이 사용됩니다.

## 동작 방식

1. 보고서 생성 시 스프레드시트에서 "보고서 양식 설정" 탭을 읽습니다.
2. 보고서 그룹의 `sections` 필드를 확인합니다.
3. 지정된 섹션 ID만 필터링하여 순서대로 조합합니다.
4. 각 섹션의 템플릿에 실제 데이터를 채워넣습니다.
5. 최종 보고서를 생성합니다.

## 문제 해결

### 섹션이 표시되지 않는 경우
- `sections` 필드에 해당 섹션 ID가 포함되어 있는지 확인
- `enabled`가 TRUE인지 확인
- `section_order`가 올바른지 확인

### 템플릿 변수가 치환되지 않는 경우
- 변수명이 정확히 일치하는지 확인 (중괄호 포함)
- 해당 섹션에서 사용 가능한 변수인지 확인

### 보고서가 생성되지 않는 경우
- "보고서 양식 설정" 탭이 존재하는지 확인
- 스프레드시트 연결 상태 확인
- 기본 템플릿으로 폴백되는지 확인 (로그 확인)



