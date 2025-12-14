-- Smart Investment Assistant 데이터베이스 스키마
-- SQLite 데이터베이스용 테이블 정의

-- 뉴스 원본 테이블
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    link TEXT UNIQUE NOT NULL,
    snippet TEXT,
    source TEXT,
    published_at DATETIME,
    vector_id TEXT,  -- ChromaDB의 벡터 ID
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    analyzed BOOLEAN DEFAULT 0,  -- 분석 완료 여부
    importance_score REAL DEFAULT 1.0,  -- 언론사 가중치 + 검색 순위 가점 기반 중요도 점수
    user_feedback_score REAL DEFAULT 0.0,  -- 사용자 피드백 기반 별도 점수 (0.0 ~ 1.0)
    feedback_applied_to_importance BOOLEAN DEFAULT 0,  -- 피드백이 일반 점수에 반영되었는지 여부
    category_name TEXT,  -- 카테고리 명
    search_keyword TEXT,  -- 검색 키워드
    search_rank INTEGER  -- 검색 노출 순위 (1부터 시작)
);

-- 분석된 이슈 테이블
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    summary TEXT,
    primary_tag TEXT NOT NULL,
    secondary_tags TEXT,  -- JSON 배열 형태로 저장
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    cluster_id INTEGER  -- 클러스터 ID (참고용)
);

-- 이슈-뉴스 매핑 테이블 (N:M 관계)
CREATE TABLE IF NOT EXISTS issue_news_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    news_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE,
    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE,
    UNIQUE(issue_id, news_id)
);

-- 인덱스 생성 (트렌드 분석 및 조회 성능 향상)
CREATE INDEX IF NOT EXISTS idx_news_link ON news(link);
CREATE INDEX IF NOT EXISTS idx_news_analyzed ON news(analyzed);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news(published_at);
CREATE INDEX IF NOT EXISTS idx_news_created_at ON news(created_at);
CREATE INDEX IF NOT EXISTS idx_news_category ON news(category_name);
CREATE INDEX IF NOT EXISTS idx_news_search_rank ON news(search_rank);

CREATE INDEX IF NOT EXISTS idx_issues_created_at ON issues(created_at);
CREATE INDEX IF NOT EXISTS idx_issues_primary_tag ON issues(primary_tag);
CREATE INDEX IF NOT EXISTS idx_issues_created_at_tag ON issues(created_at, primary_tag);

CREATE INDEX IF NOT EXISTS idx_mapping_issue_id ON issue_news_mapping(issue_id);
CREATE INDEX IF NOT EXISTS idx_mapping_news_id ON issue_news_mapping(news_id);



