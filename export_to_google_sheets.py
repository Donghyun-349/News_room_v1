"""
현재 분석된 결과를 구글 스프레드시트에 출력하는 스크립트
"""
import sys
import io
from pathlib import Path
import pandas as pd

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

from modules.sheet_exporter import SheetExporter
from database import DatabaseManager

def export_all_to_sheets():
    """모든 분석 결과를 구글 스프레드시트에 출력"""
    print("=" * 80)
    print("구글 스프레드시트로 분석 결과 내보내기")
    print("=" * 80)
    print()
    
    sheet_exporter = SheetExporter()
    db_manager = DatabaseManager()
    
    if not sheet_exporter.spreadsheet:
        print("❌ 구글 스프레드시트 연동이 설정되지 않았습니다.")
        return
    
    print("✅ 구글 스프레드시트 연결 성공")
    print()
    
    try:
        with db_manager.get_connection() as conn:
            # 1. 뉴스 수집 데이터
            print("1. 뉴스 수집 데이터 내보내는 중...")
            news_query = """
                SELECT 
                    id, category_name, search_keyword, title, link, snippet,
                    source, search_rank, published_at, created_at,
                    importance_score, analyzed
                FROM news
                ORDER BY created_at DESC
            """
            df_news = pd.read_sql_query(news_query, conn)
            if not df_news.empty:
                df_news['analyzed'] = df_news['analyzed'].map({0: '대기중', 1: '완료'})
                success = sheet_exporter.export_to_sheet(df_news, sheet_name='뉴스 수집')
                if success:
                    print(f"   ✅ {len(df_news)}개 뉴스 기록 완료")
                else:
                    print("   ❌ 뉴스 수집 데이터 기록 실패")
            else:
                print("   ⚠️  뉴스 데이터가 없습니다.")
            print()
            
            # 2. 이슈 목록 (카테고리, 스코어 포함하여 정렬)
            print("2. 이슈 목록 내보내는 중...")
            issues_query = """
                SELECT 
                    i.id,
                    COALESCE(MAX(n.category_name), 'Unknown') as category_name,
                    i.title,
                    i.summary,
                    i.created_at,
                    i.cluster_id,
                    AVG(n.importance_score) as avg_importance,
                    COUNT(m.news_id) as news_count
                FROM issues i
                LEFT JOIN issue_news_mapping m ON i.id = m.issue_id
                LEFT JOIN news n ON m.news_id = n.id
                GROUP BY i.id, i.title, i.summary, i.created_at, i.cluster_id
            """
            df_issues = pd.read_sql_query(issues_query, conn)
            if not df_issues.empty:
                # 스코어 계산
                import math
                df_issues['score'] = df_issues.apply(
                    lambda row: (row['avg_importance'] ** 3) * math.log2(row['news_count'] + 1) 
                    if row['avg_importance'] is not None and row['news_count'] is not None else 0.0,
                    axis=1
                )
                # 카테고리별, 스코어별 정렬
                df_issues = df_issues.sort_values(
                    by=['category_name', 'score'],
                    ascending=[True, False],
                    na_position='last'
                )
                # 필요한 컬럼만 선택 (카테고리, 제목, 스코어, 요약, 생성일, 클러스터 ID)
                df_issues_final = df_issues[['category_name', 'title', 'score', 'summary', 'created_at', 'cluster_id']].copy()
                df_issues_final['score'] = df_issues_final['score'].round(2)
                success = sheet_exporter.export_to_sheet(df_issues_final, sheet_name='이슈 목록')
                if success:
                    print(f"   ✅ {len(df_issues_final)}개 이슈 기록 완료")
                else:
                    print("   ❌ 이슈 목록 기록 실패")
            else:
                print("   ⚠️  이슈 데이터가 없습니다.")
            print()
            
            # 3. 이슈-뉴스 매핑 (새 형식)
            print("3. 이슈-뉴스 매핑 내보내는 중...")
            mapping_query_with_id = """
                SELECT 
                    n.category_name,
                    i.id as issue_id,
                    i.title as issue_title,
                    COUNT(*) OVER (PARTITION BY i.id) as number_of_news,
                    m.news_id,
                    n.title as news_title,
                    n.source,
                    n.link
                FROM issue_news_mapping m
                JOIN issues i ON m.issue_id = i.id
                JOIN news n ON m.news_id = n.id
                ORDER BY i.id, n.importance_score DESC
            """
            df_mapping = pd.read_sql_query(mapping_query_with_id, conn)
            if not df_mapping.empty:
                # 이슈별 점수 계산
                import math
                issue_scores = {}
                for issue_id in df_mapping['issue_id'].unique():
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT 
                            AVG(n.importance_score) as avg_importance,
                            COUNT(m.news_id) as news_count
                        FROM issues i
                        LEFT JOIN issue_news_mapping m ON i.id = m.issue_id
                        LEFT JOIN news n ON m.news_id = n.id
                        WHERE i.id = ?
                        GROUP BY i.id
                    """, (int(issue_id),))
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        avg_importance = result[0] or 1.0
                        news_count = result[1] or 0
                        score = (avg_importance ** 3) * math.log2(news_count + 1)
                        issue_scores[int(issue_id)] = score
                    else:
                        issue_scores[int(issue_id)] = 0.0
                
                # 점수 컬럼 추가
                df_mapping['score'] = df_mapping['issue_id'].map(issue_scores)
                
                # 필요한 컬럼만 선택 (Source 추가: category_name, issue_title, number_of_news, score, source, news_title, link)
                df_mapping_final = df_mapping[['category_name', 'issue_title', 'number_of_news', 'score', 'source', 'news_title', 'link']].copy()
                
                # number_of_news를 정수로 변환
                df_mapping_final['number_of_news'] = df_mapping_final['number_of_news'].astype(int)
                
                # 카테고리별, 스코어별 정렬
                df_mapping_final = df_mapping_final.sort_values(
                    by=['category_name', 'score'],
                    ascending=[True, False],
                    na_position='last'
                )
                
                success = sheet_exporter.export_to_sheet(df_mapping_final, sheet_name='이슈-뉴스 매핑')
                if success:
                    print(f"   ✅ {len(df_mapping_final)}개 매핑 기록 완료")
                else:
                    print("   ❌ 이슈-뉴스 매핑 기록 실패")
            else:
                print("   ⚠️  매핑 데이터가 없습니다.")
            print()
        
        print("=" * 80)
        print("✅ 모든 데이터를 구글 스프레드시트에 기록했습니다!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_all_to_sheets()

