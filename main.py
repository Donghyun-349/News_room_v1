"""
Smart Investment Assistant 메인 실행 파일
스케줄러 및 엔트리 포인트
"""
import os
import sys
import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# .env 파일 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config


def setup_logging():
    """로깅 시스템을 설정합니다"""
    log_config = get_config('logging', {})
    
    # 로그 레벨
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # 로그 포맷
    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 로그 파일 경로
    log_file = log_config.get('file', 'logs/app.log')
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일 핸들러 (로테이션)
    max_bytes = log_config.get('max_bytes', 10485760)  # 10MB
    backup_count = log_config.get('backup_count', 5)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info("로깅 시스템 초기화 완료")
    logging.info(f"로그 파일: {log_path.absolute()}")


def main():
    """메인 함수"""
    # 로깅 설정
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Smart Investment Assistant 시작")
    logger.info("=" * 50)
    
    # 환경 변수 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "sk-your-api-key-here":
        logger.warning("⚠️  OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    # 설정 확인
    try:
        config = get_config()
        categories = config.get('categories', [])
        if categories:
            # #region agent log
            import json
            import time
            log_path = Path(__file__).parent / ".cursor" / "debug.log"
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    category_details = [{"name":c.get('name'),"report_group":c.get('report_group'),"keywords_count":len(c.get('keywords',[]))} for c in categories]
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"main.py:94","message":"Categories loaded in main","data":{"count":len(categories),"categories":category_details},"timestamp":int(time.time()*1000)})+'\n')
            except Exception as ex:
                logger.debug(f"Debug log write failed: {ex}")
            # #endregion
            logger.info(f"카테고리 수: {len(categories)}")
        else:
            # 하위 호환성: search_groups 확인
            search_groups = config.get('search_groups')
            if search_groups:
                logger.info(f"검색 그룹 수: {len(search_groups)}")
            else:
                logger.warning("⚠️  카테고리 또는 검색 그룹이 설정되지 않았습니다.")
        
        meta_tags = config.get('meta_tags', [])
        if meta_tags:
            logger.info(f"메타 태그 수: {len(meta_tags)}")
        
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return
    
    # 뉴스 수집 및 분석 실행
    try:
        from modules.news_collector import NewsCollector
        from modules.news_analyzer import NewsAnalyzer
        from modules.sheet_exporter import SheetExporter
        from database import DatabaseManager
        import pandas as pd
        
        # 데이터베이스 리셋 설정 확인
        from modules.settings_loader import SettingsLoader
        settings_loader = SettingsLoader()
        system_settings = settings_loader.get('system_settings', {})
        reset_database = system_settings.get('reset_database', False)
        
        if reset_database:
            logger.info("데이터베이스 리셋 중...")
            db_manager = DatabaseManager()
            db_manager.reset_database()
            logger.info("✅ 데이터베이스 리셋 완료")
        else:
            logger.info("데이터베이스 리셋 건너뜀 (시스템 설정: reset_database = False)")
        
        # 1. 뉴스 수집
        logger.info("=" * 50)
        logger.info("뉴스 수집 시작")
        logger.info("=" * 50)
        collector = NewsCollector()
        # 설정 파일의 각 카테고리 max_articles 설정에 따라 수집
        collection_result = collector.collect_all(max_articles=None)
        logger.info(f"수집 완료: {collection_result['total_collected']}개 수집")
        
        # 뉴스 수집 데이터를 스프레드시트로 내보내기
        try:
            db_manager = DatabaseManager()
            sheet_exporter = SheetExporter()
            if sheet_exporter.spreadsheet:
                with db_manager.get_connection() as conn:
                    query = """
                        SELECT 
                            id, category_name, search_keyword, title, link, snippet,
                            source, search_rank, published_at, created_at,
                            importance_score, analyzed
                        FROM news
                        ORDER BY created_at DESC
                    """
                    df_news = pd.read_sql_query(query, conn)
                    if not df_news.empty:
                        df_news['analyzed'] = df_news['analyzed'].map({0: '대기중', 1: '완료'})
                        sheet_exporter.export_to_sheet(df_news, sheet_name='뉴스 수집')
                        logger.info("✅ 뉴스 수집 데이터를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.warning(f"스프레드시트 내보내기 실패 (계속 진행): {e}")
        
        # 2. 뉴스 분석 (카테고리별 순차 처리)
        logger.info("=" * 50)
        logger.info("뉴스 분석 시작 (카테고리별 순차 처리)")
        logger.info("=" * 50)
        
        analyzer = NewsAnalyzer()
        config = get_config()
        categories = config.get('categories', [])
        
        if not categories:
            logger.warning("⚠️  카테고리가 설정되지 않았습니다. 기존 방식으로 분석합니다.")
            analysis_result = analyzer.analyze_news(batch_size=100)
            logger.info(f"분석 완료: {analysis_result['issues_created']}개 이슈 생성")
        else:
            total_issues = 0
            total_news = 0
            
            for i, category in enumerate(categories, 1):
                category_name = category.get('name', 'Unknown')
                logger.info("")
                logger.info(f"[{i}/{len(categories)}] 카테고리 '{category_name}' 분석 시작")
                logger.info("-" * 50)
                
                try:
                    analysis_result = analyzer.analyze_news_by_category(
                        category_name=category_name,
                        batch_size=100
                    )
                    total_issues += analysis_result.get('issues_created', 0)
                    total_news += analysis_result.get('total_news', 0)
                    logger.info(f"✅ 카테고리 '{category_name}' 분석 완료: {analysis_result.get('issues_created', 0)}개 이슈 생성")
                except Exception as e:
                    logger.error(f"❌ 카테고리 '{category_name}' 분석 실패: {e}", exc_info=True)
                    logger.info(f"다음 카테고리로 계속 진행합니다...")
                    continue
            
            logger.info("")
            logger.info("=" * 50)
            logger.info(f"전체 분석 완료: 총 {total_issues}개 이슈 생성, {total_news}개 뉴스 분석")
            logger.info("=" * 50)
        
        # 분석 결과를 스프레드시트로 내보내기
        try:
            db_manager = DatabaseManager()
            sheet_exporter = SheetExporter()
            if sheet_exporter.spreadsheet:
                with db_manager.get_connection() as conn:
                    # 이슈 목록 (카테고리, 스코어 포함하여 정렬)
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
                        sheet_exporter.export_to_sheet(df_issues_final, sheet_name='이슈 목록')
                        logger.info(f"✅ 이슈 목록 {len(df_issues_final)}개를 스프레드시트에 기록했습니다.")
                    
                    # 이슈-뉴스 매핑 (구글 스프레드시트 출력 형식)
                    mapping_query = """
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
                    df_mapping = pd.read_sql_query(mapping_query, conn)
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
                        
                        # 필요한 컬럼만 선택: category_name, issue_title, number_of_news, score, source, news_title, link
                        df_mapping_final = df_mapping[['category_name', 'issue_title', 'number_of_news', 'score', 'source', 'news_title', 'link']].copy()
                        
                        # number_of_news를 정수로 변환
                        df_mapping_final['number_of_news'] = df_mapping_final['number_of_news'].astype(int)
                        
                        # 카테고리별, 스코어별 정렬
                        df_mapping_final = df_mapping_final.sort_values(
                            by=['category_name', 'score'],
                            ascending=[True, False],
                            na_position='last'
                        )
                        
                        sheet_exporter.export_to_sheet(df_mapping_final, sheet_name='이슈-뉴스 매핑')
                        logger.info(f"✅ 이슈-뉴스 매핑 {len(df_mapping_final)}개를 스프레드시트에 기록했습니다.")
        except Exception as e:
            logger.warning(f"분석 결과 스프레드시트 내보내기 실패: {e}")
        
        logger.info("=" * 50)
        logger.info("전체 프로세스 완료!")
        logger.info("=" * 50)
        
        # 3. 보고서 생성
        try:
            logger.info("")
            logger.info("=" * 50)
            logger.info("보고서 생성 시작")
            logger.info("=" * 50)
            
            from generate_daily_report import DailyReportGenerator
            
            report_generator = DailyReportGenerator()
            report_groups = config.get('report_groups', [])
            
            if report_groups:
                # 보고서 그룹별로 생성
                logger.info(f"보고서 그룹 {len(report_groups)}개 생성 시작")
                for i, group in enumerate(report_groups, 1):
                    group_name = group.get('name', f'Group {i}')
                    categories = group.get('categories', [])
                    logger.info(f"[{i}/{len(report_groups)}] {group_name} 생성 중... (카테고리: {', '.join(categories)})")
                
                reports = report_generator.run_by_groups()
                logger.info(f"✅ 보고서 생성 완료: {len(reports)}개 보고서 생성")
                
                # 이메일 발송
                try:
                    logger.info("")
                    logger.info("=" * 50)
                    logger.info("이메일 발송 시작")
                    logger.info("=" * 50)
                    
                    from modules.email_sender import EmailSender
                  
                    
                    # 생성된 보고서 파일 경로 수집
                    report_files = {}
                    for group_name, report_data in reports.items():
                        # report_data는 {'content': str, 'output_file': str} 형식
                        # 실제 파일명은 run() 메서드에서 {report_name}.md로 저장됨
                        # 따라서 {group_name}.md 파일을 찾음
                        possible_filenames = [
                            f"{group_name}.md",  # run() 메서드가 저장하는 파일명
                            report_data.get('output_file', f"{group_name.lower().replace(' ', '_')}.md")  # output_file
                        ]
                        
                        report_path = None
                        for filename in possible_filenames:
                            path = Path(project_root) / filename
                            if path.exists():
                                report_path = path
                                break
                        
                        if not report_path:
                            # 파일이 없으면 content를 직접 사용하여 임시 파일 생성
                            report_content = report_data.get('content', '')
                            if report_content:
                                # output_file 이름으로 저장
                                output_file = report_data.get('output_file', f"{group_name.lower().replace(' ', '_')}.md")
                                temp_path = Path(project_root) / output_file
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    f.write(report_content)
                                report_path = temp_path
                        
                        if report_path:
                            report_files[group_name] = report_path
                    
                    if report_files:
                        email_sender = EmailSender()
                        email_results = email_sender.send_reports_to_recipients(report_files)
                        logger.info(f"✅ 이메일 발송 완료: {email_results['sent_count']}개 발송, {email_results['failed_count']}개 실패")
                    else:
                        logger.warning("⚠️  발송할 보고서 파일을 찾을 수 없습니다.")
                        
                except Exception as e:
                    logger.warning(f"이메일 발송 실패 (계속 진행): {e}", exc_info=True)
            else:
                # 모든 카테고리를 하나의 보고서로 생성
                logger.info("모든 카테고리를 하나의 보고서로 생성합니다.")
                report = report_generator.run()
                if report:
                    logger.info("✅ 보고서 생성 완료: daily_market_report.md")
                    
                    # 이메일 발송
                    try:
                        logger.info("")
                        logger.info("=" * 50)
                        logger.info("이메일 발송 시작")
                        logger.info("=" * 50)
                        
                        from modules.email_sender import EmailSender
                      
                        
                        report_path = Path(project_root) / "daily_market_report.md"
                        if report_path.exists():
                            # 기본 보고서 그룹 이름 사용
                            report_files = {"Daily Market Report": report_path}
                            email_sender = EmailSender()
                            email_results = email_sender.send_reports_to_recipients(report_files)
                            logger.info(f"✅ 이메일 발송 완료: {email_results['sent_count']}개 발송, {email_results['failed_count']}개 실패")
                        else:
                            logger.warning("⚠️  보고서 파일을 찾을 수 없습니다.")
                            
                    except Exception as e:
                        logger.warning(f"이메일 발송 실패 (계속 진행): {e}", exc_info=True)
                else:
                    logger.warning("⚠️  보고서 생성 실패 (데이터 없음)")
                    
        except Exception as e:
            logger.warning(f"보고서 생성 실패 (계속 진행): {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"프로세스 실행 실패: {e}", exc_info=True)
        raise
    
    # TODO: 스케줄러 설정
    # from apscheduler.schedulers.blocking import BlockingScheduler
    # scheduler = BlockingScheduler()
    # scheduler.add_job(collect_news, 'interval', hours=1)
    # scheduler.add_job(analyze_news, 'interval', hours=2)
    # scheduler.start()
    
    logger.info("메인 프로세스 준비 완료")


if __name__ == "__main__":
    main()




