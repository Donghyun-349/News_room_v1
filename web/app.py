"""
Smart Investment Assistant Streamlit 앱
Daily Briefing과 Trend Monitor를 제공합니다.
"""
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database import DatabaseManager
from modules.trend_calculator import TrendCalculator
from web.view_components import (
    render_issue_card,
    render_trend_chart,
    render_trend_summary
)

# 페이지 설정
st.set_page_config(
    page_title="Smart Investment Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐싱 데코레이터
@st.cache_data(ttl=3600)  # 1시간 캐싱
def get_cached_today_issues(date: datetime):
    """오늘의 이슈를 캐싱하여 가져옵니다"""
    db_manager = DatabaseManager()
    return db_manager.get_today_issues(date=date)


@st.cache_data(ttl=3600)  # 1시간 캐싱
def get_cached_trend_summary(days: int):
    """트렌드 요약을 캐싱하여 가져옵니다"""
    calculator = TrendCalculator()
    return calculator.get_trend_summary(days=days)


@st.cache_data(ttl=3600)  # 1시간 캐싱
def get_cached_chart_data(days: int, top_n: int):
    """차트 데이터를 캐싱하여 가져옵니다"""
    calculator = TrendCalculator()
    return calculator.get_tag_trend_chart_data(days=days, top_n_tags=top_n)


def main():
    """메인 함수"""
    # 사이드바
    with st.sidebar:
        st.title("📊 SIA")
        st.markdown("Smart Investment Assistant")
        st.divider()
        
        st.subheader("설정")
        
        # 날짜 선택
        selected_date = st.date_input(
            "조회 날짜",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
        
        # 트렌드 기간 선택
        trend_days = st.slider(
            "트렌드 조회 기간 (일)",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        # 상위 태그 개수
        top_n_tags = st.slider(
            "상위 태그 개수",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        st.divider()
        
        # 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.caption("© 2024 Smart Investment Assistant")
    
    # 메인 탭
    tab1, tab2 = st.tabs(["📰 Daily Briefing", "📈 Trend Monitor"])
    
    # Daily Briefing 탭
    with tab1:
        st.header("📰 Daily Briefing")
        st.markdown(f"**{selected_date}** 일일 브리핑")
        
        # 이슈 가져오기
        with st.spinner("이슈를 불러오는 중..."):
            issues = get_cached_today_issues(datetime.combine(selected_date, datetime.min.time()))
        
        if not issues:
            st.info(f"{selected_date}에 생성된 이슈가 없습니다.")
        else:
            st.success(f"총 {len(issues)}개의 이슈를 찾았습니다.")
            
            # 필터 옵션
            col1, col2 = st.columns([3, 1])
            with col1:
                # 태그 필터
                all_tags = list(set([issue.get('primary_tag', 'Unknown') for issue in issues]))
                selected_tags = st.multiselect(
                    "태그 필터",
                    options=all_tags,
                    default=all_tags
                )
            
            with col2:
                # 정렬 옵션
                sort_by = st.selectbox(
                    "정렬 기준",
                    options=["생성일 (최신순)", "생성일 (오래된순)", "태그"]
                )
            
            # 필터링
            filtered_issues = [
                issue for issue in issues
                if issue.get('primary_tag', 'Unknown') in selected_tags
            ]
            
            # 정렬
            if sort_by == "생성일 (최신순)":
                filtered_issues.sort(
                    key=lambda x: x.get('created_at', datetime.min),
                    reverse=True
                )
            elif sort_by == "생성일 (오래된순)":
                filtered_issues.sort(
                    key=lambda x: x.get('created_at', datetime.min)
                )
            else:  # 태그
                filtered_issues.sort(key=lambda x: x.get('primary_tag', ''))
            
            # 이슈 카드 표시
            st.divider()
            for issue in filtered_issues:
                render_issue_card(issue, show_details=True)
    
    # Trend Monitor 탭
    with tab2:
        st.header("📈 Trend Monitor")
        st.markdown(f"최근 **{trend_days}일**간의 트렌드 분석")
        
        # 트렌드 요약
        with st.spinner("트렌드 데이터를 분석하는 중..."):
            trend_summary = get_cached_trend_summary(days=trend_days)
        
        render_trend_summary(trend_summary)
        
        st.divider()
        
        # 차트 타입 선택
        chart_type = st.radio(
            "차트 타입",
            options=["stacked_area", "line", "bar"],
            horizontal=True,
            index=0
        )
        
        # 차트 데이터 가져오기
        with st.spinner("차트 데이터를 준비하는 중..."):
            chart_data = get_cached_chart_data(days=trend_days, top_n=top_n_tags)
        
        if chart_data.get('dates'):
            render_trend_chart(chart_data, chart_type=chart_type)
            
            # 통계 정보
            st.subheader("📊 통계 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 이슈 수", f"{chart_data.get('total_issues', 0):,}개")
            with col2:
                st.metric("표시 태그 수", f"{len(chart_data.get('tags', []))}개")
            with col3:
                st.metric("조회 기간", f"{trend_days}일")
        else:
            st.warning("표시할 트렌드 데이터가 없습니다.")


if __name__ == "__main__":
    main()











