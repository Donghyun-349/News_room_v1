"""
트렌드 계산기 모듈
DB에서 가져온 Raw 데이터를 가공하여 UI에 전달하기 좋은 형태로 변환합니다.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from database import DatabaseManager

logger = logging.getLogger(__name__)


class TrendCalculator:
    """트렌드 계산기 클래스"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Args:
            db_manager: DatabaseManager 인스턴스 (None이면 새로 생성)
        """
        self.db_manager = db_manager or DatabaseManager()
    
    def get_trend_data(self, days: int = 30, include_ma: bool = True, 
                      ma_window: int = 7) -> pd.DataFrame:
        """
        태그별 트렌드 데이터를 가공하여 반환합니다.
        
        Args:
            days: 조회할 일수
            include_ma: 이동평균선 포함 여부
            ma_window: 이동평균선 윈도우 크기 (일 단위)
        
        Returns:
            DataFrame (컬럼: date, tag, count, ma_count(선택))
        """
        # DB에서 Raw 데이터 가져오기
        raw_df = self.db_manager.get_tag_trends(days=days)
        
        if raw_df.empty:
            logger.warning("트렌드 데이터가 없습니다")
            return pd.DataFrame(columns=['date', 'tag', 'count'])
        
        # 날짜 범위 생성 (빈 날짜 채우기)
        date_range = pd.date_range(
            start=raw_df['date'].min(),
            end=raw_df['date'].max(),
            freq='D'
        )
        
        # 모든 태그 목록
        all_tags = raw_df['tag'].unique()
        
        # 전체 조합 생성 (날짜 x 태그)
        full_index = pd.MultiIndex.from_product(
            [date_range, all_tags],
            names=['date', 'tag']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        # Raw 데이터와 병합 (빈 값은 0으로 채움)
        merged_df = full_df.merge(
            raw_df,
            on=['date', 'tag'],
            how='left'
        )
        merged_df['count'] = merged_df['count'].fillna(0).astype(int)
        
        # 이동평균선 계산
        if include_ma:
            merged_df = merged_df.sort_values(['tag', 'date'])
            merged_df['ma_count'] = merged_df.groupby('tag')['count'].transform(
                lambda x: x.rolling(window=ma_window, min_periods=1).mean()
            )
            merged_df['ma_count'] = merged_df['ma_count'].round(2)
        
        # 정렬
        merged_df = merged_df.sort_values(['date', 'tag'])
        
        logger.info(f"트렌드 데이터 가공 완료: {len(merged_df)}개 레코드, {len(all_tags)}개 태그")
        
        return merged_df
    
    def get_trend_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        트렌드 요약 정보를 반환합니다.
        
        Args:
            days: 조회할 일수
        
        Returns:
            요약 딕셔너리
        """
        raw_df = self.db_manager.get_tag_trends(days=days)
        
        if raw_df.empty:
            return {
                'total_issues': 0,
                'total_tags': 0,
                'top_tags': [],
                'trending_tags': [],
                'declining_tags': []
            }
        
        # 전체 통계
        total_issues = raw_df['count'].sum()
        total_tags = raw_df['tag'].nunique()
        
        # 태그별 총합
        tag_totals = raw_df.groupby('tag')['count'].sum().sort_values(ascending=False)
        top_tags = tag_totals.head(10).to_dict()
        
        # 급상승/급하락 태그 계산
        # 최근 7일 vs 그 이전 7일 비교
        recent_date = raw_df['date'].max()
        recent_7d_start = recent_date - timedelta(days=7)
        previous_7d_start = recent_7d_start - timedelta(days=7)
        
        recent_7d = raw_df[
            (raw_df['date'] >= recent_7d_start) & 
            (raw_df['date'] <= recent_date)
        ].groupby('tag')['count'].sum()
        
        previous_7d = raw_df[
            (raw_df['date'] >= previous_7d_start) & 
            (raw_df['date'] < recent_7d_start)
        ].groupby('tag')['count'].sum()
        
        # 변화율 계산
        change_rates = {}
        all_tags_in_period = set(recent_7d.index) | set(previous_7d.index)
        
        for tag in all_tags_in_period:
            recent_count = recent_7d.get(tag, 0)
            previous_count = previous_7d.get(tag, 0)
            
            if previous_count > 0:
                change_rate = ((recent_count - previous_count) / previous_count) * 100
            elif recent_count > 0:
                change_rate = 100  # 새로 등장
            else:
                change_rate = 0
            
            change_rates[tag] = {
                'recent': int(recent_count),
                'previous': int(previous_count),
                'change_rate': round(change_rate, 1),
                'change': int(recent_count - previous_count)
            }
        
        # 급상승 태그 (상승률 50% 이상 또는 +5개 이상 증가)
        trending_tags = [
            {
                'tag': tag,
                **data
            }
            for tag, data in change_rates.items()
            if data['change_rate'] >= 50 or data['change'] >= 5
        ]
        trending_tags.sort(key=lambda x: x['change_rate'], reverse=True)
        trending_tags = trending_tags[:10]
        
        # 급하락 태그 (하락률 50% 이상 또는 -5개 이상 감소)
        declining_tags = [
            {
                'tag': tag,
                **data
            }
            for tag, data in change_rates.items()
            if data['change_rate'] <= -50 or data['change'] <= -5
        ]
        declining_tags.sort(key=lambda x: x['change_rate'])
        declining_tags = declining_tags[:10]
        
        return {
            'total_issues': int(total_issues),
            'total_tags': int(total_tags),
            'top_tags': [{'tag': k, 'count': int(v)} for k, v in top_tags.items()],
            'trending_tags': trending_tags,
            'declining_tags': declining_tags,
            'period': {
                'start': raw_df['date'].min().isoformat(),
                'end': raw_df['date'].max().isoformat(),
                'days': days
            }
        }
    
    def get_tag_trend_chart_data(self, days: int = 30, 
                                 top_n_tags: int = 10) -> Dict[str, Any]:
        """
        차트 그리기에 최적화된 형태로 트렌드 데이터를 반환합니다.
        
        Args:
            days: 조회할 일수
            top_n_tags: 상위 N개 태그만 포함
        
        Returns:
            차트 데이터 딕셔너리
        """
        # 트렌드 데이터 가져오기
        trend_df = self.get_trend_data(days=days, include_ma=True)
        
        if trend_df.empty:
            return {
                'dates': [],
                'tags': [],
                'data': {}
            }
        
        # 상위 태그 선택 (전체 기간 합계 기준)
        tag_totals = trend_df.groupby('tag')['count'].sum().sort_values(ascending=False)
        top_tags = tag_totals.head(top_n_tags).index.tolist()
        
        # 상위 태그만 필터링
        filtered_df = trend_df[trend_df['tag'].isin(top_tags)]
        
        # 날짜 리스트
        dates = sorted(filtered_df['date'].unique())
        dates_str = [d.strftime('%Y-%m-%d') for d in dates]
        
        # 태그별 데이터 구성
        chart_data = {}
        for tag in top_tags:
            tag_data = filtered_df[filtered_df['tag'] == tag]
            # 날짜별로 정렬하여 값 추출
            tag_dict = dict(zip(tag_data['date'], tag_data['count']))
            values = [int(tag_dict.get(d, 0)) for d in dates]
            chart_data[tag] = values
        
        return {
            'dates': dates_str,
            'tags': top_tags,
            'data': chart_data,
            'total_issues': int(filtered_df['count'].sum())
        }
    
    def get_tag_performance(self, tag: str, days: int = 30) -> Dict[str, Any]:
        """
        특정 태그의 성과 지표를 반환합니다.
        
        Args:
            tag: 태그 이름
            days: 조회할 일수
        
        Returns:
            성과 지표 딕셔너리
        """
        raw_df = self.db_manager.get_tag_trends(days=days)
        
        if raw_df.empty:
            return {
                'tag': tag,
                'total_count': 0,
                'avg_daily': 0,
                'max_daily': 0,
                'trend': 'stable'
            }
        
        tag_df = raw_df[raw_df['tag'] == tag]
        
        if tag_df.empty:
            return {
                'tag': tag,
                'total_count': 0,
                'avg_daily': 0,
                'max_daily': 0,
                'trend': 'stable'
            }
        
        total_count = int(tag_df['count'].sum())
        avg_daily = round(tag_df['count'].mean(), 2)
        max_daily = int(tag_df['count'].max())
        
        # 트렌드 방향 계산 (최근 7일 vs 그 이전)
        recent_date = tag_df['date'].max()
        recent_7d_start = recent_date - timedelta(days=7)
        
        recent_7d_avg = tag_df[
            tag_df['date'] >= recent_7d_start
        ]['count'].mean()
        
        previous_7d_avg = tag_df[
            tag_df['date'] < recent_7d_start
        ]['count'].mean() if len(tag_df[tag_df['date'] < recent_7d_start]) > 0 else recent_7d_avg
        
        if recent_7d_avg > previous_7d_avg * 1.1:
            trend = 'rising'
        elif recent_7d_avg < previous_7d_avg * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'tag': tag,
            'total_count': total_count,
            'avg_daily': avg_daily,
            'max_daily': max_daily,
            'trend': trend,
            'recent_7d_avg': round(recent_7d_avg, 2),
            'previous_7d_avg': round(previous_7d_avg, 2)
        }


def main():
    """테스트용 메인 함수"""
    import sys
    from pathlib import Path
    
    # 프로젝트 루트를 Python 경로에 추가
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 트렌드 계산기 생성 및 실행
    calculator = TrendCalculator()
    
    # 트렌드 요약
    summary = calculator.get_trend_summary(days=30)
    print("\n트렌드 요약:")
    print(f"  총 이슈: {summary['total_issues']}개")
    print(f"  총 태그: {summary['total_tags']}개")
    print(f"\n상위 태그:")
    for item in summary['top_tags'][:5]:
        print(f"  {item['tag']}: {item['count']}개")
    
    print(f"\n급상승 태그:")
    for item in summary['trending_tags'][:5]:
        print(f"  {item['tag']}: {item['change_rate']:+.1f}% ({item['change']:+d}개)")
    
    # 차트 데이터
    chart_data = calculator.get_tag_trend_chart_data(days=30, top_n_tags=5)
    print(f"\n차트 데이터:")
    print(f"  날짜 수: {len(chart_data['dates'])}개")
    print(f"  태그 수: {len(chart_data['tags'])}개")


if __name__ == "__main__":
    main()











