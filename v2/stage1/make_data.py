import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

pd.set_option('future.no_silent_downcasting', True)

# 선택된 컬럼들
selected_cols = [
    # 기업 식별자
    'GVKEY', 'datadate', 'tic', 'cusip', 'conm', 'LPERMNO', 'LPERMCO',
    
    # 재무상태표 항목
    'atq', 'ltq', 'dlcq', 'dlttq', 'seqq', 'ceqq', 'cheq',
    
    # 손익계산서 항목
    'saleq', 'cogsq', 'niq', 'oiadpq', 'ibq', 'piq', 'txtq', 'dpq',
    
    # 주가 관련 정보
    'prccq', 'prchq', 'prclq', 'epspxq', 'epsfiq', 'cshoq',
    
    # 산업분류코드
    'sic', 'naics', 'gsector', 'gind', 'gsubind'
]

def safe_compound_return(x):
    """안전한 복리 수익률 계산"""
    # NaN 값 제거
    x_clean = x.dropna()
    if len(x_clean) == 0:
        return np.nan
    
    # 극단값 처리 (-98% 이하 수익률 제외)
    x_clean = x_clean[x_clean > -0.98]
    if len(x_clean) == 0:
        return np.nan
    
    # # 이상치 제거 (일일 수익률 100% 초과)
    # x_clean = x_clean[x_clean < 1.0]
    # if len(x_clean) == 0:
    #     return np.nan
    
    return (x_clean + 1).prod() - 1

def calculate_momentum(df):
    """
    PERMNO(기업)별로 정렬된 월별 수익률이 담긴 데이터프레임(df)에 대해,
    MOM1~MOM24를 계산하여 새로운 컬럼으로 추가한다.
    """
    df = df.copy()  # 원본 보호

    for i in range(1, 25):
        colname = f'MOM{i}'
        if i == 1:
            # MOM1 = r_{t-1} = 이전 달 수익률
            df[colname] = df['RET'].shift(1)
        else:
            # MOMi = (Π(1 + r_{t-j})) - 1, j=1..i (단, 현재 달은 제외하기 위해 shift(1))
            # rolling(i).apply(np.prod)은 윈도우 내 곱셈
            df[colname] = (
                df['RET_plus1']
                  .shift(1)                 # 현재 달은 제외 (이전 달까지)
                  .rolling(window=i)        # 과거 i개월
                  .apply(np.prod, raw=True) # 곱셈
                  - 1
            )

    return df

def convert_quarterly_to_monthly(firm_char, selected_cols):
    """
    분기별 재무제표 데이터를 월별 데이터로 변환
    """
    # 1. 선택된 컬럼만 복사
    firm_char_sel = firm_char[selected_cols].copy()
    
    # 2. datadate를 datetime 형식으로 변환 및 정렬
    firm_char_sel['datadate'] = pd.to_datetime(firm_char_sel['datadate'])
    firm_char_sel = firm_char_sel.sort_values(['LPERMNO', 'datadate'])
    
    # 3. 중복 제거 (같은 기업, 같은 날짜의 중복 데이터)
    firm_char_sel = firm_char_sel.drop_duplicates(['LPERMNO', 'datadate'])
    
    # 4. 각 기업별로 월별 데이터를 생성
    monthly_firm_char_list = []
    
    for firm, group in tqdm(firm_char_sel.groupby('LPERMNO'), desc="Converting quarterly to monthly"):
        group = group.sort_values('datadate')
        
        # 해당 기업의 데이터 기간
        start_date = group['datadate'].min()
        end_date = group['datadate'].max()
        
        # 월별 그리드 생성
        monthly_index = pd.date_range(start=start_date, end=end_date, freq='ME')
        df_monthly = pd.DataFrame({'month': monthly_index})
        
        # merge_asof를 사용하여 각 월에 대해 가장 최근 분기 데이터를 가져옴
        merged = pd.merge_asof(
            df_monthly.sort_values('month'),
            group.sort_values('datadate'),
            left_on='month',
            right_on='datadate',
            direction='backward'
        )
        
        # # 3개월 이상 오래된 데이터는 사용하지 않음
        # time_diff = (merged['month'] - merged['datadate']).dt.days
        # merged.loc[time_diff > 120, group.columns.difference(['LPERMNO'])] = np.nan
        
        # 월 정보를 Period 형식의 문자열로 변환
        merged['month'] = merged['month'].dt.to_period('M').astype(str)
        
        monthly_firm_char_list.append(merged)
    
    # 5. 모든 기업의 월별 데이터를 하나로 결합
    monthly_firm_char = pd.concat(monthly_firm_char_list, ignore_index=True)
    
    logging.info(f"Quarterly to monthly conversion completed: {len(monthly_firm_char)} rows")
    
    return monthly_firm_char

def process_nan_values(df):
    """
    데이터 타입별로 적절한 NaN 처리
    """
    df = df.copy()
    
    # 1. 수익률 관련 컬럼은 NaN 유지
    return_cols = ['RET', 'log_ret'] + [col for col in df.columns if 'MOM' in col]
    
    # 2. 재무제표 컬럼은 forward fill 후 0
    financial_cols = ['atq', 'ltq', 'dlcq', 'dlttq', 'seqq', 'ceqq', 'cheq',
                     'saleq', 'cogsq', 'niq', 'oiadpq', 'ibq', 'piq', 'txtq', 'dpq']
    
    # for col in financial_cols:
    #     if col in df.columns:
    #         df[col] = df.groupby('PERMNO')[col].ffill().fillna(0)
    
    # 3. 가격 관련 컬럼은 forward fill
    price_cols = ['PRC', 'OPENPRC', 'prccq', 'prchq', 'prclq', 'ASKHI', 'BIDLO']
    
    for col in price_cols:
        if col in df.columns:
            df[col] = df.groupby('PERMNO')[col].ffill()
    
    # 4. 주식수 관련
    share_cols = ['SHROUT', 'cshoq']
    for col in share_cols:
        if col in df.columns:
            df[col] = df.groupby('PERMNO')[col].ffill().fillna(0)
    
    # 5. 문자열/카테고리 컬럼
    string_cols = ['tic', 'cusip', 'conm', 'TICKER', 'COMNAM', 'CUSIP']
    for col in string_cols:
        if col in df.columns:
            df[col] = df.groupby('PERMNO')[col].ffill().fillna('Unknown')
    
    # 6. 산업분류 코드
    industry_cols = ['sic', 'naics', 'gsector', 'gind', 'gsubind', 'SICCD']
    for col in industry_cols:
        if col in df.columns:
            df[col] = df.groupby('PERMNO')[col].ffill().fillna('Unknown')
    
    return df

def calculate_safe_log_returns(returns):
    """
    안전한 로그 수익률 계산
    """
    # 합리적인 범위 체크 (-99% < return < 1000%)
    valid_range = (returns > -0.99) & (returns < 10)
    
    log_returns = np.where(
        valid_range,
        np.log(1 + returns),
        np.nan
    )
    
    # # 추가 이상치 제거 (일일 95% 이상 변동)
    # log_returns = np.where(
    #     abs(log_returns) > 0.95,
    #     np.nan,
    #     log_returns
    # )
    
    return log_returns

def validate_data(df, stage_name):
    """
    데이터 검증 및 로깅
    """
    logging.info(f"\n=== {stage_name} 데이터 검증 ===")
    logging.info(f"총 행 수: {len(df):,}")
    logging.info(f"고유 기업 수: {df['PERMNO'].nunique():,}")
    
    # NaN 비율 체크
    nan_ratio = df.isna().sum() / len(df) * 100
    high_nan_cols = nan_ratio[nan_ratio > 50].sort_values(ascending=False)
    if len(high_nan_cols) > 0:
        logging.warning(f"NaN 비율이 50% 이상인 컬럼:\n{high_nan_cols}")
    
    # 수익률 통계
    if 'RET' in df.columns:
        ret_stats = df['RET'].describe()
        logging.info(f"수익률 통계:\n{ret_stats}")
        
        extreme_returns = df[(df['RET'] < -0.5) | (df['RET'] > 1.0)]
        if len(extreme_returns) > 0:
            logging.warning(f"극단적 수익률 관측치: {len(extreme_returns):,} ({len(extreme_returns)/len(df)*100:.2f}%)")
    
    return df

def make_data(crsp, char):
    """
    메인 데이터 처리 함수
    """
    logging.info("데이터 처리 시작")
    
    # 1. CRSP 데이터 전처리
    logging.info("CRSP 데이터 변환 중...")
    crsp['date'] = pd.to_datetime(crsp['date'])
    crsp['month'] = crsp['date'].dt.to_period('M')
    
    # 2. 월별 집계
    agg_dict = {
        'date': 'max',
        'PERMCO': 'last',
        'SICCD': 'last',
        'TICKER': 'last',
        'COMNAM': 'last',
        'CUSIP': 'last',
        'OPENPRC': 'first',
        'ASKHI': 'last',
        'BIDLO': 'last',
        'PRC': 'last',
        'VOL': 'sum',
        'RET': safe_compound_return,
        'SHROUT': 'last'
    }
    
    logging.info("월별 데이터 집계 중...")
    monthly_crsp = crsp.groupby(['PERMNO', 'month'], as_index=False).agg(agg_dict)
    monthly_crsp['month'] = monthly_crsp['month'].astype(str)
    monthly_crsp['RET_plus1'] = 1 + monthly_crsp['RET']
    
    # 데이터 검증
    monthly_crsp = validate_data(monthly_crsp, "월별 CRSP")
    
    # 3. 모멘텀 계산
    logging.info("모멘텀 지표 계산 중...")
    monthly_crsp = (
        monthly_crsp
        .groupby('PERMNO', group_keys=False)
        .apply(calculate_momentum)
    )
    
    # 4. 분기별 재무제표를 월별로 변환
    logging.info("분기별 재무제표를 월별로 변환 중...")
    monthly_firm_char = convert_quarterly_to_monthly(char, selected_cols)
    
    # 5. 데이터 병합
    logging.info("데이터 병합 중...")
    monthly_firm_char = monthly_firm_char.rename(columns={'LPERMNO': 'PERMNO'})
    
    # 데이터 타입 통일
    monthly_crsp['PERMNO'] = monthly_crsp['PERMNO'].astype(str)
    monthly_firm_char['PERMNO'] = monthly_firm_char['PERMNO'].astype(str)
    
    # 병합
    merged_data = pd.merge(
        monthly_crsp,
        monthly_firm_char,
        on=['PERMNO', 'month'],
        how='inner'
    )
    
    # 6. NaN 처리
    logging.info("NaN 값 처리 중...")
    pre_nan_count = merged_data.isna().sum().sum()
    merged_data = process_nan_values(merged_data)
    post_nan_count = merged_data.isna().sum().sum()
    logging.info(f"NaN 처리 전: {pre_nan_count:,}, 처리 후: {post_nan_count:,}")
    
    # 7. 로그 수익률 계산
    logging.info("로그 수익률 계산 중...")
    merged_data['log_ret'] = calculate_safe_log_returns(merged_data['RET'])
    
    # 최종 데이터 검증
    merged_data = validate_data(merged_data, "최종 병합")
    
    # 8. 데이터 저장
    logging.info("데이터 저장 중...")
    
    # 메인 데이터 저장
    merged_data.to_pickle('monthly_merged_data.pkl')
    
    # 월별 로그 수익률 피벗 테이블
    log_ret_df = merged_data.pivot(
        index='PERMNO',
        columns='month',
        values='log_ret'
    )
    log_ret_df.to_pickle('data/log_returns_by_month.pkl')
    
    # 9. 일별 로그 수익률 처리
    logging.info("일별 로그 수익률 처리 중...")
    process_daily_returns()
    
    # 10. 월별 CSV 파일 저장
    os.makedirs('data/monthly', exist_ok=True)
    for month in tqdm(merged_data['month'].unique(), desc="월별 CSV 저장"):
        month_data = merged_data[merged_data['month'] == month]
        month_data.to_csv(f'data/monthly/{month}.csv', index=False)
    
    logging.info("데이터 처리 완료!")
    logging.info(f"최종 데이터 크기: {merged_data.shape}")
    logging.info(f"로그 수익률 피벗 테이블 크기: {log_ret_df.shape}")
    
    return merged_data, log_ret_df

def process_daily_returns():
    """
    일별 로그 수익률 처리
    """

    daily_crsp = pd.read_pickle('data/raw_data/inter_permno_crsp.pkl')
    daily_crsp['date'] = pd.to_datetime(daily_crsp['date'])
    daily_crsp['PERMNO'] = daily_crsp['PERMNO'].astype(str)
    
    # 안전한 로그 수익률 계산
    daily_crsp['log_ret'] = calculate_safe_log_returns(daily_crsp['RET'])
    
    # 일별 수익률 통계
    daily_stats = daily_crsp['log_ret'].describe()
    logging.info(f"일별 로그 수익률 통계:\n{daily_stats}")
    
    # 피벗 테이블 생성
    daily_log_ret_df = daily_crsp.pivot(
        index='date',
        columns='PERMNO',
        values='log_ret'
    )
    
    # 결측치 비율 체크
    missing_ratio = daily_log_ret_df.isna().sum().sum() / daily_log_ret_df.size * 100
    logging.info(f"일별 수익률 결측치 비율: {missing_ratio:.2f}%")
    
    # 저장
    os.makedirs('data', exist_ok=True)
    daily_log_ret_df.to_pickle('data/daily_returns.pkl')
    logging.info(f"일별 로그 수익률 데이터 저장 완료: {daily_log_ret_df.shape}")
    


if __name__ == '__main__':
    # 데이터 로드
    logging.info("원본 데이터 로딩 중...")
    crsp = pd.read_pickle('data/raw_data/inter_permno_crsp.pkl')
    char = pd.read_pickle('data/raw_data/inter_permno_firm_characteristics.pkl')
    
    logging.info(f"CRSP 데이터 크기: {crsp.shape}")
    logging.info(f"재무제표 데이터 크기: {char.shape}")
    
    # 데이터 처리 실행
    merged_data, log_ret_df = make_data(crsp, char)
    
    # 최종 요약 통계
    logging.info("\n=== 최종 데이터 요약 ===")
    logging.info(f"처리된 기업 수: {merged_data['PERMNO'].nunique():,}")
    logging.info(f"처리된 기간: {merged_data['month'].min()} ~ {merged_data['month'].max()}")
    logging.info(f"총 관측치 수: {len(merged_data):,}")
    
    # 모멘텀 지표 요약
    mom_cols = [col for col in merged_data.columns if 'MOM' in col]
    logging.info(f"생성된 모멘텀 지표 수: {len(mom_cols)}")
    
