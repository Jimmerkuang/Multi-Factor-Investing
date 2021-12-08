# -*- coding: utf-8 -*-
# Time   : 2021/12/7 17:15
# Author : kfu

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tushare as ts
ts.set_token('8a13051b514249491b029cb46bcf1cd4e059b83bdeb516fc53c9f630')
pro = ts.pro_api()


def get_trade_calendar():
    trade_calendar = pro.trade_cal(start_date='20100101', end_date='20211207')
    trade_calendar = tuple(trade_calendar[trade_calendar.is_open == 1]['cal_date'].astype(int).values)
    return trade_calendar


class DataCleaner(object):

    def __init__(self):
        self.data_path = "F:\Data\multi_factor"
        self.stock_daily = None
        self.trade_calendar = get_trade_calendar()

    def _clean_data(self):
        stock_daily = pd.read_parquet(os.path.join(self.data_path, 'stock_daily.parquet'))
        stock_daily_basic = pd.read_parquet(os.path.join(self.data_path, 'stock_daily_basic.parquet'))
        stock_daily = pd.merge(stock_daily, stock_daily_basic.drop('close', axis=1), on=['ts_code', 'trade_date'])
        return stock_daily.drop(['pre_close', 'change', 'pct_chg'], axis=1)

    def _fillna(self, stock_daily):
        ffill_col = ['high', 'low', 'close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']
        zerofill_col = ['vol', 'amount', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'free_share']
        stock_daily[ffill_col] = stock_daily.groupby('ts_code')[ffill_col].ffill()
        stock_daily[zerofill_col] = stock_daily.groupby('ts_code')[zerofill_col].fillna(0)
        return stock_daily

    def resample_data(self):
        stock_daily = self._clean_data()
        stock_daily['trade_date'] = pd.to_datetime(stock_daily.trade_date, format='%Y%m%d')
        stock_daily = stock_daily.set_index('trade_date').groupby('ts_code').resample(rule='1d').last().drop('ts_code', axis=1).reset_index()
        stock_daily['trade_date'] = stock_daily.trade_date.dt.year * 1e4 + stock_daily.trade_date.dt.month * 1e2 + stock_daily.trade_date.dt.day
        stock_daily = stock_daily[stock_daily.trade_date.isin(self.trade_calendar)].sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        self.stock_daily = self._fillna(stock_daily)
        self.stock_daily.to_parquet(os.path.join(self.data_path, 'resample_stock_daily.parquet'))


class SignalGenerator(object):

    def __init__(self):
        self.data_path = "F:\Data\multi_factor"
        self.signal_daily = None
        self.trade_calendar = get_trade_calendar()
        self.stock_daily = pd.read_parquet(os.path.join(self.data_path, 'resample_stock_daily.parquet'))

    def _gen_future_return(self, df):
        df['f_ret'] = df.groupby('ts_code')['close'].apply(lambda x: (x.shift(-1)/x - 1)*100)
        df['f_ret'] = df['f_ret'].where(df['f_ret'] >= -10, -10)
        df['f_ret'] = df['f_ret'].where(df['f_ret'] <= 10, 10)
        return df[df.f_ret != np.NaN]

    def _gen_signal(self, df):
        for i in [1, 2, 3, 4, 5, 10, 20, 40, 60, 90, 120, 240]:
            df[f'ret_{i}'] = df.groupby('ts_code')['close'].apply(lambda x: (x/x.shift(i)-1)*100)
        for i in [5, 10, 20, 60, 90, 120, 240]:
            df[f'vol_{i}'] = df.groupby('ts_code')['close'].apply(lambda x: ((x/x.shift(i)-1)*100).rolling(i).std())
            df[f'poshigh_{i}'] = df['close'] / df.groupby('ts_code')['high'].apply(lambda x: x.rolling(i).max())
            df[f'poslow_{i}'] = df['close'] / df.groupby('ts_code')['low'].apply(lambda x: x.rolling(i).min())
        return df

    def main(self):
        signal_daily = self._gen_signal(self.stock_daily)
        self.signal_daily = self._gen_future_return(signal_daily)
        self.signal_daily.to_parquet(os.path.join(self.data_path, 'stock_daily_signal.parquet'))


if __name__ == '__main__':
    data_cleaner = DataCleaner()
    data_cleaner.resample_data()
    signal_generator = SignalGenerator()
    signal_generator.main()
