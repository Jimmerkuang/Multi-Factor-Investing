# -*- coding: utf-8 -*-
# Time   : 2021/12/7 16:01
# Author : kfu

import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import tushare as ts
ts.set_token('8a13051b514249491b029cb46bcf1cd4e059b83bdeb516fc53c9f630')
pro = ts.pro_api()


class DataDownloader(object):

    def __init__(self):
        self.data_path = "F:\Data\multi_factor"
        self.stock_basic = self._get_stock_basic()
        self.stock_code_list = list(self.stock_basic.ts_code)
        self.stock_daily = None
        self.stock_daily_basic = None

    def _get_stock_basic(self):
        CSI800 = pro.index_weight(index_code='000906.SH', start_date='20180101', end_date='20200101').con_code.unique()
        stock_basic = pro.stock_basic(exchange='', list_status='L')
        stock_basic = stock_basic[stock_basic.ts_code.isin(CSI800)]
        stock_basic = stock_basic[-stock_basic['name'].apply(lambda x: x.startswith('*ST'))]
        stock_basic = stock_basic[stock_basic['list_date'].astype(int).values < 20100101]
        return stock_basic

    def get_stock_daily(self):
        stock_daily = [pro.daily(ts_code=code, start_date='20100101') for code in tqdm(self.stock_code_list)]
        self.stock_daily = pd.concat(stock_daily).reset_index(drop=True)
        self.stock_daily.to_parquet(os.path.join(self.data_path, 'stock_daily.parquet'))

    def get_stock_daily_basic(self):
        stock_daily_basic = [pro.daily_basic(ts_code=code, start_date='20100101') for code in tqdm(self.stock_code_list)]
        self.stock_daily_basic = pd.concat(stock_daily_basic).reset_index(drop=True)
        self.stock_daily_basic.to_parquet(os.path.join(self.data_path, 'stock_daily_basic.parquet'))


if __name__ == '__main__':
    data_downloader = DataDownloader()
    data_downloader.get_stock_daily()
    data_downloader.get_stock_daily_basic()