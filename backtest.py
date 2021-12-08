# -*- coding: utf-8 -*-
# Time   : 2021/12/8 16:25
# Author : kfu

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Backtest(object):

    def __init__(self):
        self.data_path = "F:\Data\multi_factor"
        self.score = pd.read_parquet(os.path.join(self.data_path, 'score.parquet'))
        self.group_profit = None

    def _stock_cross_section_grouping(self, score_series, bins=5):
        quantile_list = list(score_series.quantile([i for i in np.arange(1/bins, 1, 1/bins)]))
        quantile_list.insert(0, -1e10)
        quantile_list.append(1e10)
        score_group = pd.cut(score_series, quantile_list, labels=False, duplicates='drop')
        if score_group.nunique() < bins:
            score_group[:] = np.NaN
        return score_group

    def plot_profit(self):
        time_series = pd.to_datetime(pd.Series(self.group_profit.trade_date.unique()).sort_values(), format='%Y%m%d')
        plt.figure(figsize=(15, 12))
        plt.plot(time_series, self.group_profit[self.group_profit.group == 0]['cum_profit'], label='group 0')
        plt.plot(time_series, self.group_profit[self.group_profit.group == 1]['cum_profit'], label='group 1')
        plt.plot(time_series, self.group_profit[self.group_profit.group == 2]['cum_profit'], label='group 2')
        plt.plot(time_series, self.group_profit[self.group_profit.group == 3]['cum_profit'], label='group 3')
        plt.plot(time_series, self.group_profit[self.group_profit.group == 4]['cum_profit'], label='group 4')
        plt.xlabel('Year')
        plt.legend()
        plt.show()

    def main(self):
        self.score['group'] = self.score.groupby('trade_date')['score'].apply(self._stock_cross_section_grouping)
        self.score = self.score.sort_values(['ts_code', 'trade_date'])
        self.score['group'] = self.score['group'].ffill()
        self.score['profit'] = self.score.groupby('ts_code')['close'].apply(lambda x: (x.shift(-1) / x - 1).fillna(0))
        self.group_profit = self.score.groupby(['trade_date', 'group'])['profit'].apply(lambda x: x.sum() / len(x) - 0.00135).reset_index().sort_values(['group', 'trade_date'])
        self.group_profit['cum_profit'] = self.group_profit.groupby('group')['profit'].apply(lambda x: (x+1).cumprod())


if __name__ == '__main__':
    backtest = Backtest()
    backtest.main()
    backtest.plot_profit()