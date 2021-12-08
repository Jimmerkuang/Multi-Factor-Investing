# -*- coding: utf-8 -*-
# Time   : 2021/12/7 20:59
# Author : kfu

import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats.stats import spearmanr, rankdata
from gen_signal import get_trade_calendar


class Dataset(object):

    def __init__(self, train_date_list, test_date_list):
        self.train_date_list = train_date_list
        self.test_date_list = test_date_list
        self.data_path = "F:\Data\multi_factor"
        self.signal_daily = pd.read_parquet(os.path.join(self.data_path, 'stock_daily_signal.parquet'))
        self.score = pd.DataFrame()

    def prepare_data(self):
        train_data = self.signal_daily[self.signal_daily.trade_date.isin(self.train_date_list)]
        test_data = self.signal_daily[self.signal_daily.trade_date.isin(self.test_date_list)]
        x_train = train_data.drop(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'f_ret'], axis=1)
        x_test = test_data.drop(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'f_ret'], axis=1)
        y_train = train_data['f_ret']
        y_test = test_data['f_ret']
        self.score = test_data[['ts_code', 'trade_date', 'open', 'close', 'f_ret']]
        return x_train, y_train, x_test, y_test


class Trainer(object):

    def __init__(self):
        self.data_path = "F:\Data\multi_factor"
        self.daily_ic = None
        self.score = pd.DataFrame()
        self.feature_importance = pd.DataFrame()
        self.trade_calendar = [i for i in get_trade_calendar() if i >= 20100101]
        self.score_calendar = self.trade_calendar[240:]

    def save_score(self):
        self.score.to_parquet(os.path.join(self.data_path, 'score.parquet'))

    def plot_daily_ic(self):
        ic_list = []
        for date in self.score_calendar:
            temp_score = self.score[self.score.trade_date == date]
            ic_list.append(spearmanr(rankdata(temp_score['f_ret']), rankdata(temp_score['score']))[0])
        self.daily_ic = pd.DataFrame({'date': self.score_calendar, 'ic': ic_list})
        self.daily_ic['cum_ic'] = self.daily_ic.ic.cumsum()
        plt.figure(figsize=(10, 8))
        plt.plot(pd.to_datetime(self.score_calendar, format='%Y%m%d'), self.daily_ic['cum_ic'])
        plt.xlabel('Year')
        plt.legend(['Cumsum_IC'])
        plt.show()

    def _lgb_reg_model(self, x_train, y_train, x_test, y_test):
        gbm = lgb.LGBMRegressor(num_iterations=1000,
                                learning_rate=0.05,
                                max_depth=5,
                                min_data_in_leaf=20,
                                boosting='goss')
        gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=50)
        return gbm

    def train(self):
        for i in tqdm(range(0, len(self.trade_calendar)-240, 20)):
            train_date_list = self.trade_calendar[0+i:240+i]
            test_date_list = self.trade_calendar[240+i:260+i]
            lgb_dataset = Dataset(train_date_list, test_date_list)
            x_train, y_train, x_test, y_test = lgb_dataset.prepare_data()
            gbm = self._lgb_reg_model(x_train, y_train, x_test, y_test)
            lgb_dataset.score['score'] = gbm.predict(x_test)
            self.score = self.score.append(lgb_dataset.score, ignore_index=True)
            self.feature_importance = self.feature_importance.append(pd.DataFrame({'feature_name': gbm.booster_.feature_name(),
                                                                                   'importance': gbm.booster_.feature_importance(importance_type='split')}),
                                                                     ignore_index=True)
        self.feature_importance = self.feature_importance.groupby('feature_name').sum().reset_index().sort_values('importance', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    lgb_trainer = Trainer()
    lgb_trainer.train()
    lgb_trainer.plot_daily_ic()
    lgb_trainer.save_score()