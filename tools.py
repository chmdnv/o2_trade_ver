from collections import deque
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timezone


class PSAR:
    def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
        self.max_af = max_af
        self.init_af = init_af
        self.af = init_af
        self.af_step = af_step
        self.extreme_point = None
        self.high_price_trend = []
        self.low_price_trend = []
        self.high_price_window = deque(maxlen=2)
        self.low_price_window = deque(maxlen=2)

        # Lists to track results
        self.psar_list = []
        self.af_list = []
        self.ep_list = []
        self.high_list = []
        self.low_list = []
        self.trend_list = []
        self._num_days = 0

    def calc_PSAR(self, high, low):
        if self._num_days >= 3:
            psar = self._calc_PSAR()
        else:
            psar = self._init_PSAR_vals(high, low)

        psar = self._update_current_vals(psar, high, low)
        self._num_days += 1

        return psar

    def _init_PSAR_vals(self, high, low):
        if len(self.low_price_window) <= 1:
            self.trend = None
            self.extreme_point = high
            return None

        if self.high_price_window[0] < self.high_price_window[1]:
            self.trend = 1
            psar = min(self.low_price_window)
            self.extreme_point = max(self.high_price_window)
        else:
            self.trend = 0
            psar = max(self.high_price_window)
            self.extreme_point = min(self.low_price_window)

        return psar

    def _calc_PSAR(self):
        prev_psar = self.psar_list[-1]
        if self.trend == 1:  # Up
            psar = prev_psar + self.af * (self.extreme_point - prev_psar)
            psar = min(psar, min(self.low_price_window))
        else:
            psar = prev_psar - self.af * (prev_psar - self.extreme_point)
            psar = max(psar, max(self.high_price_window))

        return psar

    def _update_current_vals(self, psar, high, low):
        if self.trend == 1:
            self.high_price_trend.append(high)
        elif self.trend == 0:
            self.low_price_trend.append(low)

        psar = self._trend_reversal(psar, high, low)

        self.psar_list.append(psar)
        self.af_list.append(self.af)
        self.ep_list.append(self.extreme_point)
        self.high_list.append(high)
        self.low_list.append(low)
        self.high_price_window.append(high)
        self.low_price_window.append(low)
        self.trend_list.append(self.trend)

        return psar

    def _trend_reversal(self, psar, high, low):
        # Checks for reversals
        reversal = False
        if self.trend == 1 and psar > low:
            self.trend = 0
            psar = max(self.high_price_trend)
            self.extreme_point = low
            reversal = True
        elif self.trend == 0 and psar < high:
            self.trend = 1
            psar = min(self.low_price_trend)
            self.extreme_point = high
            reversal = True

        if reversal:
            self.af = self.init_af
            self.high_price_trend.clear()
            self.low_price_trend.clear()
        else:
            if high > self.extreme_point and self.trend == 1:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = high
            elif low < self.extreme_point and self.trend == 0:
                self.af = min(self.af + self.af_step, self.max_af)
                self.extreme_point = low

        return psar


def ts2dt(timestamp):
    """Переводит метку времени в строку дату-время МОСКОВСКОЕ!"""
    dt_object = dt.fromtimestamp(float(timestamp) / 1000.0 + 3 * 60 * 60,
                                 timezone.utc)
    time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S").partition("+")[0]
    return time_str


def add_is_raise(df, feat, periods, eps=-0.005) -> pd.Series:
    """Вычисляет, растёт ли фича"""

    df = df.reset_index().copy()
    new_name = f"{feat}_isRaise"
    df[new_name] = False

    for i in range(len(df)):
        y = df.loc[i - periods: i, feat].dropna()
        x = np.arange(len(y))

        if len(y) >= periods:
            p = np.polynomial.Polynomial.fit(x, y, 1)
            try:
                is_raise = p.convert().coef[1] > eps * y.max()
            except IndexError:
                is_raise = False

            df.loc[i, new_name] = is_raise

    return df[[new_name]]


def add_is_fall(df, feat, periods, eps=0.005) -> pd.Series:
    """Вычисляет, падает ли фича"""

    df = df.reset_index().copy()
    new_name = f"{feat}_isFall"
    df[new_name] = False

    for i in range(len(df)):
        y = df.loc[i - periods: i, feat].dropna()
        x = np.arange(len(y))

        if len(y) >= periods:
            p = np.polynomial.Polynomial.fit(x, y, 1)
            try:
                is_fall = p.convert().coef[1] < eps * y.max()
            except IndexError:
                is_fall = False

            df.loc[i, new_name] = is_fall

    return df[[new_name]]


def scope(df, feat, periods) -> pd.Series:
    """Вычисляет наклон касательной"""

    df = df.reset_index()[[feat]].copy()
    new_name = f"{feat}_scope"
    df[new_name] = np.nan

    for i in range(periods - 1, len(df)):
        y = df.iloc[i - periods + 1: i + 1][feat].dropna()
        x = np.arange(len(y))

        if len(y) >= periods:
            p = np.polynomial.Polynomial.fit(x, y, 1)
            try:
                scp = p.convert().coef[1]
            except IndexError:
                pass

            df.loc[i, new_name] = scp

    return df[[new_name]]

