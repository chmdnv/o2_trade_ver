import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timezone

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_formatted_time(timestamp):
    """Переводит метку времени в дату-время МОСКОВСКОЕ!"""
    dt_object = dt.fromtimestamp(float(timestamp) / 1000.0 + 3 * 60 * 60,
                                 timezone.utc)
    return dt_object


def ts2dt(timestamp):
    """Переводит метку времени в строку дату-время МОСКОВСКОЕ!"""
    dt_object = dt.fromtimestamp(float(timestamp) / 1000.0 + 3 * 60 * 60,
                                 timezone.utc)
    time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S").partition("+")[0]
    return time_str


def dt2ts(dtime: str) -> int:
    """Переводит строку дату-время МОСКОВСКОЕ в метку времени"""
    dt_str = str(dtime).partition(".")[0].replace("T", " ").partition("+")[0]
    dt_str = f'{dt_str}.+0300'
    return int(dt.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%z').timestamp()) * 1000


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


def add_features(init_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в DF необходимые фичи"""

    init_df = init_df.copy()
    df = init_df.copy()

    # Avg Price
    df['Price'] = df[['Open', 'Close']].apply('mean', axis=1)

    # Price Trend
    df['Price_isRaise'] = add_is_raise(df, 'Price', 3, 0.0010)
    df['Price_isFall'] = add_is_fall(df, 'Price', 3, -0.0010)
    df['Price_scope'] = scope(df, 'Price', 2)
    df['Price_scope3'] = scope(df, 'Price', 2)
    df['Price_2div'] = scope(df, 'Price_scope', 2)
    scope_eps = 0.002
    df['Price_isRaise_2div'] = df.Price_isRaise & (df.Price_2div > scope_eps * df.Price)
    df['Block_long'] = df['Price_scope3'] < 0

    # MACD
    df['EMA12'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Diff'] = df.MACD - df.Signal_Line
    df['DiffMA'] = df.Diff.rolling(10).mean().rolling(10).mean()
    df['IsDiffRaise'] = add_is_raise(df, 'DiffMA', 3, 0.001)

    # Volume
    df['Maker buy base asset volume'] = df.Volume - df['Taker buy base asset volume']
    df['TMRate'] = df['Taker buy base asset volume'] / df['Maker buy base asset volume']
    df['TMRate'] = (df.TMRate > 3).rolling(3, closed='right').max()
    df['VolumeMA'] = df.Volume.rolling(10).mean().rolling(3).mean()
    df['VolumeRaisePrc'] = (df.VolumeMA - df.VolumeMA.shift(1)) / df.VolumeMA.shift(1) * 100
    df['IsVolumePump'] = df.VolumeRaisePrc > 30
    df['IsVolumeRaise'] = add_is_raise(df, 'VolumeMA', 3, 0.01)
    df['IsVolumeFall'] = add_is_fall(df, 'VolumeMA', 3, 0.01)
    df['Is2DivVolumeRaise'] = add_is_raise(df, 'VolumeRaisePrc', 3, -0.1)
    df['Is2DivVolumeFall'] = add_is_fall(df, 'VolumeRaisePrc', 3, 0.1)
    df['GoodVol'] = (df.IsVolumeRaise & df.Is2DivVolumeRaise)
    df['BadVol'] = (df.IsVolumeFall & df.Is2DivVolumeFall)
    df['VolumeFactor'] = df.GoodVol.astype(int) - df.BadVol.astype(int)

    # Volume strength
    df['Strength'] = (df.Volume * (df.Price / df.Price.shift(1) - 1)).rolling(5).mean()
    df['Strength_isRaise'] = add_is_raise(df, 'Strength', 3, 0.001)
    df['Strength_isFall'] = add_is_fall(df, 'Strength', 3, -0.001)
    df['Str_isLocalMin'] = df.Strength_isFall.shift(1, fill_value=np.False_) & (df.Strength_isRaise) & (df.Strength < 0)
    df['Str_isLocalMax'] = df.Strength_isRaise.shift(1, fill_value=np.False_) & (df.Strength_isFall) & (df.Strength > 0)

    df[['Str_minDelta', 'Str_maxDelta']] = np.nan, np.nan
    prev_max, prev_min = 0, 0
    for i, row in df[df.Str_isLocalMin | df.Str_isLocalMax].iterrows():
        if row.Str_isLocalMin:
            delta = prev_max - row.Strength
            df.loc[i, 'Str_minDelta'] = delta
        elif row.Str_isLocalMax:
            delta = row.Strength - prev_min
            df.loc[i, 'Str_maxDelta'] = delta

    df.loc[:, ['Str_minDelta', 'Str_maxDelta']] = df[['Str_minDelta', 'Str_maxDelta']].ffill()

    for i, row in df[df.Str_isLocalMin | df.Str_isLocalMax].iterrows():
        if row.Str_isLocalMin:
            if row.Str_minDelta < df.loc[i - 1, 'Str_minDelta']:
                df.loc[i, 'Str_isLocalMin'] = np.False_
        elif row.Str_isLocalMax:
            if row.Str_maxDelta < df.loc[i - 1, 'Str_maxDelta']:
                df.loc[i, 'Str_isLocalMax'] = np.False_

    seen_max = False
    seen_min = False
    for i, row in df.iterrows():
        if row.Str_isLocalMax == True and row.Strength > 0:
            seen_max = True
            seen_min = False

        if row.Str_isLocalMin == True and row.Strength < 0:
            seen_min = True
            seen_max = False

        res = (seen_max == True) * (seen_max ^ seen_min)
        df.loc[i, 'Str_ind'] = res

    # Добавление нужных колонок в исходный DF
    new_cols = ['Price', 'Price_isRaise_2div', 'Price_isFall', 'Price_2div',
                'IsDiffRaise', 'DiffMA',
                'IsVolumePump', 'TMRate', 'GoodVol', 'Str_ind',
                'Block_long']

    init_df.loc[df.index, new_cols] = df[new_cols]

    return init_df


def ak_long_indicator(df: pd.DataFrame) -> float:
    """Вычисляет значение индикатора лонг-определённости по минутным свечам (ver.7.1)"""

    df = df.copy()
    df = add_features(df)

    row = df.iloc[-1]  # работаем с последней закрытой минутой
    indicator = sum(
        (x or 0 for x in (
            row.IsDiffRaise and row.DiffMA > 0,
            row.Price_isRaise_2div * 2,
            row.Price_isFall * (-2),
            row.IsVolumePump and row.TMRate,
            row.Str_ind,
            row.GoodVol,
        )))

    # Проверка блокировки открытия
    indicator = indicator * (not row.Block_long)

    return float(indicator)


# Тест
dir = 'e:/Documents/o2trading/data/antifrauded all/'
file_name = 'CYBERUSDT-1m-2024-08-11_06-24-18.csv'
df = pd.read_csv(dir + file_name)

# Пороги входа/выхода
tresh_top = 3
tresh_bot = -1

alert_time = df.Time_alert[0]
line_2 = alert_time + 2 * 60 * 60 * 1000

# Условия входа/выхода
enter_idxs = []
exit_idxs = []
ind = []

for i in df[df.Time.between(alert_time, line_2)].index:
    indicator = ak_long_indicator(df.loc[:i])

    # Открытие
    if indicator >= tresh_top:
        enter_idxs.append(i+1)

    # Закрытие
    elif indicator <= tresh_bot:
        exit_idxs.append(i+1)

    ind.append(indicator)

print(pd.Series(ind).value_counts(dropna=False).sort_index())
