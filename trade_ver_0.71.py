from tools import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class AKTrade:
    def __init__(self, alert_time: int, tresh_top: int = None, tresh_bot: int = None):

        # Пороги входа/выхода
        self.tresh_top = 3 if tresh_top is None else tresh_top
        self.tresh_bot = -1 if tresh_bot is None else tresh_bot

        # Время алерта и выхода из торгов
        self.alert_time = alert_time
        self.line_2 = alert_time + 120 * 60 * 1000

        # Маркер состояния сделки: opened/closed
        self.state = 'closed'
        self.last_open_price = 0

        # Маркер поиска первой сделки
        self.first_deal = True

        # Запись сюда значений индикатора по индексу
        self.indicator = pd.Series()

        # Запись сигналов
        self.signals = {}  # {'index' : ['open_long', 'Price'], ...}
        self.enter_idxs = []  # индексы открытия
        self.exit_idxs = []  # индексы закрытия

        # Подсчёт эффективности
        self.pnl_sum = 0
        self.n_deals = 0
        self.pnl_lst = []
        self.commission = 0.16  # %

    @staticmethod
    def add_features(init_df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет в DF необходимые фичи"""

        init_df = init_df.copy()
        df = init_df.reset_index().copy()

        # Avg Price
        df['Price'] = df[['Open', 'Close']].apply('mean', axis=1)

        # Price Trend
        df['Price_isRaise'] = add_is_raise(df, 'Price', 3, 0.0020)
        df['Price_isFall'] = add_is_fall(df, 'Price', 3, -0.0020)
        df['Price_scope'] = scope(df, 'Price', 3)
        df['Price_2div'] = scope(df, 'Price_scope', 3)
        scope_eps = 0.001
        df['Price_isRaise_2div'] = df.Price_isRaise & (df.Price_2div > scope_eps * df.Price)

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
        new_cols = ['Price', 'Price_isRaise_2div', 'Price_isFall',
                    'IsDiffRaise', 'DiffMA',
                    'IsVolumePump', 'TMRate', 'GoodVol', 'Str_ind',
                   ]

        init_df.loc[:, new_cols] = df[new_cols].set_index(init_df.index)

        return init_df

    def ak_long_indicator(self, df: pd.DataFrame) -> float:
        """Вычисляет значение индикатора лонг-определённости по минутным свечам (ver.7.1)"""

        df = df.copy()
        df = self.add_features(df)

        alert_price = df[df.Time > df.Time_alert].iloc[0].Price

        row = df.iloc[-1]  # работаем с последней закрытой минутой
        indicator = sum(
            (x or 0 for x in (
                row.IsDiffRaise and row.DiffMA > 0,
                row.Price_isRaise_2div * 2,
                row.Price_isFall * (-2),
                row.IsVolumePump and row.TMRate,
                row.Str_ind,
                row.GoodVol,
                ((row.Price / alert_price - 1 > 0.005) and self.first_deal) * (3 + 2)
            )))

        self.indicator[df.last_valid_index()] = indicator

        # Моментальное открытие если цена выросла на 0,5% (Проверка актуальности)
        if (row.Price / alert_price - 1 > 0.005) and (row.Time >= self.alert_time):
            self.first_deal = False

        return float(indicator)

    def check_open_close(self, indicator: float, time: int) -> dict[str: bool]:
        """Выдаёт действие (вход/выход) по значению индикатора. Также есть выход по времени"""

        res = {'open_long': False, 'close_long': False}

        # Открытие
        res['open_long'] = indicator >= self.tresh_top

        # Закрытие
        res['close_long'] = indicator <= self.tresh_bot

        # Выход по времени
        if time >= self.alert_time + 120 * 60 * 1000:
            res['close_long'] = True

        return res

    def add_stats(self):
        for i, rec in self.signals.items():
            close_ = rec[1] if rec[0] == 'close_long' else None

        if self.last_open_price and close_:
            pnl = (close_ / self.last_open_price - 1) * 100 - self.commission
            self.pnl_sum += pnl
            self.n_deals += 1
            self.pnl_lst.append(pnl)

    def get_signal(self, df: pd.DataFrame) -> str:
        """ Получить сигнал открытия: open_long, закрытия: close_long, бездействия: None"""

        indicator = self.ak_long_indicator(df.iloc[-60:])
        res = self.check_open_close(indicator, df.iloc[-1].Time)
        output = 'open_long' if res['open_long'] else ''
        output = 'close_long' if res['close_long'] else output

        if output == 'open_long':
            i = df.last_valid_index()
            if self.state == 'closed':
                self.last_open_price = df.loc[i, ['Open', 'Close']].mean()
            self.state = 'opened'
            self.signals.update({i+1: [output, df.loc[i, ['Open', 'Close']].mean()]})
            self.enter_idxs.append(i+1)

        elif output == 'close_long':
            i = df.last_valid_index()
            self.signals.update({i + 1: [output, df.loc[i, ['Open', 'Close']].mean()]})
            self.exit_idxs.append(i + 1)
            if self.state == 'opened':
                self.add_stats()
            self.state = 'closed'

        return output or None


# Тест
def main_test():
    dir = 'data/'
    file_name = 'SSVUSDT-1m-2024-07-23_11-27-25.csv'
    df = pd.read_csv(dir + file_name)

    alert_time = df.iloc[0].Time_alert

    # Проверка работы
    trade = AKTrade(alert_time)

    for i in df[df.Time.between(alert_time, alert_time + 121 * 60 * 1000)].index:

        signal = trade.get_signal(df.loc[:i])

        if signal == 'open_long':
            print(f"{ts2dt(df.loc[i, 'Time'])} open long")

        elif signal == 'close_long':
            print(f"{ts2dt(df.loc[i, 'Time'])} close long. Last PnL = {trade.pnl_lst[-1] :.2}")

    print(trade.indicator.value_counts(dropna=False).sort_index(), end='\n\n')
    print('Входы:', trade.enter_idxs)
    print('Выходы', trade.exit_idxs, end='\n\n')
    print('Суммарный PnL', trade.pnl_sum)

if __name__ == '__main__':
    main_test()


