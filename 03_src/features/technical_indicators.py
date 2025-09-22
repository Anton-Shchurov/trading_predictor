"""
Модуль для создания технических индикаторов.

Использует pandas-ta для расчета технических индикаторов.
Если pandas-ta недоступен, предоставляет базовые реализации.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas-ta not available. Using basic implementations.")


class TechnicalIndicators:
    """
    Класс-калькулятор для технических индикаторов.
    Предоставляет атомарные методы для вычисления индикаторов,
    которые принимают на вход pd.Series и возвращают pd.Series или pd.DataFrame.
    """
    def __init__(self):
        self.has_pandas_ta = HAS_PANDAS_TA

    def calculate_ema(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт EMA."""
        if not self.has_pandas_ta:
            return close_series.ewm(span=period, adjust=False, min_periods=period).mean()
        return ta.ema(close_series, length=period)

    def calculate_macd(self, close_series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """Расчёт MACD, возвращает DataFrame с MACD, гистограммой и сигнальной линией."""
        return ta.macd(close_series, fast=fast, slow=slow, signal=signal, append=False)

    def calculate_rsi(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт RSI."""
        return ta.rsi(close_series, length=period, append=False)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Расчёт ATR."""
        return ta.atr(high, low, close, length=period, append=False)

    def calculate_bbands(self, close_series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
        """Расчёт Bollinger Bands. Возвращает DataFrame с линиями, шириной и позицией."""
        return ta.bbands(close_series, length=period, std=std_dev, append=False)
    
    def calculate_donchian(self, high: pd.Series, low: pd.Series, period: int) -> pd.DataFrame:
        """Расчёт Donchian Channels."""
        return ta.donchian(high, low, lower_length=period, upper_length=period, append=False)

    def calculate_kc(self, high: pd.Series, low: pd.Series, close: pd.Series, atr_series: pd.Series, ema_period: int, atr_period: int) -> pd.DataFrame:
        """Расчёт Keltner Channels. Принимает pre-calculated ATR."""
        if not self.has_pandas_ta:
            ema = self.calculate_ema(close, ema_period)
            kc_upper = ema + (2 * atr_series)
            kc_lower = ema - (2 * atr_series)
            return pd.DataFrame({'KC_Lower': kc_lower, 'KC_Middle': ema, 'KC_Upper': kc_upper})
        
        # pandas-ta не позволяет передать внешний ATR, поэтому считаем его внутри, если нужно.
        # Но для нашего графа зависимостей лучше ручная реализация.
        ema = self.calculate_ema(close, ema_period)
        kc_upper = ema + (2 * atr_series)
        kc_lower = ema - (2 * atr_series)
        df = pd.DataFrame(index=close.index)
        df[f'KCU_{ema_period}_{atr_period}_2.0'] = kc_upper
        df[f'KCL_{ema_period}_{atr_period}_2.0'] = kc_lower
        df[f'KCM_{ema_period}_{atr_period}_2.0'] = ema
        return df

    def calculate_roc(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт Rate of Change."""
        return ta.roc(close_series, length=period, append=False)

    def calculate_zscore(self, close_series: pd.Series, window: int) -> pd.Series:
        """Расчёт Z-Score."""
        mean = close_series.rolling(window=window).mean()
        std = close_series.rolling(window=window).std().replace(0, np.nan)
        return (close_series - mean) / std

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.DataFrame:
        """Расчёт ADX."""
        return ta.adx(high, low, close, length=period, append=False)
        
    def calculate_vwap(self, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Расчёт VWAP."""
        return ta.vwap(high=close, low=close, close=close, volume=volume, length=period, append=False)

    def calculate_psar(self, high: pd.Series, low: pd.Series, close: pd.Series, af_initial: float, af_max: float) -> pd.Series:
        """Расчёт Parabolic SAR."""
        psar_df = ta.psar(high, low, close, af=af_initial, max_af=af_max, append=False)
        # Возвращаем только основную линию PSAR, а не все его компоненты
        return psar_df.iloc[:, 0]

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Добавляет Stochastic Oscillator."""
        df = df.copy()
        
        if self.has_pandas_ta:
            stoch_data = ta.stoch(df['High'], df['Low'], df['Close'], k=k_period, d=d_period)
            df['Stoch_K'] = stoch_data[f'STOCHk_{k_period}_{d_period}_3']
            df['Stoch_D'] = stoch_data[f'STOCHd_{k_period}_{d_period}_3']
        else:
            # Базовая реализация Stochastic
            lowest_low = df['Low'].rolling(window=k_period).min()
            highest_high = df['High'].rolling(window=k_period).max()
            df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
            df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df
    
    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет Commodity Channel Index (CCI)."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=period)
        else:
            # Базовая реализация CCI
            tp = (df['High'] + df['Low'] + df['Close']) / 3  # Typical Price
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        return df
    
    def add_keltner_channel(self, df: pd.DataFrame, ema_period: int = 20, 
                           atr_period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        """Добавляет Keltner Channel."""
        df = df.copy()
        
        if self.has_pandas_ta:
            kc_data = ta.kc(df['High'], df['Low'], df['Close'], 
                           length=ema_period, scalar=multiplier)
            # Привязываемся к порядку столбцов для устойчивости к форматированию float в именах
            df['KC_Lower'] = kc_data.iloc[:, 0]
            df['KC_Middle'] = kc_data.iloc[:, 1]
            df['KC_Upper'] = kc_data.iloc[:, 2]
        else:
            # Базовая реализация Keltner Channel
            ema = df['Close'].ewm(span=ema_period).mean()
            # Рассчитываем ATR для Keltner Channel
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
            
            df['KC_Middle'] = ema
            df['KC_Upper'] = ema + (atr * multiplier)
            df['KC_Lower'] = ema - (atr * multiplier)
        
        return df
    
    def add_donchian_channel(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет Donchian Channel."""
        df = df.copy()
        
        if self.has_pandas_ta:
            dc_data = ta.donchian(df['High'], df['Low'], lower_length=period, upper_length=period)
            df['DC_Lower'] = dc_data[f'DCL_{period}_{period}']
            df['DC_Middle'] = dc_data[f'DCM_{period}_{period}']
            df['DC_Upper'] = dc_data[f'DCU_{period}_{period}']
        else:
            # Базовая реализация Donchian Channel
            df['DC_Upper'] = df['High'].rolling(window=period).max()
            df['DC_Lower'] = df['Low'].rolling(window=period).min()
            df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
        
        return df
    
    def add_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Добавляет Momentum индикатор."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df[f'Momentum_{period}'] = ta.mom(df['Close'], length=period)
        else:
            # Базовая реализация Momentum
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет On-Balance Volume (OBV)."""
        df = df.copy()
        
        # Защита от отсутствующих колонок/пустых данных
        if df is None or len(df) == 0:
            return df
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            return df

        # Базовая реализация OBV (используем единый путь для предсказуемости)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        return df
    
    def add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет Chaikin Money Flow (CMF)."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=period)
        else:
            # Базовая реализация CMF
            mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
            mfv = mfv.fillna(0)  # Заполняем NaN если High == Low
            df['CMF'] = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        
        return df
    
    def add_ad_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет Accumulation/Distribution Line (A/D)."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        else:
            # Базовая реализация A/D Line
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0)  # Заполняем NaN если High == Low
            mfv = clv * df['Volume']
            df['AD'] = mfv.cumsum()
        
        return df
    
    def add_parabolic_sar(self, df: pd.DataFrame, af_initial: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        Добавляет Parabolic SAR с правильной реализацией алгоритма.
        
        Parabolic SAR (Stop and Reverse) - это технический индикатор, который следует за трендом
        и помогает определить точки разворота. Он отображается как точки выше или ниже цены.
        
        Args:
            df: DataFrame с OHLC данными
            af_initial: Начальный коэффициент ускорения (обычно 0.02)
            af_max: Максимальный коэффициент ускорения (обычно 0.2)
            
        Returns:
            DataFrame с добавленной колонкой PSAR
        """
        df = df.copy()
        
        if len(df) < 2:
            # Недостаточно данных для расчета PSAR
            df['PSAR'] = np.nan
            return df
        
        # Принудительно используем только нашу ручную реализацию для PSAR
        # pandas-ta PSAR дает слишком много NaN значений (~50%)
        # Наша реализация работает гораздо лучше (~0% NaN)
        df = self._calculate_psar_manual(df, af_initial, af_max)
        
        return df
    
    def _calculate_psar_manual(self, df: pd.DataFrame, af_initial: float, af_max: float) -> pd.DataFrame:
        """
        Ручная реализация алгоритма Parabolic SAR.
        
        Алгоритм:
        1. Начинаем с определения направления тренда
        2. PSAR начинается как предыдущий экстремум (High в медвежьем, Low в бычьем тренде)
        3. На каждом шаге: PSAR = Предыдущий_PSAR + AF * (EP - Предыдущий_PSAR)
        4. EP (Extreme Point) - это максимум в бычьем тренде или минимум в медвежьем
        5. AF увеличивается при новых экстремумах, но не превышает af_max
        6. При пересечении цены и PSAR происходит разворот тренда
        """
        length = len(df)
        high = df['High'].values
        low = df['Low'].values
        
        # Инициализируем массив PSAR значениями NaN
        psar = np.full(length, np.nan, dtype=np.float64)
        
        if length < 2:
            df['PSAR'] = psar
            return df
        
        # Определяем начальное направление тренда
        # Если первая свеча растущая, начинаем с бычьего тренда
        is_bullish = high[1] > high[0] and low[1] > low[0]
        
        # Инициализация для первого периода
        if is_bullish:
            # Бычий тренд: PSAR начинается с Low предыдущего периода
            psar[0] = low[0]
            extreme_point = high[0]  # EP = максимум
        else:
            # Медвежий тренд: PSAR начинается с High предыдущего периода  
            psar[0] = high[0]
            extreme_point = low[0]   # EP = минимум
        
        af = af_initial
        
        # Основной цикл расчета PSAR
        for i in range(1, length):
            # Сохраняем предыдущие значения
            prev_psar = psar[i-1]
            
            # Рассчитываем новый PSAR
            psar[i] = prev_psar + af * (extreme_point - prev_psar)
            
            # Проверяем разворот тренда
            trend_reversed = False
            
            if is_bullish:
                # В бычьем тренде проверяем пересечение снизу
                if low[i] <= psar[i]:
                    # Разворот: переходим к медвежьему тренду
                    is_bullish = False
                    trend_reversed = True
                    psar[i] = extreme_point  # PSAR становится предыдущим максимумом
                    extreme_point = low[i]   # Новый EP = текущий минимум
                    af = af_initial          # Сбрасываем AF
                else:
                    # Продолжаем бычий тренд
                    if high[i] > extreme_point:
                        extreme_point = high[i]  # Новый максимум
                        af = min(af + af_initial, af_max)  # Увеличиваем AF
                    
                    # SAR не может быть выше минимумов последних двух периодов в бычьем тренде
                    if i >= 1 and psar[i] > low[i-1]:
                        psar[i] = low[i-1]
                    if i >= 2 and psar[i] > low[i-2]:
                        psar[i] = low[i-2]
            else:
                # В медвежьем тренде проверяем пересечение сверху
                if high[i] >= psar[i]:
                    # Разворот: переходим к бычьему тренду
                    is_bullish = True
                    trend_reversed = True
                    psar[i] = extreme_point  # PSAR становится предыдущим минимумом
                    extreme_point = high[i]  # Новый EP = текущий максимум
                    af = af_initial          # Сбрасываем AF
                else:
                    # Продолжаем медвежий тренд
                    if low[i] < extreme_point:
                        extreme_point = low[i]   # Новый минимум
                        af = min(af + af_initial, af_max)  # Увеличиваем AF
                    
                    # SAR не может быть ниже максимумов последних двух периодов в медвежьем тренде
                    if i >= 1 and psar[i] < high[i-1]:
                        psar[i] = high[i-1]
                    if i >= 2 and psar[i] < high[i-2]:
                        psar[i] = high[i-2]
        
        # Добавляем результат в DataFrame
        df['PSAR'] = psar
        
        # Проверяем качество результата
        nan_count = np.isnan(psar).sum()
        total_count = len(psar)
        if nan_count > 0:
            print(f"Warning: PSAR calculation resulted in {nan_count}/{total_count} "
                  f"({nan_count/total_count*100:.1f}%) NaN values")
        
        return df