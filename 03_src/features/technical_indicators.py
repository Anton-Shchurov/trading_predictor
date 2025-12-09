"""
Модуль для создания технических индикаторов.

Реализует технические индикаторы на чистом Pandas/NumPy для максимальной совместимости.
Поддержка pandas-ta сохранена опционально, но встроенные реализации являются приоритетными
в контексте задачи (Строгий Pandas/NumPy).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    # print("Warning: pandas-ta not available. Using basic implementations.") 


class TechnicalIndicators:
    """
    Класс-калькулятор для технических индикаторов.
    Предоставляет атомарные методы для вычисления индикаторов.
    """
    def __init__(self):
        self.has_pandas_ta = HAS_PANDAS_TA

    def _rma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Wilder's Moving Average (RMA).
        RMA[t] = (RMA[t-1] * (n-1) + X[t]) / n
        Equivalent to EMA with alpha = 1 / length.
        """
        return series.ewm(alpha=1/length, adjust=False).mean()

    def calculate_ema(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт EMA."""
        return close_series.ewm(span=period, adjust=False, min_periods=period).mean().astype('float32')

    def calculate_macd(self, close_series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """
        Расчёт MACD.
        Возвращает DataFrame с колонками:
        MACD_{fast}_{slow}_{signal}
        MACDh_{fast}_{slow}_{signal} (Histogram)
        MACDs_{fast}_{slow}_{signal} (Signal)
        """
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Naming convention based on pandas-ta for compatibility in pipeline
        col_macd = f'MACD_{fast}_{slow}_{signal}'
        col_hist = f'MACDh_{fast}_{slow}_{signal}'
        col_signal = f'MACDs_{fast}_{slow}_{signal}'
        
        df = pd.DataFrame({
            col_macd: macd_line,
            col_hist: histogram,
            col_signal: signal_line
        }, index=close_series.index)
        
        return df.astype('float32')

    def calculate_rsi(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт RSI (Wilder's)."""
        delta = close_series.diff()
        
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        # Wilder's Smoothing
        avg_gain = self._rma(up, period)
        avg_loss = self._rma(down, period)
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NaN at start
        rsi = rsi.fillna(50) # Neutral fill or Nan? Usually NaNs at start. 
        # But 'replace(0, nan)' above keeps division safe.
        
        return rsi.astype('float32')

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Расчёт ATR (Wilder's)."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR uses RMA (Wilder's) usually
        atr = self._rma(tr, period)
        return atr.astype('float32')

    def calculate_bbands(self, close_series: pd.Series, period: int, std_dev: float) -> pd.DataFrame:
        """Расчёт Bollinger Bands."""
        mid = close_series.rolling(window=period).mean()
        std = close_series.rolling(window=period).std()
        
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        bandwidth = (upper - lower) / mid.replace(0, np.nan) * 100
        percent_b = (close_series - lower) / (upper - lower).replace(0, np.nan)
        
        # Naming convention
        col_lower = f'BBL_{period}_{std_dev}'
        col_mid = f'BBM_{period}_{std_dev}'
        col_upper = f'BBU_{period}_{std_dev}'
        col_bandwidth = f'BBB_{period}_{std_dev}'
        col_percent = f'BBP_{period}_{std_dev}'
        
        df = pd.DataFrame({
            col_lower: lower,
            col_mid: mid,
            col_upper: upper,
            col_bandwidth: bandwidth,
            col_percent: percent_b
        }, index=close_series.index)
        
        return df.astype('float32')
    
    def calculate_donchian(self, high: pd.Series, low: pd.Series, period: int) -> pd.DataFrame:
        """Расчёт Donchian Channels."""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        mid = (upper + lower) / 2
        
        col_lower = f'DCL_{period}_{period}'
        col_mid = f'DCM_{period}_{period}'
        col_upper = f'DCU_{period}_{period}'
        
        df = pd.DataFrame({
            col_lower: lower,
            col_mid: mid,
            col_upper: upper
        }, index=high.index)
        
        return df.astype('float32')

    def calculate_kc(self, high: pd.Series, low: pd.Series, close: pd.Series, atr_series: pd.Series, ema_period: int, atr_period: int) -> pd.DataFrame:
        """Расчёт Keltner Channels."""
        ema = self.calculate_ema(close, ema_period)
        ema, atr_series = ema.align(atr_series, axis=0)
        
        kc_upper = ema + (2 * atr_series)
        kc_lower = ema - (2 * atr_series)
        
        df = pd.DataFrame(index=close.index)
        df[f'KCL_{ema_period}_{atr_period}_2.0'] = kc_lower
        df[f'KCM_{ema_period}_{atr_period}_2.0'] = ema
        df[f'KCU_{ema_period}_{atr_period}_2.0'] = kc_upper
        
        return df.astype('float32')

    def calculate_roc(self, close_series: pd.Series, period: int) -> pd.Series:
        """Расчёт Rate of Change (%)."""
        return (close_series.pct_change(period) * 100).astype('float32')

    def calculate_zscore(self, close_series: pd.Series, window: int) -> pd.Series:
        """Расчёт Z-Score."""
        mean = close_series.rolling(window=window).mean()
        std = close_series.rolling(window=window).std()
        return ((close_series - mean) / std.replace(0, np.nan)).astype('float32')

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.DataFrame:
        """
        Расчёт ADX, DMP, DMN.
        """
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)
        
        tr = self.calculate_atr(high, low, close, 1).fillna(0) # TR1 for ADX calc logic usually similar to ATR
        # Better: use proper TR series calculation
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed
        atr_smooth = self._rma(tr_series, period)
        plus_di = 100 * self._rma(plus_dm, period) / atr_smooth.replace(0, np.nan)
        minus_di = 100 * self._rma(minus_dm, period) / atr_smooth.replace(0, np.nan)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = self._rma(dx, period)
        
        col_adx = f'ADX_{period}'
        col_dmp = f'DMP_{period}'
        col_dmn = f'DMN_{period}'
        
        df = pd.DataFrame({
            col_adx: adx,
            col_dmp: plus_di,
            col_dmn: minus_di
        }, index=close.index)
        
        return df.astype('float32')
        
    def calculate_vwap(self, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """
        Расчёт Rolling VWAP.
        """
        pv = close * volume
        cum_pv = pv.rolling(window=period).sum()
        cum_vol = volume.rolling(window=period).sum()
        return (cum_pv / cum_vol.replace(0, np.nan)).astype('float32')

    def calculate_psar(self, high: pd.Series, low: pd.Series, close: pd.Series, af_initial: float, af_max: float) -> pd.Series:
        """
        Расчёт Parabolic SAR (NumPy implementation).
        """
        df_temp = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
        psar_df = self._calculate_psar_manual(df_temp, af_initial, af_max)
        return psar_df['PSAR'].astype('float32')

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет OBV."""
        df = df.copy()
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            return df
        
        change = df['Close'].diff()
        direction = np.sign(change).fillna(0)
        volume_flow = direction * df['Volume']
        df['OBV'] = volume_flow.cumsum().astype('float32')
        return df

    def _calculate_psar_manual(self, df: pd.DataFrame, af_initial: float, af_max: float) -> pd.DataFrame:
        """
        NumPy реализация Parabolic SAR.
        """
        high = df['High'].values
        low = df['Low'].values
        n = len(df)
        psar = np.full(n, np.nan, dtype=np.float64)
        
        if n < 2:
            df['PSAR'] = psar
            return df
        
        trend = 1 if (high[1] > high[0]) else -1
        sar = low[0] if trend == 1 else high[0]
        ep = high[0] if trend == 1 else low[0]
        af = af_initial
        
        psar[0] = sar
        
        for i in range(1, n):
            prev_sar = psar[i-1]
            new_sar = prev_sar + af * (ep - prev_sar)
            
            if trend == 1:
                limit = low[i-1]
                if i > 1: limit = min(limit, low[i-2])
                new_sar = min(new_sar, limit)
                
                if low[i] < new_sar:
                    trend = -1
                    new_sar = ep
                    ep = low[i]
                    af = af_initial
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_initial, af_max)
            else:
                limit = high[i-1]
                if i > 1: limit = max(limit, high[i-2])
                new_sar = max(new_sar, limit)
                
                if high[i] > new_sar:
                    trend = 1
                    new_sar = ep
                    ep = high[i]
                    af = af_initial
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_initial, af_max)
            
            psar[i] = new_sar
            
        df['PSAR'] = psar
        return df

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Добавляет Stochastic Oscillator."""
        df = df.copy()
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        df['Stoch_K'] = (100 * ((df['Close'] - lowest_low) / denom)).astype('float32')
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean().astype('float32')
        return df

    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет CCI."""
        df = df.copy()
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        # Mean Absolute Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['CCI'] = ((tp - sma_tp) / (0.015 * mad.replace(0, np.nan))).astype('float32')
        return df

    def add_keltner_channel(self, df: pd.DataFrame, ema_period: int = 20, 
                           atr_period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        """Добавляет Keltner Channel."""
        df = df.copy()
        ema = df['Close'].ewm(span=ema_period).mean()
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
        
        df['KC_Middle'] = ema.astype('float32')
        df['KC_Upper'] = (ema + (atr * multiplier)).astype('float32')
        df['KC_Lower'] = (ema - (atr * multiplier)).astype('float32')
        return df

    def add_donchian_channel(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет Donchian Channel."""
        df = df.copy()
        df['DC_Upper'] = df['High'].rolling(window=period).max().astype('float32')
        df['DC_Lower'] = df['Low'].rolling(window=period).min().astype('float32')
        df['DC_Middle'] = ((df['DC_Upper'] + df['DC_Lower']) / 2).astype('float32')
        return df
    
    def add_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Добавляет Momentum."""
        df = df.copy()
        df[f'Momentum_{period}'] = (df['Close'] - df['Close'].shift(period)).astype('float32')
        return df

    def add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Добавляет CMF."""
        df = df.copy()
        mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan) * df['Volume']
        mfv = mfv.fillna(0)
        df['CMF'] = (mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum().replace(0, np.nan)).astype('float32')
        return df

    def add_ad_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет A/D Line."""
        df = df.copy()
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv = clv.fillna(0)
        mfv = clv * df['Volume']
        df['AD'] = mfv.cumsum().astype('float32')
        return df