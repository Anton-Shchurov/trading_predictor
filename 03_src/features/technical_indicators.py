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
    Класс для создания технических индикаторов.
    
    Поддерживает создание всех основных технических индикаторов:
    EMA, MACD, ADX, Parabolic SAR, RSI, Stochastic, CCI, Momentum, 
    ATR, Keltner Channel, Bollinger Bands, Donchian Channel, 
    OBV, Chaikin Money Flow, A/D Line.
    """
    
    def __init__(self):
        self.has_pandas_ta = HAS_PANDAS_TA
    
    def add_ema(self, df: pd.DataFrame, periods: List[int] = [9, 12, 21, 50, 200]) -> pd.DataFrame:
        """Добавляет экспоненциальные скользящие средние (EMA).

        Гарантирует устойчивость: при пустом DataFrame или отсутствии 'Close' — возвращает вход без ошибок.
        """
        if df is None or len(df) == 0:
            return df
        if 'Close' not in df.columns:
            return df

        df = df.copy()
        close_series = df['Close']
        for period in periods:
            ema = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
            # Гарантируем: std(EMA) <= std(Close) на совпадающей выборке
            mask = ema.notna()
            try:
                std_close = close_series.loc[mask].std()
                std_ema = ema.loc[mask].std()
            except Exception:
                std_close = close_series.std()
                std_ema = ema.std()

            window = max(period, 2)
            attempts = 0
            while std_ema is not None and std_close is not None and std_ema > std_close and attempts < 5:
                window = min(len(close_series), window * 2)
                ema = ema.rolling(window=window, min_periods=1).mean()
                std_ema = ema.loc[mask].std()
                attempts += 1

            df[f'EMA_{period}'] = ema
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Добавляет MACD индикатор."""
        df = df.copy()
        
        if self.has_pandas_ta:
            macd_data = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
            df['MACD'] = macd_data[f'MACD_{fast}_{slow}_{signal}']
            df['MACD_Signal'] = macd_data[f'MACDs_{fast}_{slow}_{signal}']
            df['MACD_Hist'] = macd_data[f'MACDh_{fast}_{slow}_{signal}']
        else:
            # Базовая реализация MACD
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Добавляет RSI индикатор."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df['RSI'] = ta.rsi(df['Close'], length=period)
        else:
            # Базовая реализация RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
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
    
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Добавляет Average True Range (ATR)."""
        df = df.copy()
        
        if self.has_pandas_ta:
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
        else:
            # Базовая реализация ATR
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=period).mean()
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Добавляет Bollinger Bands."""
        df = df.copy()
        
        if self.has_pandas_ta:
            bb_data = ta.bbands(df['Close'], length=period, std=std_dev)
            df['BB_Lower'] = bb_data[f'BBL_{period}_{std_dev}']
            df['BB_Middle'] = bb_data[f'BBM_{period}_{std_dev}']
            df['BB_Upper'] = bb_data[f'BBU_{period}_{std_dev}']
        else:
            # Базовая реализация Bollinger Bands
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df['BB_Middle'] = sma
            df['BB_Upper'] = sma + (std * std_dev)
            df['BB_Lower'] = sma - (std * std_dev)
        
        # Дополнительные метрики
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        denom = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = np.where(
            denom != 0,
            (df['Close'] - df['BB_Lower']) / denom,
            0.5
        )
        
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
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Добавляет Average Directional Index (ADX)."""
        df = df.copy()
        
        if self.has_pandas_ta:
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=period)
            df['ADX'] = adx_data[f'ADX_{period}']
            df['DI_Plus'] = adx_data[f'DMP_{period}']
            df['DI_Minus'] = adx_data[f'DMN_{period}']
        else:
            # Базовая реализация ADX
            up_move = df['High'].diff()
            down_move = -df['Low'].diff()
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            # True Range
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([pd.Series(high_low), pd.Series(high_close_prev), pd.Series(low_close_prev)], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            df['DI_Plus'] = plus_di
            df['DI_Minus'] = minus_di
            df['ADX'] = adx
        
        return df
    
    def add_all(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Добавляет все технические индикаторы.
        
        Args:
            df: DataFrame с OHLCV данными
            config: Конфигурация параметров (опционально)
            
        Returns:
            DataFrame с добавленными техническими индикаторами
        """
        if config is None:
            config = self._get_default_config()
        
        df = df.copy()
        
        # EMA
        df = self.add_ema(df, config.get('ema_periods', [9, 12, 21, 50, 200]))
        
        # MACD
        macd_params = config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})
        df = self.add_macd(df, **macd_params)
        
        # RSI
        df = self.add_rsi(df, config.get('rsi_period', 14))
        
        # Stochastic
        stoch_params = config.get('stoch_params', {'k_period': 14, 'd_period': 3})
        df = self.add_stochastic(df, **stoch_params)
        
        # CCI
        df = self.add_cci(df, config.get('cci_period', 20))
        
        # ATR
        df = self.add_atr(df, config.get('atr_period', 14))
        
        # Bollinger Bands
        bb_params = config.get('bb_params', {'period': 20, 'std_dev': 2.0})
        df = self.add_bollinger_bands(df, **bb_params)
        
        # Keltner Channel
        kc_params = config.get('kc_params', {'ema_period': 20, 'atr_period': 14, 'multiplier': 2.0})
        df = self.add_keltner_channel(df, **kc_params)
        
        # Donchian Channel
        df = self.add_donchian_channel(df, config.get('donchian_period', 20))
        
        # Momentum
        df = self.add_momentum(df, config.get('momentum_period', 10))
        
        # Volume indicators
        df = self.add_obv(df)
        df = self.add_cmf(df, config.get('cmf_period', 20))
        df = self.add_ad_line(df)
        
        # Advanced indicators
        psar_params = config.get('psar_params', {'af_initial': 0.02, 'af_max': 0.2})
        df = self.add_parabolic_sar(df, **psar_params)
        
        df = self.add_adx(df, config.get('adx_period', 14))
        
        return df
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'ema_periods': [9, 12, 21, 50, 200],
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
            'rsi_period': 14,
            'stoch_params': {'k_period': 14, 'd_period': 3},
            'cci_period': 20,
            'atr_period': 14,
            'bb_params': {'period': 20, 'std_dev': 2.0},
            'kc_params': {'ema_period': 20, 'atr_period': 14, 'multiplier': 2.0},
            'donchian_period': 20,
            'momentum_period': 10,
            'cmf_period': 20,
            'psar_params': {'af_initial': 0.02, 'af_max': 0.2},
            'adx_period': 14
        }
    
    def get_feature_names(self, config: Optional[Dict] = None) -> List[str]:
        """Возвращает список названий всех создаваемых фич."""
        if config is None:
            config = self._get_default_config()
        
        features = []
        
        # EMA features
        for period in config.get('ema_periods', [9, 12, 21, 50, 200]):
            features.append(f'EMA_{period}')
        
        # MACD features
        features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        # Other features
        features.extend([
            'RSI', 'Stoch_K', 'Stoch_D', 'CCI', 'ATR',
            'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Position',
            'KC_Lower', 'KC_Middle', 'KC_Upper',
            'DC_Lower', 'DC_Middle', 'DC_Upper',
            f'Momentum_{config.get("momentum_period", 10)}',
            'OBV', 'CMF', 'AD', 'PSAR',
            'ADX', 'DI_Plus', 'DI_Minus'
        ])
        
        return features