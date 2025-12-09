"""
Модуль для создания статистических фич.

Создает различные статистические признаки на основе ценовых данных:
ROC, rolling mean/std, Z-Score, Skewness/Kurtosis и другие.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class StatisticalFeatures:
    """
    Класс-калькулятор для статистических признаков.
    """
    def calculate_log_return(self, close_series: pd.Series, period: int) -> pd.Series:
        """
        Расчёт логарифмической доходности.
        LogRet = ln(Close[t] / Close[t-period])
        """
        return np.log(close_series / close_series.shift(period)).astype('float32')

    # Календарные признаки
    def calculate_hour_sin(self, index: pd.DatetimeIndex) -> pd.Series:
        """Синус от часа дня."""
        hour = index.hour
        return pd.Series(np.sin(2 * np.pi * hour / 24.0), index=index).astype('float32')

    def calculate_hour_cos(self, index: pd.DatetimeIndex) -> pd.Series:
        """Косинус от часа дня."""
        hour = index.hour
        return pd.Series(np.cos(2 * np.pi * hour / 24.0), index=index).astype('float32')
    
    def calculate_day_of_week(self, index: pd.DatetimeIndex) -> pd.Series:
        """День недели (0=Mon, 6=Sun)."""
        return pd.Series(index.dayofweek.astype('Int8'), index=index)

    # -----------------------------
    # Режимы/статистика распределений/прочее
    # -----------------------------
    def calculate_quantile_flag(self, series: pd.Series, q: float = 0.8, rolling_window: Optional[int] = None,
                                use_past_only: bool = True) -> pd.Series:
        """
        Бинарный флаг по квантилю q.
        
        Args:
            series: Входная серия.
            q: Квантиль (0.0-1.0).
            rolling_window: Если задан, квантиль считается в скользящем окне.
            use_past_only: Если True, используется shift(1) для исключения утечки (сравниваем с прошлым распределением).
            
        Returns:
            pd.Series (Int8): 1 если значение выше порога, иначе 0.
        """
        mask = series.notna()
        result_array = np.zeros(len(series), dtype=np.int8)
        
        if rolling_window and rolling_window > 1:
            # Для предотвращения look-ahead bias при rolling вычислении порога
            # мы хотим сравнить X[t] с Quantile(X[t-window : t-1])
            # series.rolling().quantile() включает текущий элемент по умолчанию, поэтому делаем shift базы расчёта.
            
            base = series.shift(1) if use_past_only else series
            thr_series = base.rolling(window=rolling_window, min_periods=rolling_window).quantile(q)
            
            # Сравниваем
            valid = mask & thr_series.notna()
            result_array[valid] = (series[valid] > thr_series[valid]).astype(np.int8)
            out = pd.Series(result_array, index=series.index, dtype='Int8')
            out[~valid] = pd.NA # Возвращаем NA там где нельзя было посчитать
        else:
            # Глобальный квантиль (ОСТОРОЖНО: look-ahead bias если считается по всему тесту)
            # Допустимо только если q фиксирован константой или рассчитан на трейне.
            # Здесь мы считаем по всей серии, что является утечкой данных, если серия включает тест.
            # Но функция делает то, что просят.
            threshold = series.quantile(q)
            result_array[mask] = (series[mask] > threshold).astype(np.int8)
            out = pd.Series(result_array, index=series.index, dtype='Int8')
            out[~mask] = pd.NA

        return out

    def calculate_threshold_flag(self, series: pd.Series, threshold: float = 0.0, strictly_greater: bool = True) -> pd.Series:
        """Бинарный флаг по порогу."""
        out = pd.Series(pd.NA, index=series.index, dtype='Int8')
        mask = series.notna()
        
        if strictly_greater:
            out.loc[mask] = (series.loc[mask] > threshold).astype('Int8')
        else:
            out.loc[mask] = (series.loc[mask] >= threshold).astype('Int8')
        return out

    def calculate_rolling_skew(self, series: pd.Series, window: int = 20, bias: bool = False) -> pd.Series:
        """Скользящая асимметрия (skewness)."""
        # Pandas rolling skew uses unbias=True by default? Pandas doesn't have rolling skew built-in universally 
        # but modern pandas might. Checking documentation: pandas.core.window.rolling.Rolling.skew available.
        # It's faster than apply(scipy.stats.skew).
        return series.rolling(window=window).skew().astype('float32')

    def calculate_rolling_kurtosis(self, series: pd.Series, window: int = 20, fisher: bool = True, bias: bool = False) -> pd.Series:
        """Скользящий куртозис (kurtosis)."""
        # Pandas rolling kurt uses Fisher=True (excess kurtosis) by default.
        return series.rolling(window=window).kurt().astype('float32')

    def calculate_autocorr(self, series: pd.Series, lag: int = 1, window: int = 50) -> pd.Series:
        """
        Скользящая автокорреляция.
        Векторизированная реализация через rolling correlation двух сдвинутых серий.
        """
        if lag <= 0:
            return pd.Series(np.nan, index=series.index)
            
        # Corr(X_t, X_{t-lag}) на окне window
        # Мы хотим корреляцию между серией и её лагом в окне.
        # series.rolling(window).corr(other) вычисляет попарную корреляцию в окне.
        
        lagged = series.shift(lag)
        return series.rolling(window=window).corr(lagged).astype('float32')

    def calculate_bars_since_donchian_break(self, close: pd.Series, donchian_df: pd.DataFrame, which: str = 'upper') -> pd.Series:
        """
        Возвращает количество баров с момента последнего пробоя.
        Векторизированная реализация.
        """
        if donchian_df is None or len(donchian_df.columns) == 0:
            return pd.Series(np.nan, index=close.index)

        which_lower = (which or 'upper').lower()
        if which_lower == 'upper':
            col_candidates = [c for c in donchian_df.columns if 'DCU' in c]
            bound_series = donchian_df[col_candidates[0]] if col_candidates else donchian_df.iloc[:, 2]
            # Пробой UP: Close > Upper
            # Важно: Сравниваем Close[t] с Upper[t-1] чтобы обнаружить момент пробоя уровня, который был сформирован ранее?
            # Или пробой текущего уровня?
            # Обычно: Close[t] > DonchianHigh[t-1] (ибо DonchianHigh[t] сам равен max(High[t-period:t]), 
            # и если High[t] выше всех, то DonchianHigh[t] вырастет).
            # В `technical_indicators.py` Donchian считается включая текущий бар.
            # Поэтому логично сравнивать с shift(1).
            comparator = lambda c, b: c > b
        else:
            col_candidates = [c for c in donchian_df.columns if 'DCL' in c]
            bound_series = donchian_df[col_candidates[0]] if col_candidates else donchian_df.iloc[:, 0]
            comparator = lambda c, b: c < b

        # Уровень предыдущего бара
        bound_shifted = bound_series.shift(1)
        
        # Булева маска событий
        events = comparator(close, bound_shifted)
        events = events.fillna(False)
        
        # Векторизированный подсчет "времени с последнего True"
        # 1. Индексы, где произошло событие
        # 2. Forward fill этих индексов
        # 3. Разница текущего индекса и индекса события
        
        # Создаем серию индексов
        # (Предполагаем RangeIndex или конвертируем в reset_index для подсчета расстояния)
        # Если индекс Datetime, нам нужно количество БАРОВ, а не время.
        
        n = len(close)
        arange = np.arange(n)
        
        # Массив индексов событий. Там где event=False, ставим Nan (или -1 для логики)
        # Потом делаем ffill.
        
        last_event_idx = pd.Series(np.where(events, arange, np.nan), index=close.index)
        last_event_idx = last_event_idx.ffill()
        
        # bars_since = current_idx - last_event_idx
        bars_since = arange - last_event_idx
        
        return bars_since.astype('float32')

    def calculate_session_flag(self, index: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series:
        """Флаг сессии по часу дня."""
        hour = index.hour
        if end_hour > start_hour:
            mask = (hour >= start_hour) & (hour < end_hour)
        else:
            mask = (hour >= start_hour) | (hour < end_hour)
        return pd.Series(mask, index=index, dtype='Int8')

    def calculate_session_flag_tz(self, index: pd.DatetimeIndex, tz: str, start_hour_local: int, end_hour_local: int) -> pd.Series:
        """Флаг сессии в локальном часовом поясе."""
        if index.tz is None:
            index_utc = index.tz_localize('UTC')
        else:
            index_utc = index.tz_convert('UTC')
            
        index_local = index_utc.tz_convert(tz)
        hour = index_local.hour
        
        if end_hour_local > start_hour_local:
            mask = (hour >= start_hour_local) & (hour < end_hour_local)
        else:
            mask = (hour >= start_hour_local) | (hour < end_hour_local)
            
        return pd.Series(mask, index=index, dtype='Int8')