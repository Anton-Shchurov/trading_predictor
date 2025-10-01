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
        """Расчёт логарифмической доходности."""
        return np.log(close_series / close_series.shift(period))

    # Календарные признаки
    def calculate_hour_sin(self, index: pd.DatetimeIndex) -> pd.Series:
        """Синус от часа дня."""
        hour = index.hour
        return pd.Series(np.sin(2 * np.pi * hour / 24.0), index=index)

    def calculate_hour_cos(self, index: pd.DatetimeIndex) -> pd.Series:
        """Косинус от часа дня."""
        hour = index.hour
        return pd.Series(np.cos(2 * np.pi * hour / 24.0), index=index)
    
    def calculate_day_of_week(self, index: pd.DatetimeIndex) -> pd.Series:
        """День недели."""
        return pd.Series(index.dayofweek.astype('Int8'), index=index)

    # -----------------------------
    # Режимы/статистика распределений/прочее
    # -----------------------------
    def calculate_quantile_flag(self, series: pd.Series, q: float = 0.8, rolling_window: Optional[int] = None,
                                use_past_only: bool = True) -> pd.Series:
        """
        Бинарный флаг по квантилю q.
        По умолчанию — глобальный квантиль по всей серии. Для избежания утечек можно указать rolling_window:
        тогда порог считается как (series.shift(1) если use_past_only) на окне rolling_window.
        """
        out = pd.Series(pd.NA, index=series.index, dtype='Int8')
        mask = series.notna()
        if rolling_window and rolling_window > 1:
            base = series.shift(1) if use_past_only else series
            thr_series = base.rolling(window=rolling_window, min_periods=rolling_window).quantile(q)
            cmp_mask = mask & thr_series.notna()
            out.loc[cmp_mask] = (series.loc[cmp_mask] > thr_series.loc[cmp_mask]).astype('Int8')
        else:
            threshold = series.quantile(q)
            out.loc[mask] = (series.loc[mask] > threshold).astype('Int8')
        return out

    def calculate_threshold_flag(self, series: pd.Series, threshold: float = 0.0, strictly_greater: bool = True) -> pd.Series:
        """Бинарный флаг по порогу: 1, если series > threshold (или >= если strictly_greater=False)."""
        out = pd.Series(pd.NA, index=series.index, dtype='Int8')
        mask = series.notna()
        if strictly_greater:
            out.loc[mask] = (series.loc[mask] > threshold).astype('Int8')
        else:
            out.loc[mask] = (series.loc[mask] >= threshold).astype('Int8')
        return out

    def calculate_rolling_skew(self, series: pd.Series, window: int = 20, bias: bool = False) -> pd.Series:
        """Скользящая асимметрия (skewness) ряда на окне window."""
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: stats.skew(x, bias=bias), raw=True
        )

    def calculate_rolling_kurtosis(self, series: pd.Series, window: int = 20, fisher: bool = True, bias: bool = False) -> pd.Series:
        """Скользящая куртозис (kurtosis) ряда на окне window (по умолчанию избыточная куртозис, fisher=True)."""
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: stats.kurtosis(x, fisher=fisher, bias=bias), raw=True
        )

    def calculate_autocorr(self, series: pd.Series, lag: int = 1, window: int = 50) -> pd.Series:
        """Скользящая автокорреляция ряда с лагом lag на окне window."""
        values = series.astype(float).values
        n = len(values)
        result = np.full(n, np.nan, dtype=float)
        if lag <= 0:
            return pd.Series(result, index=series.index)
        effective_window = max(window, lag + 2)
        for i in range(effective_window - 1, n):
            seg = values[i - effective_window + 1:i + 1]
            a = seg[lag:]
            b = seg[:-lag]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() >= 2:
                result[i] = np.corrcoef(a[valid], b[valid])[0, 1]
        return pd.Series(result, index=series.index)

    def calculate_bars_since_donchian_break(self, close: pd.Series, donchian_df: pd.DataFrame, which: str = 'upper') -> pd.Series:
        """
        Возвращает количество баров с момента последнего пробоя границы Дончиана.
        which: 'upper' — пробой верхней границы (close > DCU), 'lower' — пробой нижней границы (close < DCL).
        """
        if donchian_df is None or len(donchian_df.columns) == 0:
            return pd.Series(np.nan, index=close.index)

        which_lower = (which or 'upper').lower()
        if which_lower == 'upper':
            col_candidates = [c for c in donchian_df.columns if 'DCU' in c]
            comparator = np.greater
        else:
            col_candidates = [c for c in donchian_df.columns if 'DCL' in c]
            comparator = np.less

        if len(col_candidates) == 0:
            # Fallback: предполагаем порядок столбцов [Lower, Middle, Upper]
            bound_series = donchian_df.iloc[:, 2] if which_lower == 'upper' else donchian_df.iloc[:, 0]
        else:
            bound_series = donchian_df[col_candidates[0]]

        # Избежать утечки будущего: сравниваем close[i] с границей предыдущего бара
        bound_series = bound_series.shift(1)

        close_vals = close.astype(float).values
        bound_vals = bound_series.astype(float).values
        n = len(close_vals)
        out = np.full(n, np.nan, dtype=float)
        last_idx = None
        for i in range(n):
            cv = close_vals[i]
            bv = bound_vals[i]
            is_event = (not np.isnan(cv)) and (not np.isnan(bv)) and comparator(cv, bv)
            if is_event:
                out[i] = 0.0
                last_idx = i
            else:
                out[i] = np.nan if last_idx is None else float(i - last_idx)

        return pd.Series(out, index=close.index)

    def calculate_session_flag(self, index: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series:
        """Флаг сессии по часу дня [start_hour, end_hour). Предполагается часовой пояс индекса (обычно UTC/GMT)."""
        hour = index.hour
        if end_hour > start_hour:
            mask = (hour >= start_hour) & (hour < end_hour)
        else:
            # Обработка интервалов с переходом через полночь (например, 22–2)
            mask = (hour >= start_hour) | (hour < end_hour)
        return pd.Series(mask, index=index, dtype='Int8')

    def calculate_session_flag_tz(self, index: pd.DatetimeIndex, tz: str, start_hour_local: int, end_hour_local: int) -> pd.Series:
        """
        Флаг сессии в локальном часовом поясе с учётом DST.
        Если индекс без tz — трактуем как UTC.
        """
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