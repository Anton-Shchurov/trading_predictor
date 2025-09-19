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