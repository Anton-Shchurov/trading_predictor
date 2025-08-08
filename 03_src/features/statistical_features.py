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
    Класс для создания статистических фич.
    
    Поддерживает создание:
    - Rate of Change (ROC)
    - Rolling mean и standard deviation
    - Z-Score (нормализация относительно скользящего среднего)
    - Skewness и Kurtosis (асимметрия и эксцесс)
    - Различные производные статистические метрики
    """
    
    def add_roc(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Добавляет Rate of Change (ROC) для различных периодов.
        
        ROC показывает процентное изменение цены за период.
        """
        df = df.copy()
        
        for period in periods:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
        
        return df
    
    def add_rolling_mean_std(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Добавляет скользящие средние и стандартные отклонения.
        """
        df = df.copy()
        
        for window in windows:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            
            # Коэффициент вариации с защитой от деления на ноль
            mean_safe = df[f'Rolling_Mean_{window}'].replace(0, 1e-8)
            df[f'CV_{window}'] = df[f'Rolling_Std_{window}'] / mean_safe
        
        return df
    
    def add_zscore(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Добавляет Z-Score для различных окон.
        
        Z-Score показывает, на сколько стандартных отклонений текущая цена
        отличается от скользящего среднего.
        """
        df = df.copy()
        
        for window in windows:
            rolling_mean = df['Close'].rolling(window=window).mean()
            rolling_std = df['Close'].rolling(window=window).std()
            
            # Избегаем деления на ноль - заменяем нулевое std на очень маленькое значение
            rolling_std_safe = rolling_std.replace(0, 1e-8)
            
            zscore = (df['Close'] - rolling_mean) / rolling_std_safe
            df[f'ZScore_{window}'] = zscore
        
        return df
    
    def add_skew_kurt(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Добавляет асимметрию (Skewness) и эксцесс (Kurtosis).
        
        Skewness показывает асимметрию распределения.
        Kurtosis показывает "остроту" распределения.
        """
        df = df.copy()
        
        for window in windows:
            df[f'Skew_{window}'] = df['Close'].rolling(window=window).skew()
            df[f'Kurt_{window}'] = df['Close'].rolling(window=window).kurt()
        
        return df
    
    def add_price_position(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Добавляет позицию цены относительно минимума и максимума за период.
        """
        df = df.copy()
        
        for window in windows:
            rolling_min = df['Low'].rolling(window=window).min()
            rolling_max = df['High'].rolling(window=window).max()
            
            # Рассчитываем диапазон
            price_range = rolling_max - rolling_min
            
            # Позиция цены закрытия относительно диапазона
            # Избегаем деления на ноль когда max == min (нет волатильности)
            df[f'Price_Position_{window}'] = np.where(
                price_range > 1e-8,  # Если есть диапазон
                (df['Close'] - rolling_min) / price_range,  # Нормальный расчет
                0.5  # Если диапазон = 0, то цена в середине "диапазона"
            )
            
            # Расстояние от максимума и минимума (с защитой от деления на ноль)
            df[f'Distance_From_High_{window}'] = np.where(
                rolling_max > 1e-8,
                (rolling_max - df['Close']) / rolling_max,
                0
            )
            
            df[f'Distance_From_Low_{window}'] = np.where(
                rolling_min > 1e-8,
                (df['Close'] - rolling_min) / rolling_min,
                0
            )
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Добавляет различные метрики волатильности.
        """
        df = df.copy()
        
        # Рассчитываем логарифмические доходности
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in windows:
            # Реализованная волатильность (аннуализированная)
            annual = np.sqrt(252 * 24)
            df[f'Realized_Vol_{window}'] = df['Log_Returns'].rolling(window=window).std() * annual
            
            # Parkinson volatility (использует High-Low)
            hl = np.log(df['High'] / df['Low'])
            parkinson_var = (1 / (4 * np.log(2))) * hl.pow(2).rolling(window=window).mean()
            
            # Garman-Klass volatility
            co = np.log(df['Close'] / df['Open'])
            gk_var = 0.5 * hl.pow(2).rolling(window=window).mean() - (2*np.log(2)-1) * co.pow(2).rolling(window=window).mean()
            
            df[f'Parkinson_Vol_{window}'] = np.sqrt(np.clip(parkinson_var, 0, None)) * annual
            df[f'GK_Vol_{window}'] = np.sqrt(np.clip(gk_var, 0, None)) * annual
        
        return df
    
    def add_trend_strength(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Добавляет метрики силы тренда.
        """
        df = df.copy()
        
        for window in windows:
            # Линейная регрессия для определения тренда
            def linear_reg_slope(y):
                if len(y) < 2:
                    return np.nan
                x = np.arange(len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return slope
            
            def linear_reg_r2(y):
                if len(y) < 2:
                    return np.nan
                x = np.arange(len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return r_value ** 2
            
            # Наклон линии тренда
            df[f'Trend_Slope_{window}'] = df['Close'].rolling(window=window).apply(linear_reg_slope, raw=True)
            
            # R-squared для оценки силы тренда
            df[f'Trend_R2_{window}'] = df['Close'].rolling(window=window).apply(linear_reg_r2, raw=True)
            
            # Тренд направление (1 для роста, -1 для падения, 0 для боковика)
            df[f'Trend_Direction_{window}'] = np.where(df[f'Trend_Slope_{window}'] > 0, 1,
                                                     np.where(df[f'Trend_Slope_{window}'] < 0, -1, 0))
        
        return df
    
    def add_autocorrelation(self, df: pd.DataFrame, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Добавляет автокорреляции для различных лагов.
        """
        df = df.copy()
        
        # Используем логарифмические доходности для автокорреляции
        if 'Log_Returns' not in df.columns:
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        def safe_autocorr(series, lag):
            """
            Безопасный расчет автокорреляции с обработкой краевых случаев.
            """
            try:
                if len(series) <= lag + 1:  # Недостаточно данных
                    return np.nan
                
                # Убираем NaN значения
                clean_series = series.dropna()
                if len(clean_series) <= lag + 1:
                    return np.nan
                
                # Рассчитываем автокорреляцию через numpy
                x = clean_series.values
                n = len(x)
                
                if n <= lag:
                    return np.nan
                
                # Центрируем данные
                x_centered = x - np.mean(x)
                
                # Рассчитываем автокорреляцию
                autocorr = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
                
                # Проверяем на NaN (может возникнуть при нулевой дисперсии)
                if np.isnan(autocorr) or np.isinf(autocorr):
                    return 0.0
                    
                return autocorr
                
            except (ValueError, IndexError, ZeroDivisionError):
                return np.nan
        
        # Используем увеличенное окно для более стабильных результатов
        window = 30  # Увеличено с 20 до 30 для лучшей стабильности
        
        for lag in lags:
            df[f'Autocorr_{lag}'] = df['Log_Returns'].rolling(window=window).apply(
                lambda x: safe_autocorr(x, lag), raw=False
            )
        
        return df
    
    def add_entropy_features(self, df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Добавляет метрики энтропии для оценки предсказуемости.
        """
        df = df.copy()
        
        def shannon_entropy(x, bins=10):
            """Рассчитывает энтропию Шеннона."""
            if len(x) < 2:
                return np.nan
            
            hist, _ = np.histogram(x, bins=bins)
            hist = hist[hist > 0]  # Убираем нулевые значения
            probs = hist / hist.sum()
            return -np.sum(probs * np.log2(probs))
        
        # Рассчитываем доходности если их нет
        if 'Log_Returns' not in df.columns:
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in windows:
            df[f'Entropy_{window}'] = df['Log_Returns'].rolling(window=window).apply(
                shannon_entropy, raw=True
            )
        
        return df
    
    def add_all(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Добавляет все статистические фичи.
        
        Args:
            df: DataFrame с OHLCV данными
            config: Конфигурация параметров (опционально)
            
        Returns:
            DataFrame с добавленными статистическими фичами
        """
        if config is None:
            config = self._get_default_config()
        
        df = df.copy()
        
        # ROC
        df = self.add_roc(df, config.get('roc_periods', [1, 5, 10, 20]))
        
        # Rolling statistics
        df = self.add_rolling_mean_std(df, config.get('rolling_windows', [5, 10, 20, 50]))
        
        # Z-Score
        df = self.add_zscore(df, config.get('zscore_windows', [5, 10, 20]))
        
        # Skewness and Kurtosis
        df = self.add_skew_kurt(df, config.get('skew_kurt_windows', [10, 20, 50]))
        
        # Price position
        df = self.add_price_position(df, config.get('position_windows', [10, 20, 50]))
        
        # Volatility
        df = self.add_volatility_features(df, config.get('volatility_windows', [10, 20, 50]))
        
        # Trend strength
        df = self.add_trend_strength(df, config.get('trend_windows', [10, 20, 50]))
        
        # Autocorrelation
        df = self.add_autocorrelation(df, config.get('autocorr_lags', [1, 5, 10]))
        
        # Entropy
        df = self.add_entropy_features(df, config.get('entropy_windows', [10, 20]))
        
        return df
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'roc_periods': [1, 5, 10, 20],
            'rolling_windows': [5, 10, 20, 50],
            'zscore_windows': [5, 10, 20],
            'skew_kurt_windows': [10, 20, 50],
            'position_windows': [10, 20, 50],
            'volatility_windows': [10, 20, 50],
            'trend_windows': [10, 20, 50],
            'autocorr_lags': [1, 5, 10],
            'entropy_windows': [10, 20]
        }
    
    def get_feature_names(self, config: Optional[Dict] = None) -> List[str]:
        """Возвращает список названий всех создаваемых фич."""
        if config is None:
            config = self._get_default_config()
        
        features = []
        
        # ROC features
        for period in config.get('roc_periods', [1, 5, 10, 20]):
            features.append(f'ROC_{period}')
        
        # Rolling statistics
        for window in config.get('rolling_windows', [5, 10, 20, 50]):
            features.extend([
                f'Rolling_Mean_{window}',
                f'Rolling_Std_{window}',
                f'CV_{window}'
            ])
        
        # Z-Score features
        for window in config.get('zscore_windows', [5, 10, 20]):
            features.append(f'ZScore_{window}')
        
        # Skew/Kurt features
        for window in config.get('skew_kurt_windows', [10, 20, 50]):
            features.extend([f'Skew_{window}', f'Kurt_{window}'])
        
        # Position features
        for window in config.get('position_windows', [10, 20, 50]):
            features.extend([
                f'Price_Position_{window}',
                f'Distance_From_High_{window}',
                f'Distance_From_Low_{window}'
            ])
        
        # Volatility features
        features.append('Log_Returns')
        for window in config.get('volatility_windows', [10, 20, 50]):
            features.extend([
                f'Realized_Vol_{window}',
                f'Parkinson_Vol_{window}',
                f'GK_Vol_{window}'
            ])
        
        # Trend features
        for window in config.get('trend_windows', [10, 20, 50]):
            features.extend([
                f'Trend_Slope_{window}',
                f'Trend_R2_{window}',
                f'Trend_Direction_{window}'
            ])
        
        # Autocorrelation features
        for lag in config.get('autocorr_lags', [1, 5, 10]):
            features.append(f'Autocorr_{lag}')
        
        # Entropy features
        for window in config.get('entropy_windows', [10, 20]):
            features.append(f'Entropy_{window}')
        
        return features