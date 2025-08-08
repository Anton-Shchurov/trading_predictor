"""
Модуль для создания лаг-фич (lag features).

Создает лаговые признаки на основе исторических значений:
- Лаги цен (Close, Open, High, Low)
- Лаги объемов
- Лаги доходностей
- Разности и отношения лагов
- Скользящие лаги
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class LagFeatures:
    """
    Класс для создания лаг-фич.
    
    Поддерживает создание:
    - Простые лаги для любых колонок
    - Лаги доходностей
    - Разности между лагами
    - Отношения между лагами
    - Скользящие лаги (среднее за несколько лаговых периодов)
    """
    
    def add_price_lags(self, df: pd.DataFrame, 
                       columns: List[str] = ['Close'], 
                       lags: List[int] = [1, 2, 5, 10, 20, 24]) -> pd.DataFrame:
        """
        Добавляет лаги для ценовых колонок.
        
        Args:
            df: DataFrame с данными
            columns: Список колонок для создания лагов
            lags: Список лаговых периодов
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_volume_lags(self, df: pd.DataFrame, 
                        lags: List[int] = [1, 2, 5, 10, 20, 24]) -> pd.DataFrame:
        """
        Добавляет лаги для объема.
        """
        df = df.copy()
        
        if 'Volume' in df.columns:
            for lag in lags:
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    def add_return_lags(self, df: pd.DataFrame, 
                        periods: List[int] = [1, 5, 10, 20],
                        lags: List[int] = [1, 2, 5, 10]) -> pd.DataFrame:
        """
        Добавляет лаги доходностей для различных периодов.
        
        Args:
            df: DataFrame с данными
            periods: Периоды для расчета доходностей
            lags: Лаговые периоды
        """
        df = df.copy()
        
        # Сначала рассчитываем доходности для разных периодов
        for period in periods:
            return_col = f'Return_{period}'
            df[return_col] = df['Close'].pct_change(periods=period)
            
            # Затем создаем лаги для каждой доходности
            for lag in lags:
                df[f'{return_col}_Lag_{lag}'] = df[return_col].shift(lag)
        
        return df
    
    def add_lag_differences(self, df: pd.DataFrame,
                           base_column: str = 'Close',
                           lags: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Добавляет разности между текущим значением и лагами.
        
        Полезно для выявления трендов и изменений momentum.
        """
        df = df.copy()
        
        if base_column not in df.columns:
            return df
        
        for lag in lags:
            # Абсолютная разность
            df[f'{base_column}_Diff_Lag_{lag}'] = df[base_column] - df[base_column].shift(lag)
            
            # Относительная разность (в процентах)
            df[f'{base_column}_PctDiff_Lag_{lag}'] = (df[base_column] / df[base_column].shift(lag) - 1) * 100
        
        return df
    
    def add_lag_ratios(self, df: pd.DataFrame,
                       base_column: str = 'Close',
                       lags: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Добавляет отношения между текущим значением и лагами.
        """
        df = df.copy()
        
        if base_column not in df.columns:
            return df
        
        for lag in lags:
            df[f'{base_column}_Ratio_Lag_{lag}'] = df[base_column] / df[base_column].shift(lag)
        
        return df
    
    def add_rolling_lags(self, df: pd.DataFrame,
                        base_column: str = 'Close',
                        lag_periods: List[int] = [1, 5, 10],
                        rolling_windows: List[int] = [3, 5]) -> pd.DataFrame:
        """
        Добавляет скользящие средние лагов.
        
        Например, среднее значение цены за последние 3 дня,
        сдвинутое на 5 дней назад.
        """
        df = df.copy()
        
        if base_column not in df.columns:
            return df
        
        for lag in lag_periods:
            lagged_series = df[base_column].shift(lag)
            
            for window in rolling_windows:
                df[f'{base_column}_RollingLag_{lag}_{window}'] = lagged_series.rolling(window=window).mean()
        
        return df
    
    def add_lag_volatility(self, df: pd.DataFrame,
                          lags: List[int] = [1, 5, 10, 20],
                          volatility_window: int = 5) -> pd.DataFrame:
        """
        Добавляет лаги исторической волатильности.
        """
        df = df.copy()
        
        # Рассчитываем логарифмические доходности
        if 'Log_Returns' not in df.columns:
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Рассчитываем волатильность
        volatility = df['Log_Returns'].rolling(window=volatility_window).std()
        
        # Создаем лаги волатильности
        for lag in lags:
            df[f'Volatility_{volatility_window}d_Lag_{lag}'] = volatility.shift(lag)
        
        return df
    
    def add_lag_volume_price_interaction(self, df: pd.DataFrame,
                                        lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Добавляет лаги взаимодействия между ценой и объемом.
        """
        df = df.copy()
        
        if 'Volume' not in df.columns:
            return df
        
        # Volume-Price Trend (VPT)
        if 'VPT' not in df.columns:
            price_change = df['Close'].pct_change()
            df['VPT'] = (price_change * df['Volume']).cumsum()
        
        # Volume-Weighted Average Price лаги (защита от деления на ноль)
        vwap_den = df['Volume'].rolling(window=5).sum().replace(0, np.nan)
        vwap = (df['Close'] * df['Volume']).rolling(window=5).sum() / vwap_den
        
        for lag in lags:
            df[f'VPT_Lag_{lag}'] = df['VPT'].shift(lag)
            df[f'VWAP_5d_Lag_{lag}'] = vwap.shift(lag)
        
        return df
    
    def add_cross_lag_features(self, df: pd.DataFrame,
                              primary_col: str = 'Close',
                              secondary_cols: List[str] = ['Volume'],
                              lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Добавляет кросс-лаговые фичи между разными колонками.
        
        Например, корреляция между ценой сегодня и объемом 5 дней назад.
        """
        df = df.copy()
        
        for sec_col in secondary_cols:
            if sec_col not in df.columns:
                continue
                
            for lag in lags:
                # Простое произведение
                df[f'{primary_col}_{sec_col}_CrossLag_{lag}'] = df[primary_col] * df[sec_col].shift(lag)
                
                # Улучшенная скользящая корреляция
                # Используем фиксированное окно, но достаточно большое для стабильности
                min_window = max(30, lag + 10)  # Минимум 30 или lag+10
                max_window = min(50, len(df) // 10)  # Максимум 50 или 1/10 от длины
                window = max(min_window, max_window)
                
                if window >= min_window and len(df) > window + lag:
                    try:
                        # Создаем лагированную версию
                        lagged_col = df[sec_col].shift(lag)
                        
                        # Рассчитываем корреляцию только там, где есть достаточно данных
                        def safe_corr(x, y):
                            """Безопасный расчет корреляции."""
                            try:
                                # Убираем NaN
                                valid_idx = ~(pd.isna(x) | pd.isna(y))
                                if valid_idx.sum() < 5:  # Минимум 5 точек для корреляции
                                    return np.nan
                                
                                x_clean = x[valid_idx]
                                y_clean = y[valid_idx]
                                
                                # Проверяем на вариацию
                                if x_clean.std() < 1e-8 or y_clean.std() < 1e-8:
                                    return 0.0
                                
                                return np.corrcoef(x_clean, y_clean)[0, 1]
                            except (ValueError, IndexError):
                                return np.nan
                        
                        # Применяем rolling с безопасной функцией корреляции
                        corr_result = []
                        primary_series = df[primary_col]
                        
                        for i in range(len(df)):
                            if i < window + lag - 1:
                                corr_result.append(np.nan)
                            else:
                                start_idx = i - window + 1
                                end_idx = i + 1
                                
                                x_window = primary_series.iloc[start_idx:end_idx]
                                y_window = lagged_col.iloc[start_idx:end_idx]
                                
                                corr_val = safe_corr(x_window, y_window)
                                corr_result.append(corr_val)
                        
                        df[f'{primary_col}_{sec_col}_CrossCorr_Lag_{lag}'] = corr_result
                        
                    except Exception as e:
                        # В случае ошибки создаем колонку с NaN
                        df[f'{primary_col}_{sec_col}_CrossCorr_Lag_{lag}'] = np.nan
        
        return df
    
    def add_seasonal_lags(self, df: pd.DataFrame,
                         base_column: str = 'Close',
                         seasonal_periods: List[int] = [24]) -> pd.DataFrame:  # 24H для часовых данных
        """
        Добавляет сезонные лаги (суточные, недельные паттерны).
        
        Args:
            seasonal_periods: Периоды сезонности (24 для суточной для часовых данных)
        """
        df = df.copy()
        
        if base_column not in df.columns:
            return df
        
        for period in seasonal_periods:
            if len(df) > period:  # Проверяем достаточность данных
                df[f'{base_column}_Seasonal_Lag_{period}'] = df[base_column].shift(period)
                
                # Также добавляем отношение к сезонному лагу
                df[f'{base_column}_Seasonal_Ratio_{period}'] = df[base_column] / df[base_column].shift(period)
        
        return df
    
    def add_all(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Добавляет все лаг-фичи.
        
        Args:
            df: DataFrame с OHLCV данными
            config: Конфигурация параметров (опционально)
            
        Returns:
            DataFrame с добавленными лаг-фичами
        """
        if config is None:
            config = self._get_default_config()
        
        df = df.copy()
        
        # Простые лаги цен
        df = self.add_price_lags(
            df, 
            config.get('price_columns', ['Close', 'Open', 'High', 'Low']),
            config.get('price_lags', [1, 2, 5, 10, 20, 24])
        )
        
        # Лаги объема
        df = self.add_volume_lags(df, config.get('volume_lags', [1, 2, 5, 10, 20, 24]))
        
        # Лаги доходностей
        df = self.add_return_lags(
            df,
            config.get('return_periods', [1, 5, 10, 20]),
            config.get('return_lags', [1, 2, 5, 10])
        )
        
        # Разности лагов
        df = self.add_lag_differences(df, 'Close', config.get('diff_lags', [1, 5, 10, 20]))
        
        # Отношения лагов
        df = self.add_lag_ratios(df, 'Close', config.get('ratio_lags', [1, 5, 10, 20]))
        
        # Скользящие лаги
        df = self.add_rolling_lags(
            df, 
            'Close',
            config.get('rolling_lag_periods', [1, 5, 10]),
            config.get('rolling_lag_windows', [3, 5])
        )
        
        # Лаги волатильности (передаем окно явно для согласованности имён)
        df = self.add_lag_volatility(
            df,
            config.get('volatility_lags', [1, 5, 10, 20]),
            config.get('volatility_window', 5)
        )
        
        # Взаимодействие цена-объем
        df = self.add_lag_volume_price_interaction(df, config.get('volume_price_lags', [1, 5, 10]))
        
        # Кросс-лаговые фичи
        df = self.add_cross_lag_features(
            df,
            'Close',
            config.get('cross_lag_columns', ['Volume']),
            config.get('cross_lags', [1, 5, 10])
        )
        
        # Сезонные лаги
        df = self.add_seasonal_lags(
            df,
            'Close',
            config.get('seasonal_periods', [24])  # Часовые данные: суточные паттерны
        )
        
        return df
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'price_columns': ['Close', 'Open', 'High', 'Low'],
            'price_lags': [1, 2, 5, 10, 20, 24],
            'volume_lags': [1, 2, 5, 10, 20, 24],
            'return_periods': [1, 5, 10, 20],
            'return_lags': [1, 2, 5, 10],
            'diff_lags': [1, 5, 10, 20],
            'ratio_lags': [1, 5, 10, 20],
            'rolling_lag_periods': [1, 5, 10],
            'rolling_lag_windows': [3, 5],
            'volatility_lags': [1, 5, 10, 20],
            'volume_price_lags': [1, 5, 10],
            'cross_lag_columns': ['Volume'],
            'cross_lags': [1, 5, 10],
            'seasonal_periods': [24]  # Для часовых данных (суточные паттерны)
        }
    
    def get_feature_names(self, config: Optional[Dict] = None) -> List[str]:
        """Возвращает список названий всех создаваемых фич."""
        if config is None:
            config = self._get_default_config()
        
        features = []
        
        # Price lag features
        for col in config.get('price_columns', ['Close', 'Open', 'High', 'Low']):
            for lag in config.get('price_lags', [1, 2, 5, 10, 20, 24]):
                features.append(f'{col}_Lag_{lag}')
        
        # Volume lag features
        for lag in config.get('volume_lags', [1, 2, 5, 10, 20, 24]):
            features.append(f'Volume_Lag_{lag}')
        
        # Return lag features
        for period in config.get('return_periods', [1, 5, 10, 20]):
            features.append(f'Return_{period}')
            for lag in config.get('return_lags', [1, 2, 5, 10]):
                features.append(f'Return_{period}_Lag_{lag}')
        
        # Difference features
        for lag in config.get('diff_lags', [1, 5, 10, 20]):
            features.extend([
                f'Close_Diff_Lag_{lag}',
                f'Close_PctDiff_Lag_{lag}'
            ])
        
        # Ratio features
        for lag in config.get('ratio_lags', [1, 5, 10, 20]):
            features.append(f'Close_Ratio_Lag_{lag}')
        
        # Rolling lag features
        for lag in config.get('rolling_lag_periods', [1, 5, 10]):
            for window in config.get('rolling_lag_windows', [3, 5]):
                features.append(f'Close_RollingLag_{lag}_{window}')
        
        # Volatility lag features
        vol_window = config.get('volatility_window', 5)
        for lag in config.get('volatility_lags', [1, 5, 10, 20]):
            features.append(f'Volatility_{vol_window}d_Lag_{lag}')
        
        # Volume-price interaction features
        features.extend(['VPT'])
        for lag in config.get('volume_price_lags', [1, 5, 10]):
            features.extend([
                f'VPT_Lag_{lag}',
                f'VWAP_5d_Lag_{lag}'
            ])
        
        # Cross lag features
        for sec_col in config.get('cross_lag_columns', ['Volume']):
            for lag in config.get('cross_lags', [1, 5, 10]):
                features.extend([
                    f'Close_{sec_col}_CrossLag_{lag}',
                    f'Close_{sec_col}_CrossCorr_Lag_{lag}'
                ])
        
        # Seasonal lag features
        for period in config.get('seasonal_periods', [24]):
            features.extend([
                f'Close_Seasonal_Lag_{period}',
                f'Close_Seasonal_Ratio_{period}'
            ])
        
        return features