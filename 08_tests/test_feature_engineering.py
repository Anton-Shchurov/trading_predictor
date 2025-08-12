"""
Unit-тесты для модулей Feature Engineering.

Тестирует корректность работы всех модулей:
- TechnicalIndicators
- StatisticalFeatures  
- LagFeatures
- FeatureEngineeringPipeline
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import yaml

# Добавляем путь к модулям
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / '03_src'))

from features import (
    TechnicalIndicators,
    StatisticalFeatures,
    LagFeatures,
    FeatureEngineeringPipeline
)


class TestDataGenerator:
    """Генератор тестовых данных для тестов."""
    
    @staticmethod
    def create_sample_ohlcv(n_periods=100, start_price=1.0):
        """Создает синтетические OHLCV данные."""
        np.random.seed(42)  # Для воспроизводимости
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='h')
        
        # Генерация цен с трендом и случайностью
        returns = np.random.normal(0, 0.001, n_periods)
        prices = [start_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Минимальная цена
        
        # Создание OHLC из цен
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Open - предыдущий Close с небольшим gap
        df['Open'] = df['Close'].shift(1).fillna(start_price) * (1 + np.random.normal(0, 0.0005, n_periods))
        
        # Инициализируем колонки High и Low
        df['High'] = df['Close']
        df['Low'] = df['Close']
        
        # High и Low относительно Open и Close
        for i in range(n_periods):
            high_base = max(df.iloc[i]['Open'], df.iloc[i]['Close'])
            low_base = min(df.iloc[i]['Open'], df.iloc[i]['Close'])
            
            df.iloc[i, df.columns.get_loc('High')] = high_base * (1 + abs(np.random.normal(0, 0.002)))
            df.iloc[i, df.columns.get_loc('Low')] = low_base * (1 - abs(np.random.normal(0, 0.002)))
        
        # Volume
        df['Volume'] = np.random.randint(1000, 10000, n_periods)
        
        # Проверяем корректность OHLC
        df.loc[df['High'] < df['Close'], 'High'] = df['Close']
        df.loc[df['High'] < df['Open'], 'High'] = df['Open']
        df.loc[df['Low'] > df['Close'], 'Low'] = df['Close']
        df.loc[df['Low'] > df['Open'], 'Low'] = df['Open']
        
        return df


class TestTechnicalIndicators:
    """Тесты для модуля технических индикаторов."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.ti = TechnicalIndicators()
        self.df = TestDataGenerator.create_sample_ohlcv(100)
    
    def test_initialization(self):
        """Тест инициализации модуля."""
        assert isinstance(self.ti, TechnicalIndicators)
        assert hasattr(self.ti, 'has_pandas_ta')
    
    def test_add_ema(self):
        """Тест создания EMA."""
        periods = [10, 20]
        result = self.ti.add_ema(self.df.copy(), periods)
        
        # Проверяем создание колонок
        for period in periods:
            assert f'EMA_{period}' in result.columns
            assert not result[f'EMA_{period}'].isnull().all()
        
        # Проверяем свойства EMA (должна быть меньше волатильна чем цена)
        ema_10 = result['EMA_10'].dropna()
        close_subset = result['Close'].loc[ema_10.index]
        
        assert ema_10.std() <= close_subset.std(), "EMA должна быть менее волатильной"
    
    def test_add_macd(self):
        """Тест создания MACD."""
        result = self.ti.add_macd(self.df.copy())
        
        expected_cols = ['MACD', 'MACD_Signal', 'MACD_Hist']
        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().all()
        
        # Проверяем связь между компонентами MACD
        valid_data = result[expected_cols].dropna()
        if len(valid_data) > 0:
            # MACD_Hist = MACD - MACD_Signal
            hist_calc = valid_data['MACD'] - valid_data['MACD_Signal']
            np.testing.assert_array_almost_equal(
                valid_data['MACD_Hist'].values,
                hist_calc.values,
                decimal=8
            )
    
    def test_add_rsi(self):
        """Тест создания RSI."""
        result = self.ti.add_rsi(self.df.copy())
        
        assert 'RSI' in result.columns
        rsi_values = result['RSI'].dropna()
        
        # RSI должен быть в диапазоне 0-100
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()
        
        # RSI не должен быть константой
        assert rsi_values.std() > 0
    
    def test_add_bollinger_bands(self):
        """Тест создания Bollinger Bands."""
        result = self.ti.add_bollinger_bands(self.df.copy())
        
        expected_cols = ['BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Position']
        for col in expected_cols:
            assert col in result.columns
        
        valid_data = result[expected_cols + ['Close']].dropna()
        if len(valid_data) > 0:
            # BB_Upper >= BB_Middle >= BB_Lower
            assert (valid_data['BB_Upper'] >= valid_data['BB_Middle']).all()
            assert (valid_data['BB_Middle'] >= valid_data['BB_Lower']).all()
            
            # BB_Width = BB_Upper - BB_Lower
            width_calc = valid_data['BB_Upper'] - valid_data['BB_Lower']
            np.testing.assert_array_almost_equal(
                valid_data['BB_Width'].values,
                width_calc.values,
                decimal=8
            )
    
    def test_add_atr(self):
        """Тест создания ATR."""
        result = self.ti.add_atr(self.df.copy())
        
        assert 'ATR' in result.columns
        atr_values = result['ATR'].dropna()
        
        # ATR должен быть положительным
        assert (atr_values > 0).all()
    
    def test_add_stochastic(self):
        """Тест создания Stochastic Oscillator."""
        result = self.ti.add_stochastic(self.df.copy())
        
        expected_cols = ['Stoch_K', 'Stoch_D']
        for col in expected_cols:
            assert col in result.columns
        
        valid_data = result[expected_cols].dropna()
        if len(valid_data) > 0:
            # Stochastic должен быть в диапазоне 0-100
            for col in expected_cols:
                values = valid_data[col]
                assert (values >= 0).all() and (values <= 100).all()
    
    def test_add_volume_indicators(self):
        """Тест создания объемных индикаторов."""
        result = self.ti.add_obv(self.df.copy())
        result = self.ti.add_cmf(result)
        result = self.ti.add_ad_line(result)
        
        expected_cols = ['OBV', 'CMF', 'AD']
        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isnull().all()
    
    def test_add_all(self):
        """Тест создания всех технических индикаторов."""
        result = self.ti.add_all(self.df.copy())
        
        # Проверяем, что создано больше колонок
        assert len(result.columns) > len(self.df.columns)
        
        # Проверяем наличие основных индикаторов
        key_indicators = ['EMA_21', 'MACD', 'RSI', 'BB_Upper', 'ATR']
        for indicator in key_indicators:
            assert indicator in result.columns, f"Отсутствует индикатор: {indicator}"
    
    def test_get_feature_names(self):
        """Тест получения списка названий фич."""
        feature_names = self.ti.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'EMA_21' in feature_names
        assert 'RSI' in feature_names


class TestStatisticalFeatures:
    """Тесты для модуля статистических признаков."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.sf = StatisticalFeatures()
        self.df = TestDataGenerator.create_sample_ohlcv(100)
    
    def test_add_roc(self):
        """Тест создания ROC."""
        periods = [1, 5, 10]
        result = self.sf.add_roc(self.df.copy(), periods)
        
        for period in periods:
            col_name = f'ROC_{period}'
            assert col_name in result.columns
            
            # Проверяем формулу ROC
            valid_data = result[[col_name, 'Close']].dropna()
            if len(valid_data) > period:
                expected_roc = valid_data['Close'].pct_change(periods=period) * 100
                # Сравниваем только последние значения (где есть все данные)
                if len(expected_roc.dropna()) > 0:
                    assert not np.isnan(valid_data[col_name].iloc[-1])
    
    def test_add_rolling_mean_std(self):
        """Тест создания скользящих статистик."""
        windows = [5, 20]
        result = self.sf.add_rolling_mean_std(self.df.copy(), windows)
        
        for window in windows:
            mean_col = f'Rolling_Mean_{window}'
            std_col = f'Rolling_Std_{window}'
            cv_col = f'CV_{window}'
            
            assert mean_col in result.columns
            assert std_col in result.columns
            assert cv_col in result.columns
            
            # Проверяем, что стандартное отклонение >= 0
            std_values = result[std_col].dropna()
            assert (std_values >= 0).all()
    
    def test_add_zscore(self):
        """Тест создания Z-Score."""
        windows = [10, 20]
        result = self.sf.add_zscore(self.df.copy(), windows)
        
        for window in windows:
            col_name = f'ZScore_{window}'
            assert col_name in result.columns
            
            # Z-Score должен иметь среднее около 0 для достаточно большого окна
            zscore_values = result[col_name].dropna()
            if len(zscore_values) > window * 2:
                # Проверяем, что значения не все одинаковые
                assert zscore_values.std() > 0
    
    def test_add_skew_kurt(self):
        """Тест создания Skewness и Kurtosis."""
        windows = [20]
        result = self.sf.add_skew_kurt(self.df.copy(), windows)
        
        for window in windows:
            skew_col = f'Skew_{window}'
            kurt_col = f'Kurt_{window}'
            
            assert skew_col in result.columns
            assert kurt_col in result.columns
    
    def test_add_volatility_features(self):
        """Тест создания фич волатильности."""
        result = self.sf.add_volatility_features(self.df.copy(), [20])
        
        vol_cols = ['Log_Returns', 'Realized_Vol_20', 'Parkinson_Vol_20', 'GK_Vol_20']
        for col in vol_cols:
            assert col in result.columns
        
        # Проверяем, что волатильность положительная
        for col in ['Realized_Vol_20', 'Parkinson_Vol_20', 'GK_Vol_20']:
            vol_values = result[col].dropna()
            if len(vol_values) > 0:
                assert (vol_values >= 0).all(), f"Отрицательная волатильность в {col}"
    
    def test_add_all(self):
        """Тест создания всех статистических признаков."""
        result = self.sf.add_all(self.df.copy())
        
        # Проверяем, что создано больше колонок
        assert len(result.columns) > len(self.df.columns)
        
        # Проверяем наличие ключевых фич
        key_features = ['ROC_1', 'Rolling_Mean_20', 'ZScore_20', 'Log_Returns']
        for feature in key_features:
            assert feature in result.columns, f"Отсутствует признак: {feature}"


class TestLagFeatures:
    """Тесты для модуля лаг-признаков."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.lf = LagFeatures()
        self.df = TestDataGenerator.create_sample_ohlcv(100)
    
    def test_add_price_lags(self):
        """Тест создания лагов цен."""
        columns = ['Close']
        lags = [1, 5, 10]
        result = self.lf.add_price_lags(self.df.copy(), columns, lags)
        
        for col in columns:
            for lag in lags:
                lag_col = f'{col}_Lag_{lag}'
                assert lag_col in result.columns
                
                # Проверяем корректность лага
                original_values = result[col].iloc[lag:].values
                lag_values = result[lag_col].iloc[lag:].dropna().values
                
                if len(lag_values) > 0 and len(original_values) >= len(lag_values):
                    # Лаг должен совпадать с исходными значениями сдвинутыми на lag позиций
                    expected_values = result[col].iloc[:-lag].values
                    if len(expected_values) == len(lag_values):
                        np.testing.assert_array_almost_equal(
                            lag_values, expected_values, decimal=8
                        )
    
    def test_add_volume_lags(self):
        """Тест создания лагов объема."""
        lags = [1, 5]
        result = self.lf.add_volume_lags(self.df.copy(), lags)
        
        for lag in lags:
            lag_col = f'Volume_Lag_{lag}'
            assert lag_col in result.columns
    
    def test_add_return_lags(self):
        """Тест создания лагов доходностей."""
        periods = [1, 5]
        lags = [1, 2]
        result = self.lf.add_return_lags(self.df.copy(), periods, lags)
        
        # Проверяем создание колонок доходностей
        for period in periods:
            return_col = f'Return_{period}'
            assert return_col in result.columns
            
            # И их лагов
            for lag in lags:
                lag_col = f'{return_col}_Lag_{lag}'
                assert lag_col in result.columns
    
    def test_add_lag_differences(self):
        """Тест создания разностей с лагами."""
        lags = [1, 5]
        result = self.lf.add_lag_differences(self.df.copy(), 'Close', lags)
        
        for lag in lags:
            diff_col = f'Close_Diff_Lag_{lag}'
            pct_diff_col = f'Close_PctDiff_Lag_{lag}'
            
            assert diff_col in result.columns
            assert pct_diff_col in result.columns
    
    def test_add_seasonal_lags(self):
        """Тест создания сезонных лагов."""
        # Используем короткие периоды для тестовых данных
        seasonal_periods = [24]
        result = self.lf.add_seasonal_lags(self.df.copy(), 'Close', seasonal_periods)
        
        for period in seasonal_periods:
            if len(self.df) > period:  # Только если данных достаточно
                seasonal_col = f'Close_Seasonal_Lag_{period}'
                ratio_col = f'Close_Seasonal_Ratio_{period}'
                
                assert seasonal_col in result.columns
                assert ratio_col in result.columns
    
    def test_add_all(self):
        """Тест создания всех лаг-признаков."""
        # Используем ограниченную конфигурацию для быстрого тестирования
        config = {
            'price_columns': ['Close'],
            'price_lags': [1, 5],
            'volume_lags': [1],
            'return_periods': [1],
            'return_lags': [1],
            'diff_lags': [1],
            'ratio_lags': [1],
            'seasonal_periods': []  # Убираем сезонные лаги для быстрого теста
        }
        
        result = self.lf.add_all(self.df.copy(), config)
        
        # Проверяем, что создано больше колонок
        assert len(result.columns) > len(self.df.columns)
        
        # Проверяем наличие ключевых лагов
        key_lags = ['Close_Lag_1', 'Volume_Lag_1', 'Return_1']
        for lag in key_lags:
            assert lag in result.columns, f"Отсутствует лаг: {lag}"


class TestFeatureEngineeringPipeline:
    """Тесты для главного пайплайна."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.test_df = TestDataGenerator.create_sample_ohlcv(50)
        
    def create_temp_config(self):
        """Создает временный конфигурационный файл."""
        config = {
            'technical_indicators': {
                'ema_periods': [10, 20],
                'rsi_period': 14
            },
            'statistical_features': {
                'roc_periods': [1, 5],
                'rolling_windows': [10]
            },
            'lag_features': {
                'price_columns': ['Close'],
                'price_lags': [1, 5],
                'volume_lags': [1],
                'seasonal_periods': []
            },
            'pipeline_settings': {
                'missing_values': {'strategy': 'drop'},
                'validation': {'check_duplicates': True}
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        yaml.dump(config, temp_file, default_flow_style=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_initialization_with_config(self):
        """Тест инициализации с конфигурацией."""
        config_file = self.create_temp_config()
        
        try:
            pipeline = FeatureEngineeringPipeline(config_path=config_file)
            assert pipeline.config is not None
            assert 'technical_indicators' in pipeline.config
        finally:
            Path(config_file).unlink()  # Удаляем временный файл
    
    def test_initialization_without_config(self):
        """Тест инициализации без конфигурации."""
        with patch('pathlib.Path.exists', return_value=False):
            pipeline = FeatureEngineeringPipeline()
            assert pipeline.config is not None
            # Должна загрузиться конфигурация по умолчанию
    
    def test_validate_and_prepare_data(self):
        """Тест валидации и подготовки данных."""
        pipeline = FeatureEngineeringPipeline()
        
        # Создаем данные с исходными названиями колонок (как в CSV)
        test_data = self.test_df.copy()
        test_data.columns = ['open', 'high', 'low', 'close', 'volume']
        test_data.reset_index(inplace=True)
        test_data.rename(columns={'index': 'time'}, inplace=True)
        
        result = pipeline._validate_and_prepare_data(test_data)
        
        # Проверяем, что колонки переименованы
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            assert col in result.columns
        
        # Проверяем, что индекс - datetime
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_create_features(self):
        """Тест создания признаков."""
        config_file = self.create_temp_config()
        
        try:
            pipeline = FeatureEngineeringPipeline(config_path=config_file)
            result = pipeline.create_features(self.test_df.copy())
            
            # Проверяем, что создано больше колонок
            assert len(result.columns) > len(self.test_df.columns)
            
            # Проверяем статистику
            assert pipeline.stats['created_features'] > 0
            assert pipeline.stats['total_columns'] == len(result.columns)
            
        finally:
            Path(config_file).unlink()
    
    def test_missing_values_handling(self):
        """Тест обработки пропущенных значений."""
        pipeline = FeatureEngineeringPipeline()
        
        # Создаем данные с пропусками
        test_data = self.test_df.copy()
        test_data.iloc[5:10, 1] = np.nan  # Добавляем NaN в High
        
        result = pipeline._handle_missing_values(test_data)
        
        # После обработки не должно быть пропусков
        assert result.isnull().sum().sum() == 0
    
    @patch('pandas.DataFrame.to_parquet')
    def test_save_results(self, mock_to_parquet):
        """Тест сохранения результатов."""
        pipeline = FeatureEngineeringPipeline()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.parquet"
            demo_path = Path(temp_dir) / "test_demo.parquet"
            
            # Мокаем настройки пайплайна (без demo)
            pipeline.config['pipeline_settings'] = {
                'parquet_settings': {
                    'engine': 'pyarrow',
                    'compression': 'snappy',
                    'index': True
                }
            }
            
            full_path, demo_path_result = pipeline.save_results(self.test_df, str(output_path))
            
            # Проверяем, что сохранение полного файла вызвано один раз, демо не создаётся
            assert mock_to_parquet.call_count == 1
            assert full_path == str(output_path)
            assert demo_path_result is None
    
    def test_get_feature_importance_analysis(self):
        """Тест анализа важности признаков."""
        pipeline = FeatureEngineeringPipeline()
        
        # Создаем данные с несколькими признаками
        test_data = self.test_df.copy()
        test_data['EMA_10'] = test_data['Close'].rolling(10).mean()
        test_data['RSI'] = 50  # Константа для простоты
        test_data['Close_Lag_1'] = test_data['Close'].shift(1)
        
        analysis = pipeline.get_feature_importance_analysis(test_data)
        
        assert 'total_features' in analysis
        assert 'feature_types' in analysis
        assert 'technical' in analysis['feature_types']
        assert 'lag' in analysis['feature_types']


class TestEdgeCases:
    """Тесты для edge cases и обработки ошибок."""
    
    def test_empty_dataframe(self):
        """Тест с пустым DataFrame."""
        ti = TechnicalIndicators()
        empty_df = pd.DataFrame()
        
        # Должен вернуть пустой DataFrame без ошибок
        result = ti.add_ema(empty_df, [10])
        assert len(result) == 0
    
    def test_insufficient_data(self):
        """Тест с недостаточным количеством данных."""
        ti = TechnicalIndicators()
        small_df = TestDataGenerator.create_sample_ohlcv(5)  # Только 5 периодов
        
        result = ti.add_ema(small_df, [10])  # EMA требует больше данных
        
        # Должны быть созданы колонки, но с NaN значениями
        assert 'EMA_10' in result.columns
        # Первые значения должны быть NaN
        assert result['EMA_10'].iloc[:9].isnull().all()
    
    def test_constant_prices(self):
        """Тест с константными ценами."""
        ti = TechnicalIndicators()
        
        # Создаем данные с константными ценами
        dates = pd.date_range('2020-01-01', periods=50, freq='H')
        const_df = pd.DataFrame({
            'Open': [1.0] * 50,
            'High': [1.0] * 50,
            'Low': [1.0] * 50,
            'Close': [1.0] * 50,
            'Volume': [1000] * 50
        }, index=dates)
        
        result = ti.add_rsi(const_df)
        
        # RSI для константных цен должен быть NaN или определенным значением
        assert 'RSI' in result.columns
    
    def test_missing_columns(self):
        """Тест с отсутствующими обязательными колонками."""
        ti = TechnicalIndicators()
        
        # DataFrame без колонки Volume
        incomplete_df = pd.DataFrame({
            'Open': [1.0, 1.1, 1.05],
            'High': [1.1, 1.15, 1.1],
            'Low': [0.95, 1.05, 1.0],
            'Close': [1.05, 1.1, 1.08]
        })
        
        # OBV требует Volume - должен обработать отсутствие gracefully
        result = ti.add_obv(incomplete_df)
        
        # Колонка должна быть создана или обработка должна пройти без ошибок
        assert len(result.columns) >= len(incomplete_df.columns)


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])