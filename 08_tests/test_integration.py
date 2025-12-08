"""
Интеграционные тесты для Feature Engineering Pipeline.

Тестирует полный workflow и интеграцию между модулями.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, Mock

# Добавляем путь к модулям
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / '03_src'))

from features import FeatureEngineeringPipeline


class TestFullPipeline:
    """Интеграционные тесты полного пайплайна."""
    
    def create_test_csv(self, n_periods=200):
        """Создает временный CSV файл с тестовыми данными."""
        np.random.seed(42)
        
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='h')
        
        # Генерация реалистичных FOREX данных
        returns = np.random.normal(0, 0.001, n_periods)
        prices = [1.1000]  # Начальная цена EUR/USD
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.5))  # Минимальная цена
        
        # Создание OHLC
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            if i == 0:
                open_price = close_price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.0002))
            
            # High и Low относительно Open и Close
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)
            
            high = high_base * (1 + abs(np.random.normal(0, 0.001)))
            low = low_base * (1 - abs(np.random.normal(0, 0.001)))
            
            volume = np.random.randint(1000, 15000)
            
            data.append({
                'time': date.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Сохраняем во временный файл
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def create_test_config(self, profile="quick"):
        """Создает временную конфигурацию для тестов."""
        if profile == "quick":
            config = {
                'technical_indicators': {
                    'ema_periods': [10, 21],
                    'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                    'rsi_period': 14,
                    'bb_params': {'period': 20, 'std_dev': 2.0}
                },
                'statistical_features': {
                    'roc_periods': [1, 5],
                    'rolling_windows': [10, 20],
                    'zscore_windows': [20]
                },
                'lag_features': {
                    'price_columns': ['Close'],
                    'price_lags': [1, 5, 24],
                    'volume_lags': [1, 5],
                    'return_periods': [1],
                    'return_lags': [1],
                    'seasonal_periods': []  # Убираем для быстрого теста
                },
                'pipeline_settings': {
                    'missing_values': {'strategy': 'drop', 'min_periods_required': 50},
                    'validation': {'check_duplicates': True, 'check_sorting': True},
                    'demo_size': 100,
                    'parquet_settings': {
                        'engine': 'pyarrow',
                        'compression': 'snappy',
                        'index': True
                    }
                }
            }
        else:
            # Полная конфигурация для тестов производительности
            config = {
                'technical_indicators': {
                    'ema_periods': [9, 12, 21, 50, 200],
                    'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                    'rsi_period': 14,
                    'stoch_params': {'k_period': 14, 'd_period': 3},
                    'cci_period': 20,
                    'atr_period': 14,
                    'bb_params': {'period': 20, 'std_dev': 2.0},
                    'adx_period': 14,
                    'momentum_period': 10,
                    'cmf_period': 20
                },
                'statistical_features': {
                    'roc_periods': [1, 5, 10, 20],
                    'rolling_windows': [5, 10, 20, 50],
                    'zscore_windows': [5, 10, 20],
                    'skew_kurt_windows': [10, 20],
                    'volatility_windows': [10, 20],
                    'trend_windows': [10, 20]
                },
                'lag_features': {
                    'price_columns': ['Close', 'Open'],
                    'price_lags': [1, 2, 5, 10, 24],
                    'volume_lags': [1, 2, 5, 10],
                    'return_periods': [1, 5, 10],
                    'return_lags': [1, 2, 5],
                    'seasonal_periods': [24]
                },
                'pipeline_settings': {
                    'missing_values': {'strategy': 'drop', 'min_periods_required': 200},
                    'validation': {'check_duplicates': True, 'check_sorting': True}
                }
            }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        yaml.dump(config, temp_file, default_flow_style=False)
        temp_file.close()
        
        return temp_file.name
    
    def test_full_pipeline_quick(self):
        """Тест полного пайплайна с быстрой конфигурацией."""
        csv_file = self.create_test_csv(100)
        config_file = self.create_test_config("quick")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / "test_features.parquet"
                
                # Мокаем настройки путей в конфигурации
                with patch.object(FeatureEngineeringPipeline, '_load_config') as mock_config:
                    # Загружаем реальную конфигурацию и обновляем пути
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    config['pipeline_settings']['input_file'] = csv_file
                    config['pipeline_settings']['output_file'] = str(output_file)
                    config['pipeline_settings']['output_demo_file'] = str(output_file.with_suffix('.demo.parquet'))
                    
                    mock_config.return_value = config
                    
                    # Инициализируем и запускаем пайплайн
                    pipeline = FeatureEngineeringPipeline()
                    df_result, stats = pipeline.run_full_pipeline()
                    
                    # Проверяем результаты
                    assert isinstance(df_result, pd.DataFrame)
                    assert len(df_result) > 0
                    assert len(df_result.columns) > 5  # Больше чем исходные OHLCV
                    
                    # Проверяем статистику
                    assert stats['created_features'] > 0
                    assert stats['total_columns'] == len(df_result.columns)
                    assert stats['processing_time'] > 0
                    
                    # Проверяем качество данных
                    assert df_result.index.is_monotonic_increasing  # Сортировка по времени
                    assert df_result.isnull().sum().sum() == 0  # Нет пропусков после обработки
                    
                    print(f"✅ Быстрый тест пройден:")
                    print(f"   Создано признаков: {stats['created_features']}")
                    print(f"   Итого колонок: {stats['total_columns']}")
                    print(f"   Время выполнения: {stats['processing_time']:.2f} сек")
        
        finally:
            Path(csv_file).unlink()
            Path(config_file).unlink()
    
    def test_pipeline_with_different_profiles(self):
        """Тест пайплайна с разными профилями."""
        csv_file = self.create_test_csv(80)
        
        try:
            # Тест с профилем "quick"
            with patch('pathlib.Path.exists', return_value=False):
                pipeline_quick = FeatureEngineeringPipeline(profile="quick")
                
                # Мокаем загрузку данных
                test_df = pd.read_csv(csv_file)
                test_df['time'] = pd.to_datetime(test_df['time'])
                test_df = test_df.set_index('time')
                test_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                result_quick = pipeline_quick.create_features(test_df.copy())
                
                # Проверяем, что quick профиль создает меньше признаков
                assert len(result_quick.columns) > len(test_df.columns)
                assert len(result_quick.columns) < 100  # Ограничение для quick
                
            print(f"✅ Quick профиль: {len(result_quick.columns)} колонок")
        
        finally:
            Path(csv_file).unlink()
    
    def test_pipeline_error_handling(self):
        """Тест обработки ошибок в пайплайне."""
        # Тест с несуществующим файлом
        pipeline = FeatureEngineeringPipeline()
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_data("несуществующий_файл.csv")
        
        # Тест с некорректными данными
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Создаем файл с некорректной структурой
            temp_file.write("col1,col2\n1,2\n3,4\n")
            temp_file.close()
            
            try:
                df_invalid = pipeline.load_data(temp_file.name)
                # Должен загрузиться, но с предупреждениями
                assert len(df_invalid) > 0
                
            finally:
                Path(temp_file.name).unlink()
    
    def test_pipeline_memory_usage(self):
        """Тест использования памяти пайплайном."""
        csv_file = self.create_test_csv(500)  # Больший датасет
        config_file = self.create_test_config("quick")
        
        try:
            with patch.object(FeatureEngineeringPipeline, '_load_config') as mock_config:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                config['pipeline_settings']['input_file'] = csv_file
                mock_config.return_value = config
                
                pipeline = FeatureEngineeringPipeline()
                
                # Измеряем память до
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                df_result, stats = pipeline.run_full_pipeline()
                
                # Измеряем память после
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Проверяем разумное использование памяти
                data_size_mb = df_result.memory_usage(deep=True).sum() / 1024 / 1024
                
                print(f"📊 Использование памяти:")
                print(f"   Размер данных: {data_size_mb:.2f} MB")
                print(f"   Использовано памяти: {memory_used:.2f} MB")
                print(f"   Эффективность: {data_size_mb/memory_used:.2f}")
                
                # Память не должна превышать разумные пределы
                assert memory_used < 500, f"Слишком большое использование памяти: {memory_used:.2f} MB"
        
        finally:
            Path(csv_file).unlink()
            Path(config_file).unlink()
    
    def test_pipeline_performance(self):
        """Тест производительности пайплайна."""
        csv_file = self.create_test_csv(1000)  # Большой датасет
        config_file = self.create_test_config("full")
        
        try:
            import time
            
            with patch.object(FeatureEngineeringPipeline, '_load_config') as mock_config:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                config['pipeline_settings']['input_file'] = csv_file
                mock_config.return_value = config
                
                pipeline = FeatureEngineeringPipeline()
                
                start_time = time.time()
                df_result, stats = pipeline.run_full_pipeline()
                end_time = time.time()
                
                execution_time = end_time - start_time
                records_per_second = len(df_result) / execution_time
                features_per_second = len(df_result.columns) * len(df_result) / execution_time
                
                print(f"⚡ Производительность:")
                print(f"   Время выполнения: {execution_time:.2f} сек")
                print(f"   Записей в секунду: {records_per_second:.0f}")
                print(f"   Признаков в секунду: {features_per_second:.0f}")
                print(f"   Создано признаков: {stats['created_features']}")
                
                # Проверяем разумную производительность
                assert execution_time < 120, f"Слишком медленное выполнение: {execution_time:.2f} сек"
                assert records_per_second > 10, f"Слишком низкая производительность: {records_per_second:.2f} записей/сек"
        
        finally:
            Path(csv_file).unlink()
            Path(config_file).unlink()
    
    def test_pipeline_data_consistency(self):
        """Тест консистентности данных через пайплайн."""
        csv_file = self.create_test_csv(200)
        config_file = self.create_test_config("quick")
        
        try:
            with patch.object(FeatureEngineeringPipeline, '_load_config') as mock_config:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                config['pipeline_settings']['input_file'] = csv_file
                mock_config.return_value = config
                
                pipeline = FeatureEngineeringPipeline()
                
                # Запускаем пайплайн дважды
                df_result1, stats1 = pipeline.run_full_pipeline()
                
                # Создаем новый пайплайн с той же конфигурацией
                pipeline2 = FeatureEngineeringPipeline()
                pipeline2.config = config
                df_result2, stats2 = pipeline2.run_full_pipeline()
                
                # Результаты должны быть идентичными
                assert len(df_result1) == len(df_result2)
                assert len(df_result1.columns) == len(df_result2.columns)
                assert list(df_result1.columns) == list(df_result2.columns)
                
                # Проверяем числовые значения (с допуском на погрешности вычислений)
                numeric_cols = df_result1.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if not df_result1[col].equals(df_result2[col]):
                        # Проверяем с небольшим допуском
                        diff = (df_result1[col] - df_result2[col]).abs()
                        max_diff = diff.max()
                        assert max_diff < 1e-10, f"Большая разница в колонке {col}: {max_diff}"
                
                print(f"✅ Тест консистентности пройден")
                print(f"   Колонок: {len(df_result1.columns)}")
                print(f"   Записей: {len(df_result1)}")
        
        finally:
            Path(csv_file).unlink()
            Path(config_file).unlink()


class TestRealWorldScenarios:
    """Тесты реальных сценариев использования."""
    
    def test_missing_data_scenarios(self):
        """Тест различных сценариев с пропущенными данными."""
        # Создаем данные с различными типами пропусков
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'Open': np.random.uniform(1.0, 1.2, 100),
            'High': np.random.uniform(1.1, 1.3, 100),
            'Low': np.random.uniform(0.9, 1.1, 100),
            'Close': np.random.uniform(1.0, 1.2, 100),
            'Volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Добавляем разные типы пропусков
        df.iloc[10:15, 1] = np.nan  # Пропуски в середине (High)
        df.iloc[0:5, 4] = np.nan    # Пропуски в начале (Volume)
        df.iloc[-5:, 3] = np.nan    # Пропуски в конце (Close)
        
        pipeline = FeatureEngineeringPipeline()
        
        # Тест с drop стратегией
        pipeline.config = {
            'pipeline_settings': {'missing_values': {'strategy': 'drop'}},
            'technical_indicators': {'ema_periods': [10]},
            'statistical_features': {'roc_periods': [1]},
            'lag_features': {'price_columns': ['Close'], 'price_lags': [1]}
        }
        
        result = pipeline.create_features(df)
        
        # После drop не должно быть пропусков
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(df)  # Часть данных должна быть удалена
        
        print(f"✅ Тест пропущенных данных: {len(df)} -> {len(result)} записей")
    
    def test_extreme_values(self):
        """Тест с экстремальными значениями."""
        dates = pd.date_range('2020-01-01', periods=50, freq='h')
        
        # Создаем данные с экстремальными значениями
        df = pd.DataFrame({
            'Open': [1.0] * 50,
            'High': [1.1] * 50,
            'Low': [0.9] * 50,
            'Close': [1.0] * 50,
            'Volume': [1000] * 50
        }, index=dates)
        
        # Добавляем экстремальные значения
        df.iloc[25, df.columns.get_loc('Close')] = 100.0  # Очень высокая цена
        df.iloc[26, df.columns.get_loc('Close')] = 0.01   # Очень низкая цена
        df.iloc[27, df.columns.get_loc('Volume')] = 1000000  # Очень большой объем
        
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.create_features(df)
        
        # Проверяем, что пайплайн справился с экстремальными значениями
        assert len(result) > 0
        assert not result.isin([np.inf, -np.inf]).any().any()  # Нет бесконечностей
        
        print(f"✅ Тест экстремальных значений пройден")


if __name__ == "__main__":
    # Запуск интеграционных тестов
    pytest.main([__file__, "-v", "--tb=short", "-s"])