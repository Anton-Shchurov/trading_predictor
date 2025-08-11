"""
Главный пайплайн для Feature Engineering.

Объединяет все модули создания признаков и обеспечивает:
- Загрузку и валидацию данных
- Последовательное применение всех типов фич
- Обработку пропущенных значений
- Сохранение результатов в различных форматах
- Детальное логирование процесса
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

from .technical_indicators import TechnicalIndicators
from .statistical_features import StatisticalFeatures
from .lag_features import LagFeatures


class FeatureEngineeringPipeline:
    """
    Главный класс для создания всех признаков.
    
    Координирует работу всех модулей создания признаков,
    обеспечивает загрузку конфигурации, валидацию данных
    и сохранение результатов.
    """
    
    def __init__(self, config_path: Optional[str] = None, profile: str = "full"):
        """
        Инициализация пайплайна.
        
        Args:
            config_path: Путь к файлу конфигурации
            profile: Профиль конфигурации ("quick", "full", "experimental")
        """
        self.profile = profile
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Инициализация модулей
        self.technical_indicators = TechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        self.lag_features = LagFeatures()
        
        # Статистика выполнения
        self.stats = {
            'original_columns': 0,
            'created_features': 0,
            'total_columns': 0,
            'processing_time': 0,
            'data_shape': (0, 0)
        }
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Загружает конфигурацию из YAML файла."""
        if config_path is None:
            # Путь по умолчанию
            current_dir = Path(__file__).parent.parent.parent
            config_path = current_dir / "04_configs" / "feature_engineering.yml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Применяем профиль если указан
            if self.profile in config.get('profiles', {}):
                profile_config = config['profiles'][self.profile]
                if profile_config.get('inherit_from') != 'default':
                    # Обновляем конфигурацию профилем
                    for section, values in profile_config.items():
                        if section in config and isinstance(values, dict):
                            config[section].update(values)
            
            return config
        
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'technical_indicators': {},
            'statistical_features': {},
            'lag_features': {},
            'pipeline_settings': {
                'input_file': "01_data/raw/EURUSD_2010-2024_H1_OANDA.csv",
                'output_file': "01_data/processed/eurusd_features.parquet",
                'output_demo_file': "01_data/processed/eurusd_features_demo.parquet",
                'demo_size': 10000,
                'missing_values': {'strategy': 'keep_all', 'min_periods_required': 200, 'drop_all_nan_only': False},
                'validation': {'check_duplicates': True, 'check_sorting': True, 'max_missing_ratio': 0.1}
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Настраивает логирование."""
        logger = logging.getLogger('FeatureEngineering')
        
        if not logger.handlers:  # Избегаем дублирования handlers
            # Уровень логирования
            log_level = self.config.get('pipeline_settings', {}).get('logging', {}).get('level', 'INFO')
            logger.setLevel(getattr(logging, log_level))
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (если указан путь)
            log_file = self.config.get('pipeline_settings', {}).get('logging', {}).get('log_file')
            if log_file:
                try:
                    # Определяем корневую папку проекта
                    current_file = Path(__file__).resolve()
                    project_root = current_file.parent.parent.parent
                    
                    # Создаем абсолютный путь к логу от корня проекта
                    log_path = project_root / log_file
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    
                    logger.debug(f"Logging to: {log_path}")
                except Exception as e:
                    logger.warning(f"Could not setup file logging: {e}")
        
        return logger
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Загружает и валидирует исходные данные.
        
        Args:
            file_path: Путь к файлу с данными
            
        Returns:
            DataFrame с загруженными и валидированными данными
        """
        if file_path is None:
            file_path = self.config['pipeline_settings']['input_file']
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            # Загружаем данные
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Базовая валидация
            df = self._validate_and_prepare_data(df)
            
            # Обновляем статистику
            self.stats['original_columns'] = len(df.columns)
            self.stats['data_shape'] = df.shape
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидирует и подготавливает данные для обработки."""
        self.logger.info("Validating and preparing data...")
        
        # Проверяем обязательные колонки
        expected_cols = self.config.get('metadata', {}).get('expected_input_columns', 
                                                           ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            # Попытка автоматического маппинга
            column_mapping = self._auto_map_columns(df.columns, expected_cols)
            if column_mapping:
                df = df.rename(columns=column_mapping)
                self.logger.info(f"Auto-mapped columns: {column_mapping}")
                # Перепроверяем какие колонки всё ещё отсутствуют после маппинга
                missing_cols = [col for col in expected_cols if col not in df.columns]
            
            # Выводим WARNING только если после маппинга всё ещё есть недостающие колонки
            if missing_cols:
                available_cols = df.columns.tolist()
                self.logger.warning(f"Missing expected columns after auto-mapping: {missing_cols}")
                self.logger.info(f"Available columns: {available_cols}")
        
        # Обработка временной колонки
        if 'Time' in df.columns:
            try:
                df['Time'] = pd.to_datetime(df['Time'])
                df = df.set_index('Time')
                self.logger.info("Time column converted to datetime index")
            except Exception as e:
                self.logger.warning(f"Could not process Time column: {e}")
        
        # Проверка на дубликаты только по времени (более точная логика)
        validation_config = self.config.get('pipeline_settings', {}).get('validation', {})
        if validation_config.get('check_duplicates', True) and isinstance(df.index, pd.DatetimeIndex):
            # Проверяем дубликаты только если они действительно есть в том же времени
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate rows with same timestamp. Removing...")
                # Подробная информация о дубликатах
                dup_times = df.index[df.index.duplicated(keep=False)]
                if len(dup_times) > 0:
                    unique_dup_times = dup_times.value_counts().head(3)
                    self.logger.info(f"Examples of duplicated timestamps: {dict(unique_dup_times)}")
                df = df[~df.index.duplicated(keep='first')]
        
        # Проверка сортировки по времени
        if validation_config.get('check_sorting', True) and isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                self.logger.info("Sorting data by time...")
                df = df.sort_index()
        
        # Проверка на пропущенные значения
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        max_missing_ratio = validation_config.get('max_missing_ratio', 0.1)
        
        if missing_ratio > max_missing_ratio:
            self.logger.warning(f"High missing values ratio: {missing_ratio:.2%} > {max_missing_ratio:.2%}")
        
        self.logger.info(f"Data validation completed. Final shape: {df.shape}")
        return df
    
    def _auto_map_columns(self, available_cols: List[str], expected_cols: List[str]) -> Dict[str, str]:
        """Автоматическое сопоставление названий колонок."""
        mapping = {}
        
        # Словарь возможных вариантов названий
        variants = {
            'Open': ['open', 'OPEN', 'o', 'O'],
            'High': ['high', 'HIGH', 'h', 'H'],
            'Low': ['low', 'LOW', 'l', 'L'],
            'Close': ['close', 'CLOSE', 'c', 'C'],
            'Volume': ['volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V'],
            'Time': ['time', 'TIME', 'Date', 'date', 'DATE', 'Datetime', 'datetime', 'DATETIME', 'timestamp', 'Timestamp']
        }
        
        for expected_col in expected_cols:
            if expected_col in variants:
                for variant in variants[expected_col]:
                    if variant in available_cols:
                        mapping[variant] = expected_col
                        break
        
        return mapping
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает финальный набор признаков для MVP согласно спецификации.
        
        Args:
            df: DataFrame с исходными данными
            
        Returns:
            DataFrame с 33 признаками (колонка "close" и признаки вида "f_*")
        """
        self.logger.info("Starting MVP feature creation...")

        start_time = pd.Timestamp.now()
        original_shape = df.shape

        # Защита: требуем базовые колонки
        base_required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_base = [c for c in base_required if c not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required columns for feature engineering: {missing_base}")

        # 1) Базовые тех. серии, которые будут использованы далее
        # EMA 10/20/50/200
        try:
            df = self.technical_indicators.add_ema(df, periods=[10, 20, 50, 200])
        except Exception as e:
            self.logger.error(f"Failed to compute EMA: {e}")
            raise

        # ATR14 и ATR20 (сохраняем отдельно)
        try:
            df = self.technical_indicators.add_atr(df, period=14)
            atr14 = df['ATR'].copy()
            df = self.technical_indicators.add_atr(df, period=20)
            atr20 = df['ATR'].copy()
        except Exception as e:
            self.logger.error(f"Failed to compute ATR: {e}")
            raise

        # Bollinger Bands 20
        try:
            df = self.technical_indicators.add_bollinger_bands(df, period=20, std_dev=2.0)
        except Exception as e:
            self.logger.error(f"Failed to compute Bollinger Bands: {e}")
            raise

        # Keltner Channel (EMA20 + ATR20)
        try:
            df = self.technical_indicators.add_keltner_channel(df, ema_period=20, atr_period=20, multiplier=2.0)
        except Exception as e:
            self.logger.error(f"Failed to compute Keltner Channel: {e}")
            raise

        # Donchian 20
        try:
            df = self.technical_indicators.add_donchian_channel(df, period=20)
        except Exception as e:
            self.logger.error(f"Failed to compute Donchian Channel: {e}")
            raise

        # MACD (12,26,9)
        try:
            df = self.technical_indicators.add_macd(df, fast=12, slow=26, signal=9)
        except Exception as e:
            self.logger.error(f"Failed to compute MACD: {e}")
            raise

        # ADX/DI (14)
        try:
            df = self.technical_indicators.add_adx(df, period=14)
        except Exception as e:
            self.logger.error(f"Failed to compute ADX/DI: {e}")
            raise

        # PSAR
        try:
            df = self.technical_indicators.add_parabolic_sar(df)
        except Exception as e:
            self.logger.error(f"Failed to compute PSAR: {e}")
            raise

        # RSI (5 и 14)
        try:
            # сначала 5, сохраняем
            df = self.technical_indicators.add_rsi(df, period=5)
            df['RSI_5'] = df['RSI']
            # затем 14, сохраняем
            df = self.technical_indicators.add_rsi(df, period=14)
            df['RSI_14'] = df['RSI']
        except Exception as e:
            self.logger.error(f"Failed to compute RSI: {e}")
            raise

        # 2) Вспомогательные вычисления
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Лог-доходности
        log_ret_1 = np.log(close / close.shift(1))

        # Интра-дневной (anchored per day) VWAP: cum(price*vol)/cum(vol) по дням
        if isinstance(df.index, pd.DatetimeIndex):
            grp = df.index.normalize()
            vwap_num = (close * volume).groupby(grp).cumsum()
            vwap_den = volume.groupby(grp).cumsum()
            vwap = vwap_num / vwap_den.replace(0, np.nan)
        else:
            # fallback: 20-bar rolling VWAP
            vwap = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()

        # 3) Сборка итоговых признаков (33 колонки)
        eps = 1e-12
        features = pd.DataFrame(index=df.index)

        # Базовый уровень цены
        features['close'] = close

        # EMA
        features['f_ema_10'] = df.get('EMA_10')
        features['f_ema_20'] = df.get('EMA_20')
        features['f_ema_50'] = df.get('EMA_50')
        features['f_ema_200'] = df.get('EMA_200')

        # Трендовые
        features['f_close_minus_ema_20_over_atr14'] = np.where(
            atr14.abs() > eps, (close - features['f_ema_20']) / atr14, 0.0
        )
        features['f_ema20_slope'] = features['f_ema_20'].diff()
        features['f_macd_12_26_9_hist'] = df.get('MACD_Hist')
        features['f_di_diff_14'] = df.get('DI_Plus') - df.get('DI_Minus')
        features['f_adx_14'] = df.get('ADX')

        # Импульс / осцилляторы
        features['f_rsi_5'] = df.get('RSI_5')
        features['f_rsi_14'] = df.get('RSI_14')
        features['f_roc_5'] = close.pct_change(periods=5) * 100.0

        # Волатильность и диапазон
        features['f_atr_14_pct'] = np.where(close.abs() > eps, atr14 / close, np.nan)
        features['f_return_std_20'] = log_ret_1.rolling(window=20).std()
        features['f_range_over_atr'] = np.where(
            atr14.abs() > eps, (high - low) / atr14, np.nan
        )

        dc_low = df.get('DC_Lower')
        dc_up = df.get('DC_Upper')
        dc_range = (dc_up - dc_low)
        features['f_donchian_pos_20'] = np.where(
            dc_range.abs() > eps, (close - dc_low) / dc_range, 0.5
        )

        features['f_keltner_pos_20'] = np.where(
            (atr20 * 2).abs() > eps, (close - features['f_ema_20']) / (2.0 * atr20), 0.0
        )

        bb_low = df.get('BB_Lower')
        bb_up = df.get('BB_Upper')
        bb_mid = df.get('BB_Middle')
        bb_width = (bb_up - bb_low)
        features['f_percentB_20'] = np.where(
            bb_width.abs() > eps, (close - bb_low) / bb_width, 0.5
        )
        features['f_bandwidth_20'] = np.where(
            bb_mid.abs() > eps, bb_width / bb_mid, np.nan
        )

        # Статистика / лаги
        def log_ret(period: int) -> pd.Series:
            return np.log(close / close.shift(period))

        features['f_ret_1'] = log_ret(1)
        features['f_ret_6'] = log_ret(6)
        features['f_ret_12'] = log_ret(12)
        features['f_ret_24'] = log_ret(24)

        mean20 = features['f_ret_1'].rolling(window=20).mean()
        std20 = features['f_ret_1'].rolling(window=20).std().replace(0, np.nan)
        features['f_ret_1_z20'] = (features['f_ret_1'] - mean20) / std20

        features['f_rolling_mean_ret_5'] = features['f_ret_1'].rolling(window=5).mean()
        features['f_rolling_std_ret_5'] = features['f_ret_1'].rolling(window=5).std()

        mean50 = close.rolling(window=50).mean()
        std50 = close.rolling(window=50).std().replace(0, np.nan)
        features['f_zscore_close_50'] = (close - mean50) / std50

        # Прочие
        features['f_vwap_dev_over_atr14'] = np.where(
            atr14.abs() > eps, (close - vwap) / atr14, np.nan
        )
        if 'PSAR' in df.columns:
            features['f_psar_trend'] = (close > df['PSAR']).astype(int)
        else:
            features['f_psar_trend'] = np.nan

        # Календарные признаки
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
            features['f_hour_of_day_sin'] = np.sin(2 * np.pi * hours / 24.0)
            features['f_hour_of_day_cos'] = np.cos(2 * np.pi * hours / 24.0)
            features['f_dow'] = df.index.dayofweek
        else:
            features['f_hour_of_day_sin'] = np.nan
            features['f_hour_of_day_cos'] = np.nan
            features['f_dow'] = np.nan

        # Удалим возможные бесконечности
        features = features.replace([np.inf, -np.inf], np.nan)

        # Обработка пропусков по стратегии из конфигурации
        features = self._handle_missing_values(features)

        # Статистика
        end_time = pd.Timestamp.now()
        self.stats.update({
            'created_features': len(features.columns),
            'total_columns': len(features.columns),
            'processing_time': (end_time - start_time).total_seconds(),
            'data_shape': features.shape
        })

        self.logger.info(f"MVP features created in {self.stats['processing_time']:.2f} seconds")
        self.logger.info(f"Final feature set shape: {features.shape}")

        return features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает пропущенные значения согласно конфигурации."""
        self.logger.info("Handling missing values...")
        
        missing_config = self.config.get('pipeline_settings', {}).get('missing_values', {})
        strategy = missing_config.get('strategy', 'drop')
        min_periods = missing_config.get('min_periods_required', 200)
        drop_all_nan_only = missing_config.get('drop_all_nan_only', False)
        
        initial_shape = df.shape
        initial_missing = df.isnull().sum().sum()
        
        self.logger.info(f"Initial missing values: {initial_missing:,} ({initial_missing/(df.shape[0]*df.shape[1])*100:.2f}%)")
        
        if strategy == 'keep_all':
            # Не удаляем никакие строки, сохраняем все данные с NaN
            self.logger.info("Keeping all rows including those with missing values")
        elif strategy == 'drop':
            if drop_all_nan_only:
                # Удаляем только строки где ВСЕ значения NaN
                before_rows = len(df)
                df = df.dropna(how='all')
                after_rows = len(df)
                self.logger.info(f"Dropped {before_rows - after_rows} rows where ALL values were NaN")
                
                # Теперь удаляем строки где слишком много NaN в признаках (исключая исходные OHLCV)
                original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                feature_cols = [col for col in df.columns if col not in original_cols]
                
                if feature_cols:
                    # Удаляем строки где более 50% признаков NaN
                    missing_threshold = len(feature_cols) * 0.5
                    before_rows = len(df)
                    df = df[df[feature_cols].isnull().sum(axis=1) <= missing_threshold]
                    after_rows = len(df)
                    self.logger.info(f"Dropped {before_rows - after_rows} rows with >50% missing features")
            else:
                # Стандартное удаление всех строк с любыми NaN
                df = df.dropna()
        elif strategy == 'forward_fill':
            # Заполнение вперед
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            # Заполнение назад
            df = df.fillna(method='bfill')
        
        # Убеждаемся, что у нас достаточно данных
        if len(df) < min_periods:
            self.logger.warning(f"Insufficient data after cleaning: {len(df)} < {min_periods}")
        
        dropped_rows = initial_shape[0] - df.shape[0]
        final_missing = df.isnull().sum().sum()
        
        if dropped_rows > 0:
            self.logger.info(f"Dropped {dropped_rows} rows with missing values")
        
        self.logger.info(f"Final missing values: {final_missing:,} ({final_missing/(df.shape[0]*df.shape[1])*100:.2f}%)")
        
        return df
    
    def save_results(self, df: pd.DataFrame, 
                    output_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Сохраняет результаты в формате Parquet.
        
        Args:
            df: DataFrame с признаками
            output_path: Путь для сохранения (опционально)
            create_demo: Создавать ли демо-файл
            
        Returns:
            Кортеж (путь_к_полному_файлу, путь_к_демо_файлу)
        """
        self.logger.info("Saving results...")
        
        if output_path is None:
            output_path = self.config['pipeline_settings']['output_file']
        
        # Создаем директорию если нужно
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки Parquet
        parquet_settings = self.config.get('pipeline_settings', {}).get('parquet_settings', {})
        engine = parquet_settings.get('engine', 'pyarrow')
        compression = parquet_settings.get('compression', 'snappy')
        index = parquet_settings.get('index', True)
        
        try:
            # Сохраняем полный файл
            df.to_parquet(
                output_path,
                engine=engine,
                compression=compression,
                index=index
            )
            self.logger.info(f"Full dataset saved to: {output_path}")

            # Больше не создаём demo-файл; возвращаем None на его месте
            return output_path, None
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def run_full_pipeline(self, input_path: Optional[str] = None, 
                         output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Запускает полный пайплайн обработки.
        
        Args:
            input_path: Путь к входным данным
            output_path: Путь для сохранения результатов
            
        Returns:
            Кортеж (DataFrame с признаками, статистика выполнения)
        """
        self.logger.info("="*60)
        self.logger.info("STARTING FULL FEATURE ENGINEERING PIPELINE")
        self.logger.info("="*60)
        
        try:
            # 1. Загрузка данных
            df = self.load_data(input_path)
            
            # 2. Создание признаков
            df_with_features = self.create_features(df)
            
            # 3. Сохранение результатов
            full_path, demo_path = self.save_results(df_with_features, output_path)
            
            # 4. Финальная статистика
            self._log_final_stats(df_with_features)
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            
            return df_with_features, self.stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _log_final_stats(self, df: pd.DataFrame):
        """Выводит финальную статистику выполнения."""
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("="*50)
        self.logger.info(f"Original columns: {self.stats['original_columns']}")
        self.logger.info(f"Created features: {self.stats['created_features']}")
        self.logger.info(f"Total columns: {self.stats['total_columns']}")
        self.logger.info(f"Final dataset shape: {self.stats['data_shape']}")
        self.logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Информация о пропущенных значениях
        missing_count = df.isnull().sum().sum()
        missing_ratio = missing_count / (df.shape[0] * df.shape[1])
        self.logger.info(f"Missing values: {missing_count} ({missing_ratio:.2%})")
        
        self.logger.info("="*50)
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Проводит базовый анализ важности признаков.
        
        Returns:
            Словарь с метриками важности признаков
        """
        self.logger.info("Conducting feature importance analysis...")
        
        # Исключаем исходные OHLCV колонки из анализа
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in original_cols]
        
        analysis = {
            'total_features': len(feature_cols),
            'feature_types': {
                'technical': len([col for col in feature_cols if any(indicator in col for indicator in 
                                ['EMA', 'MACD', 'RSI', 'BB_', 'ATR', 'ADX', 'CCI', 'OBV'])]),
                'statistical': len([col for col in feature_cols if any(stat in col for stat in 
                                  ['ROC', 'Rolling', 'ZScore', 'Skew', 'Kurt', 'Volatility'])]),
                'lag': len([col for col in feature_cols if 'Lag' in col or 'Seasonal' in col])
            },
            'missing_values_by_type': {},
            'correlation_summary': {}
        }
        
        # Анализ пропущенных значений по типам
        for feature_type, count in analysis['feature_types'].items():
            type_cols = []
            if feature_type == 'technical':
                type_cols = [col for col in feature_cols if any(indicator in col for indicator in 
                           ['EMA', 'MACD', 'RSI', 'BB_', 'ATR', 'ADX', 'CCI', 'OBV'])]
            elif feature_type == 'statistical':
                type_cols = [col for col in feature_cols if any(stat in col for stat in 
                           ['ROC', 'Rolling', 'ZScore', 'Skew', 'Kurt', 'Volatility'])]
            elif feature_type == 'lag':
                type_cols = [col for col in feature_cols if 'Lag' in col or 'Seasonal' in col]
            
            if type_cols:
                missing_ratio = df[type_cols].isnull().sum().sum() / (len(type_cols) * len(df))
                analysis['missing_values_by_type'][feature_type] = missing_ratio
        
        return analysis