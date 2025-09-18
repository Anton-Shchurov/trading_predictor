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
import re
import fnmatch

from .technical_indicators import TechnicalIndicators
from .statistical_features import StatisticalFeatures
from .lag_features import LagFeatures
from .target_labels import create_binary_labels


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
            
            # Применяем реестр датасетов (dataset.active/items)
            config = self._apply_dataset_overrides(config)
            
            return config
        
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_project_root(self) -> Path:
        """Возвращает корневую папку проекта (относительно этого файла)."""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent
    
    def _resolve_to_project_root(self, path_str: str) -> Path:
        """Преобразует относительный путь в абсолютный от корня проекта."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._get_project_root() / p
    
    def _apply_dataset_overrides(self, config: Dict) -> Dict:
        """Подменяет input_file и метаданные согласно dataset.active/items в YAML."""
        try:
            dataset_cfg = (config.get('dataset') or {})
            active_key = dataset_cfg.get('active')
            items = dataset_cfg.get('items') or {}
            active_ds = items.get(active_key) if isinstance(items, dict) else None
            if active_ds:
                ds_name = active_ds.get('name') or active_key
                ds_path = active_ds.get('path')
                # Обновляем входной файл пайплайна
                if ds_path:
                    config.setdefault('pipeline_settings', {})
                    config['pipeline_settings']['input_file'] = ds_path
                # Прокидываем имя датасета в метаданные
                metadata = config.setdefault('metadata', {})
                metadata.setdefault('dataset_name', ds_name)
        except Exception:
            # Мягкая деградация: в случае любых проблем просто возвращаем исходную конфигурацию
            return config
        return config
    
    def _slugify(self, text: str) -> str:
        """Простая нормализация строки под имя файла."""
        text = (text or '').strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text, flags=re.IGNORECASE)
        text = re.sub(r"_+", "_", text)
        return text.strip('_')
    
    def _resolve_feature_set(self) -> Tuple[Optional[str], Dict[str, List[str]]]:
        """Возвращает активный набор признаков и сводные include/exclude паттерны с учетом наследования."""
        pipeline_cfg = self.config.get('pipeline_settings') or {}
        fs_name = pipeline_cfg.get('feature_set')
        sets = self.config.get('feature_sets') or {}
        if not fs_name or not isinstance(sets, dict) or fs_name not in sets:
            return None, {'include': [], 'exclude': []}
        resolved: Dict[str, List[str]] = {'include': [], 'exclude': []}
        visited: set = set()
        
        def collect(name: str):
            if not name or name in visited:
                return
            node = sets.get(name) or {}
            visited.add(name)
            parent = node.get('inherit_from')
            if parent:
                collect(parent)
            resolved['include'] += node.get('include_patterns') or node.get('include') or []
            resolved['exclude'] += node.get('exclude_patterns') or node.get('exclude') or []
        
        collect(fs_name)
        return fs_name, resolved
    
    def _filter_columns_by_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Отбирает колонки по активному набору признаков:
        - Всегда сохраняет 'Close' и целевую колонку (если есть)
        - Из OHLCV сохраняется только 'Close'
        - Остальные колонки фильтруются по include/exclude маскам набора
        """
        fs_name, patterns = self._resolve_feature_set()
        
        # Базовые колонки: только Close
        selected_columns: List[str] = []
        base_keep = {'Close'}
        
        # Добавляем целевую колонку если она уже есть в данных
        tgt_cfg = self.config.get('target') or {}
        target_name = tgt_cfg.get('name', 'y_bs')
        if target_name in df.columns:
            base_keep.add(target_name)
        
        # Колонки OHLCV, которые исключаем (кроме Close)
        ohlcv_exclude = {'Open', 'High', 'Low', 'Volume'}
        
        include_patterns = patterns['include'] if fs_name else ['*']
        exclude_patterns = patterns['exclude'] if fs_name else []
        
        def match_any(name: str, pats: List[str]) -> bool:
            return any(fnmatch.fnmatch(name, p) for p in pats) if pats else False
        
        for col in df.columns:
            if col in base_keep:
                selected_columns.append(col)
                continue
            if col in ohlcv_exclude:
                continue
            # Остальные колонки — это признаки
            if match_any(col, include_patterns) and not match_any(col, exclude_patterns):
                selected_columns.append(col)
        
        # Гарантируем порядок столбцов как в исходном df
        filtered = df.loc[:, [c for c in df.columns if c in selected_columns]]
        self.logger.info(f"Feature set applied: {fs_name or 'all'} | Columns kept: {len(filtered.columns)}/{len(df.columns)}")
        return filtered
    
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
        # Нормализуем путь относительно корня проекта
        resolved_path = self._resolve_to_project_root(file_path)
        
        self.logger.info(f"Loading data from: {resolved_path}")
        
        try:
            # Загружаем данные
            df = pd.read_csv(str(resolved_path))
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Базовая валидация
            df = self._validate_and_prepare_data(df)
            
            # Обновляем статистику
            self.stats['original_columns'] = len(df.columns)
            self.stats['data_shape'] = df.shape
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {resolved_path}")
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
        Создает признаки строго по параметрам YAML-конфига.
        
        Args:
            df: DataFrame с исходными данными (OHLCV).
            
        Returns:
            DataFrame с исходными колонками и добавленными техническими,
            статистическими и лаговыми признаками согласно конфигу.
        """
        self.logger.info("Starting config-driven feature creation...")

        start_time = pd.Timestamp.now()
        original_columns = list(df.columns)

        # Требуем базовые колонки
        base_required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_base = [c for c in base_required if c not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required columns for feature engineering: {missing_base}")

        # Конфигурации для блоков
        ti_cfg = self.config.get('technical_indicators', {}) or {}
        sf_cfg = self.config.get('statistical_features', {}) or {}
        lf_cfg = self.config.get('lag_features', {}) or {}

        # 1) Технические индикаторы
        try:
            df = self.technical_indicators.add_all(df, ti_cfg)
        except Exception as e:
            self.logger.error(f"Failed to compute technical indicators: {e}")
            raise

        # 2) Статистические признаки
        try:
            df = self.statistical_features.add_all(df, sf_cfg)
        except Exception as e:
            self.logger.error(f"Failed to compute statistical features: {e}")
            raise

        # 3) Лаг-признаки
        try:
            df = self.lag_features.add_all(df, lf_cfg)
        except Exception as e:
            self.logger.error(f"Failed to compute lag features: {e}")
            raise

        # Очистка бесконечностей и обработка пропусков
        df = df.replace([np.inf, -np.inf], np.nan)
        df = self._handle_missing_values(df)

        # Обновление статистики
        end_time = pd.Timestamp.now()
        created_count = len(df.columns) - len(original_columns)
        self.stats.update({
            'created_features': max(created_count, 0),
            'total_columns': len(df.columns),
            'processing_time': (end_time - start_time).total_seconds(),
            'data_shape': df.shape
        })

        self.logger.info(
            f"Config-driven features created in {self.stats['processing_time']:.2f} seconds"
        )
        self.logger.info(f"Final dataset shape (with features): {df.shape}")

        return df
    
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
            # Не заполняем пропуски, сохраняем исходные NaN для последующей фильтрации на следующих этапах
            self.logger.info("Preserving missing values (keep_all mode): no filling, no dropping at this stage")
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
            # Также заполняем ведущие NaN назад, чтобы не осталось пропусков в начале
            df = df.fillna(method='bfill')
        elif strategy == 'backward_fill':
            # Заполнение назад
            df = df.fillna(method='bfill')
            df = df.fillna(method='ffill')
        
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
        
        # Базовое имя файла
        base_output = output_path or (self.config.get('pipeline_settings', {}) or {}).get('output_file')
        if not base_output:
            # Формируем дефолтное имя на основе имени датасета
            ds_name = (self.config.get('metadata', {}) or {}).get('dataset_name') \
                or (self.config.get('dataset', {}) or {}).get('active', 'dataset')
            ds_slug = self._slugify(ds_name)
            base_output = f"01_data/processed/{ds_slug}__features.parquet"

        # Добавляем суффикс набора признаков к имени файла, чтобы не перезатирать разные версии
        fs_name = (self.config.get('pipeline_settings') or {}).get('feature_set')
        p = Path(base_output)
        stem = p.stem
        if fs_name and fs_name not in stem:
            if stem.endswith('_features'):
                new_stem = f"{stem[:-len('_features')]}_{fs_name}_features"
            else:
                new_stem = f"{stem}_{fs_name}"
            p = p.with_name(new_stem + p.suffix)

        # Нормализуем путь сохранения
        resolved_output = self._resolve_to_project_root(str(p))
        
        # Создаем директорию если нужно
        output_dir = resolved_output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки Parquet
        parquet_settings = self.config.get('pipeline_settings', {}).get('parquet_settings', {})
        engine = parquet_settings.get('engine', 'pyarrow')
        compression = parquet_settings.get('compression', 'snappy')
        index = parquet_settings.get('index', True)
        
        try:
            # Сохраняем полный файл
            df.to_parquet(
                str(resolved_output),
                engine=engine,
                compression=compression,
                index=index
            )
            self.logger.info(f"Full dataset saved to: {resolved_output}")

            return str(resolved_output), None
            
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

            # 2.1 Создание целевой переменной (добавляем прямо в итоговый датасет)
            tgt_cfg = (self.config.get('target') or {})
            if tgt_cfg.get('type') == 'binary_buy_sell':
                self.logger.info("Creating binary target y_bs...")
                labels = create_binary_labels(
                    close=df_with_features['Close'],
                    horizon=int(tgt_cfg.get('horizon', 5)),
                    return_debug=False,
                )
                target_name = tgt_cfg.get('name', 'y_bs')
                # Присоединяем и приводим тип
                df_with_features[target_name] = labels['y_bs']
                # Удаляем хвостовые NaN по целевой (последние H баров)
                before_rows = len(df_with_features)
                df_with_features = df_with_features[df_with_features[target_name].notna()].copy()
                df_with_features[target_name] = df_with_features[target_name].astype('Int8')
                self.logger.info(f"Target '{target_name}' created. Dropped {before_rows - len(df_with_features)} tail rows without label")
            
            # 3. Фильтрация признаков по активному набору
            df_selected = self._filter_columns_by_feature_set(df_with_features)

            # 3.1 Обновляем статистику под выбранный набор
            tgt_cfg = (self.config.get('target') or {})
            target_name = tgt_cfg.get('name', 'y_bs')
            base_cols = {'Close'}
            if target_name in df_selected.columns:
                base_cols.add(target_name)
            created_included = len([c for c in df_selected.columns if c not in base_cols])
            self.stats['created_features'] = created_included
            self.stats['total_columns'] = len(df_selected.columns)
            self.stats['data_shape'] = df_selected.shape

            # 4. Сохранение результатов
            full_path, demo_path = self.save_results(df_selected, output_path)
            
            # 5. Финальная статистика
            self._log_final_stats(df_selected)
            
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