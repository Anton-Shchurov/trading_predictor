"""
Главный пайплайн для Feature Engineering.
Объединяет все модули создания признаков и обеспечивает:
- Загрузку и валидацию данных
- Декларативное создание признаков на основе графа зависимостей (DAG)
- Обработку пропущенных значений
- Сохранение результатов в различных форматах
- Детальное логирование процесса
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from collections import deque
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
    
    Координирует работу всех модулей-калькуляторов,
    обеспечивает загрузку конфигурации, строит и выполняет
    граф зависимостей признаков, валидирует данные и сохраняет результаты.
    """
    
    def __init__(self, config_path: Optional[str] = None, profile: str = "full"):
        """
        Инициализация пайплайна.
        
        Args:
            config_path: Путь к файлу конфигурации
            profile: Профиль конфигурации ("quick", "full")
        """
        self.profile = profile
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Инициализация калькуляторов и регистрация методов
        self.technical_indicators = TechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        self.lag_features = LagFeatures()
        self._method_registry = self._register_methods()
        
        # Статистика выполнения
        self.stats = {
            'original_columns': 0,
            'created_features': 0,
            'total_columns': 0,
            'processing_time': 0,
            'data_shape': (0, 0)
        }

    def _register_methods(self) -> Dict[str, Any]:
        """Собирает все публичные `calculate_*` методы из калькуляторов в один словарь."""
        registry = {}
        calculators = [
            self.technical_indicators,
            self.statistical_features,
            self.lag_features
        ]
        for calc in calculators:
            for method_name in dir(calc):
                if method_name.startswith('calculate_'):
                    key = method_name.replace('calculate_', '')
                    registry[key] = getattr(calc, method_name)
        
        # Добавляем специальные/внешние методы
        registry['binary_target'] = create_binary_labels

        return registry

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
            return {}
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return {}
    
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
        if not fs_name or not isinstance(sets, dict) or len(sets) == 0:
            return None, {'include': [], 'exclude': []}

        # Строгая проверка: допускаем только формат fset-N во входном значении
        if not re.match(r'^fset-\d+$', str(fs_name)):
            msg = (
                f"Некорректное имя набора признаков: '{fs_name}'. Используйте формат 'fset-N', например 'fset-5'."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Ищем соответствующий ключ в конфиге: ключ может быть записан с подчёркиванием
        canonical_name: Optional[str] = None
        if fs_name in sets:
            canonical_name = fs_name
        else:
            alt = fs_name.replace('-', '_')
            if alt in sets:
                canonical_name = alt
        if canonical_name is None:
            maybe = [k for k in sets.keys() if isinstance(k, str) and k.replace('_', '-') == fs_name]
            hint = f" Возможно, вы имели в виду: '{maybe[0]}'" if maybe else ""
            msg = f"Набор признаков '{fs_name}' не найден в конфиге.{hint}"
            self.logger.error(msg)
            raise KeyError(msg)
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
        
        collect(canonical_name)
        # Возвращаем исходное (строгое) имя fs_name для логов/сохранения, а паттерны собраны по canonical_name
        return fs_name, resolved
    
    def _filter_columns_by_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Отбирает колонки по активному набору признаков:
        - Всегда сохраняет целевую колонку (если есть)
        - Колонку 'close' сохраняет только если включен флаг
          pipeline_settings.always_include_close, либо она явно
          попадает по include/exclude маскам набора
        - Остальные колонки фильтруются по include/exclude маскам набора
        """
        fs_name, patterns = self._resolve_feature_set()
        if not fs_name:
            return df # Если feature_set не задан, возвращаем все сгенерированные признаки

        selected_columns: List[str] = []
        
        # Базовые колонки, которые сохраняем без учёта паттернов
        ps = (self.config.get('pipeline_settings', {}) or {})
        keep_close = bool(ps.get('always_include_close', True))
        keep_target = bool(ps.get('include_target', True))
        # Дополнительные базовые колонки, которые нужно сохранить всегда
        extra_base_keep = set()
        for col in ps.get('base_keep_columns', []) or []:
            if isinstance(col, str):
                extra_base_keep.add(col)
        base_keep = set()
        if keep_close:
            base_keep.add('close')
        # Целевая метка всегда должна сохраняться, если она рассчитана
        target_name = 'y_bs'
        if keep_target and target_name in df.columns:
            base_keep.add(target_name)
        base_keep |= extra_base_keep

        include_patterns = patterns.get('include', [])
        exclude_patterns = patterns.get('exclude', [])
        
        def match_any(name: str, pats: List[str]) -> bool:
            return any(fnmatch.fnmatch(name, p) for p in pats) if pats else False
        
        for col in df.columns:
            if col in base_keep:
                selected_columns.append(col)
                continue
            
            if match_any(col, include_patterns) and not match_any(col, exclude_patterns):
                selected_columns.append(col)
        
        # Гарантируем присутствие целевой метки даже при нестандартных паттернах (если включена в настройках)
        if keep_target and target_name in df.columns and target_name not in selected_columns:
            selected_columns.append(target_name)

        filtered = df[selected_columns]
        self.logger.info(f"Feature set applied: '{fs_name}' | Columns kept: {len(filtered.columns)}/{len(df.columns)}")
        return filtered

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
            
            # Файловое логирование отключено: оставляем только вывод в консоль
        
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
    
    def _execute_declarative_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Строит и выполняет DAG генерации признаков на основе `feature_definitions`.
        """
        self.logger.info("Starting declarative feature pipeline execution...")
        definitions = self.config.get('feature_definitions', {})
        if not definitions:
            self.logger.warning("`feature_definitions` not found in config. No features will be generated.")
            return df

        # 1. Построение графа зависимостей и определение порядка выполнения
        all_deps = {name: set(defn.get('needs', [])) | set(defn.get('inputs', [])) for name, defn in definitions.items()}
        graph = {name: deps for name, deps in all_deps.items() if name in definitions}
        in_degree = {name: 0 for name in graph}
        adj = {name: [] for name in graph}

        for u, deps in graph.items():
            for v_dep in deps:
                # v_dep может быть внешней зависимостью (напр. 'Close'), которой нет в графе
                if v_dep in adj:
                    adj[v_dep].append(u)
                    in_degree[u] += 1
        
        queue = deque([name for name in graph if in_degree[name] == 0])
        execution_order = []
        while queue:
            u = queue.popleft()
            execution_order.append(u)
            for v_neighbor in adj.get(u, []):
                in_degree[v_neighbor] -= 1
                if in_degree[v_neighbor] == 0:
                    queue.append(v_neighbor)

        if len(execution_order) != len(graph):
            unprocessed = set(graph.keys()) - set(execution_order)
            raise RuntimeError(f"Cycle detected in feature dependency graph. Unprocessed nodes: {unprocessed}")
            
        self.logger.info(f"Execution order determined for {len(execution_order)} features.")

        # 2. Выполнение графа
        results_cache = {col.lower(): df[col] for col in df.columns}
        results_cache.update({col: df[col] for col in df.columns}) # Для `High`, `Low`
        results_cache['index'] = df.index

        for name in execution_order:
            if name in results_cache: continue
            
            defn = definitions[name]
            method_name = defn.get('method')
            params = defn.get('params', {})
            
            args_list = [results_cache[dep] for dep in defn.get('inputs', []) + defn.get('needs', [])]

            self.logger.debug(f"Computing feature '{name}' with method '{method_name}'")

            if method_name == 'formula':
                formula = defn['formula']
                # np.nan и pd.NA доступны в eval контексте
                local_scope = {'nan': np.nan, 'NA': pd.NA}
                eval_locals = {**results_cache, **local_scope}
                try:
                    result = pd.eval(formula, engine='python', local_dict=eval_locals)
                except Exception as exc:
                    self.logger.debug(
                        "pd.eval failed for feature '%s' with formula '%s': %s. Falling back to python eval.",
                        name,
                        formula,
                        exc,
                    )
                    safe_globals = {'np': np, 'pd': pd, 'nan': np.nan, 'NA': pd.NA}
                    result = eval(formula, safe_globals, eval_locals)



            elif method_name in self._method_registry:
                func = self._method_registry[method_name]
                # Специальная обработка для календарных фич, которым нужен индекс
                if method_name in ['hour_sin', 'hour_cos', 'day_of_week', 'session_flag', 'session_flag_tz']:
                    result = func(index=results_cache['index'], **params)
                else:
                    result = func(*args_list, **params)
            else:
                raise NotImplementedError(f"Method '{method_name}' not found in registry.")

            results_cache[name] = result
        
        # 3. Сборка финального DataFrame
        final_df = pd.DataFrame(index=df.index)
        for name, defn in definitions.items():
            if not defn.get('is_intermediate', False):
                result = results_cache[name]
                if isinstance(result, pd.DataFrame):
                    # Для методов, возвращающих DataFrame (напр., MACD, target)
                    # Если имя колонки совпадает с именем фичи, берем ее
                    if name in result.columns:
                       final_df[name] = result[name]
                    else: # Иначе присоединяем весь DataFrame (для y_bs)
                        final_df = final_df.join(result)
                else:
                    final_df[name] = result

        return final_df

    def save_results(self, df: pd.DataFrame, 
                    output_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Сохраняет результаты в формате Parquet.
        
        Args:
            df: DataFrame с признаками
            output_path: Путь для сохранения (опционально)
            
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
        # Используем нормализованное имя набора признаков (если доступно)
        resolved_fs_name, _ = self._resolve_feature_set()
        fs_name = resolved_fs_name or (self.config.get('pipeline_settings') or {}).get('feature_set')
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
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. Загрузка и валидация данных
            df = self.load_data(input_path)
            
            # 2. Декларативное создание признаков
            df_with_features = self._execute_declarative_pipeline(df)

            # 2.1 Гарантируем наличие целевой метки y_bs, даже если она не была собрана по каким-либо причинам
            if 'y_bs' not in df_with_features.columns:
                try:
                    self.logger.warning("Target 'y_bs' not found after feature generation. Computing fallback target from 'close'.")
                    # Получаем источник close
                    if 'close' in df_with_features.columns:
                        close_series = df_with_features['close']
                    elif 'Close' in df.columns:
                        close_series = df['Close']
                    elif 'close' in df.columns:
                        close_series = df['close']
                    else:
                        close_series = None

                    if close_series is not None:
                        # Читаем горизонт из конфигурации, по умолчанию 5
                        y_params = ((self.config.get('feature_definitions', {}) or {}).get('y_bs', {}) or {}).get('params', {})
                        horizon = int(y_params.get('horizon', 5))
                        y_df = create_binary_labels(close_series, horizon=horizon, return_debug=False)
                        df_with_features = df_with_features.join(y_df[['y_bs']])
                        self.logger.info("Fallback target 'y_bs' computed and appended.")
                    else:
                        self.logger.error("Cannot compute fallback 'y_bs': 'close' series not available.")
                except Exception as err:
                    self.logger.error(f"Failed to compute fallback target 'y_bs': {err}")
            
            # 3. Фильтрация признаков по активному набору
            df_selected = self._filter_columns_by_feature_set(df_with_features)
            end_time = pd.Timestamp.now()
            
            # 3.1 Обновляем статистику под выбранный набор
            # Базовые колонки, не считаем их созданными признаками
            base_cols = set()
            if 'close' in df_selected.columns:
                base_cols.add('close')
            target_name = 'y_bs' # Имя жестко задано в декларативном конфиге
            if target_name in df_selected.columns:
                base_cols.add(target_name)

            created_included = len([c for c in df_selected.columns if c not in base_cols])
            self.stats['created_features'] = created_included
            self.stats['total_columns'] = len(df_selected.columns)
            self.stats['data_shape'] = df_selected.shape
            self.stats['processing_time'] = (end_time - start_time).total_seconds()


            # 4. Сохранение результатов
            full_path, demo_path = self.save_results(df_selected, output_path)
            # Сохраняем служебные пути и активный набор признаков в статистику
            try:
                resolved_fs, _ = self._resolve_feature_set()
            except Exception:
                resolved_fs = None
            self.stats['output_path'] = full_path
            if resolved_fs:
                self.stats['feature_set'] = resolved_fs
            
            # 5. Финальная статистика
            self._log_final_stats(df_selected)
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            
            return df_selected, self.stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
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
        if df.size > 0:
            missing_ratio = missing_count / df.size
        else:
            missing_ratio = 0
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