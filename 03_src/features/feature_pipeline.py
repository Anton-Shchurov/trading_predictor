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
            current_dir = Path(__file__).parent.parent.parent
            config_path = current_dir / "04_configs" / "feature_engineering.yml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Применяем профиль
            if self.profile in config.get('profiles', {}):
                profile_config = config['profiles'][self.profile]
                if profile_config.get('inherit_from') != 'default':
                    for section, values in profile_config.items():
                        if section in config and isinstance(values, dict):
                            config[section].update(values)
            
            # Применяем реестр датасетов
            config = self._apply_dataset_overrides(config)
            return config
        
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using default configuration.")
            return {}
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
            return {}
    
    def _get_project_root(self) -> Path:
        """Возвращает корневую папку проекта."""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent
    
    def _resolve_to_project_root(self, path_str: str) -> Path:
        """Преобразует относительный путь в абсолютный."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._get_project_root() / p
    
    def _apply_dataset_overrides(self, config: Dict) -> Dict:
        """Подменяет input_file и метаданные согласно dataset.active."""
        try:
            dataset_cfg = (config.get('dataset') or {})
            active_key = dataset_cfg.get('active')
            items = dataset_cfg.get('items') or {}
            active_ds = items.get(active_key) if isinstance(items, dict) else None
            if active_ds:
                ds_name = active_ds.get('name') or active_key
                ds_path = active_ds.get('path')
                if ds_path:
                    config.setdefault('pipeline_settings', {})
                    config['pipeline_settings']['input_file'] = ds_path
                metadata = config.setdefault('metadata', {})
                metadata.setdefault('dataset_name', ds_name)
        except Exception:
            return config
        return config
    
    def _slugify(self, text: str) -> str:
        """Простая нормализация строки."""
        text = (text or '').strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text, flags=re.IGNORECASE)
        text = re.sub(r"_+", "_", text)
        return text.strip('_')
    
    def _resolve_feature_set(self) -> Tuple[Optional[str], Dict[str, List[str]]]:
        """Возвращает активный набор признаков с учетом наследования."""
        pipeline_cfg = self.config.get('pipeline_settings') or {}
        fs_name = pipeline_cfg.get('feature_set')
        sets = self.config.get('feature_sets') or {}
        if not fs_name or not isinstance(sets, dict) or len(sets) == 0:
            return None, {'include': [], 'exclude': []}

        if not re.match(r'^fset-\d+$', str(fs_name)):
            msg = f"Некорректное имя набора признаков: '{fs_name}'. Используйте формат 'fset-N'."
            self.logger.error(msg)
            raise ValueError(msg)

        canonical_name = None
        if fs_name in sets:
            canonical_name = fs_name
        else:
            alt = fs_name.replace('-', '_')
            if alt in sets:
                canonical_name = alt
        
        if canonical_name is None:
            raise KeyError(f"Набор признаков '{fs_name}' не найден.")
            
        resolved = {'include': [], 'exclude': []}
        visited = set()
        
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
        return fs_name, resolved
    
    def _filter_columns_by_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Отбирает колонки по активному набору признаков."""
        fs_name, patterns = self._resolve_feature_set()
        if not fs_name:
            return df

        selected_columns = []
        ps = (self.config.get('pipeline_settings', {}) or {})
        keep_close = bool(ps.get('always_include_close', True))
        keep_target = bool(ps.get('include_target', True))
        
        base_keep = set()
        if keep_close: base_keep.add('close')
        target_name = 'y_bs'
        if keep_target: base_keep.add(target_name)
        for col in ps.get('base_keep_columns', []) or []:
             if isinstance(col, str): base_keep.add(col)

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
        
        if keep_target and target_name in df.columns and target_name not in selected_columns:
            selected_columns.append(target_name)

        filtered = df[selected_columns]
        self.logger.info(f"Feature set applied: '{fs_name}' | Columns kept: {len(filtered.columns)}/{len(df.columns)}")
        return filtered

    def _setup_logging(self) -> logging.Logger:
        """Настраивает логирование."""
        logger = logging.getLogger('FeatureEngineering')
        if not logger.handlers:
            log_level = self.config.get('pipeline_settings', {}).get('logging', {}).get('level', 'INFO')
            logger.setLevel(getattr(logging, log_level))
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Загружает и валидирует исходные данные."""
        if file_path is None:
            file_path = self.config['pipeline_settings']['input_file']
        resolved_path = self._resolve_to_project_root(file_path)
        self.logger.info(f"Loading data from: {resolved_path}")
        
        try:
            df = pd.read_csv(str(resolved_path))
            self.logger.info(f"Data loaded. Shape: {df.shape}")
            df = self._validate_and_prepare_data(df)
            self.stats['original_columns'] = len(df.columns)
            self.stats['data_shape'] = df.shape
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидирует и подготавливает данные."""
        self.logger.info("Validating data...")
        expected_cols = self.config.get('metadata', {}).get('expected_input_columns', ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            column_mapping = self._auto_map_columns(df.columns, expected_cols)
            if column_mapping:
                df = df.rename(columns=column_mapping)
                self.logger.info(f"Auto-mapped columns: {column_mapping}")
        
        if 'Time' in df.columns:
            try:
                df['Time'] = pd.to_datetime(df['Time'])
                df = df.set_index('Time')
            except Exception as e:
                self.logger.warning(f"Could not process Time column: {e}")
        
        validation_config = self.config.get('pipeline_settings', {}).get('validation', {})
        if validation_config.get('check_duplicates', True) and isinstance(df.index, pd.DatetimeIndex):
            df = df[~df.index.duplicated(keep='first')]
        
        if validation_config.get('check_sorting', True) and isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
        
        return df
    
    def _auto_map_columns(self, available_cols: List[str], expected_cols: List[str]) -> Dict[str, str]:
        """Автоматическое сопоставление названий колонок."""
        mapping = {}
        variants = {
            'Open': ['open', 'OPEN', 'o', 'O'],
            'High': ['high', 'HIGH', 'h', 'H'],
            'Low': ['low', 'LOW', 'l', 'L'],
            'Close': ['close', 'CLOSE', 'c', 'C'],
            'Volume': ['volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V'],
            'Time': ['time', 'TIME', 'Date', 'date', 'DATE', 'Timestamp']
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
        Строит и выполняет DAG генерации признаков.
        ОПТИМИЗАЦИЯ: Предотвращение фрагментации DataFrame путем сбора результатов в словарь.
        """
        self.logger.info("Starting declarative feature pipeline execution...")
        definitions = self.config.get('feature_definitions', {})
        if not definitions:
            return df

        # 1. Построение графа зависимостей
        all_deps = {name: set(defn.get('needs', [])) | set(defn.get('inputs', [])) for name, defn in definitions.items()}
        graph = {name: deps for name, deps in all_deps.items() if name in definitions}
        in_degree = {name: 0 for name in graph}
        adj = {name: [] for name in graph}

        for u, deps in graph.items():
            for v_dep in deps:
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
            raise RuntimeError(f"Cycle detected in feature dependency graph.")
            
        self.logger.info(f"Execution order determined for {len(execution_order)} features.")

        # 2. Выполнение графа
        # Кэш результатов: используем словарь для хранения серий
        # Инициализируем исходными колонками
        results_cache = {col.lower(): df[col] for col in df.columns}
        results_cache.update({col: df[col] for col in df.columns})
        results_cache['index'] = df.index

        # Список для сбора финальных колонок, которые пойдут в DataFrame
        # Начинаем с исходных колонок или создаем новый DataFrame в конце?
        # Лучше создать новый чистый DataFrame для результатов, чтобы не тянуть лишнее
        
        for name in execution_order:
            if name in results_cache: continue
            
            defn = definitions[name]
            method_name = defn.get('method')
            params = defn.get('params', {})
            
            args_list = [results_cache[dep] for dep in defn.get('inputs', []) + defn.get('needs', [])]

            if method_name == 'formula':
                formula = defn['formula']
                local_scope = {'nan': np.nan, 'NA': pd.NA}
                eval_locals = {**results_cache, **local_scope}
                try:
                    result = pd.eval(formula, engine='python', local_dict=eval_locals)
                except Exception as exc:
                    self.logger.warning(f"pd.eval failed for {name}: {exc}. Fallback to eval.")
                    safe_globals = {'np': np, 'pd': pd, 'nan': np.nan, 'NA': pd.NA}
                    result = eval(formula, safe_globals, eval_locals)

            elif method_name in self._method_registry:
                func = self._method_registry[method_name]
                if method_name in ['hour_sin', 'hour_cos', 'day_of_week', 'session_flag', 'session_flag_tz']:
                    result = func(index=results_cache['index'], **params)
                else:
                    result = func(*args_list, **params)
            else:
                self.logger.warning(f"Method '{method_name}' not found for feature '{name}'")
                continue

            results_cache[name] = result
        
        # 3. Сборка финального DataFrame без фрагментации
        # Собираем данные в dict: {col_name: series}
        final_data_dict = {}
        
        # Сначала добавляем исходные данные, которые нужны (или просто индексируем по ним)
        # Если мы хотим вернуть DF с исходными данными + фичи:
        for col in df.columns:
            final_data_dict[col] = df[col]
            
        for name, defn in definitions.items():
            if not defn.get('is_intermediate', False):
                result = results_cache.get(name)
                if result is None:
                    continue
                    
                if isinstance(result, pd.DataFrame):
                    # Если функция вернула DataFrame (напр MACD), распаковываем колонки
                    for col in result.columns:
                        final_data_dict[col] = result[col]
                else:
                    final_data_dict[name] = result

        # Создаем DataFrame один раз
        self.logger.info("Constructing final DataFrame...")
        final_df = pd.DataFrame(final_data_dict, index=df.index)
        
        return final_df

    def save_results(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Сохраняет результаты в формате Parquet."""
        self.logger.info("Saving results...")
        
        base_output = output_path or (self.config.get('pipeline_settings', {}) or {}).get('output_file')
        if not base_output:
            ds_name = (self.config.get('metadata', {}) or {}).get('dataset_name') \
                or (self.config.get('dataset', {}) or {}).get('active', 'dataset')
            ds_slug = self._slugify(ds_name)
            base_output = f"01_data/processed/{ds_slug}__features.parquet"

        resolved_fs_name, _ = self._resolve_feature_set()
        fs_name = resolved_fs_name or (self.config.get('pipeline_settings') or {}).get('feature_set')
        p = Path(base_output)
        stem = p.stem
        if fs_name and fs_name not in stem:
            new_stem = f"{stem[:-len('_features')]}_{fs_name}_features" if stem.endswith('_features') else f"{stem}_{fs_name}"
            p = p.with_name(new_stem + p.suffix)

        resolved_output = self._resolve_to_project_root(str(p))
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        
        parquet_settings = self.config.get('pipeline_settings', {}).get('parquet_settings', {})
        
        # Принудительный кастинг float64 -> float32 для экономии памяти перед сохранением
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            self.logger.info(f"Casting {len(float_cols)} columns to float32 for storage optimization.")
            df[float_cols] = df[float_cols].astype('float32')

        df.to_parquet(
            str(resolved_output),
            engine=parquet_settings.get('engine', 'pyarrow'),
            compression=parquet_settings.get('compression', 'snappy'),
            index=parquet_settings.get('index', True)
        )
        self.logger.info(f"Saved to: {resolved_output}")
        return str(resolved_output), None
    
    def run_full_pipeline(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Запускает полный пайплайн."""
        self.logger.info("STARTING FULL FEATURE ENGINEERING PIPELINE")
        start_time = pd.Timestamp.now()
        
        try:
            df = self.load_data(input_path)
            
            # Декларативное создание (теперь возвращает monolithic DF)
            df_with_features = self._execute_declarative_pipeline(df)

            # Fallback target calc
            if 'y_bs' not in df_with_features.columns:
                try:
                    self.logger.warning("Target 'y_bs' missing. Computing fallback.")
                    col_close = df_with_features.get('close') if 'close' in df_with_features else df_with_features.get('Close')
                    if col_close is not None:
                         y_params = ((self.config.get('feature_definitions', {}) or {}).get('y_bs', {}) or {}).get('params', {})
                         horizon = int(y_params.get('horizon', 5))
                         y_df = create_binary_labels(col_close, horizon=horizon)
                         # join безопасно
                         df_with_features = df_with_features.join(y_df[['y_bs']])
                except Exception as err:
                    self.logger.error(f"Fallback target failed: {err}")
            
            df_selected = self._filter_columns_by_feature_set(df_with_features)
            
            self.stats['processing_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            self.stats['created_features'] = len(df_selected.columns) - 5 # Approximate
            self.stats['total_columns'] = len(df_selected.columns)
            
            self.save_results(df_selected, output_path)
            
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            return df_selected, self.stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise