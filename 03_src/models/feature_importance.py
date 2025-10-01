from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

# Переиспользуем валидацию и сортировку из существующего пайплайна
# Импорт из соседнего файла без обязательной пакетной структуры
try:
    from .modeling_pipeline import _ensure_sorted_index, _validate_features  # type: ignore
except Exception:
    import importlib.util
    import sys
    _mp_path = Path(__file__).resolve().parent / "modeling_pipeline.py"
    spec = importlib.util.spec_from_file_location("_modeling_pipeline", str(_mp_path))
    if spec and spec.loader:
        _mp = importlib.util.module_from_spec(spec)
        sys.modules["_modeling_pipeline"] = _mp
        spec.loader.exec_module(_mp)  # type: ignore[arg-type]
        _ensure_sorted_index = _mp._ensure_sorted_index  # type: ignore[attr-defined]
        _validate_features = _mp._validate_features  # type: ignore[attr-defined]
    else:
        raise ImportError("Не удалось импортировать modeling_pipeline для функций валидации фич")


@dataclass
class ImportanceConfig:
    scoring: str = "f1_macro"
    n_repeats: int = 10
    random_state: int = 42
    shap_max_samples: int = 3000


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_current_experiment_id(models_cfg_path: Path) -> str:
    cfg = _load_yaml(models_cfg_path)
    exp_id = (cfg.get("experiment") or {}).get("current_id")
    if not exp_id:
        raise ValueError("В конфиге моделей отсутствует experiment.current_id")
    return str(exp_id)


def _get_enabled_models(models_cfg_path: Path) -> List[str]:
    cfg = _load_yaml(models_cfg_path)
    models = []
    for name, mcfg in (cfg.get("models") or {}).items():
        if (mcfg or {}).get("enabled", False):
            models.append(name)
    return models


def _find_experiment_dir(reports_dir: Path, exp_id: str) -> Path:
    # Ищем директории вида exp_0006_YYYYMMDD_HHMMSS
    candidates = sorted(
        [p for p in reports_dir.glob(f"{exp_id}_*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Не найдена директория отчёта для эксперимента: {exp_id}")
    return candidates[0]


def _load_test_split(experiment_dir: Path, features_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    split_fp = experiment_dir / "split_dates.yml"
    if not split_fp.exists():
        raise FileNotFoundError(f"Не найден файл с информацией о разбиении: {split_fp}")
    split_meta = _load_yaml(split_fp)
    test_start = split_meta.get("test_start")
    if not test_start:
        # Если дата не указана (возможен случай с ratios без сохранения), попробуем взять последнюю треть
        test_start = None

    if not features_path.exists():
        raise FileNotFoundError(f"Не найден файл признаков: {features_path}")
    df = pd.read_parquet(features_path)
    df = _ensure_sorted_index(df)
    X_all, y_all = _validate_features(df)

    if test_start is None or str(test_start).lower() == "null":
        # Разобьём по тем же долям, что и в split_meta (если есть)
        ratios = (split_meta.get("ratios") or split_meta.get("splits", {}).get("ratios") or {"train": 0.7, "valid": 0.15, "test": 0.15})
        n = len(X_all)
        n_train = int(n * float(ratios.get("train", 0.7)))
        n_valid = int(n * float(ratios.get("valid", 0.15)))
        X_test = X_all.iloc[n_train + n_valid :]
        y_test = y_all.iloc[n_train + n_valid :]
    else:
        X_test = X_all.loc[str(test_start) :]
        y_test = y_all.loc[str(test_start) :]

    return X_test, y_test


def _load_model(model_dir: Path, model_name: str):
    # Для CatBoost предпочитаем бинарный формат .cbm, затем joblib
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Не удалось импортировать CatBoost: {e}")
        cbm_fp = model_dir / "model.cbm"
        if cbm_fp.exists():
            model = CatBoostClassifier()
            model.load_model(str(cbm_fp))
            return model

    # Универсальная загрузка через joblib (для LGBM/XGB и CatBoost, если нет .cbm)
    joblib_fp = model_dir / "model.joblib"
    if joblib_fp.exists():
        return joblib.load(joblib_fp)

    # Если это CatBoost и .cbm отсутствует, сообщаем об ошибке
    raise FileNotFoundError(f"Не найден файл модели в {model_dir}")


def _native_importance(model_name: str, model, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.Series:
    cols = X.columns
    if model_name == "lightgbm":
        # Количество сплитов (как в примере отчёта)
        try:
            import lightgbm as lgb  # noqa: F401
        except Exception:
            pass
        values = getattr(model, "feature_importances_", None)
        if values is None:
            # Попробуем через booster
            try:
                values = model.booster_.feature_importance(importance_type="split")  # type: ignore[attr-defined]
            except Exception:
                values = np.zeros(len(cols), dtype=float)
        return pd.Series(values, index=cols, dtype=float)

    if model_name == "xgboost":
        try:
            # feature_importances_ обычно нормированы
            values = getattr(model, "feature_importances_", None)
            if values is None:
                # Попробуем через booster (вернёт словарь f{idx}: score)
                booster = model.get_booster()
                score_map = booster.get_score(importance_type="gain")
                # Преобразуем в массив по порядку колонок
                values = np.array([score_map.get(f"f{i}", 0.0) for i in range(len(cols))], dtype=float)
        except Exception:
            values = np.zeros(len(cols), dtype=float)
        return pd.Series(values, index=cols, dtype=float)

    if model_name == "catboost":
        # Многоступенчатые попытки: PredictionValuesChange -> LossFunctionChange -> FeatureImportance
        try:
            from catboost import Pool  # type: ignore
        except Exception:
            return pd.Series(np.zeros(len(cols), dtype=float), index=cols, dtype=float)

        # Пытаемся создать Pool с именами признаков; метки добавляем, если доступны
        try:
            pool = Pool(X, label=y if y is not None else None, feature_names=cols.tolist())
        except Exception:
            pool = None  # если не удалось, попробуем тип без пула

        values = None  # type: ignore[assignment]

        # 1) PredictionValuesChange (требует пул и обычно метки)
        if pool is not None and y is not None:
            try:
                values = model.get_feature_importance(pool=pool, type="PredictionValuesChange")
            except Exception:
                values = None

        # 2) LossFunctionChange (также требует y)
        if values is None and pool is not None and y is not None:
            try:
                values = model.get_feature_importance(pool=pool, type="LossFunctionChange")
            except Exception:
                values = None

        # 3) FeatureImportance (не требует пула)
        if values is None:
            try:
                values = model.get_feature_importance(type="FeatureImportance")
            except Exception:
                values = None

        if values is None:
            values = np.zeros(len(cols), dtype=float)

        return pd.Series(np.asarray(values, dtype=float), index=cols, dtype=float)

    # Для неизвестной модели — нули
    return pd.Series(np.zeros(len(cols), dtype=float), index=cols, dtype=float)


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, cfg: ImportanceConfig) -> pd.Series:
    try:
        r = permutation_importance(
            model,
            X,
            y,
            scoring=cfg.scoring,
            n_repeats=cfg.n_repeats,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
        return pd.Series(r.importances_mean, index=X.columns, dtype=float)
    except Exception:
        # Повторим без параллелизма как устойчивый фолбэк
        try:
            r = permutation_importance(
                model,
                X,
                y,
                scoring=cfg.scoring,
                n_repeats=cfg.n_repeats,
                n_jobs=1,
                random_state=cfg.random_state,
            )
            return pd.Series(r.importances_mean, index=X.columns, dtype=float)
        except Exception:
            return pd.Series(np.zeros(X.shape[1], dtype=float), index=X.columns, dtype=float)


def _ensure_fi_dir(exp_dir: Path) -> Path:
    fi_dir = exp_dir / "feature_importance"
    fi_dir.mkdir(parents=True, exist_ok=True)
    return fi_dir


def _save_raw_and_summary(fi_map: Dict[str, pd.Series], save_dir: Path) -> Tuple[Path, Path]:
    # Собираем общий датафрейм по всем сериям
    df = pd.DataFrame(fi_map).fillna(0.0)

    # Порядок колонок: native:..., permutation:...
    native_cols = [c for c in df.columns if c.startswith("native:")]
    perm_cols = [c for c in df.columns if c.startswith("permutation:")]
    ordered_cols = native_cols + perm_cols
    df = df[ordered_cols]

    raw_fp = save_dir / "raw_importances.csv"
    df.to_csv(raw_fp)

    # Сводка (как в примере: сохранить сырые + агрегаты)
    summary_extra = pd.DataFrame(index=df.index)
    summary_extra["avg_native"] = df[native_cols].mean(axis=1) if native_cols else 0.0
    summary_extra["avg_perm"] = df[perm_cols].mean(axis=1) if perm_cols else 0.0
    if native_cols and perm_cols:
        summary_extra["avg_total"] = pd.concat([df[native_cols], df[perm_cols]], axis=1).mean(axis=1)
    elif native_cols:
        summary_extra["avg_total"] = summary_extra["avg_native"]
    else:
        summary_extra["avg_total"] = summary_extra["avg_perm"]

    summary_extra["zero_native"] = (df[native_cols].abs().sum(axis=1) == 0.0) if native_cols else True
    summary_extra["zero_perm"] = (df[perm_cols].abs().sum(axis=1) == 0.0) if perm_cols else True
    summary_extra["all_zero"] = summary_extra["zero_native"] & summary_extra["zero_perm"]

    summary_df = pd.concat([df, summary_extra], axis=1)
    summary_fp = save_dir / "summary_importances.csv"
    summary_df.to_csv(summary_fp)

    return raw_fp, summary_fp


def compute_feature_importance_for_experiment(
    experiment_id: Optional[str] = None,
    models_subset: Optional[List[str]] = None,
    features_path: Optional[str] = None,
    reports_dir: str = "06_reports",
    models_cfg_path: str = "04_configs/models.yml",
    cfg: Optional[ImportanceConfig] = None,
) -> Dict[str, str]:
    """Вычисляет важность признаков для моделей эксперимента и сохраняет результаты.

    Возвращает пути к сохранённым файлам raw_importances.csv и summary_importances.csv.
    """
    imp_cfg = cfg or ImportanceConfig()
    root = _project_root()
    # Попытаемся автоопределить путь к файлу признаков из experiments.csv по ID эксперимента
    reports_dp = root / reports_dir
    models_cfg_fp = root / models_cfg_path

    exp_id = experiment_id or _get_current_experiment_id(models_cfg_fp)
    exp_dir = _find_experiment_dir(reports_dp, f"{exp_id}")
    data_dir = root / "01_data/processed"

    if features_path is None or str(features_path).lower() == "auto":
        features_fp = _infer_features_path_from_experiments_csv(exp_id, reports_dp, data_dir)
    else:
        features_fp = root / features_path

    enabled_models = _get_enabled_models(models_cfg_fp)
    models_to_process = [m for m in (models_subset or enabled_models)]

    X_test, y_test = _load_test_split(exp_dir, features_fp)
    fi_dir = _ensure_fi_dir(exp_dir)

    fi_map: Dict[str, pd.Series] = {}

    for model_name in models_to_process:
        model_dir = exp_dir / model_name
        if not model_dir.exists():
            # Модель для этого эксперимента не обучалась или не сохранена
            continue
        model = _load_model(model_dir, model_name)

        # Native
        native_ser = _native_importance(model_name, model, X_test, y_test)
        fi_map[f"native:{model_name}_native"] = native_ser

        # Permutation
        perm_ser = _permutation_importance(model, X_test, y_test, imp_cfg)
        # Короткое имя метрики для консистентности с примером (f1_macro -> f1m)
        metric_suffix = "f1m" if imp_cfg.scoring == "f1_macro" else imp_cfg.scoring.replace("_", "")
        fi_map[f"permutation:{model_name}_perm_{metric_suffix}"] = perm_ser

        # SHAP графики (если возможно)
        try:
            _save_shap_plots(model_name, model, X_test, fi_dir, max_samples=imp_cfg.shap_max_samples)
        except Exception:
            # Мягко игнорируем ошибки SHAP, чтобы не ломать весь расчёт важностей
            pass

    raw_fp, summary_fp = _save_raw_and_summary(fi_map, fi_dir)

    # Дополнительно сохраним метаданные запуска
    meta = {
        "experiment_dir": str(exp_dir),
        "experiment_id": exp_id,
        "models": models_to_process,
        "scoring": imp_cfg.scoring,
        "n_repeats": imp_cfg.n_repeats,
        "random_state": imp_cfg.random_state,
        "features_path": str(features_fp),
        "raw_importances": str(raw_fp),
        "summary_importances": str(summary_fp),
    }
    with open(fi_dir / "fi_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"raw_importances": str(raw_fp), "summary_importances": str(summary_fp)}


__all__ = [
    "ImportanceConfig",
    "compute_feature_importance_for_experiment",
]


# ===== Доп. утилиты =====

def _infer_features_path_from_experiments_csv(exp_id: str, reports_dp: Path, data_dir: Path) -> Path:
    """Определяет путь к parquet с фичами по соответствию в 06_reports/experiments.csv.

    Ожидается, что колонка ID содержит, например, "EXP_0006", а колонка Feature set — что-то вроде "fset-4".
    Файл фич ищется как *{feature_set}*_features.parquet в data_dir.
    """
    csv_fp = reports_dp / "experiments.csv"
    if not csv_fp.exists():
        raise FileNotFoundError(f"Не найден файл экспериментов: {csv_fp}")

    df = pd.read_csv(csv_fp)
    if "ID" not in df.columns or "Feature set" not in df.columns:
        raise ValueError("В experiments.csv отсутствуют необходимые колонки 'ID' и 'Feature set'")

    # Нормализуем ID (exp_0006 -> EXP_0006)
    exp_upper = exp_id.upper()
    row = df.loc[df["ID"].str.upper() == exp_upper]
    if row.empty:
        raise ValueError(f"В experiments.csv не найден эксперимент с ID={exp_upper}")
    feature_set = str(row.iloc[0]["Feature set"])  # например, fset-4

    # Поиск файла признаков
    candidates = sorted(data_dir.rglob(f"*{feature_set}*_features.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Не найден файл признаков в {data_dir} по шаблону '*{feature_set}*_features.parquet'"
        )
    # Берём самый новый (по имени/алфавиту)
    return candidates[-1]


def _save_shap_plots(model_name: str, model, X: pd.DataFrame, out_dir: Path, max_samples: int = 3000) -> None:
    """Сохраняет SHAP summary (beeswarm) и bar графики для деревьев, если установлен shap.

    Для CatBoost используем get_feature_importance(..., type='ShapValues'), для LightGBM/XGBoost — shap.TreeExplainer.
    """
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    # Сэмплируем, чтобы ускорить
    if len(X) > max_samples:
        X_sample = X.iloc[-max_samples:]
    else:
        X_sample = X

    shap_values = None
    expected_value = None

    if model_name == "catboost":
        try:
            from catboost import Pool  # type: ignore
            pool = Pool(X_sample, feature_names=X_sample.columns.tolist())
            shap_arr = model.get_feature_importance(pool=pool, type="ShapValues")
            # Последняя колонка — базовый прогноз (expected value)
            shap_values = np.array(shap_arr)[:, :-1]
            expected_value = np.array(shap_arr)[:, -1].mean()
        except Exception:
            shap_values = None
    else:
        try:
            # Для LGBM/XGB используем TreeExplainer
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample, check_additivity=False)
            if isinstance(sv, list):
                # бинарный случай: берём класс 1
                shap_values = sv[1]
                try:
                    expected_value = explainer.expected_value[1]  # type: ignore[index]
                except Exception:
                    expected_value = None
            else:
                shap_values = sv
                try:
                    expected_value = explainer.expected_value  # type: ignore[assignment]
                except Exception:
                    expected_value = None
        except Exception:
            shap_values = None

    if shap_values is None:
        return

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    beeswarm_fp = out_dir / f"shap_summary_{model_name}.png"
    plt.savefig(beeswarm_fp, dpi=200, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    bar_fp = out_dir / f"shap_bar_{model_name}.png"
    plt.savefig(bar_fp, dpi=200, bbox_inches="tight")
    plt.close()


