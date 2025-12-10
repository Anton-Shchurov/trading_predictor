from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve

# Реюз внутренней логики пайплайна моделирования
from .modeling_pipeline import (
    SplitConfig,
    _compute_class_weights,
    _confusion_and_report,
    _ensure_sorted_index,
    _evaluate,
    _fit_and_eval_model,
    _predict_catboost,
    _predict_xgboost,
    _save_cv_indices,
    _temporal_train_valid_test_split,
    _time_series_folds,
)


# =============================
# Вспомогательные утилиты
# =============================


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _infer_run_dir(project_root: Path, reports_dir: Path) -> Path:
    """Создаёт подпапку запуска 06_reports/exp_<id>_<ts>, если не передали run_dir.

    Идентификатор эксперимента берём из 04_configs/models.yml → experiment.current_id.
    """
    models_cfg = _load_yaml(project_root / "04_configs" / "models.yml")
    exp_id = ((models_cfg.get("experiment") or {}).get("current_id") or "exp_0000")
    prefix = str(exp_id).lower()
    run_dir = reports_dir / f"{prefix}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    return run_dir


# =============================
# Определение пространства поиска Optuna
# =============================


_FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _parse_and_suggest(trial: optuna.trial.Trial, name: str, spec: Any) -> Any:
    """Поддержка спецификаций вида:
    - "int(a, b)"
    - "uniform(a, b)"
    - "loguniform(a, b)"
    - списки => categorical
    - скаляры => возвращаются как есть (не тюним)
    """
    if isinstance(spec, (int, float)):
        return spec
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if not isinstance(spec, str):
        return spec

    s = spec.strip().replace(" ", "")
    m = re.match(fr"^int\(({_FLOAT_RE}),({_FLOAT_RE})\)$", s)
    if m:
        lo, hi = m.groups()
        return trial.suggest_int(name, int(float(lo)), int(float(hi)))

    m = re.match(fr"^uniform\(({_FLOAT_RE}),({_FLOAT_RE})\)$", s)
    if m:
        lo, hi = m.groups()
        return trial.suggest_float(name, float(lo), float(hi))

    m = re.match(fr"^loguniform\(({_FLOAT_RE}),({_FLOAT_RE})\)$", s)
    if m:
        lo, hi = m.groups()
        return trial.suggest_float(name, float(lo), float(hi), log=True)

    # по умолчанию — возвращаем как есть
    return spec


def _sample_params(trial: optuna.trial.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for p_name, p_spec in (space or {}).items():
        params[p_name] = _parse_and_suggest(trial, p_name, p_spec)
    return params


# =============================
# Графики метрик
# =============================


def _save_confusion_matrix_png(path: Path, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, cm: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    if cm is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=[0, 1])
        disp.plot(ax=ax)
    elif y_true is not None and y_pred is not None:
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    else:
        plt.close(fig)
        return
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_roc_curve_png(path: Path, y_true: np.ndarray, y_proba: Optional[np.ndarray]) -> None:
    if y_proba is None:
        return
    y_score = None
    if isinstance(y_proba, np.ndarray):
        if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            y_score = y_proba[:, 1]
        elif y_proba.ndim == 1:
            y_score = y_proba
    if y_score is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================
# Основная функция тюнинга
# =============================


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    test_metrics: Dict[str, float]
    test_details: Dict[str, Any]
    study_trials: List[Dict[str, Any]]


def _get_pruner(name: Optional[str]) -> optuna.pruners.BasePruner:
    name = (name or "median").lower()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "nopruner":
        return optuna.pruners.NopPruner()
    # по умолчанию — median
    return optuna.pruners.MedianPruner()


def _list_enabled_models(models_cfg: Dict) -> List[str]:
    return [name for name, cfg in (models_cfg.get("models") or {}).items() if (cfg or {}).get("enabled", False)]


def _merge_params(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    merged.update(updates or {})
    return merged


def _objective_factory(
    model_name: str,
    space: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    common_cfg: Dict[str, Any],
    X_trval: pd.DataFrame,
    y_trval: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    class_weights: Dict[int, float],
    primary_metric: str,
):
    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = _sample_params(trial, space)
        model_cfg = {
            "params": _merge_params((base_model_cfg.get("params") or {}), trial_params),
            "fit": (base_model_cfg.get("fit") or {}),
        }

        fold_scores: List[float] = []
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X_trval.iloc[tr_idx], y_trval.iloc[tr_idx]
            X_va, y_va = X_trval.iloc[va_idx], y_trval.iloc[va_idx]
            _, metrics = _fit_and_eval_model(
                model_name,
                X_tr,
                y_tr,
                X_va,
                y_va,
                class_weights,
                model_cfg,
                common_cfg,
            )
            score = float(metrics.get(primary_metric, np.nan))
            fold_scores.append(score)

        return float(np.nanmean(fold_scores))

    return objective


def _final_fit_and_test(
    model_name: str,
    best_params: Dict[str, Any],
    base_model_cfg: Dict[str, Any],
    common_cfg: Dict[str, Any],
    X_trval: pd.DataFrame,
    y_trval: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_weights: Dict[int, float],
) -> Tuple[Dict[str, float], Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    final_cfg = {
        "params": _merge_params((base_model_cfg.get("params") or {}), best_params),
        "fit": (base_model_cfg.get("fit") or {}),
    }
    clf, _ = _fit_and_eval_model(
        model_name, X_trval, y_trval, X_test, y_test, class_weights, final_cfg, common_cfg
    )

    # Предсказания на тесте (для ROC и метрик)
    if model_name == "catboost":
        y_pred_test, y_proba_test = _predict_catboost(clf, X_test)
    elif model_name == "xgboost":
        y_pred_test, y_proba_test = _predict_xgboost(clf, X_test)
    else:
        y_proba_test = clf.predict_proba(X_test)
        y_pred_test = clf.predict(X_test)

    test_metrics = _evaluate(y_test.values, y_pred_test, y_proba_test)
    test_details = _confusion_and_report(y_test.values, y_pred_test)
    return test_metrics, test_details, y_pred_test, y_proba_test


def _study_to_records(study: optuna.Study) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for t in study.trials:
        rec = {
            "number": t.number,
            "state": str(t.state),
            "value": None if t.value is None else float(t.value),
            "params": {k: (None if v is None else (float(v) if isinstance(v, (int, float)) else v)) for k, v in t.params.items()},
        }
        records.append(rec)
    return records


def _update_models_yaml(models_yaml_path: Path, best_params_map: Dict[str, Dict[str, Any]]) -> None:
    original_text = models_yaml_path.read_text(encoding="utf-8")
    backup_path = models_yaml_path.with_suffix(models_yaml_path.suffix + f".{_timestamp()}.bak")
    backup_path.write_text(original_text, encoding="utf-8")

    try:
        from ruamel.yaml import YAML
        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        data = yaml_parser.load(original_text)
        
        models_section = data.get("models", {})
        for name, best_params in (best_params_map or {}).items():
            if name in models_section:
                # Update best_params section directly
                if "best_params" not in models_section[name]:
                    models_section[name]["best_params"] = {}
                # Update keys one by one to preserve structure/comments if exist
                for k, v in best_params.items():
                    models_section[name]["best_params"][k] = v
                # Automatically enable use_best_params
                models_section[name]["use_best_params"] = True
        
        with open(models_yaml_path, "w", encoding="utf-8") as f:
            yaml_parser.dump(data, f)
            
    except ImportError:
        print("Warning: ruamel.yaml not found, falling back to pyyaml (comments might be lost).")
        data = _load_yaml(models_yaml_path)
        models_section = data.setdefault("models", {})
        for name, best_params in (best_params_map or {}).items():
            model_cfg = models_section.setdefault(name, {})
            model_cfg["best_params"] = best_params
            model_cfg["use_best_params"] = True
        _save_yaml(models_yaml_path, data)


def tune_hyperparameters(
    models: Optional[List[str]] = None,
    n_trials: Optional[int] = None,
    config_path: str = "04_configs/hyperparameter_tuning.yml",
    features_path: str = "01_data/processed/eurusd_features.parquet",
    splits_config_path: str = "04_configs/splits.yml",
    models_config_path: str = "04_configs/models.yml",
    reports_dir: str = "06_reports",
    run_dir: Optional[str] = None,
    update_models_yaml: bool = True,
) -> Dict[str, Any]:
    """Тюнинг гиперпараметров для включённых моделей (или заданного списка).

    Возвращает словарь с результатами по моделям и путями к артефактам.
    """
    project_root = Path(__file__).resolve().parents[2]
    # models_fp is needed early for data config
    models_fp = project_root / models_config_path
    models_cfg = _load_yaml(models_fp)
    
    # Read data config
    data_cfg = models_cfg.get("data", {})
    target_col = data_cfg.get("target_column", "y_bs")
    
    # Resolve features path: priority to config if default arg is used
    if features_path == "01_data/processed/eurusd_features.parquet" and data_cfg.get("features_path"):
        features_fp = project_root / data_cfg.get("features_path")
    else:
        features_fp = project_root / features_path

    splits_fp = project_root / splits_config_path
    reports_dir_path = project_root / reports_dir
    reports_dir_path.mkdir(parents=True, exist_ok=True)

    # RUN_DIR согласно общей схеме
    if run_dir is None:
        run_dir_path = _infer_run_dir(project_root, reports_dir_path)
    else:
        run_dir_path = Path(run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)

    # Данные
    if not features_fp.exists():
        raise FileNotFoundError(f"Не найден файл признаков: {features_fp}")
    df = pd.read_parquet(features_fp)
    df = _ensure_sorted_index(df)
    X, y = _validate_features(df, target_col=target_col)

    # Конфиг сплитов и фолдов
    cfg_raw = _load_yaml(splits_fp)
    cfg = SplitConfig(
        method=cfg_raw.get("splits", {}).get("method", "ratios"),
        ratios=cfg_raw.get("splits", {}).get("ratios", {"train": 0.7, "valid": 0.15, "test": 0.15}),
        dates=cfg_raw.get("splits", {}).get("dates", {}),
        n_splits=int(cfg_raw.get("time_series_cv", {}).get("n_splits", 5)),
        max_train_size=cfg_raw.get("time_series_cv", {}).get("max_train_size"),
        test_size=cfg_raw.get("time_series_cv", {}).get("test_size"),
        gap=int(cfg_raw.get("time_series_cv", {}).get("gap", 0)),
        artifacts=cfg_raw.get("artifacts", {}),
    )

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), _ = _temporal_train_valid_test_split(X, y, cfg)
    X_trval = pd.concat([X_train, X_valid])
    y_trval = pd.concat([y_train, y_valid])
    folds = _time_series_folds(X_trval, y_trval, cfg)

    # Артефакты фолдов для воспроизводимости
    if cfg.artifacts.get("save_cv_indices", True):
        _save_cv_indices(folds, run_dir_path / "cv_indices.json", X_trval.index)

    # Балансировка классов
    # Check pipeline settings for class weights usage
    pipeline_settings = models_cfg.get("pipeline_settings", {})
    if pipeline_settings.get("use_class_weights", False):
        class_weights = _compute_class_weights(y_train)
    else:
        class_weights = {0: 1.0, 1: 1.0}

    # Конфиги моделей и тюнинга
    enabled = _list_enabled_models(models_cfg)
    model_list = enabled if models is None else list(models)
    common_cfg = models_cfg.get("common", {})

    hpt_cfg = models_cfg # Tuning config is now part of models.yml
    optuna_cfg = hpt_cfg.get("optuna", {})
    if n_trials is None:
        n_trials = int(optuna_cfg.get("n_trials", 50))
    pruner = _get_pruner(optuna_cfg.get("pruner"))
    primary_metric = (models_cfg.get("metrics", {}).get("primary") or "f1")

    results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    best_params_map: Dict[str, Dict[str, Any]] = {}

    for model_name in model_list:
        m_cfg_full = (models_cfg.get("models", {}).get(model_name) or {})
        base_model_cfg = {
            "params": m_cfg_full.get("default_params", {}),
            "fit": m_cfg_full.get("fit", {})
        }
        
        # Read space from hp_tuning_params
        space = m_cfg_full.get("hp_tuning_params", {})
        if not space:
             print(f"Warning: No hp_tuning_params found for {model_name}, skipping tuning.")
             continue

        study = optuna.create_study(direction="maximize", pruner=pruner)

        objective = _objective_factory(
            model_name,
            space,
            base_model_cfg,
            common_cfg,
            X_trval,
            y_trval,
            folds,
            class_weights,
            primary_metric,
        )
        timeout = optuna_cfg.get("timeout")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_trial.params if study.best_trial is not None else {}
        best_score = float(study.best_value) if study.best_trial is not None else float("nan")
        best_params_map[model_name] = dict(best_params)

        # Финальная оценка на тесте
        test_metrics, test_details, y_pred_test, y_proba_test = _final_fit_and_test(
            model_name,
            best_params,
            base_model_cfg,
            common_cfg,
            X_trval,
            y_trval,
            X_test,
            y_test,
            class_weights,
        )

        # Сохранение графиков
        cm_path = run_dir_path / "plots" / f"cm_{model_name}.png"
        _save_confusion_matrix_png(cm_path, y_true=y_test.values, y_pred=y_pred_test)
        roc_path = run_dir_path / "plots" / f"roc_{model_name}.png"
        _save_roc_curve_png(roc_path, y_true=y_test.values, y_proba=y_proba_test)

        # Соберём результаты по модели
        model_result = {
            "best_score_cv": best_score,
            "best_params": best_params,
            "test_metrics": test_metrics,
            "test_details": test_details,
            "trials": _study_to_records(study),
        }
        results[model_name] = model_result

        # Строка сводной таблицы
        summary_rows.append(
            {
                "model": model_name,
                "cv_primary": primary_metric,
                "cv_primary_best": best_score,
                "test_f1": test_metrics.get("f1"),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_logloss": test_metrics.get("logloss"),
                "test_roc_auc": test_metrics.get("roc_auc"),
                "n_trials": len(study.trials),
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }
        )

        # Сохранение частных артефактов по модели
        _save_json(run_dir_path / "metrics" / f"optuna_study_{model_name}.json", _study_to_records(study))

    # Сохранение агрегированных артефактов запуска
    _save_json(run_dir_path / "metrics" / "hyperparameter_tuning_results.json", results)
    pd.DataFrame(summary_rows).to_csv(run_dir_path / "metrics" / "hyperparameter_tuning_summary.csv", index=False)
    _save_yaml(run_dir_path / "metrics" / "best_params.yml", best_params_map)
    # Save config snapshot
    _save_yaml(run_dir_path / "metrics" / "experiment_config.yaml", models_cfg)

    # Обновление models.yml
    if update_models_yaml and best_params_map:
        _update_models_yaml(models_fp, best_params_map)

    return {
        "run_dir": str(run_dir_path),
        "results": results,
        "best_params": best_params_map,
    }


# Валидация признаков — импортируем после основного определения, чтобы избежать цикличных импортаций
from .modeling_pipeline import _validate_features  # noqa: E402


