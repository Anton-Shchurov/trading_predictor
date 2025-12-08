### Конфигурация проекта

Папка содержит все YAML‑конфиги для подготовки признаков и моделирования.

- `feature_engineering.yml` — декларативное описание признаков, наборов признаков, профилей, датасетов и общих настроек пайплайна
- `splits.yml` — временные разрезы train/valid/test и параметры TimeSeriesSplit
- `models.yml` — включение/настройка моделей (XGBoost/LightGBM/CatBoost) для бинарной задачи (Sell=0, Buy=1)
- `oanda_config.yml` — параметры доступа к OANDA (опционально, если вы тянете сырые данные из API)

Все пути в конфигах интерпретируются относительно корня проекта.

### Настройка OANDA API (опционально)

1) Скопируйте пример:
```bash
cp oanda_config.example.yml oanda_config.yml
```
2) Укажите ваш API‑токен (`YOUR_OANDA_API_TOKEN_HERE`) и при необходимости URL/инструмент/таймфрейм.

⚠️ Файл `oanda_config.yml` приватный и уже исключён из VCS через `.gitignore`.

### Требования

Для работы с YAML:
```bash
pip install PyYAML
```

---

## 1) Управление признаками: декларативный DAG (`feature_engineering.yml`)

Фичи описываются декларативно и вычисляются по графу зависимостей. Основные секции:

1. `feature_definitions` — словарь всех признаков (узлов DAG)
   - `method`: имя расчётного метода. Доступные методы собираются из публичных функций вида `calculate_*` в модулях `technical_indicators`, `statistical_features`, `lag_features`, а также специальные методы:
     - `identity`, `formula`, `binary_target`, `hour_sin`, `hour_cos`, `day_of_week` и технические индикаторы (`ema`, `atr`, `macd`, `adx`, `bbands`, `kc`, `donchian`, `rsi`, `roc`, `log_return`, `vwap`, `psar`, `zscore`, и др.)
   - `inputs`: список имен входных серий, которые передаются в метод как позиционные аргументы (часто это сырые столбцы или ранее созданные фичи)
   - `needs`: мягкие зависимости для корректного порядка вычислений (не обязательно передаются как аргументы)
   - `params`: параметры метода (словарь)
   - `formula`: Python/pandas выражение для метода `formula`. В контексте доступны имена зависимостей, `nan` (эквивалент `numpy.nan`) и `NA`.
   - `is_intermediate`: true/false — помечает промежуточный узел, который не попадёт в итоговый датасет

   Пример:
```yaml
feature_definitions:
  close: { method: identity, inputs: [Close] }
  _ema_20_raw: { method: ema, inputs: [close], params: { period: 20 }, is_intermediate: true }
  _atr_14_raw: { method: atr, inputs: [High, Low, close], params: { period: 14 }, is_intermediate: true }

  f_close_minus_ema_20_over_atr14:
    method: formula
    needs: [_ema_20_raw, _atr_14_raw]
    formula: "(close - _ema_20_raw) / _atr_14_raw.replace(0, nan)"

  y_bs: { method: binary_target, inputs: [close], params: { horizon: 5 } }  # целевая 0/1
```

2. `pipeline_settings` — глобальные параметры пайплайна
   - `input_file`: путь к исходным данным CSV (может быть переопределён реестром датасетов)
   - `output_file` или (`output_dir` + `output_pattern`): куда сохранять Parquet
   - `feature_set`: активный набор признаков (см. раздел `feature_sets`)
   - `parquet_settings`: `engine`, `compression`, `index`
   - `missing_values`: стратегия обработки пропусков: `keep_all` | `drop` | `forward_fill` | `backward_fill` (по умолчанию сохранение всего для анализа)
   - `validation`: базовые проверки (дубликаты по времени, сортировка, доля пропусков)
   - `logging`: уровень логирования (по умолчанию вывод только в консоль)

3. `feature_sets` — декларативный выбор колонок по маскам
   - `include_patterns`/`exclude_patterns`: шаблоны fnmatch (например, `f_rsi_*`, `f_*atr*`)
   - `inherit_from`: наследование наборов. Всегда сохраняются `close` и целевая `y_bs` (если она есть)

4. `metadata` — вспомогательные сведения (ожидаемые входные колонки, версии, зависимости).

5. `dataset` — реестр датасетов
   - `active`: ключ активного датасета из `items`
   - `items.{key}.path`: путь к CSV; при выборе датасета автоматически подменяется `pipeline_settings.input_file` и в `metadata.dataset_name` попадает читаемое имя

6. `profiles` — профили настроек
   - Пример: `quick` (укороченные окна/периоды) и `full`. Профиль накладывает значения на соответствующие секции основного конфига

Ключевые моменты выполнения:
- Порядок вычисления фич строится автоматически по зависимостям `inputs`/`needs`. Циклы запрещены.
- Для `formula` сначала пробуется `pandas.eval`, при ошибке — безопасный `eval` с ограниченным контекстом.
- Итоговый датафрейм включает все узлы, не помеченные `is_intermediate: true`.
- Затем применяется фильтрация колонок по активному `feature_set`.

Частые операции:
- Сменить активный набор признаков: `pipeline_settings.feature_set: fset-1 | fset-2 | ...`
- Добавить новую фичу: объявите её в `feature_definitions` и при необходимости добавьте маску в `feature_sets.<name>.include_patterns`
- Переключить датасет: `dataset.active: <key>` (пути берутся из `dataset.items`)
- Изменить горизонт целевой переменной: `feature_definitions.y_bs.params.horizon`
- Переименовать/добавить выходной файл: задайте `output_file` или `output_dir` + `output_pattern`. Суффикс активного набора (`feature_set`) автоматически добавляется к имени файла

---

## 2) Разбиения по времени (`splits.yml`)

Секция `splits` управляет тем, как формируются train/valid/test:

```yaml
splits:
  method: ratios  # ratios | dates
  ratios: { train: 0.70, valid: 0.15, test: 0.15 }
  dates:  # альтернативный способ (YYYY-MM-DD HH:MM:SS)
    train_end: null
    valid_end: null
    test_start: null  # обязательно при method: dates
```

TimeSeries CV на train+valid:
```yaml
time_series_cv:
  n_splits: 5
  max_train_size: null
  test_size: null
  gap: 5
```

Артефакты разбиения:
```yaml
artifacts:
  save_cv_indices: true
  save_split_dates: true
  cv_indices_path: cv_indices.json       # опционально (по умолчанию это имя)
  split_dates_path: split_dates.yml      # опционально (по умолчанию это имя)
```

Эти настройки читаются в `modeling_pipeline`; индексы фолдов и даты разбиений сохраняются в папку отчёта (`06_reports`).

---

## 3) Конфиг моделей (`models.yml`)

Основные разделы:

```yaml
task: binary               # бинарная классификация
classes: [sell, buy]       # метки: Sell=0, Buy=1

metrics:
  primary: f1
  others: [balanced_accuracy, accuracy, logloss]

common:
  random_state: 42
  early_stopping_rounds: 50
  eval_metric: logloss     # для XGBoost; в LGBM — metric: binary_logloss

models:
  xgboost:
    enabled: true
    params: { ... }
    fit: { verbose: false }
  lightgbm:
    enabled: true
    params: { ... }
    fit: { verbose: -1 }
  catboost:
    enabled: true
    params: { ... }
```

Как это используется:
- Список моделей определяется флагом `enabled: true`.
- Веса классов считаются автоматически по частотам класса на train (0/1) и применяются корректно к каждому алгоритму:
  - XGBoost — через `sample_weight` при обучении
  - LightGBM — через `class_weight`
  - CatBoost — через `class_weights` в параметрах
- Ранняя остановка включена везде, с учётом версии библиотеки (есть резервные режимы, если колбэки не поддерживаются).
- Оценка метрик проводится на фолдах CV и финально на тесте; агрегированный отчёт сохраняется в `06_reports`.

Требование к данным для моделирования:
- Итоговый датасет должен содержать бинарную цель `y_bs` (0/1) и признаки, начинающиеся с `f_` (а также `close`). Если `y_bs` отсутствует — необходимо пересоздать фичи с целевой меткой в `feature_engineering.yml`.

---

## 4) Быстрые рецепты

- Включить/выключить модель:
  - В `models.yml` установить `models.<name>.enabled: true|false`
- Переключить стратегию разбиения:
  - В `splits.yml` сменить `splits.method` на `ratios` или `dates` и задать соответствующие поля
- Сменить активный набор признаков:
  - В `feature_engineering.yml` задать `pipeline_settings.feature_set`
- Добавить фичу по формуле:
  - В `feature_definitions` создать узел с `method: formula`, заполнить `needs`/`inputs` и `formula`
- Настройка логов:
  - `pipeline_settings.logging.level: INFO|DEBUG|WARNING`

---

## 5) Примечания

- Имена колонок входных данных автоматически приводятся к стандартным (`Open`, `High`, `Low`, `Close`, `Volume`, `Time`) при возможном совпадении с известными вариантами.
- При сохранении результатов к имени файла добавляется суффикс активного набора признаков, чтобы не перезаписывать разные версии.
- Если задействован реестр датасетов (`dataset.active`), путь к входному файлу в `pipeline_settings.input_file` будет автоматически подменён.
