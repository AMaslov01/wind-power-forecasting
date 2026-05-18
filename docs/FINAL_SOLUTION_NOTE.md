# Пояснительная записка к финальному решению

## 1. Что запускать проверяющему

Главная идея упаковки: проверяющий не должен разбираться в истории экспериментов.
В корне пакета есть один вход:

```bash
bash run_solution.sh
```

Команда не требует тяжелого ML-стека: быстрый путь использует только стандартную
библиотеку Python, проверяет входные файлы, восстанавливает финальные прогнозы
из `model_weights/`, валидирует формат и пишет отчет:

- `outputs/predictions_q1.csv` — 2126 почасовых значений для периода
  01.01.2026-31.03.2026;
- `outputs/predictions_may18.csv` — 24 почасовых значения на 18.05.2026;
- `outputs/RUN_REPORT.json` — checksums, версии библиотек, статистика выходов.

Полное переобучение доступно одной дополнительной командой. В этом режиме
`run_solution.sh` создает `.venv` при необходимости и устанавливает зависимости
из `physics/requirements.txt`:

```bash
bash run_solution.sh --retrain
```

Для Colab/GPU:

```bash
bash run_solution.sh --retrain --gpu
```

## 2. История решения и финальный выбор

По истории коммитов видно, что решение прошло несколько больших веток:

- `de3284f` — regime-first hygiene: аккуратное разделение ветровых режимов и
  отказ от смешивания всех условий в одну модель.
- `9636bb5` — regime-v2/calibration: более сложные per-regime правила, но на
  публичном лидерборде они ухудшили shape прогноза.
- `b198f6e`, `8cbe50d`, `a560f0e`, `e8c2b61` — April-May 2026 actual adapter.
  Сильный HGB-adapter хорошо улучшал post-Q1 validation, но при переносе в Q1
  создавал слишком много нулей и ломал public score. Поэтому в финале оставлен
  только безопасный scalar transfer с guardrails; если mean shift, max correction
  или zero count опасны, перенос автоматически пропускается.
- `4c779cb` — multi-regime context: дополнительные режимы по направлению/сдвигу
  ветра ухудшили CV и public shape, поэтому не вошли в финал.
- `cc246f7` — weather-dynamics V14 no-CDS: лучший стабильный кандидат. Он
  использует компактную погодную динамику, но не зависит от Copernicus/CDS.
- `5f59e9c` — V15 compact/sign/solar: эксперимент был откатан коммитом
  `b649886`, потому что не дал улучшения относительно стабильного V14.

Финальный якорь: `hgb_family_gfs_so_ups_nasa_regime_weather_dyn_q1_scalar_v14_nocds`.

## 3. Данные

Базовые данные хакатона:

- `train_dataset.csv`: история 2022-2025 с целевой переменной
  `Выработка. Результирующий расчет`;
- `valid_features.csv`: признаки Q1 2026 без target;
- `3888f9f2-9bda-4b2c-94af-5562668bce86_test_dataset.csv`: operational
  post-Q1 файл, где известные строки April-May 2026 используются для безопасной
  диагностики адаптера, а 24 пустые строки дают прогноз на 18.05.2026.

Эти CSV не публикуются в GitHub. Они лежат только локально и в ZIP-пакете.

Дополнительные данные:

- Open-Meteo historical forecast, в финальном запуске `gfs_only`: прогнозная
  погода, ближе к inference-сценарию, чем ретроспективная reanalysis.
- Meteostat cache: фактическая/историческая погода рядом с локацией, допускается
  partial coverage, gaps заполняются аккуратно.
- NASA POWER: открытый источник температуры, давления, влажности, ветра и
  wind-power-density proxy.
- SO UPS RES monthly reports: месячный контекст по ВИЭ/ограничениям Ростовской
  области. В финале используется `legacy_best_gap`: 45 надежных месяцев, где
  спорные 2023-01, 2023-05, 2024-05 исключены. Это оказалось лучше clean-48,
  потому что новые 3 месяца сдвигали средний прогноз вниз и ухудшали public
  shape.

Copernicus/CDS и Renewables Ninja оставлены как расширяемость, но не входят в
default path: CDS full-range fetch зависал, а проверяющему нужен надежный запуск.

## 4. Физическая модель и spline/power-curve слой

Физический baseline — это не отдельная финальная модель, а сильный anchor-признак.
Он строится как spline-like / piecewise cubic power curve:

1. Из скоростей на высотах 10/80/120/180 м строится rotor-equivalent wind speed.
2. На участке ниже cut-in мощность равна нулю.
3. Между cut-in и rated используется гладкая кубическая ramp-кривая
   `((v - v_cut_in) / (v_rated - v_cut_in)) ** cubic_exponent`.
4. В rated-zone мощность ограничена `p_rated_per_turbine`.
5. После cut-out мощность обнуляется/ограничивается физическими bounds.
6. Затем применяются density correction, efficiency factor и число доступных
   турбин.

В финальном preset `public_best` параметры такие:

- `v_cut_in = 2.3393`;
- `v_rated = 10.8310`;
- `v_cut_out = 23.9636`;
- `p_rated_per_turbine = 3.3261`;
- `cubic_exponent = 1.4983`;
- `efficiency_factor = 1.1196`;
- `density_correction_exp = 0.4541`;
- `ice_threshold = 3.2054`.

В коде также сохранены empirical curve / monotone curve modules. Они исследуют
PCWG-style binned monotone curve, но финальный active model set — `hgb_family`,
поэтому эти модули остаются как расширяемость, а не как главный путь.

## 5. Основные группы признаков

### Calendar/time

- `month`, `hour_of_day`;
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos`;
- `is_night`.

Зачем: сезонность и суточные режимы ветра/погоды.

### Raw meteo

- wind speeds на разных высотах;
- wind direction на разных высотах;
- gusts, pressure, temperature, precipitation, cloud cover;
- boundary layer / CAPE / freezing level где доступны.

Зачем: основной физический драйвер выработки.

### Direction encoding

Направление ветра циклическое, поэтому вместо “0 рядом с 360” используются:

- `wd_10m_sin/cos`;
- `wd_80m_sin/cos`;
- `wd_120m_sin/cos`;
- `wd_180m_sin/cos`;
- `direction_shear_120_80`.

Зачем: деревьям легче учить циклическое направление и directional shear.

### Physics features

- `wind_speed_eq`;
- `wind_speed_eq_cubed`;
- `P_physics_per_turbine`;
- `P_physics_farm`;
- `cp_effective`;
- `is_below_cutin`, `is_in_partial`, `is_in_rated`, `is_above_cutout`;
- `wind_speed_to_cutin`, `wind_speed_to_rated`, `wind_speed_to_cutout`;
- physics variants: manufacturer/public_best/hub80 deltas.

Зачем: модели не нужно “с нуля” открывать закон мощности ветра; она учит
ошибку/коррекцию поверх физического prior.

### Technical / turbine-operation features

- `repair_monthly_avg`;
- `n_working_monthly_avg`;
- `availability_fraction`;
- latent availability calibration;
- `ice_risk`;
- `turbulence_intensity`;
- `gust_factor`;
- `temp_gradient_120_80`;
- `is_stable`.

Зачем: выработка зависит не только от ветра, но и от доступности турбин,
обледенения, порывистости, стабильности атмосферы и ограничений.

### External-source guarded blends

Для внешней погоды строятся:

- source blend values;
- delta между локальным прогнозом и external source;
- agreement score;
- consensus/spread для forecast families.

Зачем: если источники согласны, сигнал усиливается; если источник выбивается,
модель видит disagreement и не обязана ему верить.

### Temporal weather features

Без target history, только X-признаки:

- lag/lead на 1/2/3/6/12 часов;
- rolling means 3/7/13;
- rolling std 12/24;
- first differences;
- deltas 3/6 часов.

Зачем: ветер инерционен; shape фронта/порывов важнее одной точки времени.

### Weather dynamics, перенесенные из Kaggle-практики

Из решений Hill of Towie и `jhyland01/kaggle_ts-forecasting` перенесена идея:
не тащить огромную “kitchen sink” матрицу, а дать деревьям короткую динамику
входной погоды:

- EMA span 3/6;
- rolling median 3/7;
- rolling trend 6/12;
- relative std;
- disagreement между GFS/NASA/CDS-like источниками;
- positive/negative/absolute source deltas и ratios.

Зачем: эти признаки ловят смену режима ветра, source bias и uncertainty.
Graph/RNN идеи из Kaggle notebooks не вошли, потому что у нас нет панели по
отдельным турбинам/станциям.

### SO UPS RES monthly context

Признаки месячного регионального контекста:

- installed MW;
- monthly generation MWh;
- YTD generation MWh;
- curtailment hours;
- max curtailment MW;
- capacity factor.

Зачем: помогает отличить метео-причины от ограничений/сетевых режимов.

## 6. Модели и ансамбль

Финальный `hgb_family` содержит 10 моделей:

- CatBoost direct/residual variants: `cat7_direct`, `cat7_residual`,
  `cat5_direct`, `cat5_residual`;
- HGB direct/residual variants: `hgb_direct`, `hgb_direct_smooth`,
  `hgb_direct_deep`, `hgb_residual_smooth`;
- regime experts: `hgb_regime_direct`, `hgb_regime_residual`.

Direct-модели учат target напрямую. Residual-модели учат
`target - P_physics_farm`, то есть поправку к физике.

Regime models делят часы по operating regime:

- ниже cut-in;
- partial-load;
- rated;
- около cut-out.

Это важно, потому что ошибка модели при слабом ветре, в кубической зоне и на
плато rated — разные задачи.

Веса ансамбля оптимизируются по OOF Q1-folds 2023/2024/2025. Итоговые prediction
bounds всегда зажаты в `[0, 90.09]`.

## 7. Guardrails

`run_solution.sh` и `validate_solution.py` закрывают типовые failure modes:

- missing data files → понятная ошибка со списком файлов;
- missing `model_weights` → подсказка запустить `--retrain`;
- broken output shape → immediate failure;
- NaN/out-of-range → immediate failure;
- optional LightGBM не блокирует default path на macOS без `libomp`;
- Copernicus/CDS не вызывается в default path;
- `.gitignore` не дает случайно отправить hackathon data, outputs, weights и
  deliverables в публичный GitHub.

## 8. Что именно отдавать

Публичный GitHub:

- код;
- `README.md`;
- `run_solution.sh`;
- `docs/FINAL_SOLUTION_NOTE.md`;
- `physics/`;
- `dataset/README.md`;
- без исходных CSV и без `model_weights`.

ZIP/локальная папка:

- все из GitHub;
- `dataset/*.csv`;
- `model_weights/`;
- `outputs/predictions_q1.csv`;
- `outputs/predictions_may18.csv`;
- `reports/solution_note.txt`;
- `DATA_MANIFEST.json`.

Готовая локальная папка собрана как:

```text
deliverables/final_solution_v14_nocds/
```
