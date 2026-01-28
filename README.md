# netcalc_dc_release

## Состав

- `netcalc_dc/` — пакет с ядром системы (импортируется скриптами):
  - `topology.py` — генерация топологий Clos/Fat-Tree/Spine-Leaf
  - `flows.py` — генерация потоков (классы трафика, SLA)
  - `arrival_service.py` — модели b1/b2-кривых
  - `router.py` — модель назначения маршрутов
  - `heuristics.py` — greedy / local_search / GRASP / GA
  - `ilp.py` — ILP-бейзлайн
  - `evaluate.py` — запуск одного эксперимента
  - `plots.py` — построение графиков
  - `main.py` — сохранение отчетов
- `scripts/` — исполняемые сценарии (подхватывают пакет `netcalc_dc/` через sys.path):
  - `suite.py` — пакетный прогон экспериментов
  - `time_suite.py` — прогон с измерением времени
  - `time_suite_failures.py` — прогон с отказами агрегации + время
  - `failure_scenarios.py` — одиночные сценарии отказов/усиления
  - `route_samples.py` — визуализация маршрутов
  - `scenario_routes.py` — набор фиксированных маршрутовых сценариев
  - `inspect_routes.py` — просмотр маршрутов
- `configs/` — базовые конфиги
- `docs/` — подробная документация (pipeline и описание модели)
- `requirements.txt` — зависимости

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Базовый запуск экспериментов

```bash
python3 scripts/suite.py -o experiment_suite
```

## Пользовательская топология (JSON)

Можно задавать топологию в JSON. Формат:

```json
{
  "nodes": ["a", "b", "c"],
  "edges": [
    {"u": "a", "v": "b", "capacity": 10.0, "latency": 0.2},
    {"u": "b", "v": "c", "capacity": 10.0, "latency": 0.2}
  ]
}
```

Пример сохранён в `configs/custom_topology.json`.  
Для использования укажите в YAML:

```yaml
topology:
  type: custom
  path: "configs/custom_topology.json"
```

### Замер времени алгоритмов

```bash
python3 scripts/time_suite.py -o experiments_time --repetitions 10 --workers 4
```

### Отказы агрегации + время

```bash
python3 scripts/time_suite_failures.py -o experiments_time_fail \
  --include clos_n1_8 fat_tree_k4 \
  --repetitions 10 --workers 4 --k-paths 5 --loads 0.3 0.5 0.7
```

### Построение графиков (EN подписи)

```bash
python3 scripts/time_suite.py -o experiments_time --plots-only --lang en
```

## Структура выходных результатов

Каждый сценарий сохраняется в отдельную папку:

```
experiment_suite/
  clos_n1_8/
    k_paths_5/
      config.yaml
      metrics.csv
      report.json
      delay_vs_load.png
      sla_violations.png
```

`metrics.csv` содержит:
- `max_delay`, `avg_delay`
- `sla_violations`
- `algorithm`, `load_factor`
- (в time_suite) `duration_ms` — время выполнения алгоритма

## Примечания

- SLA задается в `flows.py` через классы трафика.
- Ограничение SLA **не является жестким** в ILP; оно проверяется постфактум.
- При `r_i >= R_e` для ребра задержка считается бесконечной.
