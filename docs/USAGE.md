# Usage Guide

## 1) Конфигурация

Конфиг задается в YAML. Обязательные блоки:

- `topology`: тип и параметры
- `flows`: число потоков и seed
- `algorithms`: список эвристик и ILP
- `evaluation`: коэффициенты нагрузки, k_paths, repetitions

Пример:
```yaml
topology:
  type: clos
  stages: [8, 4, 2]
  capacity: 10.0
  latency: 0.2
flows:
  count: 32
  seed: 2024
algorithms:
  heuristics: [greedy, local_search, grasp, ga]
  ilp:
    enabled: true
    timeout: 600
evaluation:
  repetitions: 10
  load_factors: [0.3, 0.5, 0.7, 0.9, 1.0]
  k_paths: 5
```

### Пользовательская топология (JSON)

Поддерживается формат:
```json
{
  "nodes": ["a", "b", "c"],
  "edges": [
    {"u": "a", "v": "b", "capacity": 10.0, "latency": 0.2},
    {"u": "b", "v": "c", "capacity": 10.0, "latency": 0.2}
  ]
}
```

Пример: `configs/custom_topology.json`.  
В конфиге:
```yaml
topology:
  type: custom
  path: "configs/custom_topology.json"
```

## 2) Логика эксперимента

1. Генерация топологии (Clos/Fat-Tree/Spine-Leaf).
2. Генерация потоков (3 класса трафика с SLA).
3. Масштабирование нагрузки по λ.
4. Генерация k кратчайших путей (Yen).
5. Фильтрация по пропускной способности и задержке.
6. Назначение маршрутов эвристиками и ILP.
7. Сбор метрик: `max_delay`, `avg_delay`, `sla_violations`, `duration_ms`.

## 3) Сценарии отказов

Скрипт `time_suite_failures.py` удаляет один или несколько агрегационных коммутаторов и прогоняет эксперимент по той же схеме.

Пример:
```bash
python3 scripts/time_suite_failures.py -o experiments_time_fail \
  --include clos_n1_8 \
  --repetitions 5 \
  --k-paths 5 \
  --loads 0.5 0.7 \
  --agg-counts 1 2
```

## 4) Визуализация маршрутов

```bash
python3 scripts/route_samples.py --config configs/config.yaml \
  --output routes_example.json --load-factor 0.7 --flows 5
```

Результаты сохраняются в папку `routes_example/` с JSON/PNG/MD.

## 5) Построение графиков

Для результатов `time_suite.py`:
```bash
python3 scripts/time_suite.py -o experiments_time --plots-only --lang en
```

## 6) Типовые метрики

- `max_delay`: верхняя оценка задержки (Network Calculus)
- `avg_delay`: средняя задержка по потокам
- `sla_violations`: доля потоков, нарушающих SLA
- `duration_ms`: время работы алгоритма
