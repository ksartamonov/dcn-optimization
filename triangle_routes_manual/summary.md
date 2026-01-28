# Примеры маршрутов (custom)
* Коэффициент нагрузки: λ=1.0
* k_paths = 3

## greedy
| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |
| --- | --- | --- | --- | --- | --- |
| flow_0 | node0 → node1 | node0 → node1 | 1 | 5.000 | 0.256 |
| flow_1 | node0 → node2 | node0 → node2 | 1 | 5.000 | 0.256 |
| flow_2 | node1 → node0 | node1 → node0 | 1 | 5.000 | 0.256 |
| flow_3 | node1 → node2 | node1 → node2 | 1 | 5.000 | 0.256 |
| flow_4 | node2 → node0 | node2 → node0 | 1 | 5.000 | 0.256 |
| flow_5 | node2 → node1 | node2 → node1 | 1 | 5.000 | 0.256 |

## local_search
| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |
| --- | --- | --- | --- | --- | --- |
| flow_0 | node0 → node1 | node0 → node1 | 1 | 5.000 | 0.256 |
| flow_1 | node0 → node2 | node0 → node2 | 1 | 5.000 | 0.256 |
| flow_2 | node1 → node0 | node1 → node0 | 1 | 5.000 | 0.256 |
| flow_3 | node1 → node2 | node1 → node2 | 1 | 5.000 | 0.256 |
| flow_4 | node2 → node0 | node2 → node0 | 1 | 5.000 | 0.256 |
| flow_5 | node2 → node1 | node2 → node1 | 1 | 5.000 | 0.256 |

## grasp
| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |
| --- | --- | --- | --- | --- | --- |
| flow_0 | node0 → node1 | node0 → node1 | 1 | 5.000 | 0.256 |
| flow_1 | node0 → node2 | node0 → node2 | 1 | 5.000 | 0.256 |
| flow_2 | node1 → node0 | node1 → node0 | 1 | 5.000 | 0.256 |
| flow_3 | node1 → node2 | node1 → node2 | 1 | 5.000 | 0.256 |
| flow_4 | node2 → node0 | node2 → node0 | 1 | 5.000 | 0.256 |
| flow_5 | node2 → node1 | node2 → node1 | 1 | 5.000 | 0.256 |

## ga
| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |
| --- | --- | --- | --- | --- | --- |
| flow_0 | node0 → node1 | node0 → node1 | 1 | 5.000 | 0.256 |
| flow_1 | node0 → node2 | node0 → node2 | 1 | 5.000 | 0.256 |
| flow_2 | node1 → node0 | node1 → node0 | 1 | 5.000 | 0.256 |
| flow_3 | node1 → node2 | node1 → node2 | 1 | 5.000 | 0.256 |
| flow_4 | node2 → node0 | node2 → node0 | 1 | 5.000 | 0.256 |
| flow_5 | node2 → node1 | node2 → node1 | 1 | 5.000 | 0.256 |

## ilp
| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |
| --- | --- | --- | --- | --- | --- |
| flow_0 | node0 → node1 | node0 → node1 | 1 | 5.000 | 0.256 |
| flow_1 | node0 → node2 | node0 → node2 | 1 | 5.000 | 0.256 |
| flow_2 | node1 → node0 | node1 → node0 | 1 | 5.000 | 0.256 |
| flow_3 | node1 → node2 | node1 → node2 | 1 | 5.000 | 0.256 |
| flow_4 | node2 → node0 | node2 → node0 | 1 | 5.000 | 0.256 |
| flow_5 | node2 → node1 | node2 → node1 | 1 | 5.000 | 0.256 |
