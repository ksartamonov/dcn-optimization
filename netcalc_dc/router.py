"""
Функции для оценки маршрутов и проверки допустимости решений.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .arrival_service import (
    ArrivalCurve,
    ServiceCurve,
    aggregate_service_curve,
    solve_time_stopping,
    TimeStoppingError,
)
from .flows import Flow
from .topology import extract_edge_params


Route = List[str]
Assignment = Dict[str, Route]


def service_curve_for_path(graph, path: Sequence[str]) -> ServiceCurve:
    """Строит эквивалентную сервис-кривую для заданного маршрута."""
    edges = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph[u][v]
        edges.append(ServiceCurve(rate=edge_data["capacity"], latency=edge_data["latency"]))
    return aggregate_service_curve(edges)


def base_delay(flow: Flow, path: Sequence[str], graph) -> float:
    """Верхняя оценка задержки без учёта взаимодействия потоков."""
    service = service_curve_for_path(graph, path)
    return flow.arrival.delay_upper_bound(service)


def path_capacity_margin(flow: Flow, path: Sequence[str], graph) -> float:
    """Находит минимальный запас пропускной способности маршрута."""
    min_margin = float("inf")
    for u, v in zip(path[:-1], path[1:]):
        capacity = graph[u][v]["capacity"]
        margin = capacity - flow.arrival.rate
        min_margin = min(min_margin, margin)
    return min_margin


def is_path_feasible(flow: Flow, path: Sequence[str], graph) -> bool:
    """Проверяет условие r < R на всём маршруте."""
    return path_capacity_margin(flow, path, graph) > 0


def compute_edge_loads(
    flows: Sequence[Flow],
    assignment: Assignment,
    graph,
) -> Dict[Tuple[str, str], float]:
    """
    Возвращает словарь нагрузок на рёбрах при заданном назначении маршрутов.
    """
    loads: Dict[Tuple[str, str], float] = {}
    for flow in flows:
        path = assignment.get(flow.flow_id)
        if not path:
            continue
        for u, v in zip(path[:-1], path[1:]):
            loads[(u, v)] = loads.get((u, v), 0.0) + flow.arrival.rate
    return loads


def influence_matrix(
    flows: Sequence[Flow],
    assignment: Assignment,
    graph,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует матрицу влияния A и базовый вектор a для метода time-stopping.
    """
    n = len(flows)
    A = np.zeros((n, n), dtype=float)
    a = np.zeros(n, dtype=float)

    edge_cache: Dict[str, List[Tuple[str, str]]] = {}
    for idx, flow in enumerate(flows):
        path = assignment[flow.flow_id]
        edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
        edge_cache[flow.flow_id] = edges
        params = extract_edge_params(graph, path)
        min_capacity = min(p.capacity for p in params)
        total_latency = sum(p.latency for p in params)
        margin = min_capacity - flow.arrival.rate
        if margin <= 0:
            a[idx] = float("inf")
        else:
            a[idx] = total_latency + flow.arrival.burst / margin

    for i, flow_i in enumerate(flows):
        for j, flow_j in enumerate(flows):
            if i == j:
                continue
            shared = set(edge_cache[flow_i.flow_id]) & set(edge_cache[flow_j.flow_id])
            if not shared:
                continue
            max_ratio = 0.0
            for u, v in shared:
                capacity = graph[u][v]["capacity"]
                ratio = flow_j.arrival.rate / capacity
                max_ratio = max(max_ratio, ratio)
            A[i, j] = max_ratio

    return A, a


def evaluate_assignment(
    flows: Sequence[Flow],
    assignment: Assignment,
    graph,
) -> Dict[str, float]:
    """
    Возвращает словарь метрик для заданного маршрутизационного решения.
    """
    base_delays = []
    max_margin = float("inf")
    for flow in flows:
        path = assignment[flow.flow_id]
        base_delays.append(base_delay(flow, path, graph))
        max_margin = min(max_margin, path_capacity_margin(flow, path, graph))

    try:
        influence, base_vector = influence_matrix(flows, assignment, graph)
        delays = solve_time_stopping(influence, base_vector)
    except TimeStoppingError:
        delays = np.full(len(flows), float("inf"))

    max_delay = float(np.max(delays))
    avg_delay = float(np.mean(delays))
    sla_violations = sum(
        delay > flow.sla_delay for delay, flow in zip(delays, flows)
    ) / max(len(flows), 1)

    return {
        "max_delay": max_delay,
        "avg_delay": avg_delay,
        "sla_violations": sla_violations,
        "min_capacity_margin": max_margin,
    }
