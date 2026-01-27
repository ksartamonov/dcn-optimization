"""
ILP-постановка задачи минимизации максимальной задержки для малых инстансов.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .flows import Flow
from . import router


def solve_min_max_delay(
    graph,
    flows: List[Flow],
    candidates: Dict[str, List[List[str]]],
    *,
    time_limit: int | None = None,
) -> Dict[str, List[str]]:
    """
    Решает задачу min-max задержки с использованием PuLP.

    Возвращает назначение маршрутов или выбрасывает исключение, если решатель недоступен.
    """
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - зависящее от окружения
        raise RuntimeError("PuLP is required for ILP baseline") from exc

    problem = pulp.LpProblem("MinMaxDelay", pulp.LpMinimize)
    decision_vars: Dict[Tuple[str, int], pulp.LpVariable] = {}

    for flow in flows:
        for idx, _ in enumerate(candidates[flow.flow_id]):
            decision_vars[(flow.flow_id, idx)] = pulp.LpVariable(
                f"x_{flow.flow_id}_{idx}", lowBound=0, upBound=1, cat="Binary"
            )

    z_var = pulp.LpVariable("Z", lowBound=0)
    problem += z_var

    # Каждому потоку назначается ровно один маршрут
    for flow in flows:
        problem += (
            pulp.lpSum(decision_vars[(flow.flow_id, idx)] for idx in range(len(candidates[flow.flow_id])))
            == 1,
            f"assign_{flow.flow_id}",
        )

    # Ограничения по пропускной способности рёбер
    edge_constraints: Dict[Tuple[str, str], pulp.LpAffineExpression] = {}
    for flow in flows:
        for idx, path in enumerate(candidates[flow.flow_id]):
            var = decision_vars[(flow.flow_id, idx)]
            for u, v in zip(path[:-1], path[1:]):
                edge_constraints.setdefault((u, v), 0)
                edge_constraints[(u, v)] += flow.arrival.rate * var
    for (u, v), expr in edge_constraints.items():
        capacity = graph[u][v]["capacity"]
        problem += expr <= capacity, f"capacity_{u}_{v}"

    # Ограничения по задержке
    for flow in flows:
        for idx, path in enumerate(candidates[flow.flow_id]):
            delay = router.base_delay(flow, path, graph)
            var = decision_vars[(flow.flow_id, idx)]
            problem += delay * var <= z_var, f"delay_{flow.flow_id}_{idx}"

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD()
    status = problem.solve(solver)
    if status != pulp.LpStatusOptimal and status != pulp.LpStatusNotSolved:
        raise RuntimeError(f"ILP solver failed with status: {pulp.LpStatus[status]}")

    assignment: Dict[str, List[str]] = {}
    for flow in flows:
        for idx, path in enumerate(candidates[flow.flow_id]):
            if pulp.value(decision_vars[(flow.flow_id, idx)]) > 0.5:
                assignment[flow.flow_id] = list(path)
                break
        else:  # pragma: no cover - защита от численных ошибок
            assignment[flow.flow_id] = list(candidates[flow.flow_id][0])
    return assignment
