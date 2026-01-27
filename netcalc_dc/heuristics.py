"""
Эвристические алгоритмы маршрутизации потоков.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .flows import Flow
from . import router


Route = List[str]
CandidateRoutes = Dict[str, List[Route]]
Assignment = Dict[str, Route]


def _edge_iter(path: Sequence[str]) -> Iterable[Tuple[str, str]]:
    return zip(path[:-1], path[1:])


def greedy_routing(
    graph,
    flows: Sequence[Flow],
    candidates: CandidateRoutes,
) -> Assignment:
    """
    Жадный алгоритм: обрабатывает потоки по убыванию скорости и выбирает
    маршрут с минимальной задержкой, не нарушающий ограничение по ёмкости.
    """
    assignment: Assignment = {}
    edge_loads: Dict[Tuple[str, str], float] = defaultdict(float)
    for flow in sorted(flows, key=lambda f: f.arrival.rate, reverse=True):
        best_path = None
        best_delay = math.inf
        for path in candidates.get(flow.flow_id, []):
            if not router.is_path_feasible(flow, path, graph):
                continue
            feasible = True
            for u, v in _edge_iter(path):
                capacity = graph[u][v]["capacity"]
                if edge_loads[(u, v)] + flow.arrival.rate >= capacity:
                    feasible = False
                    break
            if not feasible:
                continue
            delay = router.base_delay(flow, path, graph)
            if delay < best_delay:
                best_delay = delay
                best_path = path
        if best_path is None:
            # fallback: минимальная задержка без учёта текущей загрузки
            candidate_paths = candidates.get(flow.flow_id, [])
            if not candidate_paths:
                raise ValueError(f"No candidate paths for flow {flow.flow_id}")
            best_path = min(
                candidate_paths, key=lambda p: router.base_delay(flow, p, graph)
            )
        assignment[flow.flow_id] = list(best_path)
        for u, v in _edge_iter(best_path):
            edge_loads[(u, v)] += flow.arrival.rate
    return assignment


def local_search(
    graph,
    flows: Sequence[Flow],
    candidates: CandidateRoutes,
    initial: Assignment,
    *,
    iterations: int = 20,
) -> Assignment:
    """Локальный поиск с перестановкой маршрутов."""
    best_assignment = {k: list(v) for k, v in initial.items()}
    best_score = router.evaluate_assignment(flows, best_assignment, graph)["max_delay"]
    rng = random.Random(0)
    for _ in range(iterations):
        flow = rng.choice(flows)
        current_path = best_assignment[flow.flow_id]
        improved = False
        for path in candidates.get(flow.flow_id, []):
            if path == current_path:
                continue
            trial_assignment = {k: list(v) for k, v in best_assignment.items()}
            trial_assignment[flow.flow_id] = list(path)
            metrics = router.evaluate_assignment(flows, trial_assignment, graph)
            score = metrics["max_delay"] + 10.0 * metrics["sla_violations"]
            if score < best_score:
                best_assignment = trial_assignment
                best_score = score
                improved = True
                break
        if not improved:
            continue
    return best_assignment


def grasp(
    graph,
    flows: Sequence[Flow],
    candidates: CandidateRoutes,
    *,
    iterations: int = 30,
    alpha: float = 0.3,
) -> Assignment:
    """
    GRASP: на каждой итерации строит решение с рандомизацией и применяет локальный поиск.
    """
    rng = random.Random(42)
    best_assignment: Assignment | None = None
    best_score = math.inf

    for _ in range(iterations):
        # Стохастический жадный выбор
        temp_assignment: Assignment = {}
        edge_loads: Dict[Tuple[str, str], float] = defaultdict(float)
        for flow in sorted(flows, key=lambda f: f.arrival.rate, reverse=True):
            ranked: List[Tuple[float, Route]] = []
            for path in candidates.get(flow.flow_id, []):
                if not router.is_path_feasible(flow, path, graph):
                    continue
                feasible = True
                for u, v in _edge_iter(path):
                    capacity = graph[u][v]["capacity"]
                    if edge_loads[(u, v)] + flow.arrival.rate >= capacity:
                        feasible = False
                        break
                if not feasible:
                    continue
                delay = router.base_delay(flow, path, graph)
                ranked.append((delay, path))
            if not ranked:
                ranked = [
                    (router.base_delay(flow, path, graph), path)
                    for path in candidates.get(flow.flow_id, [])
                ]
            ranked.sort(key=lambda item: item[0])
            rcl_len = max(1, int(len(ranked) * alpha))
            delay, path = rng.choice(ranked[:rcl_len])
            temp_assignment[flow.flow_id] = list(path)
            for u, v in _edge_iter(path):
                edge_loads[(u, v)] += flow.arrival.rate

        improved = local_search(graph, flows, candidates, temp_assignment, iterations=5)
        metrics = router.evaluate_assignment(flows, improved, graph)
        score = metrics["max_delay"] + 10.0 * metrics["sla_violations"]
        if score < best_score:
            best_score = score
            best_assignment = improved

    if best_assignment is None:
        raise RuntimeError("GRASP failed to construct any assignment")
    return best_assignment


@dataclass
class GAParams:
    population: int = 60
    generations: int = 200
    crossover: float = 0.8
    mutation: float = 0.1
    tournament: int = 3
    seed: int = 7


def _fitness(
    graph,
    flows: Sequence[Flow],
    candidates: CandidateRoutes,
    chromosome: Sequence[int],
) -> float:
    assignment: Assignment = {}
    for gene, flow in zip(chromosome, flows):
        routes = candidates[flow.flow_id]
        assignment[flow.flow_id] = list(routes[gene % len(routes)])
    metrics = router.evaluate_assignment(flows, assignment, graph)
    penalty = 1000.0 if math.isinf(metrics["max_delay"]) else 0.0
    return metrics["max_delay"] + 20.0 * metrics["sla_violations"] + penalty


def genetic_algorithm(
    graph,
    flows: Sequence[Flow],
    candidates: CandidateRoutes,
    params: GAParams | None = None,
) -> Assignment:
    """Генетический алгоритм с кодированием маршрутов индексами."""
    params = params or GAParams()
    rng = random.Random(params.seed)
    population: List[List[int]] = []
    for _ in range(params.population):
        chromosome = [
            rng.randrange(len(candidates[flow.flow_id])) for flow in flows
        ]
        population.append(chromosome)

    def tournament_select() -> List[int]:
        contenders = rng.sample(population, params.tournament)
        best = min(contenders, key=lambda chrom: _fitness(graph, flows, candidates, chrom))
        return list(best)

    for _ in range(params.generations):
        new_population: List[List[int]] = []
        while len(new_population) < params.population:
            parent1 = tournament_select()
            parent2 = tournament_select()
            if rng.random() < params.crossover:
                point = rng.randrange(1, len(flows))
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1, child2 = parent1, parent2
            for child in (child1, child2):
                if rng.random() < params.mutation:
                    idx = rng.randrange(len(flows))
                    child[idx] = rng.randrange(len(candidates[flows[idx].flow_id]))
            new_population.extend([child1, child2])
        population = new_population[: params.population]

    best_chromosome = min(population, key=lambda chrom: _fitness(graph, flows, candidates, chrom))
    assignment: Assignment = {}
    for gene, flow in zip(best_chromosome, flows):
        assignment[flow.flow_id] = list(candidates[flow.flow_id][gene % len(candidates[flow.flow_id])])
    return assignment
