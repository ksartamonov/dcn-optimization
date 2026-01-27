"""
Генерация и работа с топологиями центров обработки данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import json
import random

import networkx as nx


Edge = Tuple[str, str]


@dataclass(frozen=True)
class LinkParams:
    """Параметры канала: пропускная способность и латентность."""

    capacity: float
    latency: float


def _add_edge(graph: nx.DiGraph, u: str, v: str, params: LinkParams) -> None:
    graph.add_edge(u, v, capacity=params.capacity, latency=params.latency)
    graph.add_edge(v, u, capacity=params.capacity, latency=params.latency)


def generate_clos(
    stages: Sequence[int],
    capacity: float,
    latency: float,
    *,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Генерирует трёхуровневую Clos-топологию.

    :param stages: количество коммутаторов на уровнях (доступ, агрегация, ядро)
    :param capacity: пропускная способность всех каналов (Гбит/с)
    :param latency: латентность каналов (мс)
    """
    if len(stages) != 3:
        raise ValueError("Clos topology requires exactly three stage sizes")
    rng = random.Random(seed)
    graph = nx.DiGraph()
    access, aggregation, core = stages

    access_nodes = [f"edge{i}" for i in range(access)]
    aggregation_nodes = [f"agg{i}" for i in range(aggregation)]
    core_nodes = [f"core{i}" for i in range(core)]
    servers = [f"server{i}" for i in range(access * 2)]

    graph.add_nodes_from(access_nodes + aggregation_nodes + core_nodes + servers)

    params = LinkParams(capacity=capacity, latency=latency)
    for idx, server in enumerate(servers):
        edge = access_nodes[idx // 2]
        _add_edge(graph, server, edge, params)

    for edge in access_nodes:
        agg_subset = rng.sample(aggregation_nodes, k=min(len(aggregation_nodes), 4))
        for agg in agg_subset:
            _add_edge(graph, edge, agg, params)

    for agg in aggregation_nodes:
        for core_node in core_nodes:
            _add_edge(graph, agg, core_node, params)

    return graph


def generate_fat_tree(
    k: int,
    capacity_profile: Dict[str, float],
    latency_profile: Dict[str, float],
) -> nx.DiGraph:
    """
    Генерирует классическую Fat-Tree топологию с параметром k.

    :param k: число портов на коммутаторах доступа и агрегации, k должно быть четным
    :param capacity_profile: словарь с ключами access/aggregation/core
    :param latency_profile: словарь с ключами access/aggregation/core
    """
    if k % 2 != 0 or k < 2:
        raise ValueError("k must be an even integer >= 2")
    graph = nx.DiGraph()
    pods = k
    access_per_pod = k // 2
    aggregation_per_pod = k // 2
    core_switches = (k // 2) ** 2

    for pod in range(pods):
        for idx in range(access_per_pod):
            graph.add_node(f"edge_{pod}_{idx}")
        for idx in range(aggregation_per_pod):
            graph.add_node(f"agg_{pod}_{idx}")

    for idx in range(core_switches):
        graph.add_node(f"core_{idx}")

    # Добавляем серверы
    for pod in range(pods):
        for edge_idx in range(access_per_pod):
            edge_name = f"edge_{pod}_{edge_idx}"
            for host in range(access_per_pod):
                server_name = f"server_{pod}_{edge_idx}_{host}"
                graph.add_node(server_name)
                _add_edge(
                    graph,
                    server_name,
                    edge_name,
                    LinkParams(
                        capacity=capacity_profile["access"],
                        latency=latency_profile["access"],
                    ),
                )

    # Соединяем уровень доступа и агрегации
    for pod in range(pods):
        for edge_idx in range(access_per_pod):
            edge_name = f"edge_{pod}_{edge_idx}"
            for agg_idx in range(aggregation_per_pod):
                agg_name = f"agg_{pod}_{agg_idx}"
                _add_edge(
                    graph,
                    edge_name,
                    agg_name,
                    LinkParams(
                        capacity=capacity_profile["access"],
                        latency=latency_profile["access"],
                    ),
                )

    # Соединяем агрегацию с ядром
    for pod in range(pods):
        for agg_idx in range(aggregation_per_pod):
            agg_name = f"agg_{pod}_{agg_idx}"
            for group in range(access_per_pod):
                core_idx = agg_idx * access_per_pod + group
                core_name = f"core_{core_idx}"
                _add_edge(
                    graph,
                    agg_name,
                    core_name,
                    LinkParams(
                        capacity=capacity_profile["aggregation"],
                        latency=latency_profile["aggregation"],
                    ),
                )
                _add_edge(
                    graph,
                    core_name,
                    agg_name,
                    LinkParams(
                        capacity=capacity_profile["core"],
                        latency=latency_profile["core"],
                    ),
                )

    return graph


def generate_spine_leaf(
    leaf_count: int,
    spine_count: int,
    servers_per_leaf: int,
    capacity_profile: Dict[str, float],
    latency_profile: Dict[str, float],
) -> nx.DiGraph:
    """
    Строит топологию класса spine-leaf.

    :param leaf_count: число leaf-коммутаторов (уровень доступа)
    :param spine_count: число spine-коммутаторов (уровень ядра)
    :param servers_per_leaf: число серверов, подключаемых к каждому leaf-коммутатору
    :param capacity_profile: словарь с ключами ``access`` (server→leaf) и ``fabric`` (leaf↔spine)
    :param latency_profile: словарь с ключами ``access`` и ``fabric``, задержки в мс
    """
    if leaf_count <= 0 or spine_count <= 0 or servers_per_leaf <= 0:
        raise ValueError("leaf_count, spine_count and servers_per_leaf must be positive integers")
    for key in ("access", "fabric"):
        if key not in capacity_profile or key not in latency_profile:
            raise ValueError(f"capacity_profile and latency_profile must contain '{key}' entry for spine-leaf topology")

    graph = nx.DiGraph()
    leaf_nodes = [f"leaf_{idx}" for idx in range(leaf_count)]
    spine_nodes = [f"spine_{idx}" for idx in range(spine_count)]

    graph.add_nodes_from(leaf_nodes + spine_nodes)

    # Серверы на каждой leaf-стойке
    for leaf_idx, leaf in enumerate(leaf_nodes):
        for srv_idx in range(servers_per_leaf):
            server = f"server_{leaf_idx}_{srv_idx}"
            graph.add_node(server)
            _add_edge(
                graph,
                server,
                leaf,
                LinkParams(
                    capacity=capacity_profile["access"],
                    latency=latency_profile["access"],
                ),
            )

    # Соединение leaf ↔ spine (полносвязная схема)
    for leaf in leaf_nodes:
        for spine in spine_nodes:
            _add_edge(
                graph,
                leaf,
                spine,
                LinkParams(
                    capacity=capacity_profile["fabric"],
                    latency=latency_profile["fabric"],
                ),
            )

    return graph


def load_custom_topology(path: Path) -> nx.DiGraph:
    """
    Загружает пользовательскую топологию в формате JSON.

    Формат файла:
    {
        "nodes": ["n1", "n2", ...],
        "edges": [
            {"u": "n1", "v": "n2", "capacity": 10.0, "latency": 0.2},
            ...
        ]
    }
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    graph.add_nodes_from(data["nodes"])
    for edge in data["edges"]:
        params = LinkParams(capacity=edge["capacity"], latency=edge["latency"])
        _add_edge(graph, edge["u"], edge["v"], params)
    return graph


def k_shortest_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    k: int,
    weight: str = "latency",
) -> List[List[str]]:
    """
    Возвращает k кратчайших путей между двумя узлами графа (без повторов).
    """
    if source == target:
        return [[source]]
    try:
        generator: Iterator[List[str]] = nx.shortest_simple_paths(
            graph, source, target, weight=weight
        )
    except nx.NetworkXNoPath:
        return []
    result: List[List[str]] = []
    for _ in range(k):
        try:
            result.append(next(generator))
        except StopIteration:
            break
    return result


def annotate_random_capacity(
    graph: nx.DiGraph,
    *,
    capacity_range: Tuple[float, float],
    latency_range: Tuple[float, float],
    seed: int | None = None,
) -> None:
    """
    Перезаписывает параметры каналов случайными значениями в заданных диапазонах.
    """
    rng = random.Random(seed)
    for u, v in graph.edges():
        capacity = rng.uniform(*capacity_range)
        latency = rng.uniform(*latency_range)
        graph[u][v]["capacity"] = capacity
        graph[u][v]["latency"] = latency


def extract_edge_params(graph: nx.DiGraph, path: Sequence[str]) -> List[LinkParams]:
    """Возвращает список параметров каналов вдоль маршрута."""
    params: List[LinkParams] = []
    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        params.append(LinkParams(capacity=edge["capacity"], latency=edge["latency"]))
    return params
