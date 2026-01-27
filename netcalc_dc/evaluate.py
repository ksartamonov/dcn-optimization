"""
Запуск экспериментальных сценариев по YAML-конфигурации.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import json
import time

import pandas as pd
import yaml

from . import heuristics, router, topology
from .flows import Flow, generate_flows


@dataclass
class ExperimentConfig:
    topology: Dict
    flows: Dict
    algorithms: Dict
    evaluation: Dict


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        topology=data["topology"],
        flows=data["flows"],
        algorithms=data["algorithms"],
        evaluation=data["evaluation"],
    )


def build_topology(config: Dict) -> "nx.DiGraph":
    topo_type = config["type"]
    if topo_type == "fat-tree":
        return topology.generate_fat_tree(
            k=config["k"],
            capacity_profile=config["capacity_profile"],
            latency_profile=config["latency_profile"],
        )
    if topo_type == "clos":
        return topology.generate_clos(
            stages=config["stages"],
            capacity=config["capacity"],
            latency=config["latency"],
            seed=config.get("seed"),
        )
    if topo_type in {"spine-leaf", "spine_leaf"}:
        return topology.generate_spine_leaf(
            leaf_count=config["leaf_count"],
            spine_count=config["spine_count"],
            servers_per_leaf=config["servers_per_leaf"],
            capacity_profile=config["capacity_profile"],
            latency_profile=config["latency_profile"],
        )
    if topo_type == "custom":
        return topology.load_custom_topology(Path(config["path"]))
    raise ValueError(f"Unsupported topology type: {topo_type}")


def build_candidates(graph, flows: Sequence[Flow], k_paths: int) -> Dict[str, List[List[str]]]:
    candidates: Dict[str, List[List[str]]] = {}
    for flow in flows:
        paths = topology.k_shortest_paths(graph, flow.source, flow.target, k_paths)
        if not paths:
            raise RuntimeError(f"No feasible paths for flow {flow.flow_id}")
        candidates[flow.flow_id] = paths
    return candidates


def scale_flows(flows: Sequence[Flow], factor: float) -> List[Flow]:
    scaled = []
    for flow in flows:
        scaled.append(
            Flow(
                flow_id=flow.flow_id,
                source=flow.source,
                target=flow.target,
                arrival=type(flow.arrival)(  # reuse ArrivalCurve dataclass
                    rate=flow.arrival.rate * factor,
                    burst=flow.arrival.burst * factor,
                ),
                sla_delay=flow.sla_delay,
                class_name=flow.class_name,
            )
        )
    return scaled


def run_algorithms(
    graph,
    flows: Sequence[Flow],
    candidates: Dict[str, List[List[str]]],
    algorithms_cfg: Dict,
) -> Dict[str, router.Assignment]:
    assignments: Dict[str, router.Assignment] = {}

    heuristics_list = algorithms_cfg.get("heuristics", [])
    for name in heuristics_list:
        try:
            if name == "greedy":
                assignments["greedy"] = heuristics.greedy_routing(graph, flows, candidates)
            elif name == "local_search":
                greedy = assignments.get("greedy") or heuristics.greedy_routing(graph, flows, candidates)
                assignments["local_search"] = heuristics.local_search(
                    graph, flows, candidates, greedy
                )
            elif name == "grasp":
                assignments["grasp"] = heuristics.grasp(graph, flows, candidates)
            elif name == "ga":
                assignments["ga"] = heuristics.genetic_algorithm(graph, flows, candidates)
            else:
                raise ValueError(f"Unsupported heuristic: {name}")
        except RuntimeError as exc:
            print(f"[warning] heuristic '{name}' failed: {exc}")
        except ValueError as exc:
            print(f"[warning] heuristic '{name}' skipped: {exc}")

    ilp_cfg = algorithms_cfg.get("ilp", {})
    if ilp_cfg.get("enabled"):
        from .ilp import solve_min_max_delay

        try:
            assignments["ilp"] = solve_min_max_delay(
                graph,
                list(flows),
                candidates,
                time_limit=ilp_cfg.get("timeout"),
            )
        except RuntimeError as exc:
            print(f"[warning] ILP baseline failed: {exc}")
    return assignments


def run_algorithms_timed(
    graph,
    flows: Sequence[Flow],
    candidates: Dict[str, List[List[str]]],
    algorithms_cfg: Dict,
) -> Tuple[Dict[str, router.Assignment], Dict[str, float]]:
    """Запускает алгоритмы и возвращает назначения и время (мс) по каждому алгоритму."""
    assignments: Dict[str, router.Assignment] = {}
    durations_ms: Dict[str, float] = {}

    heuristics_list = algorithms_cfg.get("heuristics", [])
    for name in heuristics_list:
        try:
            start = time.perf_counter()
            if name == "greedy":
                assignments["greedy"] = heuristics.greedy_routing(graph, flows, candidates)
            elif name == "local_search":
                greedy = assignments.get("greedy")
                if greedy is None:
                    greedy_start = time.perf_counter()
                    greedy = heuristics.greedy_routing(graph, flows, candidates)
                    assignments["greedy"] = greedy
                    durations_ms.setdefault("greedy", (time.perf_counter() - greedy_start) * 1000.0)
                assignments["local_search"] = heuristics.local_search(
                    graph, flows, candidates, greedy
                )
            elif name == "grasp":
                assignments["grasp"] = heuristics.grasp(graph, flows, candidates)
            elif name == "ga":
                assignments["ga"] = heuristics.genetic_algorithm(graph, flows, candidates)
            else:
                raise ValueError(f"Unsupported heuristic: {name}")
            durations_ms[name] = (time.perf_counter() - start) * 1000.0
        except RuntimeError as exc:
            print(f"[warning] heuristic '{name}' failed: {exc}")
        except ValueError as exc:
            print(f"[warning] heuristic '{name}' skipped: {exc}")

    ilp_cfg = algorithms_cfg.get("ilp", {})
    if ilp_cfg.get("enabled"):
        from .ilp import solve_min_max_delay

        try:
            start = time.perf_counter()
            assignments["ilp"] = solve_min_max_delay(
                graph,
                list(flows),
                candidates,
                time_limit=ilp_cfg.get("timeout"),
            )
            durations_ms["ilp"] = (time.perf_counter() - start) * 1000.0
        except RuntimeError as exc:
            print(f"[warning] ILP baseline failed: {exc}")
    return assignments, durations_ms


ProgressCallback = Callable[[int, int, int, int, float, str], None]


def run_experiment(
    config_path: Path,
    output_dir: Path | None = None,
    *,
    progress_cb: ProgressCallback | None = None,
    measure_timings: bool = False,
) -> pd.DataFrame:
    """
    Запускает полный эксперимент и возвращает таблицу результатов.
    """
    cfg = load_config(config_path)
    graph = build_topology(cfg.topology)
    repetitions = cfg.evaluation.get("repetitions", 1)
    load_factors = cfg.evaluation.get("load_factors", [1.0])
    k_paths = cfg.evaluation.get("k_paths", 5)

    all_records = []
    route_examples: Dict[str, Dict] = {}
    base_seed = cfg.flows.get("seed", 0)

    load_total = len(load_factors)

    for rep in range(repetitions):
        flows_seed = base_seed + rep
        base_flows = generate_flows(
            list(graph.nodes),
            cfg.flows["count"],
            seed=flows_seed,
        )
        candidates = build_candidates(graph, base_flows, k_paths)
        algorithms_results = {}

        for load_idx, factor in enumerate(load_factors):
            if progress_cb:
                progress_cb(rep, repetitions, load_idx, load_total, factor, "start")
            try:
                scaled = scale_flows(base_flows, factor)
                if measure_timings:
                    assignments, durations_ms = run_algorithms_timed(
                        graph, scaled, candidates, cfg.algorithms
                    )
                else:
                    assignments = run_algorithms(graph, scaled, candidates, cfg.algorithms)
                    durations_ms = {}
            except Exception:
                if progress_cb:
                    progress_cb(rep, repetitions, load_idx, load_total, factor, "error")
                raise
            if not algorithms_results:
                algorithms_results = assignments
            for algo_name, assignment in assignments.items():
                metrics = router.evaluate_assignment(scaled, assignment, graph)
                all_records.append(
                    {
                        "repetition": rep,
                        "load_factor": factor,
                        "algorithm": algo_name,
                        "k_paths": k_paths,
                        "duration_ms": durations_ms.get(algo_name),
                        **metrics,
                    }
                )
                if algo_name not in route_examples:
                    for flow in scaled:
                        path = assignment.get(flow.flow_id)
                        if path:
                            route_examples[algo_name] = {
                                "repetition": rep,
                                "load_factor": factor,
                                "flow_id": flow.flow_id,
                                "class": flow.class_name,
                                "rate": flow.arrival.rate,
                                "sla": flow.sla_delay,
                                "route": path,
                            }
                            break
            if progress_cb:
                progress_cb(rep, repetitions, load_idx, load_total, factor, "done")

    results = pd.DataFrame.from_records(all_records)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "metrics.csv"
        results.to_csv(csv_path, index=False)
        if route_examples:
            routes_path = output_dir / "route_examples.json"
            routes_path.write_text(json.dumps(route_examples, indent=2), encoding="utf-8")
    return results
