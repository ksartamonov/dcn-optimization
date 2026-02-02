from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    REPO_ROOT = PROJECT_ROOT.parent
    for path in (PROJECT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from netcalc_dc.evaluate import (  # type: ignore
        build_topology,
        build_candidates,
        load_config,
        run_algorithms,
        scale_flows,
    )
    from netcalc_dc.flows import flows_from_config  # type: ignore
    from netcalc_dc import router  # type: ignore
    from netcalc_dc.arrival_service import solve_time_stopping, TimeStoppingError  # type: ignore
else:
    from .evaluate import (
        build_topology,
        build_candidates,
        load_config,
        run_algorithms,
        scale_flows,
    )
    from .flows import flows_from_config
    from . import router
    from .arrival_service import solve_time_stopping, TimeStoppingError


def _pick_flow_id(flows, flow_id: str | None) -> str:
    if flow_id:
        return flow_id
    return flows[0].flow_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Строит зависимость задержки выбранного потока от коэффициента нагрузки"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Путь к config.yaml (можно с ручными потоками)",
    )
    parser.add_argument(
        "--algorithm",
        default="ga",
        help="Алгоритм назначения путей (greedy/local_search/grasp/ga/ilp)",
    )
    parser.add_argument(
        "--flow-id",
        help="ID потока (по умолчанию первый в конфиге)",
    )
    parser.add_argument(
        "--loads",
        nargs="+",
        type=float,
        help="Список коэффициентов нагрузки (если не задано, берётся из config)",
    )
    parser.add_argument(
        "--k-paths",
        type=int,
        help="Переопределить k_paths из конфига",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("flow_delay_curve.png"),
        help="Файл для сохранения графика",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph = build_topology(cfg.topology)
    flows = flows_from_config(cfg.flows, list(graph.nodes), seed=cfg.flows.get("seed"))
    target_flow_id = _pick_flow_id(flows, args.flow_id)
    flow_index = {f.flow_id: i for i, f in enumerate(flows)}.get(target_flow_id)
    if flow_index is None:
        raise ValueError(f"Flow id {target_flow_id} not found in config")

    loads: Sequence[float] = args.loads or cfg.evaluation.get("load_factors", [1.0])
    k_paths = args.k_paths or cfg.evaluation.get("k_paths", 5)

    candidates = build_candidates(graph, flows, k_paths)
    delays = []

    for load in loads:
        scaled = scale_flows(flows, load)
        assignments = run_algorithms(graph, scaled, candidates, cfg.algorithms)
        if args.algorithm not in assignments:
            raise ValueError(f"Algorithm {args.algorithm} not available in config")
        assignment = assignments[args.algorithm]

        try:
            influence, base_vec = router.influence_matrix(scaled, assignment, graph)
            per_flow = solve_time_stopping(influence, base_vec)
            delay = float(per_flow[flow_index])
        except TimeStoppingError:
            delay = float("inf")

        delays.append(delay)

    plt.figure(figsize=(6, 4))
    plt.plot(loads, delays, marker="o", linewidth=2, color="#d32f2f")
    plt.xlabel("Load factor (ρ)")
    plt.ylabel("Delay bound (ms)")
    plt.title(f"Delay vs load for {target_flow_id} ({args.algorithm})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":  # pragma: no cover
    main()
