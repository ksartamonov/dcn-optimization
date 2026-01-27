"""Вспомогательный скрипт для вывода примеров маршрутов по результатам алгоритмов."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import sys

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from netcalc_dc.evaluate import (  # type: ignore
        load_config,
        build_topology,
        build_candidates,
        scale_flows,
        run_algorithms,
    )
    from netcalc_dc.flows import generate_flows  # type: ignore
else:
    from .evaluate import (
        load_config,
        build_topology,
        build_candidates,
        scale_flows,
        run_algorithms,
    )
    from .flows import generate_flows


def collect_routes(
    config_path: Path,
    *,
    load_factor: float | None = None,
    flows_limit: int = 3,
) -> dict[str, list[tuple[str, list[str]]]]:
    cfg = load_config(config_path)
    graph = build_topology(cfg.topology)
    repetitions = cfg.evaluation.get("repetitions", 1)
    load_factors: Sequence[float] = cfg.evaluation.get("load_factors", [1.0])
    target_load = load_factor or load_factors[0]

    base_seed = cfg.flows.get("seed", 0)
    flows = generate_flows(list(graph.nodes), cfg.flows["count"], seed=base_seed)
    candidates = build_candidates(graph, flows, cfg.evaluation.get("k_paths", 5))
    scaled = scale_flows(flows, target_load)
    assignments = run_algorithms(graph, scaled, candidates, cfg.algorithms)

    result: dict[str, list[tuple[str, list[str]]]] = {}
    for algo_name, assignment in assignments.items():
        samples: list[tuple[str, list[str]]] = []
        for flow in scaled[:flows_limit]:
            path = assignment.get(flow.flow_id)
            if not path:
                continue
            samples.append((flow.flow_id, path))
        result[algo_name] = samples
    return result


def print_routes(routes: dict[str, list[tuple[str, list[str]]]]) -> None:
    for algo, samples in routes.items():
        print(f"\n=== {algo} ===")
        if not samples:
            print("(нет данных)")
            continue
        for flow_id, path in samples:
            print(f"{flow_id}: {' -> '.join(path)}")


def main() -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Показ примеров маршрутов для сценария")
    parser.add_argument("--config", type=Path, required=True, help="Путь к config.yaml")
    parser.add_argument(
        "--load-factor",
        type=float,
        help="Использовать заданное значение λ (по умолчанию первый элемент evaluation.load_factors)",
    )
    parser.add_argument(
        "--flows-limit",
        type=int,
        default=3,
        help="Сколько потоков выводить для каждого алгоритма",
    )
    args = parser.parse_args()

    routes = collect_routes(
        args.config,
        load_factor=args.load_factor,
        flows_limit=args.flows_limit,
    )
    print_routes(routes)


if __name__ == "__main__":  # pragma: no cover
    main()
