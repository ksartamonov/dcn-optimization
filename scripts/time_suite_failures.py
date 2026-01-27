from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

import networkx as nx

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    import importlib

    suite = importlib.import_module("suite")  # type: ignore
    from netcalc_dc import evaluate  # type: ignore
    from netcalc_dc import plots  # type: ignore
else:
    from . import suite
    from . import evaluate
    from . import plots


def _graph_to_json(graph: nx.DiGraph) -> dict:
    nodes = list(graph.nodes())
    edges = []
    for u, v in graph.edges():
        data = graph[u][v]
        edges.append(
            {"u": u, "v": v, "capacity": data["capacity"], "latency": data["latency"]}
        )
    return {"nodes": nodes, "edges": edges}


def _save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.write_text(
        json.dumps(_graph_to_json(graph), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _aggregation_nodes(graph: nx.DiGraph) -> list[str]:
    return sorted(node for node in graph.nodes() if node.startswith("agg"))


def _remove_nodes(graph: nx.DiGraph, nodes: Sequence[str]) -> nx.DiGraph:
    modified = graph.copy()
    modified.remove_nodes_from([n for n in nodes if n in modified])
    return modified


def _build_custom_config(base_config: dict, custom_path: Path) -> dict:
    cfg = dict(base_config)
    cfg["topology"] = {"type": "custom", "path": str(custom_path)}
    return cfg


def _run_one_rep(
    cfg_path: Path,
    base_dir: Path,
    scenario_label: str,
    k_paths: int,
    rep: int,
):
    if pd is None:
        raise RuntimeError("Требуется pandas: pip install -r requirements.txt")

    rep_dir = base_dir / f"rep_{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = evaluate.run_experiment(cfg_path, rep_dir, measure_timings=True)
    df = df.copy()
    df["rep"] = rep
    df["repetition"] = rep
    df["scenario"] = scenario_label
    df["k_paths"] = k_paths
    return df


def run_with_failures(
    output: Path,
    include: Iterable[str] | None = None,
    *,
    repetitions: int = 10,
    workers: int = 1,
    k_paths_override: Sequence[int] | None = None,
    load_factors_override: Sequence[float] | None = None,
    heuristics_override: Sequence[str] | None = None,
    disable_ilp: bool = False,
    make_plots: bool = True,
    agg_counts: Sequence[int] = (1,),
    agg_nodes_override: Sequence[str] | None = None,
) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("Требуется pandas: pip install -r requirements.txt")

    specs = suite.default_scenarios()
    if include:
        include_set = set(include)
        specs = [s for s in specs if s.slug in include_set]

    out_root = output
    out_root.mkdir(parents=True, exist_ok=True)
    summary_frames = []

    for spec in specs:
        selected_k_paths = list(k_paths_override or spec.k_paths_values)
        for k_paths in selected_k_paths:
            # Базовый конфиг на один повтор; seed корректируем для каждого rep отдельно
            base_config = spec.build_config(
                k_paths=k_paths,
                repetitions=1,
                load_factors=load_factors_override,
            )
            if heuristics_override is not None:
                base_config["algorithms"]["heuristics"] = list(heuristics_override)
            if disable_ilp:
                base_config["algorithms"]["ilp"]["enabled"] = False

            base_graph = evaluate.build_topology(base_config["topology"])
            agg_nodes = _aggregation_nodes(base_graph)

            failure_sets: list[tuple[str, list[str]]] = []
            if agg_nodes_override:
                failure_sets.append(("agg_custom", list(agg_nodes_override)))
            else:
                for count in agg_counts:
                    if count <= 0:
                        continue
                    failure_sets.append((f"agg_fail_{count}", agg_nodes[:count]))

            for fail_slug, to_remove in failure_sets:
                scenario_label = f"{spec.slug}_{fail_slug}"
                cfg_dir = out_root / f"{spec.slug}_k{k_paths}_{fail_slug}"
                cfg_dir.mkdir(parents=True, exist_ok=True)

                modified = _remove_nodes(base_graph, to_remove)
                custom_path = cfg_dir / "modified_topology.json"
                _save_graph(modified, custom_path)

                failure_config = _build_custom_config(base_config, custom_path)

                df_list = []
                if workers <= 1:
                    for rep in range(repetitions):
                        rep_dir = cfg_dir / f"rep_{rep:02d}"
                        rep_dir.mkdir(parents=True, exist_ok=True)
                        rep_config = dict(failure_config)
                        rep_config["flows"] = dict(failure_config["flows"])
                        rep_config["flows"]["seed"] = spec.flow_seed + rep
                        rep_cfg_path = rep_dir / "config.yaml"
                        suite._dump_yaml(rep_cfg_path, rep_config)
                        df_list.append(
                            _run_one_rep(rep_cfg_path, cfg_dir, scenario_label, k_paths, rep)
                        )
                else:
                    rep_tasks = []
                    for rep in range(repetitions):
                        rep_dir = cfg_dir / f"rep_{rep:02d}"
                        rep_dir.mkdir(parents=True, exist_ok=True)
                        rep_config = dict(failure_config)
                        rep_config["flows"] = dict(failure_config["flows"])
                        rep_config["flows"]["seed"] = spec.flow_seed + rep
                        rep_cfg_path = rep_dir / "config.yaml"
                        suite._dump_yaml(rep_cfg_path, rep_config)
                        rep_tasks.append((rep_cfg_path, cfg_dir, scenario_label, k_paths, rep))
                    with ProcessPoolExecutor(max_workers=workers) as ex:
                        futures = [ex.submit(_run_one_rep, *task) for task in rep_tasks]
                        for fut in as_completed(futures):
                            df_list.append(fut.result())

                if not df_list:
                    continue
                df_all = pd.concat(df_list, ignore_index=True)
                df_all.to_csv(cfg_dir / "metrics_time.csv", index=False)
                summary_frames.append(df_all)
                if make_plots:
                    plots.plot_delay_vs_load_dispersion(df_all, cfg_dir)
                    plots.plot_sla_violations(df_all, cfg_dir)
                    if "duration_ms" in df_all.columns:
                        plots.plot_duration_vs_load_dispersion(df_all, cfg_dir)

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        combined.to_csv(out_root / "combined_metrics_time.csv", index=False)
        print(f"Готово. Результаты и времена сохранены в {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Эксперименты с отказами агрегационных коммутаторов и замером времени"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Каталог для результатов")
    parser.add_argument("--include", nargs="+", help="Список slug'ов сценариев")
    parser.add_argument("--workers", type=int, default=1, help="Число параллельных воркеров (processes)")
    parser.add_argument("--repetitions", type=int, default=10, help="Число повторов на сценарий")
    parser.add_argument(
        "--k-paths",
        nargs="+",
        type=int,
        help="Переопределить список k для всех сценариев (например: --k-paths 5)",
    )
    parser.add_argument(
        "--loads",
        nargs="+",
        type=float,
        help="Переопределить список коэффициентов нагрузки (например: --loads 0.5 0.7)",
    )
    parser.add_argument(
        "--heuristics",
        nargs="+",
        help="Переопределить список эвристик (например: --heuristics greedy local_search)",
    )
    parser.add_argument(
        "--no-ilp",
        action="store_true",
        help="Отключить ILP для всех сценариев",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики после прогона",
    )
    parser.add_argument(
        "--agg-counts",
        nargs="+",
        type=int,
        default=[1],
        help="Сколько агрегационных коммутаторов удалять (например: --agg-counts 1 2)",
    )
    parser.add_argument(
        "--agg-nodes",
        nargs="+",
        help="Явный список узлов агрегации для удаления (например: --agg-nodes agg2 agg_0_1)",
    )
    args = parser.parse_args()

    run_with_failures(
        args.output,
        include=args.include,
        repetitions=args.repetitions,
        workers=args.workers,
        k_paths_override=tuple(args.k_paths) if args.k_paths else None,
        load_factors_override=tuple(args.loads) if args.loads else None,
        heuristics_override=tuple(args.heuristics) if args.heuristics else None,
        disable_ilp=args.no_ilp,
        make_plots=not args.no_plots,
        agg_counts=tuple(args.agg_counts),
        agg_nodes_override=tuple(args.agg_nodes) if args.agg_nodes else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
