"""
Сценарии отказов для сетевых топологий.

Позволяет убрать из графа выбранные spine/core-коммутаторы или конкретные рёбра,
после чего сохранить топологию в JSON и прогнать эксперимент для оценки SLA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import json
import sys

import networkx as nx

try:  # pragma: no cover - импорт может отсутствовать при первом запуске
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

if __package__ in (None, ""):
    # запуск из каталога netcalc_dc
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from netcalc_dc.evaluate import load_config, build_topology, run_experiment  # type: ignore
    from netcalc_dc.main import save_reports  # type: ignore
    from netcalc_dc import plots  # type: ignore
else:
    from .evaluate import load_config, build_topology, run_experiment
    from .main import save_reports
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


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.write_text(json.dumps(_graph_to_json(graph), indent=2), encoding="utf-8")


def remove_nodes(graph: nx.DiGraph, prefixes: Sequence[str]) -> nx.DiGraph:
    modified = graph.copy()
    to_remove = [node for node in modified.nodes() if any(node.startswith(p) for p in prefixes)]
    modified.remove_nodes_from(to_remove)
    return modified


def remove_edges(graph: nx.DiGraph, edges: Iterable[tuple[str, str]]) -> nx.DiGraph:
    modified = graph.copy()
    for u, v in edges:
        if modified.has_edge(u, v):
            modified.remove_edge(u, v)
        if modified.has_edge(v, u):
            modified.remove_edge(v, u)
    return modified


def scale_edges(
    graph: nx.DiGraph,
    *,
    edge_scales: Iterable[tuple[str, str, float]] | None = None,
    prefix_scales: Iterable[tuple[str, float]] | None = None,
    node_prefix_scales: Iterable[tuple[str, float]] | None = None,
) -> nx.DiGraph:
    """
    Масштабирует пропускные способности рёбер.

    - edge_scales: конкретные рёбра (u, v, factor);
    - prefix_scales: все рёбра, у которых оба конца начинаются с указанного префикса;
    - node_prefix_scales: рёбра, инцидентные узлам с префиксом.
    """
    modified = graph.copy()

    def _apply_scale(u: str, v: str, factor: float) -> None:
        if modified.has_edge(u, v):
            modified[u][v]["capacity"] *= factor

    if edge_scales:
        for u, v, factor in edge_scales:
            _apply_scale(u, v, factor)

    if prefix_scales:
        for prefix, factor in prefix_scales:
            for u, v in list(modified.edges()):
                if u.startswith(prefix) and v.startswith(prefix):
                    _apply_scale(u, v, factor)

    if node_prefix_scales:
        for prefix, factor in node_prefix_scales:
            for u, v in list(modified.edges()):
                if u.startswith(prefix) or v.startswith(prefix):
                    _apply_scale(u, v, factor)

    return modified


def run_failure_scenario(
    config_path: Path,
    output_dir: Path,
    *,
    remove_node_prefixes: Sequence[str] | None = None,
    remove_edge_pairs: Iterable[tuple[str, str]] | None = None,
    edge_scales: Iterable[tuple[str, str, float]] | None = None,
    prefix_scales: Iterable[tuple[str, float]] | None = None,
    node_prefix_scales: Iterable[tuple[str, float]] | None = None,
) -> tuple["pd.DataFrame", int]:
    if pd is None:  # pragma: no cover
        raise RuntimeError(
            "Для запуска сценария отказов требуется pandas. Установите зависимости командой "
            "pip install -r netcalc_dc/requirements.txt"
        )

    cfg = load_config(config_path)
    graph = build_topology(cfg.topology)

    if remove_node_prefixes:
        graph = remove_nodes(graph, remove_node_prefixes)
    if remove_edge_pairs:
        graph = remove_edges(graph, remove_edge_pairs)
    if any((edge_scales, prefix_scales, node_prefix_scales)):
        graph = scale_edges(
            graph,
            edge_scales=edge_scales,
            prefix_scales=prefix_scales,
            node_prefix_scales=node_prefix_scales,
        )

    custom_path = output_dir / "modified_topology.json"
    save_graph(graph, custom_path)

    temp_config = output_dir / "failure_config.yaml"
    def _format(value: object) -> str:
        if isinstance(value, str):
            return f"\"{value}\""
        return str(value)

    yaml_lines = [
        "topology:",
        "  type: custom",
        f"  path: \"{custom_path}\"",
        "flows:",
    ]
    for key, value in cfg.flows.items():
        yaml_lines.append(f"  {key}: {_format(value)}")
    yaml_lines.append("algorithms:")
    yaml_lines.append("  heuristics:")
    for name in cfg.algorithms.get("heuristics", []):
        yaml_lines.append(f"    - {name}")
    ilp_cfg = cfg.algorithms.get("ilp", {})
    yaml_lines.append("  ilp:")
    for key, value in ilp_cfg.items():
        yaml_lines.append(f"    {key}: {_format(value)}")
    yaml_lines.append("evaluation:")
    for key, value in cfg.evaluation.items():
        yaml_lines.append(f"  {key}: {_format(value)}")
    temp_config.write_text("\n".join(yaml_lines), encoding="utf-8")
    try:
        df = run_experiment(temp_config, output_dir)
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            "Полученная топология оказалась несвязной: "
            f"{exc}. Проверьте, что после удаления узлов/рёбер сохраняются пути между серверами."
        ) from exc
    k_paths_value = cfg.evaluation.get("k_paths", 5)
    if isinstance(k_paths_value, (list, tuple)):
        if len(k_paths_value) != 1:
            raise ValueError(
                "failure_scenarios.py поддерживает конфигурации с одним значением k_paths"
            )
        k_paths_value = k_paths_value[0]
    if "k_paths" not in df.columns:
        df = df.copy()
        df["k_paths"] = int(k_paths_value)
    return df, int(k_paths_value)


def _write_outputs(
    df: "pd.DataFrame",
    output_dir: Path,
    *,
    persist: bool,
    build_plots: bool,
) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("Для построения отчётов требуется pandas.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if "k_paths" not in df.columns:
        raise ValueError("Таблица метрик должна содержать столбец 'k_paths'.")

    if persist:
        df.to_csv(output_dir / "metrics_failure.csv", index=False)
        for k_value in sorted(df["k_paths"].unique()):
            subdir = output_dir / f"k_paths_{int(k_value)}"
            subdir.mkdir(parents=True, exist_ok=True)
            df[df["k_paths"] == k_value].to_csv(subdir / "metrics.csv", index=False)

    save_reports(df, output_dir)

    if build_plots:
        try:
            plots.plot_delay_vs_load(df, output_dir)
            plots.plot_sla_violations(df, output_dir)
            plots.plot_sla_heatmap(df, output_dir / "heatmaps")
        except Exception as exc:  # pragma: no cover
            print(f"[warning] не удалось построить графики: {exc}")


def _load_existing_results(output_dir: Path) -> "pd.DataFrame":
    if pd is None:  # pragma: no cover
        raise RuntimeError("Для построения графиков требуется pandas.")

    frames: list[pd.DataFrame] = []

    combined = output_dir / "combined_metrics.csv"
    if combined.exists():
        frames.append(pd.read_csv(combined))

    metrics_failure = output_dir / "metrics_failure.csv"
    if metrics_failure.exists():
        frames.append(pd.read_csv(metrics_failure))

    for subdir in ("baseline", "failure"):
        sub_path = output_dir / subdir
        metrics_file = sub_path / "metrics_failure.csv"
        if metrics_file.exists():
            df_sub = pd.read_csv(metrics_file)
            df_sub = df_sub.copy()
            df_sub.setdefault("scenario", subdir)
            frames.append(df_sub)
        for path in sorted(sub_path.glob("k_paths_*/metrics.csv")):
            df = pd.read_csv(path)
            if "k_paths" not in df.columns:
                try:
                    k_value = int(path.parent.name.split("_")[-1])
                except ValueError as exc:
                    raise ValueError(f"Не удалось определить k_paths для файла {path}") from exc
                df["k_paths"] = k_value
            df.setdefault("scenario", subdir)
            frames.append(df)

    for path in sorted(output_dir.glob("k_paths_*/metrics.csv")):
        df = pd.read_csv(path)
        if "k_paths" not in df.columns:
            try:
                k_value = int(path.parent.name.split("_")[-1])
            except ValueError as exc:
                raise ValueError(f"Не удалось определить k_paths для файла {path}") from exc
            df["k_paths"] = k_value
        frames.append(df)

    if not frames:
        raise FileNotFoundError("В каталоге не найдено файлов metrics.csv.")

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(inplace=True)
    if "k_paths" not in combined.columns:
        raise ValueError("Не удалось определить значения k_paths в существующих данных.")
    return combined


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Запуск эксперимента с отключением коммутаторов/рёбер"
    )
    parser.add_argument("-c", "--config", type=Path, help="Базовый config.yaml")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Каталог для сохранения результатов"
    )
    parser.add_argument(
        "--remove-prefix",
        nargs="+",
        help="Префиксы узлов, которые нужно удалить (например, core_ или spine_).",
    )
    parser.add_argument(
        "--remove-edge",
        nargs=2,
        action="append",
        metavar=("U", "V"),
        help="Удалить конкретное ребро (можно указать несколько раз).",
    )
    parser.add_argument(
        "--scale-edge",
        nargs=3,
        action="append",
        metavar=("U", "V", "FACTOR"),
        help="Масштабировать пропускную способность конкретного ребра (например, core_0 core_1 0.5).",
    )
    parser.add_argument(
        "--scale-prefix",
        nargs=2,
        action="append",
        metavar=("PREFIX", "FACTOR"),
        help="Масштабировать рёбра, у которых оба конца начинаются с префикса (например, core_ 0.5).",
    )
    parser.add_argument(
        "--scale-node-prefix",
        nargs=2,
        action="append",
        metavar=("PREFIX", "FACTOR"),
        help="Масштабировать рёбра, инцидентные узлам с префиксом (например, agg_ 0.8).",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="Дополнительно запустить эксперимент без отказов для сравнения.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Построить графики по существующим данным и не запускать эксперимент.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики (оставить только CSV).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        df_existing = _load_existing_results(args.output)
        _write_outputs(df_existing, args.output, persist=False, build_plots=not args.no_plots)
        print("Графики построены на основе существующих данных.")
    else:
        if args.config is None:
            parser.error("--config обязателен при запуске эксперимента (без --plots-only)")
        summaries = []

        if args.with_baseline:
            baseline_dir = args.output / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            df_base, _ = run_failure_scenario(
                args.config,
                baseline_dir,
                remove_node_prefixes=[],
                remove_edge_pairs=None,
            )
            df_base = df_base.copy()
            df_base["scenario"] = "baseline"
            _write_outputs(df_base, baseline_dir, persist=True, build_plots=not args.no_plots)
            summaries.append(df_base)

        failure_dir = args.output / "failure"
        failure_dir.mkdir(parents=True, exist_ok=True)

        def _parse_edge_scales(values: list[list[str]] | None) -> list[tuple[str, str, float]]:
            if not values:
                return []
            result = []
            for u, v, factor in values:
                result.append((u, v, float(factor)))
            return result

        def _parse_prefix_scales(values: list[list[str]] | None) -> list[tuple[str, float]]:
            if not values:
                return []
            return [(prefix, float(factor)) for prefix, factor in values]

        df_failure, _ = run_failure_scenario(
            args.config,
            failure_dir,
            remove_node_prefixes=args.remove_prefix or [],
            remove_edge_pairs=[tuple(edge) for edge in args.remove_edge] if args.remove_edge else None,
            edge_scales=_parse_edge_scales(args.scale_edge),
            prefix_scales=_parse_prefix_scales(args.scale_prefix),
            node_prefix_scales=_parse_prefix_scales(args.scale_node_prefix),
        )
        df_failure = df_failure.copy()
        df_failure["scenario"] = "failure"
        _write_outputs(df_failure, failure_dir, persist=True, build_plots=not args.no_plots)
        summaries.append(df_failure)

        if pd is not None and summaries:
            combined = pd.concat(summaries, ignore_index=True)
            combined.to_csv(args.output / "combined_metrics.csv", index=False)

        print(
            "Эксперимент завершён. Результаты сохранены в подкаталогах baseline/ и failure/."
            if args.with_baseline
            else "Эксперимент завершён. Метрики и графики сохранены в каталоге."
        )
