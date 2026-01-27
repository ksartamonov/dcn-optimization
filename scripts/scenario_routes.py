"""
#TODO: поправить
Сценарные иллюстрации маршрутов для разных топологий и отказов/усилений.

Строит 6 картинок:
 1) fat-tree (базовый)
 2) fat-tree с вырезанным core
 3) fat-tree с усиленным одним core
 4) clos (базовый)
 5) clos с вырезанным core
 6) clos с усиленным одним core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import networkx as nx

import sys

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from netcalc_dc.evaluate import load_config, build_topology  # type: ignore
    import route_samples as _route_samples  # type: ignore

    summarize_routes = _route_samples.summarize_routes
    plot_topology_with_routes = _route_samples.plot_topology_with_routes
else:
    from .evaluate import load_config, build_topology
    from .route_samples import summarize_routes, plot_topology_with_routes


def _graph_to_json(graph: nx.DiGraph) -> dict:
    edges = []
    for u, v in graph.edges():
        data = graph[u][v]
        edges.append(
            {"u": u, "v": v, "capacity": data.get("capacity", 1.0), "latency": data.get("latency", 0.0)}
        )
    return {"nodes": list(graph.nodes()), "edges": edges}


def _save_custom_config(base_cfg, graph: nx.DiGraph, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    topo_path = dest_dir / "topology.json"
    topo_path.write_text(json.dumps(_graph_to_json(graph), indent=2), encoding="utf-8")

    def _fmt(value: object) -> str:
        if isinstance(value, str):
            return f"\"{value}\""
        return str(value)

    lines = [
        "topology:",
        "  type: custom",
        f"  path: \"{topo_path}\"",
        "flows:",
    ]
    for k, v in base_cfg.flows.items():
        lines.append(f"  {k}: {_fmt(v)}")
    lines.append("algorithms:")
    lines.append("  heuristics:")
    for name in base_cfg.algorithms.get("heuristics", []):
        lines.append(f"    - {name}")
    lines.append("  ilp:")
    for k, v in base_cfg.algorithms.get("ilp", {}).items():
        lines.append(f"    {k}: {_fmt(v)}")
    lines.append("evaluation:")
    for k, v in base_cfg.evaluation.items():
        lines.append(f"  {k}: {_fmt(v)}")

    cfg_path = dest_dir / "config_custom.yaml"
    cfg_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg_path


def _resolve_config(path: Path) -> Path:
    if path.exists():
        return path
    candidate = (Path(__file__).resolve().parent.parent / "configs" / path.name)
    if candidate.exists():
        return candidate
    return path


def _remove_one_node(graph: nx.DiGraph, prefer_prefixes: tuple[str, ...]) -> nx.DiGraph:
    modified = graph.copy()
    candidates = []
    # сначала точечно agg_1/aggregation_1
    explicit = [n for n in modified.nodes if n.startswith(("agg_1", "aggregation_1"))]
    if explicit:
        candidates = sorted(explicit)
    else:
        for pref in prefer_prefixes:
            pref_nodes = sorted([n for n in modified.nodes if n.startswith(pref)])
            if pref_nodes:
                # Сначала пытаемся удалить узел с индексом "1", затем остальные
                pref_nodes_sorted = sorted(pref_nodes, key=lambda n: (not n.endswith("1") and "_1" not in n, n))
                candidates = pref_nodes_sorted
                break
    if candidates:
        modified.remove_node(candidates[0])
    return modified


def _remove_specific_node(graph: nx.DiGraph, names: tuple[str, ...]) -> nx.DiGraph:
    modified = graph.copy()
    for name in names:
        alt = name.replace("_", "")
        for candidate in list(modified.nodes):
            if candidate == name or candidate == alt:
                modified.remove_node(candidate)
                return modified
    return modified


def _boost_one_core(graph: nx.DiGraph, factor: float = 4.0) -> nx.DiGraph:
    modified = graph.copy()
    cores = sorted([n for n in modified.nodes if n.startswith(("core", "spine"))])
    if not cores:
        return modified
    target = cores[0]
    for u, v in list(modified.edges(target)):
        if modified.has_edge(u, v):
            modified[u][v]["capacity"] = modified[u][v].get("capacity", 1.0) * factor
    for u, v in list(modified.in_edges(target)):
        if modified.has_edge(u, v):
            modified[u][v]["capacity"] = modified[u][v].get("capacity", 1.0) * factor
    return modified


def _boost_one_agg(graph: nx.DiGraph, factor: float = 6.0) -> nx.DiGraph:
    modified = graph.copy()
    aggs = sorted([n for n in modified.nodes if n.startswith(("agg", "aggregation"))])
    if not aggs:
        return modified
    target = aggs[0]
    for u, v in list(modified.edges(target)):
        if modified.has_edge(u, v):
            modified[u][v]["capacity"] = modified[u][v].get("capacity", 1.0) * factor
    for u, v in list(modified.in_edges(target)):
        if modified.has_edge(u, v):
            modified[u][v]["capacity"] = modified[u][v].get("capacity", 1.0) * factor
    return modified


def _build_and_plot(
    cfg_path: Path,
    out_dir: Path,
    *,
    load_factor: float,
    flows: int,
    base_config: Path | None = None,
) -> None:
    summary = summarize_routes(cfg_path, load_factor, flows)
    plot_topology_with_routes(cfg_path, summary, out_dir, base_config_path=base_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Строит 6 картинок маршрутов по сценариям")
    parser.add_argument("-o", "--output", type=Path, default=Path("scenario_routes"), help="Каталог для результатов")
    parser.add_argument("--fat-tree-config", type=Path, default=Path("config.yaml"), help="Базовый fat-tree config")
    parser.add_argument("--clos-config", type=Path, default=Path("config_clos.yaml"), help="Базовый clos config")
    parser.add_argument("--load-factor", type=float, default=0.7, help="λ для примеров маршрутов")
    parser.add_argument("--flows", type=int, default=5, help="Сколько потоков брать в примеры")
    args = parser.parse_args()

    args.fat_tree_config = _resolve_config(args.fat_tree_config)
    args.clos_config = _resolve_config(args.clos_config)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # 1) fat-tree базовый
    _build_and_plot(args.fat_tree_config, out / "fat-tree_base", load_factor=args.load_factor, flows=args.flows)

    base_cfg_ft = load_config(args.fat_tree_config)
    g_ft = build_topology(base_cfg_ft.topology)

    # 2) fat-tree: отказ agg_2 (агрегация, подальше от agg_1)
    g_ft_agg3 = _remove_specific_node(g_ft, names=("agg_2", "agg2", "aggregation_2"))
    cfg_ft_agg3 = _save_custom_config(base_cfg_ft, g_ft_agg3, out / "fat-tree_agg3_failure")
    _build_and_plot(cfg_ft_agg3, out / "fat-tree_agg3_failure", load_factor=args.load_factor, flows=args.flows, base_config=args.fat_tree_config)

    # 3) fat-tree: усиленный один agg (agg_0)
    g_ft_boost = _boost_one_agg(g_ft, factor=8.0)
    cfg_ft_boost = _save_custom_config(base_cfg_ft, g_ft_boost, out / "fat-tree_boosted")
    _build_and_plot(cfg_ft_boost, out / "fat-tree_boosted", load_factor=args.load_factor, flows=args.flows, base_config=args.fat_tree_config)

    # 4) clos базовый
    _build_and_plot(args.clos_config, out / "clos_base", load_factor=args.load_factor, flows=args.flows)

    base_cfg_clos = load_config(args.clos_config)
    g_clos = build_topology(base_cfg_clos.topology)

    # 5) clos: отказ agg_2
    g_clos_agg3 = _remove_specific_node(g_clos, names=("agg_2", "agg2", "aggregation_2"))
    cfg_clos_agg3 = _save_custom_config(base_cfg_clos, g_clos_agg3, out / "clos_agg3_failure")
    _build_and_plot(cfg_clos_agg3, out / "clos_agg3_failure", load_factor=args.load_factor, flows=args.flows, base_config=args.clos_config)

    # 6) clos: усиленный один agg (agg_0)
    g_clos_boost = _boost_one_agg(g_clos, factor=8.0)
    cfg_clos_boost = _save_custom_config(base_cfg_clos, g_clos_boost, out / "clos_boosted")
    _build_and_plot(cfg_clos_boost, out / "clos_boosted", load_factor=args.load_factor, flows=args.flows, base_config=args.clos_config)

    print(f"Готово. Картинки и описания лежат в {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
