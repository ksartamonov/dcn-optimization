"""
Утилита для формирования наглядных примеров маршрутов на основе конфигурации.

Позволяет для выбранного коэффициента нагрузки вывести маршруты, построенные
каждым алгоритмом, а также их базовые задержки и SLA потоков.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import statistics
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import networkx as nx

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    REPO_ROOT = PROJECT_ROOT.parent
    for path in (PROJECT_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from netcalc_dc.evaluate import (  # type: ignore
        build_candidates,
        build_topology,
        load_config,
        run_algorithms,
        scale_flows,
    )
    from netcalc_dc.flows import flows_from_config  # type: ignore
    from netcalc_dc import router  # type: ignore
else:
    from .evaluate import (
        build_candidates,
        build_topology,
        load_config,
        run_algorithms,
        scale_flows,
    )
    from .flows import flows_from_config
    from . import router


def summarize_routes(
    config_path: Path,
    load_factor: float | None,
    sample_size: int,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    graph = build_topology(cfg.topology)

    base_flows = flows_from_config(
        cfg.flows,
        list(graph.nodes),
        seed=cfg.flows.get("seed"),
    )

    k_paths = cfg.evaluation.get("k_paths", 5)
    candidates = build_candidates(graph, base_flows, k_paths)

    manual_mode = bool(cfg.flows.get("manual") or cfg.flows.get("path"))
    factors = cfg.evaluation.get("load_factors", [1.0])
    if manual_mode:
        chosen_factor = 1.0
    else:
        chosen_factor = load_factor if load_factor is not None else factors[0]
    scaled_flows = scale_flows(base_flows, chosen_factor)

    if manual_mode:
        sample_size = min(sample_size, len(scaled_flows))

    assignments = run_algorithms(graph, scaled_flows, candidates, cfg.algorithms)

    summary: Dict[str, Any] = {
        "topology": cfg.topology["type"],
        "k_paths": k_paths,
        "load_factor": chosen_factor,
        "flows_in_sample": sample_size,
        "algorithms": [],
    }

    for algo_name, assignment in assignments.items():
        algo_entry: Dict[str, Any] = {"algorithm": algo_name, "routes": []}
        for flow in scaled_flows[:sample_size]:
            path = assignment.get(flow.flow_id)
            if not path:
                continue
            algo_entry["routes"].append(
                {
                    "flow_id": flow.flow_id,
                    "class": flow.class_name,
                    "source": flow.source,
                    "target": flow.target,
                    "path": path,
                    "hop_count": len(path) - 1,
                    "sla_delay": flow.sla_delay,
                    "base_delay": router.base_delay(flow, path, graph),
                }
            )
        summary["algorithms"].append(algo_entry)

    return summary


def save_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Примеры маршрутов ({summary['topology']})")
    lines.append(f"* Коэффициент нагрузки: λ={summary['load_factor']}")
    lines.append(f"* k_paths = {summary['k_paths']}")
    lines.append("")
    for algo in summary["algorithms"]:
        lines.append(f"## {algo['algorithm']}")
        routes = algo["routes"]
        if not routes:
            lines.append("_Маршруты отсутствуют (алгоритм не построил решение)._")
            lines.append("")
            continue
        lines.append("| Поток | Источник → Приёмник | Маршрут | Хопы | SLA (мс) | Базовая задержка (мс) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in routes:
            lines.append(
                f"| {entry['flow_id']} "
                f"| {entry['source']} → {entry['target']} "
                f"| {' → '.join(entry['path'])} "
                f"| {entry['hop_count']} "
                f"| {entry['sla_delay']:.3f} "
                f"| {entry['base_delay']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _layered_layout(graph: nx.DiGraph) -> Dict[str, tuple[float, float]]:
    """
    Упорядочивает уровни: core/spine → agg → edge/leaf → servers.
    Подходит для Clos, Fat-Tree и производных custom-графов с такими именами узлов.
    """
    core = [n for n in graph if n.startswith("core") or n.startswith("spine")]
    agg = [n for n in graph if n.startswith("agg")]
    edge = [n for n in graph if n.startswith("edge") or n.startswith("leaf")]
    servers = [n for n in graph if n.startswith("server")]

    def _assign(nodes: List[str], y: float, order: List[str] | None = None) -> Dict[str, tuple[float, float]]:
        seq = order or sorted(nodes)
        n = max(len(seq), 1)
        return {node: (idx / max(n - 1, 1), y) for idx, node in enumerate(seq)}

    pos: Dict[str, tuple[float, float]] = {}
    pos.update(_assign(core, 1.0))
    pos.update(_assign(agg, 0.66))
    pos.update(_assign(edge, 0.33))

    server_order: List[str] = []
    seen = set()
    for edge_name in sorted(edge):
        neighbors = [nbr for nbr in graph.predecessors(edge_name) if nbr.startswith("server")]
        if not neighbors:
            neighbors = [nbr for nbr in graph.successors(edge_name) if nbr.startswith("server")]
        neighbors.sort(key=lambda name: int(''.join(filter(str.isdigit, name)) or 0))
        for srv in neighbors:
            if srv not in seen:
                server_order.append(srv)
                seen.add(srv)
    for srv in sorted(servers, key=lambda name: int(''.join(filter(str.isdigit, name)) or 0)):
        if srv not in seen:
            server_order.append(srv)
            seen.add(srv)

    pos.update(_assign(servers, 0.0, order=server_order))
    return pos


def _color_by_level(node: str) -> str:
    if node.startswith("server"):
        return "#90CAF9"
    if node.startswith("edge") or node.startswith("leaf"):
        return "#66BB6A"
    if node.startswith("agg"):
        return "#FFB74D"
    if node.startswith("core") or node.startswith("spine"):
        return "#BA68C8"
    return "#B0BEC5"


def _compute_node_sizes(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Подбирает размеры узлов пропорционально характерной ёмкости инцидентных рёбер.
    """
    caps = [data.get("capacity", 1.0) for _, _, data in graph.edges(data=True)]
    base = max(1.0, float(statistics.median(caps)) if caps else 1.0)
    sizes: Dict[str, float] = {}
    for node in graph.nodes:
        if node.startswith("server"):
            sizes[node] = 180.0
            continue
        incident = [
            graph[u][v].get("capacity", base)
            for u, v in graph.edges(node)
        ] + [
            graph[u][v].get("capacity", base)
            for u, v in graph.in_edges(node)
        ]
        cap = max(incident) if incident else base
        scale = cap / base
        sizes[node] = max(140.0, min(500.0, 220.0 * scale))
    return sizes


def plot_topology_with_routes(
    config_path: Path,
    summary: Dict[str, Any],
    output_dir: Path,
    base_config_path: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    graph = build_topology(cfg.topology)
    base_cfg = load_config(base_config_path) if base_config_path else cfg
    base_graph = build_topology(base_cfg.topology)

    node_sizes_active = _compute_node_sizes(graph)
    node_sizes = {}
    for n in base_graph.nodes:
        node_sizes[n] = node_sizes_active.get(n, 220.0)

    topology_type = base_cfg.topology.get("type", "")

    def _has_layered_names(g: nx.DiGraph) -> bool:
        return any(
            any(n.startswith(prefix) for n in g.nodes)
            for prefix in ("core", "spine", "agg", "edge", "leaf")
        )

    layout_graph = base_graph if base_config_path else graph
    is_custom = topology_type == "custom"
    is_small_custom = is_custom and len(layout_graph.nodes) <= 8
    if topology_type in {"clos", "fat-tree", "spine_leaf", "spine-leaf"} or _has_layered_names(layout_graph):
        pos = _layered_layout(layout_graph)
    elif is_small_custom:
        pos = nx.circular_layout(layout_graph)
    else:
        pos = nx.spring_layout(layout_graph, seed=42)

    base_nodes = list(base_graph.nodes)
    node_labels = {}
    label_pos = {}
    for node in base_nodes:
        if node.startswith("server"):
            short = node.replace("server", "srv")
        else:
            short = node
        node_labels[node] = short
        label_pos[node] = pos[node]

    if is_small_custom:
        # выносим подписи чуть наружу, чтобы не перекрывать кружки
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        for node, (x, y) in pos.items():
            vx, vy = x - cx, y - cy
            norm = (vx ** 2 + vy ** 2) ** 0.5 or 1.0
            offset = 0.12
            label_pos[node] = (x + offset * vx / norm, y + offset * vy / norm)
    level_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="core/spine", markerfacecolor="#BA68C8"),
        plt.Line2D([0], [0], marker="o", color="w", label="aggregation", markerfacecolor="#FFB74D"),
        plt.Line2D([0], [0], marker="o", color="w", label="access", markerfacecolor="#66BB6A"),
        plt.Line2D([0], [0], marker="o", color="w", label="servers", markerfacecolor="#90CAF9"),
    ]

    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    for algo_entry in summary["algorithms"]:
        algo_dir = output_dir / algo_entry["algorithm"]
        algo_dir.mkdir(parents=True, exist_ok=True)
        if is_custom:
            plt.figure(figsize=(6, 5) if is_small_custom else (8, 6))
            # базовая топология тонкими серыми линиями
            nx.draw_networkx_edges(
                graph,
                pos,
                alpha=0.2,
                arrows=False,
                width=0.6,
                edge_color="#9e9e9e",
            )
        else:
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_edges(
                graph,
                pos,
                alpha=0.2,
                arrows=False,
                width=1.0,
            )
        failed_nodes = [
            n for n in base_nodes if (n not in graph.nodes) or graph.degree(n) == 0
        ]
        healthy_nodes = [n for n in base_nodes if n not in failed_nodes]
        size_override = 520.0 if is_small_custom else None
        if is_custom:
            # минимальные точки-узлы без заливки
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=healthy_nodes,
                node_color="white",
                edgecolors="#212121",
                linewidths=1.2,
                node_size=140 if is_small_custom else 180,
            )
        else:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=healthy_nodes,
                node_color=[_color_by_level(n) for n in healthy_nodes],
                node_size=[(size_override or node_sizes[n]) for n in healthy_nodes],
            )
        if failed_nodes:
            if not is_custom:
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=failed_nodes,
                    node_color="#d32f2f",
                    node_size=[(size_override or node_sizes[n]) for n in failed_nodes],
                    node_shape="x",
                    linewidths=2.5,
                    edgecolors="#b71c1c",
                )
        nx.draw_networkx_labels(
            graph,
            label_pos if is_custom else pos,
            labels=node_labels,
            font_size=10 if is_custom else 7,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.15)
            if is_custom
            else None,
        )

        flow_handles = []
        node_highlights: Dict[str, List[str]] = defaultdict(list)
        arc_rads = [-0.45, -0.32, -0.2, -0.08, 0.08, 0.2, 0.32, 0.45]
        edge_order: Dict[tuple[str, str], int] = defaultdict(int)
        for idx, route in enumerate(algo_entry["routes"]):
            color = palette[idx % len(palette)]
            shifted_pos = {node: pos[node] for node in route["path"]}
            if node not in failed_nodes:
                for node in route["path"]:
                    node_highlights[node].append(color)
            edges = list(zip(route["path"][:-1], route["path"][1:]))
            if is_custom:
                # рисуем каждое ребро отдельно с собственным смещением, чтобы линии не накладывались
                for u, v in edges:
                    order = edge_order[(u, v)]
                    edge_order[(u, v)] += 1
                    rad = arc_rads[order % len(arc_rads)]
                    # дополнительное небольшое параллельное смещение вдоль нормали
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx * dx + dy * dy) ** 0.5 or 1.0
                    nxn, nyn = -dy / length, dx / length
                    offset = 0.03 * (order - 0.5)
                    shifted = {
                        u: (x1 + nxn * offset, y1 + nyn * offset),
                        v: (x2 + nxn * offset, y2 + nyn * offset),
                    }
                    nx.draw_networkx_edges(
                        graph,
                        {**pos, **shifted},
                        edgelist=[(u, v)],
                        width=3,
                        edge_color=color,
                        alpha=0.85,
                        arrows=False,
                        connectionstyle=f"arc3,rad={rad}",
                    )
            else:
                nx.draw_networkx_edges(
                    graph,
                    shifted_pos,
                    edgelist=edges,
                    width=2,
                    edge_color=color,
                    alpha=0.8,
                    arrows=False,
                    connectionstyle="arc3,rad=0.0",
                )
            flow_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2,
                    label=f"{route['flow_id']} ({route['source']}→{route['target']})",
                )
            )

        ax = plt.gca()
        for node, colors in node_highlights.items():
            cx, cy = pos[node]
            total = len(colors)
            if total == 0:
                continue
            radius = 0.08 if is_custom else (0.05 if is_small_custom else 0.03)
            for wedge_idx, wedge_color in enumerate(colors):
                theta1 = 360.0 * wedge_idx / total
                theta2 = 360.0 * (wedge_idx + 1) / total
                wedge = Wedge(
                    center=(cx, cy),
                    r=radius,
                    theta1=theta1,
                    theta2=theta2,
                    facecolor=wedge_color,
                    edgecolor="black",
                    linewidth=0.4,
                )
                ax.add_patch(wedge)

        handles = flow_handles + ([] if is_custom else level_handles)
        ax = plt.gca()
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            ncol=2,
            fontsize=7,
            bbox_to_anchor=(1.15, 1.12),
        )
        legend.get_frame().set_alpha(0.9)
        plt.title(
            f"Маршруты: {algo_entry['algorithm']} (λ={summary['load_factor']}, k_paths={summary['k_paths']})"
        )
        plt.axis("off")
        out_path = algo_dir / "routes.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        summary_path_json = algo_dir / "routes.json"
        summary_path_md = algo_dir / "routes.md"
        algo_summary = {**summary, "algorithms": [algo_entry]}
        summary_path_json.write_text(json.dumps(algo_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        save_markdown(algo_summary, summary_path_md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Формирует наглядные примеры маршрутов по конфигурации"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Путь к config.yaml",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help="Базовая конфигурация для раскладки (нужна, если из неё удалены узлы)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("route_samples.json"),
        help="Файл для сохранения JSON (по умолчанию route_samples.json)",
    )
    parser.add_argument(
        "--load-factor",
        type=float,
        help="Коэффициент нагрузки λ (по умолчанию берётся первое значение из конфигурации)",
    )
    parser.add_argument(
        "--flows",
        type=int,
        default=3,
        help="Сколько потоков включить в примеры маршрутов",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Старт экспорта маршрутов для %s", args.config)
    cfg = load_config(args.config)
    manual_mode = bool(cfg.flows.get("manual") or cfg.flows.get("path"))
    if manual_mode and args.flows:
        logging.info(
            "Ручные потоки заданы в конфигурации: параметр --flows игнорируется."
        )
    if manual_mode and args.load_factor is not None:
        logging.info(
            "Ручные потоки заданы в конфигурации: параметр --load-factor игнорируется (используется 1.0)."
        )
    summary = summarize_routes(args.config, args.load_factor, args.flows)
    logging.info(
        "Завершено построение маршрутов: алгоритмов=%d, потоков в выборке=%d",
        len(summary["algorithms"]),
        summary["flows_in_sample"],
    )
    base_dir = args.output
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_markdown(summary, base_dir / "summary.md")
    plot_topology_with_routes(args.config, summary, base_dir, base_config_path=args.base_config)
    logging.info("Маршруты сохранены в каталоге %s", base_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
