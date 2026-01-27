"""
Генерация и загрузка потоков трафика для экспериментов.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json
import random

from .arrival_service import ArrivalCurve


@dataclass(frozen=True)
class Flow:
    """Описание потока между парой виртуальных машин."""

    flow_id: str
    source: str
    target: str
    arrival: ArrivalCurve
    sla_delay: float
    class_name: str


DEFAULT_CLASSES: Dict[str, Dict[str, float]] = {
    "interactive": {"rate_mean": 1.5, "rate_std": 0.3, "burst_mean": 0.4, "sla": 1.5},
    "batch": {"rate_mean": 2.5, "rate_std": 0.5, "burst_mean": 0.7, "sla": 3.0},
    "background": {"rate_mean": 0.8, "rate_std": 0.2, "burst_mean": 0.3, "sla": 5.0},
}


def _sample_positive_gaussian(rng: random.Random, mean: float, std: float) -> float:
    value = rng.gauss(mean, std)
    return max(0.01, value)


def identify_terminal_nodes(nodes: Iterable[str]) -> List[str]:
    """Эвристически определяет серверные узлы по префиксу 'server'."""
    terminals = [node for node in nodes if node.startswith("server")]
    return terminals or list(nodes)


def generate_flows(
    graph_nodes: Sequence[str],
    count: int,
    *,
    classes: Dict[str, Dict[str, float]] | None = None,
    seed: int | None = None,
) -> List[Flow]:
    """
    Генерирует набор потоков между случайными парами терминальных узлов.
    """
    rng = random.Random(seed)
    classes = classes or DEFAULT_CLASSES
    terminals = identify_terminal_nodes(graph_nodes)
    if len(terminals) < 2:
        raise ValueError("Not enough terminal nodes to generate flows")

    class_names = list(classes.keys())
    flows: List[Flow] = []
    for idx in range(count):
        src, dst = rng.sample(terminals, 2)
        class_name = rng.choice(class_names)
        params = classes[class_name]
        rate = _sample_positive_gaussian(rng, params["rate_mean"], params["rate_std"])
        burst = _sample_positive_gaussian(rng, params["burst_mean"], params["burst_mean"] / 2)
        flow = Flow(
            flow_id=f"flow_{idx}",
            source=src,
            target=dst,
            arrival=ArrivalCurve(rate=rate, burst=burst),
            sla_delay=params["sla"],
            class_name=class_name,
        )
        flows.append(flow)
    return flows


def load_flows(path: Path) -> List[Flow]:
    """
    Загружает описание потоков в формате JSON.

    Формат:
    [
        {"id": "...", "source": "...", "target": "...", "rate": 1.5,
         "burst": 0.2, "sla": 2.0, "class": "interactive"},
        ...
    ]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    flows: List[Flow] = []
    for item in data:
        flows.append(
            Flow(
                flow_id=item["id"],
                source=item["source"],
                target=item["target"],
                arrival=ArrivalCurve(rate=item["rate"], burst=item["burst"]),
                sla_delay=item["sla"],
                class_name=item.get("class", "custom"),
            )
        )
    return flows
