"""
Визуализация результатов экспериментов.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _labels(language: str) -> dict:
    if language == "en":
        return {
            "load": "Load factor",
            "delay": "Maximum delay, ms",
            "duration": "Runtime, ms",
            "algorithm": "Algorithm",
            "sla": "SLA violation ratio",
            "heatmap_title": "SLA heatmap",
            "k_paths": "k_paths",
            "lambda": "Load factor λ",
        }
    return {
        "load": "Фактор загрузки",
        "delay": "Максимальная задержка, мс",
        "duration": "Время выполнения, мс",
        "algorithm": "Алгоритм",
        "sla": "Доля нарушений SLA",
        "heatmap_title": "SLA-heatmap",
        "k_paths": "k_paths",
        "lambda": "Фактор нагрузки λ",
    }


def _style_map(algorithms: Iterable[str]) -> dict:
    styles = [
        {"linestyle": "-", "marker": "o"},
        {"linestyle": "--", "marker": "^"},
        {"linestyle": ":", "marker": "s"},
        {"linestyle": "-.", "marker": "D"},
        {"linestyle": (0, (3, 1, 1, 1)), "marker": "v"},
    ]
    algo_list = list(algorithms)
    mapping = {}
    for idx, algo in enumerate(algo_list):
        mapping[algo] = styles[idx % len(styles)]
    return mapping


def plot_delay_vs_load(results: pd.DataFrame, output_dir: Path, *, language: str = "ru") -> Path:
    """Строит график зависимости максимальной задержки от фактора загрузки."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(8, 5))
    style_map = _style_map(sorted(results["algorithm"].unique()))
    for algo, subset in results.groupby("algorithm"):
        subset = subset.sort_values("load_factor")
        ax.plot(
            subset["load_factor"],
            subset["max_delay"],
            label=algo,
            linestyle=style_map[algo]["linestyle"],
            marker=style_map[algo]["marker"],
        )
    ax.set_xlabel(labels["load"])
    ax.set_ylabel(labels["delay"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=labels["algorithm"])
    fig.tight_layout()
    path = output_dir / "delay_vs_load.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_delay_vs_load_dispersion(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    language: str = "ru",
) -> Path:
    """График максимальной задержки с дисперсией (mean ± std)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(language)
    grouped = (
        results.groupby(["algorithm", "load_factor"])["max_delay"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    style_map = _style_map(sorted(grouped["algorithm"].unique()))
    for algo, subset in grouped.groupby("algorithm"):
        subset = subset.sort_values("load_factor")
        ax.plot(
            subset["load_factor"],
            subset["mean"],
            label=algo,
            linestyle=style_map[algo]["linestyle"],
            marker=style_map[algo]["marker"],
        )
        std = subset["std"].fillna(0.0)
        ax.fill_between(
            subset["load_factor"],
            subset["mean"] - std,
            subset["mean"] + std,
            alpha=0.2,
        )
    ax.set_xlabel(labels["load"])
    ax.set_ylabel(labels["delay"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=labels["algorithm"])
    fig.tight_layout()
    path = output_dir / "delay_vs_load_std.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_duration_vs_load_dispersion(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    language: str = "ru",
) -> Path:
    """График времени выполнения (median и межквартильный размах) по нагрузке."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(language)
    grouped = (
        results.groupby(["algorithm", "load_factor"])["duration_ms"]
        .quantile([0.25, 0.5, 0.75])
        .unstack(level=-1)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    style_map = _style_map(sorted(grouped["algorithm"].unique()))
    for algo, subset in grouped.groupby("algorithm"):
        subset = subset.sort_values("load_factor")
        q25 = subset[0.25].fillna(0.0)
        q50 = subset[0.5].fillna(0.0)
        q75 = subset[0.75].fillna(0.0)
        ax.plot(
            subset["load_factor"],
            q50,
            label=algo,
            linestyle=style_map[algo]["linestyle"],
            marker=style_map[algo]["marker"],
        )
        ax.fill_between(
            subset["load_factor"],
            q25,
            q75,
            alpha=0.2,
        )
    ax.set_xlabel(labels["load"])
    ax.set_ylabel(labels["duration"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title=labels["algorithm"])
    fig.tight_layout()
    path = output_dir / "duration_vs_load_std.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sla_heatmap(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    algorithms: Iterable[str] | None = None,
    language: str = "ru",
) -> list[Path]:
    """
    Строит тепловые карты доли нарушений SLA по сетке (lambda, algorithm).
    Возвращает список путей к сохранённым изображениям.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(language)
    plots: list[Path] = []
    if algorithms is None:
        algorithms = sorted(results["algorithm"].unique())
    for algorithm in algorithms:
        subset = results[results["algorithm"] == algorithm]
        if subset.empty:
            continue
        if "k_paths" not in subset.columns:
            raise ValueError("results DataFrame must contain 'k_paths' column for heatmap generation")
        pivot = subset.pivot_table(
            index="load_factor",
            columns="k_paths",
            values="sla_violations",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            pivot.sort_index(),
            annot=True,
            fmt=".2f",
            cmap="Reds",
            cbar_kws={"label": labels["sla"]},
            ax=ax,
        )
        ax.set_title(f"{labels['heatmap_title']}: {algorithm}")
        ax.set_xlabel(labels["k_paths"])
        ax.set_ylabel(labels["lambda"])
        fig.tight_layout()
        path = output_dir / f"sla_heatmap_{algorithm}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)
    return plots


def plot_sla_violations(results: pd.DataFrame, output_dir: Path, *, language: str = "ru") -> Path:
    """Строит график доли нарушений SLA для разных алгоритмов."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=results,
        x="algorithm",
        y="sla_violations",
        hue="load_factor",
        ax=ax,
    )
    ax.set_xlabel(labels["algorithm"])
    ax.set_ylabel(labels["sla"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = output_dir / "sla_violations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
