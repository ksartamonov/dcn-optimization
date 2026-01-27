"""
Автоматизированный запуск набора экспериментов согласно методике глав~\ref{ch:implementation}
и~\ref{ch:experiment_setup} дипломной работы.

Модуль формирует YAML-конфигурации для типовых сценариев (Clos, Fat-Tree, пользовательские
топологии), запускает `evaluate.run_experiment`, сохраняет метрики, отчёты и графики в
отдельных каталогах. Используется для пакетного прогонa экспериментов без ручного вмешательства.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import copy
import datetime as dt
import json
import sys
import yaml

if __package__ in (None, ""):
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from netcalc_dc import evaluate, plots  # type: ignore
    from netcalc_dc.main import save_reports  # type: ignore
else:
    from . import evaluate, plots
    from .main import save_reports


HeuristicList = Sequence[str]
_COLOR_CODES = {
    "info": "36",
    "progress": "34",
    "success": "32",
    "warning": "33",
    "error": "31",
    "muted": "90",
}
_COLOR_ENABLED = sys.stdout.isatty()


def _color(text: str, style: str) -> str:
    code = _COLOR_CODES.get(style)
    if not _COLOR_ENABLED or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Описание одного сценария: набор параметров топологии, трафика и алгоритмов.

    Поля соответствуют структуре разделов главы~\ref{ch:experiment_setup}:
    - topology: Clos/Fat-Tree/пользовательский граф.
    - flows: генерация потоков с нагрузкой 32--512 в зависимости от масштаба сети.
    - algorithms: набор эвристик и опционально ILP-бейзлайн (глава~\ref{ch:implementation}).
    - evaluation: повторения, коэффициенты загрузки, число $k$-кратчайших путей.
    """

    slug: str
    description: str
    topology: Mapping[str, Any]
    flow_count: int
    flow_seed: int
    heuristics: HeuristicList
    ilp_enabled: bool
    ilp_timeout: Optional[int]
    load_factors: Sequence[float]
    repetitions: int
    k_paths_values: Sequence[int]
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def build_config(
        self,
        *,
        k_paths: int,
        repetitions: Optional[int] = None,
        load_factors: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Собирает словарь конфигурации в формате, совместимом с `config.yaml`.
        """
        topology_cfg = copy.deepcopy(dict(self.topology))
        algorithms_cfg: Dict[str, Any] = {"heuristics": list(self.heuristics)}
        algorithms_cfg["ilp"] = {
            "enabled": self.ilp_enabled,
        }
        if self.ilp_timeout is not None:
            algorithms_cfg["ilp"]["timeout"] = self.ilp_timeout

        return {
            "topology": topology_cfg,
            "flows": {
                "count": self.flow_count,
                "seed": self.flow_seed,
            },
            "algorithms": algorithms_cfg,
            "evaluation": {
                "repetitions": repetitions if repetitions is not None else self.repetitions,
                "load_factors": list(load_factors) if load_factors is not None else list(self.load_factors),
                "k_paths": k_paths,
            },
        }


def _default_heuristics() -> List[str]:
    return ["greedy", "local_search", "grasp", "ga"]


def _make_clos_spec(access: int, *, seed: int, description_suffix: str = "") -> ScenarioSpec:
    aggregation = max(4, access // 2)
    core = max(2, aggregation // 2)
    topology_cfg = {
        "type": "clos",
        "stages": [access, aggregation, core],
        "capacity": 10.0,
        "latency": 0.2,
        "seed": seed,
    }
    server_count = access * 2
    flow_count = min(512, max(32, server_count * 2))
    desc = (
        f"Clos-топология с уровнями ({access}, {aggregation}, {core}), приблизительно {server_count} серверов. "
        f"{description_suffix}".strip()
    )
    return ScenarioSpec(
        slug=f"clos_n1_{access}",
        description=desc,
        topology=topology_cfg,
        flow_count=flow_count,
        flow_seed=seed,
        heuristics=_default_heuristics(),
        ilp_enabled=access <= 16,
        ilp_timeout=300 if access <= 16 else None,
        load_factors=(0.3, 0.5, 0.7, 0.9, 1.0),
        repetitions=5,
        k_paths_values=(5, 10),
        extra_metadata={
            "topology_family": "Clos",
            "access_switches": access,
            "aggregation_switches": aggregation,
            "core_switches": core,
            "approx_servers": server_count,
        },
    )


def _make_fat_tree_spec(k: int, *, seed: int) -> ScenarioSpec:
    topology_cfg = {
        "type": "fat-tree",
        "k": k,
        "capacity_profile": {"access": 10.0, "aggregation": 20.0, "core": 40.0},
        "latency_profile": {"access": 0.2, "aggregation": 0.3, "core": 0.4},
    }
    server_count = (k ** 3) // 4
    flow_count = min(512, max(32, server_count * 2))
    return ScenarioSpec(
        slug=f"fat_tree_k{k}",
        description=(
            f"Fat-Tree с параметром k={k} (≈{server_count} серверов), профили скоростей "
            "и задержек из раздела~\\ref{sec:setup_topologies}."
        ),
        topology=topology_cfg,
        flow_count=flow_count,
        flow_seed=seed,
        heuristics=_default_heuristics(),
        ilp_enabled=k <= 4,
        ilp_timeout=600 if k <= 4 else None,
        load_factors=(0.3, 0.5, 0.7, 0.9, 1.0),
        repetitions=5,
        k_paths_values=(5, 10),
        extra_metadata={
            "topology_family": "Fat-Tree",
            "k": k,
            "approx_servers": server_count,
        },
    )


def _make_spine_leaf_spec(
    leaf_count: int,
    spine_count: int,
    *,
    servers_per_leaf: int,
    seed: int,
) -> ScenarioSpec:
    topology_cfg = {
        "type": "spine-leaf",
        "leaf_count": leaf_count,
        "spine_count": spine_count,
        "servers_per_leaf": servers_per_leaf,
        "capacity_profile": {"access": 10.0, "fabric": 40.0},
        "latency_profile": {"access": 0.2, "fabric": 0.35},
    }
    total_servers = leaf_count * servers_per_leaf
    return ScenarioSpec(
        slug=f"spine_leaf_L{leaf_count}_S{spine_count}",
        description=(
            f"Spine-Leaf с {leaf_count} leaf и {spine_count} spine-коммутаторами "
            f"({total_servers} серверов). Конфигурация отражает современную фабрику ЦОД."
        ),
        topology=topology_cfg,
        flow_count=min(512, max(32, total_servers * 2)),
        flow_seed=seed,
        heuristics=_default_heuristics(),
        ilp_enabled=leaf_count * spine_count <= 32,
        ilp_timeout=600 if leaf_count * spine_count <= 32 else None,
        load_factors=(0.3, 0.5, 0.7, 0.9, 1.0),
        repetitions=5,
        k_paths_values=(5, 8),
        extra_metadata={
            "topology_family": "Spine-Leaf",
            "leaf_switches": leaf_count,
            "spine_switches": spine_count,
            "servers_per_leaf": servers_per_leaf,
            "approx_servers": total_servers,
        },
    )


def _discover_custom_topologies(root: Path, *, seed: int) -> List[ScenarioSpec]:
    """
    Поиск пользовательских топологий в каталоге `data/custom_topologies`.
    Для каждой найденной JSON-схемы создаётся отдельный ScenarioSpec.
    """
    specs: List[ScenarioSpec] = []
    custom_dir = root / "data" / "custom_topologies"
    if not custom_dir.exists():
        return specs
    for path in sorted(custom_dir.glob("*.json")):
        topology_cfg = {
            "type": "custom",
            "path": str(path),
        }
        name = path.stem.replace(" ", "_")
        slug = f"custom_{name}"
        specs.append(
            ScenarioSpec(
                slug=slug,
                description=f"Пользовательская топология из файла {path.name}.",
                topology=topology_cfg,
                flow_count=256,
                flow_seed=seed,
                heuristics=_default_heuristics(),
                ilp_enabled=False,
                ilp_timeout=None,
                load_factors=(0.3, 0.5, 0.7, 0.9, 1.0),
                repetitions=5,
                k_paths_values=(5, 10),
                extra_metadata={
                    "topology_family": "Custom",
                    "source_file": str(path),
                },
            )
        )
    return specs


def default_scenarios() -> List[ScenarioSpec]:
    """
    Возвращает набор сценариев, отражающий методику глав~\ref{ch:experiment_setup} и~\ref{ch:analysis}.
    """
    base_seed = 2024
    clos_specs = [
        _make_clos_spec(access, seed=base_seed + idx)
        for idx, access in enumerate((8, 16, 24, 32, 128, 256, 512))
    ]
    fat_tree_specs = [
        _make_fat_tree_spec(k, seed=base_seed + 100 + idx)
        for idx, k in enumerate((4, 6, 8, 12, 14, 16))
    ]
    spine_specs = [
        _make_spine_leaf_spec(
            leaf_count=8,
            spine_count=4,
            servers_per_leaf=8,
            seed=base_seed + 150,
        ),
    ]
    project_root = Path(__file__).resolve().parent.parent
    custom_specs = _discover_custom_topologies(project_root, seed=base_seed + 200)
    return clos_specs + fat_tree_specs + spine_specs + custom_specs


def _dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def _dump_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def run_suite(
    output_root: Path,
    *,
    scenarios: Optional[Sequence[ScenarioSpec]] = None,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    k_paths_override: Optional[Sequence[int]] = None,
    make_plots: bool = True,
    skip_completed: bool = True,
    repetitions_override: Optional[int] = None,
    load_factors_override: Optional[Sequence[float]] = None,
) -> Dict[str, List[Path]]:
    """
    Запускает серию экспериментов, создавая подкаталог для каждого сочетания сценария и k-paths.

    :param output_root: базовый каталог, куда складываются результаты.
    :param scenarios: пользовательский список сценариев (по умолчанию используется `default_scenarios()`).
    :param include: подмножество slug'ов, которые нужно оставить.
    :param exclude: slug'и, которые следует пропустить.
    :param k_paths_override: альтернативный набор значений k для всех сценариев.
    :param make_plots: строить ли визуализации (стр.~\ref{sec:impl_repro}).
    :returns: словарь slug → список каталогов с результатами.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_list = list(scenarios or default_scenarios())
    if include:
        include_set = {slug for slug in include}
        scenario_list = [spec for spec in scenario_list if spec.slug in include_set]
    if exclude:
        exclude_set = {slug for slug in exclude}
        scenario_list = [spec for spec in scenario_list if spec.slug not in exclude_set]

    index: Dict[str, List[Path]] = {}
    for spec in scenario_list:
        scenario_frames = []
        selected_k_paths = list(k_paths_override or spec.k_paths_values)
        for k_paths in selected_k_paths:
            scenario_dir = output_root / spec.slug / f"k_paths_{k_paths}"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = scenario_dir / "metrics.csv"
            if skip_completed and metrics_path.exists():
                print(
                    _color("⏭  ", "warning")
                    + f"{spec.slug} (k_paths={k_paths}) — результаты найдены, пропуск",
                    flush=True,
                )
                index.setdefault(spec.slug, []).append(scenario_dir)
                continue

            print(
                _color("▶  ", "info")
                + f"{spec.slug} (k_paths={k_paths}) — запуск сценария",
                flush=True,
            )

            config = spec.build_config(
                k_paths=k_paths,
                repetitions=repetitions_override,
                load_factors=load_factors_override,
            )
            config_path = scenario_dir / "config.yaml"
            _dump_yaml(config_path, config)

            metadata = {
                "slug": spec.slug,
                "description": spec.description,
                "k_paths": k_paths,
                "generated_at": dt.datetime.utcnow().isoformat() + "Z",
                "extra": dict(spec.extra_metadata),
                "repetitions": config["evaluation"]["repetitions"],
                "load_factors": config["evaluation"]["load_factors"],
            }
            metadata_path = scenario_dir / "metadata.json"
            _dump_json(metadata_path, metadata)

            def _progress(
                rep_idx: int,
                rep_total: int,
                load_idx: int,
                load_total: int,
                load_factor: float,
                status: str,
            ) -> None:
                prefix = "    "
                label = f"rep {rep_idx + 1}/{rep_total}, λ={load_factor:.2f} ({load_idx + 1}/{load_total})"
                if status == "start":
                    symbol = _color("→ ", "muted")
                elif status == "done":
                    symbol = _color("✓ ", "success")
                elif status == "error":
                    symbol = _color("✖ ", "error")
                else:
                    symbol = ""
                print(f"{prefix}{symbol}{label}", flush=True)

            try:
                results = evaluate.run_experiment(
                    config_path,
                    scenario_dir,
                    progress_cb=_progress,
                )
            except Exception as exc:
                print(
                    _color("✖  ", "error")
                    + f"{spec.slug} (k_paths={k_paths}) — ошибка: {exc}",
                    flush=True,
                )
                raise

            results = results.copy()
            if "k_paths" not in results.columns:
                results["k_paths"] = k_paths
            scenario_frames.append(results)

            save_reports(results, scenario_dir)
            if make_plots:
                plots.plot_delay_vs_load(results, scenario_dir)
                plots.plot_sla_violations(results, scenario_dir)

            print(
                _color("✔  ", "success")
                + f"{spec.slug} (k_paths={k_paths}) — завершено, записано {len(results)} строк",
                flush=True,
            )

            index.setdefault(spec.slug, []).append(scenario_dir)

        if make_plots and scenario_frames:
            try:
                import pandas as pd  # type: ignore
            except ImportError:
                print(
                    _color("⚠  ", "warning")
                    + f"{spec.slug} — пропуск построения heatmap (не установлен pandas)",
                    flush=True,
                )
            else:
                combined = pd.concat(scenario_frames, ignore_index=True)
                heatmap_dir = (output_root / spec.slug) / "heatmaps"
                try:
                    heatmaps = plots.plot_sla_heatmap(combined, heatmap_dir)
                except Exception as exc:  # pragma: no cover
                    print(
                        _color("⚠  ", "warning")
                        + f"{spec.slug} — не удалось построить heatmap: {exc}",
                        flush=True,
                    )
                else:
                    for path in heatmaps:
                        print(
                            _color("↳  ", "info")
                            + f"{spec.slug}: heatmap {path.name}",
                            flush=True,
                        )
    return index


def generate_heatmaps_from_existing(
    output_root: Path,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    algorithms: Optional[Iterable[str]] = None,
) -> None:
    """
    Строит тепловые карты SLA по уже сохранённым результатам (без повторного запуска экспериментов).
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Для построения heatmap требуется пакет pandas") from exc

    output_root = output_root.resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Каталог с результатами не найден: {output_root}")

    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None

    for scenario_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        slug = scenario_dir.name
        if include_set and slug not in include_set:
            continue
        if exclude_set and slug in exclude_set:
            continue

        frames = []
        for metrics_path in sorted(scenario_dir.glob("k_paths_*/metrics.csv")):
            try:
                k_paths_str = metrics_path.parent.name.split("_")[-1]
                k_paths = int(k_paths_str)
            except (ValueError, IndexError):
                print(
                    _color("⚠  ", "warning")
                    + f"Пропуск файла {metrics_path}: не удалось определить k_paths",
                    flush=True,
                )
                continue
            df = pd.read_csv(metrics_path)
            if "k_paths" not in df.columns:
                df["k_paths"] = k_paths
            frames.append(df)

        if not frames:
            print(
                _color("⚠  ", "warning")
                + f"{slug}: не найдено файлов metrics.csv для построения heatmap",
                flush=True,
            )
            continue

        combined = pd.concat(frames, ignore_index=True)
        heatmap_dir = scenario_dir / "heatmaps"
        try:
            plots.plot_sla_heatmap(combined, heatmap_dir, algorithms=algorithms)
        except Exception as exc:  # pragma: no cover
            print(
                _color("⚠  ", "warning")
                + f"{slug}: ошибка построения heatmap — {exc}",
                flush=True,
            )
        else:
            print(
                _color("✔  ", "success")
                + f"{slug}: heatmap сохранены в {heatmap_dir.relative_to(output_root)}",
                flush=True,
            )


def main() -> None:
    """
    CLI-обёртка: `python -m netcalc_dc.suite --output experiments`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Пакетный запуск экспериментов для дипломного стенда"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("experiment_suite"),
        help="Каталог для сохранения результатов (будут созданы подпапки по сценариям)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Запустить только указанные сценарии (slug).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Пропустить перечисленные сценарии (slug).",
    )
    parser.add_argument(
        "--k-paths",
        nargs="+",
        type=int,
        help="Переопределить список значений k для всех сценариев.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики (ускоряет прогон).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать сценарии, даже если уже есть готовые результаты.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Переопределить число повторов для всех сценариев (по умолчанию из спецификаций).",
    )
    parser.add_argument(
        "--loads",
        nargs="+",
        type=float,
        help="Переопределить список коэффициентов нагрузки для всех сценариев.",
    )
    parser.add_argument(
        "--heatmaps-only",
        action="store_true",
        help="Построить тепловые карты по существующим результатам и завершить без запуска экспериментов.",
    )
    args = parser.parse_args()

    if args.heatmaps_only:
        generate_heatmaps_from_existing(
            args.output,
            include=args.include,
            exclude=args.exclude,
        )
        return

    run_suite(
        args.output,
        include=args.include,
        exclude=args.exclude,
        k_paths_override=tuple(args.k_paths) if args.k_paths else None,
        make_plots=not args.no_plots,
        skip_completed=not args.force,
        repetitions_override=args.repetitions,
        load_factors_override=tuple(args.loads) if args.loads else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
