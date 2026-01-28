from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
import yaml

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


def _run_one_rep(cfg_path: Path, base_dir: Path, spec_slug: str, k_paths: int, rep: int):
    """Запуск одного повтора в отдельной подпапке."""
    if pd is None:
        raise RuntimeError("Требуется pandas: pip install -r requirements.txt")

    rep_dir = base_dir / f"rep_{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = evaluate.run_experiment(cfg_path, rep_dir, measure_timings=True)
    df = df.copy()
    df["rep"] = rep
    df["repetition"] = rep
    df["scenario"] = spec_slug
    df["k_paths"] = k_paths
    return df


def run_with_timing(
    output: Path,
    include: Iterable[str] | None = None,
    *,
    config_path: Path | None = None,
    repetitions: int = 10,
    workers: int = 1,
    k_paths_override: Sequence[int] | None = None,
    load_factors_override: Sequence[float] | None = None,
    heuristics_override: Sequence[str] | None = None,
    disable_ilp: bool = False,
    make_plots: bool = True,
    language: str = "ru",
) -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("Требуется pandas: pip install -r requirements.txt")

    specs = []
    base_config_from_file = None
    if config_path is not None:
        base_config_from_file = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        slug = config_path.stem
        specs = [slug]
    else:
        specs = suite.default_scenarios()
        if include:
            include_set = set(include)
            specs = [s for s in specs if s.slug in include_set]

    out_root = output
    out_root.mkdir(parents=True, exist_ok=True)
    summary_frames = []

    for spec in specs:
        if base_config_from_file is not None:
            selected_k_paths = list(
                k_paths_override
                or [base_config_from_file.get("evaluation", {}).get("k_paths", 5)]
            )
            spec_slug = spec
            flow_seed_base = base_config_from_file.get("flows", {}).get("seed", 0)
        else:
            selected_k_paths = list(k_paths_override or spec.k_paths_values)
            spec_slug = spec.slug
            flow_seed_base = spec.flow_seed

        for k_paths in selected_k_paths:
            cfg_dir = out_root / f"{spec_slug}_k{k_paths}"
            cfg_dir.mkdir(parents=True, exist_ok=True)

            # Базовый конфиг на один повтор; seed корректируем для каждого rep отдельно
            if base_config_from_file is not None:
                base_config = dict(base_config_from_file)
                base_config["evaluation"] = dict(base_config.get("evaluation", {}))
                base_config["evaluation"]["k_paths"] = k_paths
                base_config["evaluation"]["repetitions"] = 1
                if load_factors_override is not None:
                    base_config["evaluation"]["load_factors"] = list(load_factors_override)
            else:
                base_config = spec.build_config(
                    k_paths=k_paths,
                    repetitions=1,
                    load_factors=load_factors_override,
                )
            if heuristics_override is not None:
                base_config["algorithms"]["heuristics"] = list(heuristics_override)
            if disable_ilp:
                base_config["algorithms"]["ilp"]["enabled"] = False

            df_list = []

            if workers <= 1:
                # Последовательно
                for rep in range(repetitions):
                    rep_dir = cfg_dir / f"rep_{rep:02d}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    rep_config = dict(base_config)
                    rep_config["flows"] = dict(base_config["flows"])
                    rep_config["flows"]["seed"] = flow_seed_base + rep
                    rep_cfg_path = rep_dir / "config.yaml"
                    suite._dump_yaml(rep_cfg_path, rep_config)
                    df_list.append(_run_one_rep(rep_cfg_path, cfg_dir, spec_slug, k_paths, rep))
            else:
                # Параллельно по rep
                rep_tasks = []
                for rep in range(repetitions):
                    rep_dir = cfg_dir / f"rep_{rep:02d}"
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    rep_config = dict(base_config)
                    rep_config["flows"] = dict(base_config["flows"])
                    rep_config["flows"]["seed"] = flow_seed_base + rep
                    rep_cfg_path = rep_dir / "config.yaml"
                    suite._dump_yaml(rep_cfg_path, rep_config)
                    rep_tasks.append((rep_cfg_path, cfg_dir, spec_slug, k_paths, rep))
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [
                        ex.submit(_run_one_rep, *task)
                        for task in rep_tasks
                    ]
                    for fut in as_completed(futures):
                        df_list.append(fut.result())

            df_all = pd.concat(df_list, ignore_index=True)
            df_all.to_csv(cfg_dir / "metrics_time.csv", index=False)
            summary_frames.append(df_all)
            if make_plots:
                plots.plot_delay_vs_load_dispersion(df_all, cfg_dir, language=language)
                plots.plot_sla_violations(df_all, cfg_dir, language=language)
                if "duration_ms" in df_all.columns:
                    plots.plot_duration_vs_load_dispersion(df_all, cfg_dir, language=language)

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        combined.to_csv(out_root / "combined_metrics_time.csv", index=False)
        print(f"Готово. Результаты и времена сохранены в {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Серия экспериментов с измерением времени")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Каталог для результатов")
    parser.add_argument("--include", nargs="+", help="Список slug'ов сценариев")
    parser.add_argument(
        "--config",
        type=Path,
        help="Использовать конкретный config.yaml (ручные потоки и кастомные топологии)",
    )
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
        "--plots-only",
        action="store_true",
        help="Построить графики по существующим metrics_time.csv и выйти",
    )
    parser.add_argument(
        "--lang",
        choices=("ru", "en"),
        default="ru",
        help="Язык подписей на графиках (ru/en)",
    )

    args = parser.parse_args()
    if args.plots_only:
        out_root = args.output
        include_set = set(args.include) if args.include else None
        for cfg_dir in sorted(out_root.glob("*_k*")):
            if not cfg_dir.is_dir():
                continue
            scenario_name = cfg_dir.name.split("_k", 1)[0]
            if include_set and scenario_name not in include_set:
                continue
            metrics_path = cfg_dir / "metrics_time.csv"
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            plots.plot_delay_vs_load_dispersion(df, cfg_dir, language=args.lang)
            plots.plot_sla_violations(df, cfg_dir, language=args.lang)
            if "duration_ms" in df.columns:
                plots.plot_duration_vs_load_dispersion(df, cfg_dir, language=args.lang)
        print(f"Графики построены в {out_root}")
        return
    run_with_timing(
        args.output,
        include=args.include,
        config_path=args.config,
        repetitions=args.repetitions,
        workers=args.workers,
        k_paths_override=tuple(args.k_paths) if args.k_paths else None,
        load_factors_override=tuple(args.loads) if args.loads else None,
        heuristics_override=tuple(args.heuristics) if args.heuristics else None,
        disable_ilp=args.no_ilp,
        make_plots=not args.no_plots,
        language=args.lang,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
