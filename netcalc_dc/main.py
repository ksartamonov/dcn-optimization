"""
CLI для запуска экспериментов по оптимизации задержек.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import evaluate, plots


def save_reports(results, output_dir: Path) -> None:
    """Создаёт сводные отчёты по метрикам."""

    def _table_markdown(df) -> str:
        cols = list(df.columns)
        header = "|" + "|".join(str(col) for col in cols) + "|"
        separator = "|" + "|".join("---" for _ in cols) + "|"
        rows = []
        for _, row in df.iterrows():
            rows.append("|" + "|".join(str(row[col]) for col in cols) + "|")
        return "\n".join([header, separator, *rows])

    metrics = ["max_delay", "avg_delay", "sla_violations"]
    finite_results = results.replace([np.inf, -np.inf], np.nan)

    summary = (
        finite_results.groupby("algorithm")[metrics]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary_path = output_dir / "summary_by_algorithm.csv"
    summary.to_csv(summary_path)

    per_load = (
        finite_results.groupby(["load_factor", "algorithm"])[metrics]
        .mean()
        .reset_index()
        .round(4)
    )
    per_load_path = output_dir / "summary_by_load.csv"
    per_load.to_csv(per_load_path, index=False)

    per_load_display = per_load.copy()
    for metric in ["max_delay", "avg_delay"]:
        mask = results.groupby(["load_factor", "algorithm"])[metric].apply(lambda x: np.isinf(x).any())
        mask = mask.reset_index(name="has_inf")
        for _, row in mask.iterrows():
            if row["has_inf"]:
                sel = (per_load_display["load_factor"] == row["load_factor"]) & (
                    per_load_display["algorithm"] == row["algorithm"]
                )
                per_load_display.loc[sel, metric] = "∞"
    per_load_display["sla_violations"] = per_load_display["sla_violations"].round(4)
    per_load_display["load_factor"] = per_load_display["load_factor"].map(lambda x: f"{x:.2f}")

    def _format_load_value(v):
        if isinstance(v, str):
            return v
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "—"
            return f"{float(v):.4f}"
        return str(v)

    for col in ["max_delay", "avg_delay", "sla_violations"]:
        per_load_display[col] = per_load_display[col].apply(_format_load_value)

    report_path = output_dir / "report.md"
    best_candidates = summary["max_delay_mean"].dropna()
    if not best_candidates.empty:
        best_algo = best_candidates.idxmin()
        best_row = summary.loc[best_algo]
        best_lines = (
            f"- Название: **{best_algo}**\n"
            f"- Средняя max задержка: {best_row['max_delay_mean']:.4f} мс\n"
            f"- Средняя доля нарушений SLA: {best_row['sla_violations_mean']:.4f}\n"
        )
    else:
        best_lines = "- Нет алгоритмов с конечной средней задержкой\n"

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Отчёт по эксперименту\n\n")
        fh.write(f"Сгенерировано записей: **{len(results)}**\n\n")
        fh.write("## Лучший алгоритм по средней максимальной задержке\n\n")
        fh.write(best_lines + "\n")
        fh.write("## Средние метрики по алгоритмам\n\n")
        summary_display = summary.reset_index()

        def _format_algo_value(v):
            if isinstance(v, str):
                return v
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return "∞"
                return f"{float(v):.4f}"
            return str(v)

        for col in summary_display.columns[1:]:
            summary_display[col] = summary_display[col].apply(_format_algo_value)
        fh.write(_table_markdown(summary_display))
        fh.write("\n\n")
        fh.write("## Средние метрики по факторам загрузки\n\n")
        fh.write(_table_markdown(per_load_display))
        fh.write("\n\n")

        fh.write("## Интерпретация графиков\n\n")
        fh.write(
            "- Столбчатая диаграмма показывает долю нарушений SLA: более тёмные столбцы соответствуют большей нагрузке. "
            "При λ ≥ 0.8 большинство жадных эвристик перестают быть допустимыми (значение 1.0).\n"
        )
        fh.write(
            "- Линейный график максимальной задержки иллюстрирует рост латентности при увеличении нагрузки. "
            "Маркер «∞» означает, что алгоритм не нашёл маршрутов с ограниченной задержкой на данном уровне нагрузки.\n\n"
        )

        fh.write("### Итоги по нагрузкам\n\n")
        for load in sorted(results["load_factor"].unique()):
            subset = finite_results[results["load_factor"] == load]
            best = subset.groupby("algorithm")["max_delay"].mean().dropna()
            if best.empty:
                fh.write(f"- λ={load:.2f}: нет алгоритмов с конечной задержкой\n")
                continue
            winner = best.idxmin()
            sla = subset[subset["algorithm"] == winner]["sla_violations"].mean()
            fh.write(
                f"- λ={load:.2f}: минимальную среднюю задержку показывает **{winner}** "
                f"({best[winner]:.3f} мс), средняя доля нарушений SLA ≈ {sla:.3f}\n"
            )

    print(f"Сводные таблицы сохранены: {summary_path.name}, {per_load_path.name}")
    print(f"Markdown-отчёт: {report_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Эксперименты по динамической оптимизации сетевого ресурса"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Путь к YAML-конфигурации сценария",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results"),
        help="Каталог для сохранения результатов",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не строить графики (по умолчанию графики сохраняются)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate.run_experiment(args.config, args.output)
    print(f"Сохранено {len(results)} строк результатов в {args.output}")
    save_reports(results, args.output)
    if not args.no_plots:
        plot1 = plots.plot_delay_vs_load(results, args.output)
        plot2 = plots.plot_sla_violations(results, args.output)
        try:
            heatmaps = plots.plot_sla_heatmap(results, args.output)
        except Exception as exc:  # pragma: no cover
            heatmaps = []
            print(f"[warning] heatmap generation failed: {exc}")
        names = [plot1.name, plot2.name] + [p.name for p in heatmaps]
        print(f"Графики сохранены: {', '.join(names)}")


if __name__ == "__main__":
    main()
