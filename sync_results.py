#!/usr/bin/env python3
"""Auto-generate benchmark tables in README.md, insight.md, and Report_SER.tex.

Reads from:
  - benchmark_results_gpu.json    (main benchmark)
  - DFAT_Hybrid_Fusion/ablation_results.json (ablation study)

Updates sections between marker tags in each target file:
  README.md / insight.md :  <!-- START_BENCHMARK_TABLE --> ... <!-- END_BENCHMARK_TABLE -->
                            <!-- START_ABLATION_TABLE --> ... <!-- END_ABLATION_TABLE -->
  Report_SER.tex         :  %% START_BENCHMARK_TABLE ... %% END_BENCHMARK_TABLE
                            %% START_ABLATION_TABLE ... %% END_ABLATION_TABLE
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
BENCHMARK_PATH = ROOT / "benchmark_results_gpu.json"
ABLATION_PATH = ROOT / "DFAT_Hybrid_Fusion" / "ablation_results.json"


def load_json(path):
    if not path.exists():
        print(f"  ⚠ Not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val, decimals=4):
    """Format a float to fixed decimal string."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def fmt_pm(mean, std, decimals=4):
    """Format mean ± std."""
    if std is not None and std > 0:
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


# ==================== MARKDOWN GENERATORS ====================

def _get_metric(r, new_key, old_key):
    """Get a metric from either new-format or old-format results."""
    return r.get(new_key, r.get(old_key))


def generate_md_benchmark(data):
    """Generate a Markdown table for the benchmark."""
    lines = []
    lines.append("| Method | Category | wF1 (mean ± std) | mF1 | Acc | E2E Latency |")
    lines.append("|--------|----------|------------------|-----|-----|-------------|")

    primary = data.get("primary_models", [])

    for r in data["ranked_results"]:
        name = r["method"]
        cat = "**Primary**" if name in primary else "Secondary"
        wf1_mean = _get_metric(r, "f1_weighted_mean", "f1_weighted")
        wf1_std = r.get("f1_weighted_std", 0)
        wf1 = fmt_pm(wf1_mean, wf1_std)
        mf1 = fmt(_get_metric(r, "f1_macro_mean", "f1_macro"))
        acc = fmt(_get_metric(r, "accuracy_mean", "accuracy"))

        lat = r.get("latency", {})
        e2e = lat.get("total_e2e_ms_per_sample")
        note = lat.get("note", "")
        if e2e is not None:
            lat_str = f"{e2e:.2f} ms"
        elif note:
            lat_str = note.split(";")[0]
        else:
            lat_str = "—"

        bold = "**" if name in primary else ""
        lines.append(f"| {bold}{name}{bold} | {cat} | {wf1} | {mf1} | {acc} | {lat_str} |")

    return "\n".join(lines)


def generate_md_ablation(data):
    """Generate a Markdown table for the ablation study."""
    if data is None:
        return "_Ablation results not yet available. Run `DFAT_Hybrid_Fusion/ablation_study.py`._"

    lines = []
    lines.append("| Configuration | Ensemble wF1 | Ensemble mF1 | Acc |")
    lines.append("|---------------|-------------|-------------|-----|")

    for r in data.get("ablation_results", []):
        ens = r.get("ensemble", {})
        wf1 = fmt(ens.get("f1_weighted"))
        mf1 = fmt(ens.get("f1_macro"))
        acc = fmt(ens.get("accuracy"))
        lines.append(f"| {r['config']} | {wf1} | {mf1} | {acc} |")

    return "\n".join(lines)


# ==================== LATEX GENERATORS ====================

def generate_tex_benchmark(data):
    """Generate a LaTeX tabular for the benchmark."""
    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{wF1} & \textbf{$\sigma$} & \textbf{mF1} & \textbf{Acc} \\")
    lines.append(r"\midrule")

    primary = data.get("primary_models", [])

    # Group by category
    current_cat = None
    for r in data["ranked_results"]:
        name = r["method"]
        is_primary = name in primary

        if is_primary and current_cat != "primary":
            lines.append(r"\multicolumn{5}{l}{\textit{Primary Models}} \\")
            current_cat = "primary"
        elif not is_primary and current_cat != "secondary":
            if current_cat is not None:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{l}{\textit{Secondary Baselines}} \\")
            current_cat = "secondary"

        wf1 = fmt(_get_metric(r, "f1_weighted_mean", "f1_weighted"))
        std = fmt(r.get("f1_weighted_std", 0))
        mf1 = fmt(_get_metric(r, "f1_macro_mean", "f1_macro"))
        acc = fmt(_get_metric(r, "accuracy_mean", "accuracy"))

        prefix = r"\textbf{" if is_primary else ""
        suffix = "}" if is_primary else ""

        lines.append(f"\\quad {prefix}{name}{suffix} & {wf1} & {std} & {mf1} & {acc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_tex_ablation(data):
    """Generate a LaTeX tabular for the ablation study."""
    if data is None:
        return r"% Ablation results not yet available"

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{wF1} & \textbf{mF1} & \textbf{Acc} \\")
    lines.append(r"\midrule")

    for r in data.get("ablation_results", []):
        ens = r.get("ensemble", {})
        wf1 = fmt(ens.get("f1_weighted"))
        mf1 = fmt(ens.get("f1_macro"))
        acc = fmt(ens.get("accuracy"))
        cfg = r["config"].replace("_", r"\_").replace("%", r"\%")
        lines.append(f"{cfg} & {wf1} & {mf1} & {acc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ==================== FILE UPDATERS ====================

def update_between_markers(content, start_marker, end_marker, replacement):
    """Replace content between start_marker and end_marker (exclusive)."""
    pattern = re.compile(
        re.escape(start_marker) + r"\n.*?" + re.escape(end_marker),
        re.DOTALL,
    )
    new_block = f"{start_marker}\n{replacement}\n{end_marker}"
    if pattern.search(content):
        # Use lambda to avoid regex interpretation of backslashes in replacement
        return pattern.sub(lambda m: new_block, content)
    else:
        print(f"    ⚠ Markers not found: {start_marker}")
        return content


def update_file(filepath, bench_data, ablation_data, is_latex=False):
    """Update a file with auto-generated tables."""
    if not filepath.exists():
        print(f"  ⚠ File not found: {filepath}")
        return

    content = filepath.read_text(encoding="utf-8")

    if is_latex:
        bench_start = "%% START_BENCHMARK_TABLE"
        bench_end = "%% END_BENCHMARK_TABLE"
        abl_start = "%% START_ABLATION_TABLE"
        abl_end = "%% END_ABLATION_TABLE"
        bench_table = generate_tex_benchmark(bench_data) if bench_data else ""
        abl_table = generate_tex_ablation(ablation_data) if ablation_data else ""
    else:
        bench_start = "<!-- START_BENCHMARK_TABLE -->"
        bench_end = "<!-- END_BENCHMARK_TABLE -->"
        abl_start = "<!-- START_ABLATION_TABLE -->"
        abl_end = "<!-- END_ABLATION_TABLE -->"
        bench_table = generate_md_benchmark(bench_data) if bench_data else ""
        abl_table = generate_md_ablation(ablation_data) if ablation_data else ""

    if bench_data:
        content = update_between_markers(content, bench_start, bench_end, bench_table)
    if ablation_data or not is_latex:
        content = update_between_markers(content, abl_start, abl_end, abl_table)

    filepath.write_text(content, encoding="utf-8")
    print(f"  ✓ Updated: {filepath.name}")


def main():
    print("=" * 60)
    print("SYNC RESULTS → README.md, insight.md, Report_SER.tex")
    print("=" * 60)

    bench_data = load_json(BENCHMARK_PATH)
    ablation_data = load_json(ABLATION_PATH)

    if bench_data is None:
        print("Cannot proceed without benchmark_results_gpu.json.")
        return

    update_file(ROOT / "README.md", bench_data, ablation_data, is_latex=False)
    update_file(ROOT / "insight.md", bench_data, ablation_data, is_latex=False)
    update_file(ROOT / "Report_SER.tex", bench_data, ablation_data, is_latex=True)

    print("\n✓ All files synchronized from benchmark_results_gpu.json")


if __name__ == "__main__":
    main()
