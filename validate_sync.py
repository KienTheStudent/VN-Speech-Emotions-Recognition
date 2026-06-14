#!/usr/bin/env python3
"""
Validates that the markdown and latex files exactly match the generated
tables from the JSON source of truth.
"""

import sys
from pathlib import Path

# Import generators from sync_results
from sync_results import (
    load_json, generate_md_benchmark, generate_md_ablation,
    generate_tex_benchmark, generate_tex_ablation, update_between_markers,
    BENCHMARK_PATH, ABLATION_PATH, ROOT
)

def check_file(filepath, bench_data, ablation_data, is_latex=False):
    if not filepath.exists():
        print(f"  ⚠ File not found: {filepath}")
        return False

    original_content = filepath.read_text(encoding="utf-8")
    content = original_content

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

    if content != original_content:
        print(f"❌ MISMATCH FOUND IN {filepath.name}!")
        print("The documentation is out of sync with the JSON source of truth.")
        print("Please run `python sync_results.py` to fix this.")
        return False
        
    print(f"✓ {filepath.name} is in sync.")
    return True


def main():
    print("=" * 60)
    print("VALIDATING DOCUMENTATION SYNCHRONIZATION")
    print("=" * 60)

    bench_data = load_json(BENCHMARK_PATH)
    ablation_data = load_json(ABLATION_PATH)

    if bench_data is None:
        print("❌ benchmark_results_gpu.json not found!")
        sys.exit(1)

    all_good = True
    all_good &= check_file(ROOT / "README.md", bench_data, ablation_data, is_latex=False)
    all_good &= check_file(ROOT / "insight.md", bench_data, ablation_data, is_latex=False)
    all_good &= check_file(ROOT / "Report_SER.tex", bench_data, ablation_data, is_latex=True)

    if not all_good:
        sys.exit(1)
        
    print("\n✓ CI PASS: All documentation matches the JSON data.")

if __name__ == "__main__":
    main()
