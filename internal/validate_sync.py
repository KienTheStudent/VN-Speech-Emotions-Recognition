#!/usr/bin/env python3
"""
Validates that the markdown and latex files exactly match the generated
tables from the JSON source of truth.
"""

import sys
from pathlib import Path

# Import generators from sync_results
from sync_results import (
    load_json,
    generate_md_benchmark,
    generate_md_ablation,
    generate_tex_benchmark,
    generate_tex_ablation,
    update_between_markers,
    BENCHMARK_PATH,
    ABLATION_PATH,
    ROOT,
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


    from split_validator import validate_manifest
    try:
        current_checksum = validate_manifest()
    except Exception as e:
        print(f"❌ split_manifest validation failed: {e}")
        sys.exit(1)

    import json
    for model_meta in ["DFAT_Hybrid_Fusion/dualstream_model/metadata.json", "ECAPA/emotion_model/metadata.json"]:
        meta_path = ROOT / model_meta
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("split_checksum") != current_checksum:
                print(f"❌ {model_meta} checksum mismatch! The model was trained on a different data split.")
                sys.exit(1)
            if "n_seeds" not in meta or "seeds" not in meta:
                print(f"❌ {model_meta} is missing seeds/n_seeds schema.")
                sys.exit(1)

    bench_data = load_json(BENCHMARK_PATH)
    ablation_data = load_json(ABLATION_PATH)

    if bench_data is None:
        print("❌ benchmark_results_gpu.json not found!")
        sys.exit(1)

    all_good = True
    all_good &= check_file(
        ROOT / "README.md", bench_data, ablation_data, is_latex=False
    )
    all_good &= check_file(
        ROOT / "insight.md", bench_data, ablation_data, is_latex=False
    )
    all_good &= check_file(
        ROOT / "Report_SER.tex", bench_data, ablation_data, is_latex=True
    )

    # Validate SER.ipynb
    nb_path = ROOT / "internal" / "SER.ipynb"
    nb_content_before = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
    import subprocess

    try:
        subprocess.run(
            ["python", str(ROOT / "internal" / "sync_notebook.py")], check=True, capture_output=True
        )
        nb_content_after = (
            nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
        )
        if nb_content_before != nb_content_after:
            print(f"❌ MISMATCH FOUND IN SER.ipynb!")
            print(
                "The notebook is out of sync. Please run `python sync_results.py` to fix this."
            )
            if nb_path.exists() and nb_content_before:
                nb_path.write_text(nb_content_before, encoding="utf-8")
            all_good = False
        else:
            print(f"✓ SER.ipynb is in sync.")
    except Exception as e:
        print(f"❌ Failed to validate notebook: {e}")
        all_good = False

    if not all_good:
        sys.exit(1)

    print("\n✓ CI PASS: All documentation matches the JSON data.")


if __name__ == "__main__":
    main()
