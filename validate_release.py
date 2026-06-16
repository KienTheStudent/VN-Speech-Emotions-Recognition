#!/usr/bin/env python3
"""
Release Validation Script (Thesis Grade).
Validates:
1. Manifest provenance and checksum (no overlap).
2. Deep model metadata provenance (must have split_checksum matching the manifest).
3. Documentation synchronization (Markdown, LaTeX, Notebook).
"""

import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ensure internal package is importable
sys.path.insert(0, str(ROOT))

from internal.split_validator import validate_manifest
from internal.sync_results import (
    load_json, generate_md_benchmark, generate_md_ablation,
    generate_tex_benchmark, generate_tex_ablation, update_between_markers,
    BENCHMARK_PATH, ABLATION_PATH
)

def check_file(filepath, bench_data, ablation_data, is_latex=False):
    if not filepath.exists():
        print(f"  [FAIL] File not found: {filepath}")
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
        print(f"[FAIL] MISMATCH FOUND IN {filepath.name}!")
        print("The documentation is out of sync with the JSON source of truth.")
        print("Please run `python internal/sync_results.py` to fix this.")
        return False
        
    print(f"[PASS] {filepath.name} is in sync.")
    return True

def validate_model_provenance(manifest_checksum):
    """Ensure all deep models were trained on the exact current manifest."""
    all_good = True
    models_to_check = [
        ("ECAPA-TDNN", ROOT / "ECAPA" / "emotion_model" / "metadata.json"),
        ("DFAT", ROOT / "DFAT_Hybrid_Fusion" / "dualstream_model" / "metadata.json")
    ]
    
    for name, meta_path in models_to_check:
        if not meta_path.exists():
            print(f"[FAIL] {name} metadata not found at {meta_path}!")
            all_good = False
            continue
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        if "split_checksum" not in meta:
            print(f"[FAIL] {name} metadata is missing 'split_checksum'! Provenance cannot be verified.")
            all_good = False
            continue
            
        if meta["split_checksum"] != manifest_checksum:
            print(f"[FAIL] {name} split_checksum mismatch!")
            print(f"   Expected: {manifest_checksum}")
            print(f"   Got:      {meta['split_checksum']}")
            all_good = False
            continue
            
        print(f"[PASS] {name} provenance verified (checksum: {manifest_checksum[:8]}...).")
        
    return all_good

def main():
    print("=" * 60)
    print("RELEASE VALIDATION: PROVENANCE & SYNCHRONIZATION")
    print("=" * 60)

    # 1. Validate Manifest
    try:
        manifest_checksum = validate_manifest()
        print(f"[PASS] split_manifest.json is valid (checksum: {manifest_checksum[:8]}...).")
    except Exception as e:
        print(f"[FAIL] Manifest validation failed: {e}")
        sys.exit(1)

    # 2. Validate Model Provenance
    print("\n--- Validating Model Provenance ---")
    if not validate_model_provenance(manifest_checksum):
        print("\n[FAIL] Model provenance validation failed! Models must be retrained on the current split.")
        sys.exit(1)

    # 3. Validate Documentation Sync
    print("\n--- Validating Documentation Sync ---")
    bench_data = load_json(BENCHMARK_PATH)
    ablation_data = load_json(ABLATION_PATH)

    if bench_data is None:
        print("[FAIL] benchmark_results_gpu.json not found!")
        sys.exit(1)

    all_good = True
    all_good &= check_file(ROOT / "README.md", bench_data, ablation_data, is_latex=False)
    all_good &= check_file(ROOT / "insight.md", bench_data, ablation_data, is_latex=False)
    all_good &= check_file(ROOT / "Report_SER.tex", bench_data, ablation_data, is_latex=True)

    # Validate SER.ipynb
    nb_path = ROOT / "internal" / "SER.ipynb"
    nb_content_before = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
    try:
        subprocess.run([sys.executable, str(ROOT / "internal" / "sync_notebook.py")], check=True, capture_output=True)
        nb_content_after = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
        if nb_content_before != nb_content_after:
            print(f"[FAIL] MISMATCH FOUND IN SER.ipynb!")
            print("The notebook is out of sync. Please run `python internal/sync_results.py` to fix this.")
            if nb_path.exists() and nb_content_before:
                nb_path.write_text(nb_content_before, encoding="utf-8")
            all_good = False
        else:
            print(f"[PASS] SER.ipynb is in sync.")
    except Exception as e:
        print(f"[FAIL] Failed to validate notebook: {e}")
        all_good = False

    if not all_good:
        print("\n[FAIL] Release validation failed!")
        sys.exit(1)
        
    print("\n[PASS] CI PASS: Provenance is strict and all documentation matches the JSON data.")

if __name__ == "__main__":
    main()
