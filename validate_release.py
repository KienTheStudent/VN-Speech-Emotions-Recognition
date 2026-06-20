#!/usr/bin/env python3
"""
Release Validation Script (Thesis Grade) — Cross-Artifact Integrity Checker.

Validates:
1. Manifest provenance and checksum (no speaker overlap).
2. Deep model metadata provenance (split_checksum must match manifest).
3. Documentation synchronization (LaTeX tables vs JSON sources).
4. Figure artifact existence (all report_images/*.png referenced in LaTeX exist).
5. Method name consistency (LaTeX method names match JSON method names).
6. Obsolete reference detection.
"""

import sys
import json
import re
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

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


def validate_figure_existence():
    """Check that all figure files referenced in LaTeX actually exist."""
    tex_path = ROOT / "Report_SER.tex"
    content = tex_path.read_text(encoding="utf-8")

    # Find all includegraphics references
    pattern = r"\\includegraphics.*?\{(.+?)\}"
    references = re.findall(pattern, content)

    all_good = True
    for ref in references:
        fig_path = ROOT / ref
        if not fig_path.exists():
            print(f"[FAIL] Missing figure: {ref}")
            all_good = False
        else:
            print(f"[PASS] Figure exists: {ref}")

    return all_good


def validate_method_names():
    """Check that method names in LaTeX match those in benchmark JSON."""
    bench_path = ROOT / "benchmark_results_gpu.json"
    tex_path = ROOT / "Report_SER.tex"

    with open(bench_path, "r") as f:
        bench = json.load(f)

    json_methods = set(r["method"] for r in bench["ranked_results"])
    tex_content = tex_path.read_text(encoding="utf-8")

    all_good = True
    for method in json_methods:
        if method not in tex_content:
            print(f"[WARN] Method '{method}' from JSON not found verbatim in LaTeX.")
            all_good = False
        else:
            print(f"[PASS] Method '{method}' found in LaTeX.")

    return all_good


def validate_split_checksum():
    """Verify the split manifest checksum matches its stored value."""
    manifest_path = ROOT / "split_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    stored_checksum = manifest.get("checksum", "")
    # Recompute from the split data
    split_data = json.dumps({
        "train_indices": manifest["train_indices"],
        "val_indices": manifest["val_indices"],
        "test_indices": manifest["test_indices"]
    }, sort_keys=True)
    computed = hashlib.sha256(split_data.encode()).hexdigest()

    if stored_checksum and stored_checksum != computed:
        print(f"[FAIL] Split manifest checksum mismatch!")
        print(f"   Stored:   {stored_checksum[:16]}...")
        print(f"   Computed: {computed[:16]}...")
        return False

    print(f"[PASS] Split manifest checksum verified.")
    return True


def validate_no_obsolete_refs():
    """Scan LaTeX for references to known obsolete artifacts."""
    tex_path = ROOT / "Report_SER.tex"
    content = tex_path.read_text(encoding="utf-8")

    obsolete_patterns = [
        "sync_results.py",  # Old name, should reference sync_notebook.py or generate_report_figures.py
        "cryptographically secured",  # Overblown language from earlier drafts
        "Whisper-tiny"  # Obsolete since we replaced it with Whisper-small for inferior ablation
    ]

    all_good = True
    for pat in obsolete_patterns:
        if pat in content:
            print(f"[WARN] Potentially obsolete reference found: '{pat}'")
            all_good = False

    if all_good:
        print("[PASS] No obsolete references detected.")
    return all_good


def validate_per_sample_predictions():
    """Verify inference metadata artifact exists and is complete."""
    preds_path = ROOT / "per_sample_predictions.json"
    manifest_path = ROOT / "split_manifest.json"

    if not preds_path.exists():
        print(f"[FAIL] Inference metadata not found at {preds_path}")
        return False

    with open(preds_path, "r") as f:
        preds = json.load(f)
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    expected_len = len(manifest["test_indices"])
    if len(preds) != expected_len:
        print(f"[FAIL] Inference metadata count mismatch! Expected {expected_len}, got {len(preds)}")
        return False
        
    print(f"[PASS] Inference metadata verified ({len(preds)} test samples).")
    return True


def main():
    print("=" * 60)
    print("RELEASE VALIDATION: CROSS-ARTIFACT INTEGRITY CHECK")
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
        print("\n[FAIL] Model provenance validation failed!")
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

    # 4. Figure Existence
    print("\n--- Validating Figure Artifacts ---")
    all_good &= validate_figure_existence()

    # 5. Method Name Consistency
    print("\n--- Validating Method Names ---")
    all_good &= validate_method_names()

    # 6. Obsolete Reference Scan
    print("\n--- Scanning for Obsolete References ---")
    all_good &= validate_no_obsolete_refs()

    # 7. Inference Metadata Scan
    print("\n--- Validating Inference Metadata ---")
    all_good &= validate_per_sample_predictions()

    # 7. Notebook Sync
    print("\n--- Validating Notebook ---")
    import subprocess
    nb_path = ROOT / "internal" / "SER.ipynb"
    nb_content_before = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
    try:
        subprocess.run([sys.executable, str(ROOT / "internal" / "sync_notebook.py")], check=True, capture_output=True)
        nb_content_after = nb_path.read_text(encoding="utf-8") if nb_path.exists() else ""
        if nb_content_before != nb_content_after:
            print(f"[FAIL] MISMATCH FOUND IN SER.ipynb!")
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

    print("\n" + "=" * 60)
    print("[PASS] ALL CHECKS PASSED: Cross-artifact integrity verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
