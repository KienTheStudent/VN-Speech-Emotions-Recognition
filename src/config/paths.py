import os
from pathlib import Path

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data & Manifests
SPLIT_MANIFEST_PATH = ROOT_DIR / "split_manifest.json"

# Models & Checkpoints
ECAPA_MODEL_DIR = ROOT_DIR / "ECAPA" / "emotion_model"
ECAPA_CHECKPOINT = ECAPA_MODEL_DIR / "best_ecapa_model.pth"
ECAPA_METADATA = ECAPA_MODEL_DIR / "metadata.json"

DFAT_MODEL_DIR = ROOT_DIR / "DFAT_Hybrid_Fusion" / "dualstream_model"
DFAT_METADATA = DFAT_MODEL_DIR / "metadata.json"

# Benchmark Outputs
BENCHMARK_RESULTS_PATH = ROOT_DIR / "benchmark_results_gpu.json"
ABLATION_RESULTS_PATH = ROOT_DIR / "ablation_results.json"

# Report Outputs
REPORT_DIR = ROOT_DIR / "report_images"
