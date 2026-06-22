#!/bin/bash
set -e

echo "Starting downstream tasks..."

conda activate ColabVENV

echo "0. Pre-transcribing all audio using batched ASR..."
python scripts/batch_transcribe.py --asr_model vinai/PhoWhisper-large --batch_size 4

echo "0.5. Training DFAT model..."
python DFAT_Hybrid_Fusion/train_dualstream.py --asr_model vinai/PhoWhisper-large

echo "1. Running Ablation Study..."
python DFAT_Hybrid_Fusion/ablation_study.py --asr_model vinai/PhoWhisper-large

echo "2. Regenerating Benchmark Results..."
python benchmark_methods_gpu.py

echo "3. Extracting Inference Metadata..."
python scripts/extract_inference_metadata.py --model_dir DFAT_Hybrid_Fusion/dualstream_model

echo "4. Generating Report Figures..."
python generate_report_figures.py

echo "5. Syncing Notebook..."
python internal/sync_notebook.py

echo "6. Running Validation..."
python validate_release.py

echo "7. Compiling LaTeX Report..."
pdflatex -interaction=nonstopmode Report_SER.tex
pdflatex -interaction=nonstopmode Report_SER.tex

echo "All downstream tasks completed successfully!"
