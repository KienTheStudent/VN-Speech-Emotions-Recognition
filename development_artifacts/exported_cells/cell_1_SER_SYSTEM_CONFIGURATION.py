# ==================== SER SYSTEM CONFIGURATION ====================
# MODE has 2 options:
# - "demo": Run super fast, automatically load saved benchmark results (benchmark_results_gpu.json)
#           and trained models/checkpoints (best_ecapa_model.pth, dualstream_model/*.pkl).
#           Evaluate (Inference) on the leak-free Test set and plot charts/confusion matrices immediately.
# - "retrain": Retrain all models from scratch (Classical ML, ECAPA-TDNN, DFAT Hybrid Fusion)
#              on the full dataset (5,280 samples) using the shared split_manifest.json.
MODE = "demo" # Change to "retrain" to retrain from scratch

import warnings
warnings.filterwarnings('ignore')
