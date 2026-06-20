#!/usr/bin/env python3
"""
Generate publication-quality architecture diagrams for the SER report.
Outputs PNG files to report_images/.

Usage:
    conda run -n ColabVENV python scripts/generate_diagrams.py
"""

import graphviz
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "report_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Shared style ────────────────────────────────────────────────────────────

GRAPH_ATTR = {
    "rankdir": "LR",
    "bgcolor": "white",
    "fontname": "Helvetica",
    "fontsize": "11",
    "dpi": "300",
    "margin": "0.2",
    "nodesep": "0.35",
    "ranksep": "0.6",
}

INPUT_STYLE  = {"shape": "box",       "style": "rounded,filled", "fillcolor": "#E8F5E9", "fontname": "Helvetica", "fontsize": "10"}
PROC_STYLE   = {"shape": "box",       "style": "rounded,filled", "fillcolor": "#E3F2FD", "fontname": "Helvetica", "fontsize": "10"}
MODEL_STYLE  = {"shape": "box",       "style": "rounded,filled", "fillcolor": "#FFF3E0", "fontname": "Helvetica", "fontsize": "10"}
OUTPUT_STYLE = {"shape": "box",       "style": "rounded,filled", "fillcolor": "#FCE4EC", "fontname": "Helvetica", "fontsize": "10"}
EDGE_STYLE   = {"color": "#455A64",   "arrowsize": "0.7", "penwidth": "1.2"}
FUSION_STYLE = {"shape": "diamond",   "style": "filled",        "fillcolor": "#F3E5F5", "fontname": "Helvetica", "fontsize": "9"}

# ─── 1. MFCC + RandomForest ─────────────────────────────────────────────────

def draw_mfcc_rf():
    g = graphviz.Digraph("mfcc_rf", format="png", graph_attr=GRAPH_ATTR)
    g.attr(label="Pipeline A: MFCC + Random Forest", labelloc="t", fontsize="13", fontname="Helvetica Bold")

    g.node("wav",     "Raw Waveform\n(16 kHz mono)",        **INPUT_STYLE)
    g.node("mfcc",    "Librosa MFCC\n(40 coefficients)",    **PROC_STYLE)
    g.node("pool",    "Temporal\nMean-Pooling",             **PROC_STYLE)
    g.node("feat",    "40-d Feature\nVector",               **PROC_STYLE)
    g.node("rf",      "Random Forest\n(300 estimators)",    **MODEL_STYLE)
    g.node("out",     "Emotion\nPrediction",                **OUTPUT_STYLE)

    for src, dst in [("wav","mfcc"), ("mfcc","pool"), ("pool","feat"), ("feat","rf"), ("rf","out")]:
        g.edge(src, dst, **EDGE_STYLE)

    path = str(OUTPUT_DIR / "arch_mfcc_rf")
    g.render(path, cleanup=True)
    print(f"  ✓ {path}.png")


# ─── 2. Simplified ECAPA-TDNN ────────────────────────────────────────────────

def draw_ecapa():
    g = graphviz.Digraph("ecapa", format="png", graph_attr=GRAPH_ATTR)
    g.attr(label="Pipeline B: Simplified ECAPA-TDNN", labelloc="t", fontsize="13", fontname="Helvetica Bold")

    g.node("wav",     "Raw Waveform\n(16 kHz mono)",         **INPUT_STYLE)
    g.node("mel",     "Log-Mel\nSpectrogram\n(80 bins)",     **PROC_STYLE)
    g.node("conv",    "1-D Conv\nFrontend\n(256 ch)",        **MODEL_STYLE)
    g.node("tdnn",    "3x Dilated\nTDNN Blocks\n(d=1,2,3)",  **MODEL_STYLE)
    g.node("se",      "Squeeze-and-\nExcitation",            **MODEL_STYLE)
    g.node("cat",     "Multi-Layer\nAggregation\n(cat + 1x1 Conv)", **MODEL_STYLE)
    g.node("stat",    "Attentive\nStat Pooling\n(mean+std)", **PROC_STYLE)
    g.node("emb",     "192-d\nEmbedding",                   **PROC_STYLE)
    g.node("cls",     "Dense\nClassifier",                   **MODEL_STYLE)
    g.node("out",     "Emotion\nPrediction",                 **OUTPUT_STYLE)

    for src, dst in [("wav","mel"),("mel","conv"),("conv","tdnn"),("tdnn","se"),
                     ("se","cat"),("cat","stat"),("stat","emb"),("emb","cls"),("cls","out")]:
        g.edge(src, dst, **EDGE_STYLE)

    path = str(OUTPUT_DIR / "arch_ecapa")
    g.render(path, cleanup=True)
    print(f"  ✓ {path}.png")


# ─── 3. DFAT Dual-Stream Fusion ──────────────────────────────────────────────

def draw_dfat():
    g = graphviz.Digraph("dfat", format="png", graph_attr={
        **GRAPH_ATTR,
        "rankdir": "LR",
        "ranksep": "0.5",
        "nodesep": "0.25",
    })
    g.attr(label="Pipeline C: DFAT Dual-Stream ASR-Assisted Fusion", labelloc="t", fontsize="13", fontname="Helvetica Bold")

    # ── Input
    g.node("wav", "Raw Waveform\n(16 kHz mono)", **INPUT_STYLE)

    # ── Acoustic stream (top)
    with g.subgraph(name="cluster_acoustic") as s:
        s.attr(label="Acoustic Stream", style="dashed", color="#1565C0", fontname="Helvetica", fontsize="10")
        s.node("wavlm",  "WavLM\nbase-plus\n(frozen)", **MODEL_STYLE)
        s.node("a_emb",  "768-d\nAcoustic\nEmbedding", **PROC_STYLE)

    # ── Linguistic stream (bottom)
    with g.subgraph(name="cluster_linguistic") as s:
        s.attr(label="Linguistic Stream (ASR-derived)", style="dashed", color="#C62828", fontname="Helvetica", fontsize="10")
        s.node("whisper","Whisper\nsmall\n(ASR)", **MODEL_STYLE)
        s.node("txt",    "Vietnamese\nTranscript",  **PROC_STYLE)
        s.node("seg",    "Underthesea\nSegmentation", **PROC_STYLE)
        s.node("bert",   "PhoBERT\nbase-v2\n(frozen)", **MODEL_STYLE)
        s.node("l_emb",  "768-d\nLinguistic\nEmbedding", **PROC_STYLE)

    # ── Fusion
    g.node("concat",  "Early Fusion\n(Concat 1536-d)", **FUSION_STYLE)

    # ── Ensemble classifiers
    with g.subgraph(name="cluster_ensemble") as s:
        s.attr(label="Late Ensemble", style="dashed", color="#6A1B9A", fontname="Helvetica", fontsize="10")
        s.node("lr",  "Logistic\nRegression",   **MODEL_STYLE)
        s.node("rf",  "Random\nForest",         **MODEL_STYLE)
        s.node("xgb", "XGBoost",                **MODEL_STYLE)

    g.node("ens",  "Weighted\nEnsemble", **FUSION_STYLE)
    g.node("out",  "Emotion\nPrediction",       **OUTPUT_STYLE)

    # ── Edges
    g.edge("wav", "wavlm",  **EDGE_STYLE)
    g.edge("wavlm", "a_emb", **EDGE_STYLE)
    g.edge("wav", "whisper", **EDGE_STYLE)
    g.edge("whisper", "txt", **EDGE_STYLE)
    g.edge("txt", "seg",    **EDGE_STYLE)
    g.edge("seg", "bert",   **EDGE_STYLE)
    g.edge("bert", "l_emb", **EDGE_STYLE)
    g.edge("a_emb", "concat", **EDGE_STYLE)
    g.edge("l_emb", "concat", **EDGE_STYLE)
    g.edge("concat", "lr",  **EDGE_STYLE)
    g.edge("concat", "rf",  **EDGE_STYLE)
    g.edge("concat", "xgb", **EDGE_STYLE)
    g.edge("lr",  "ens",    **EDGE_STYLE)
    g.edge("rf",  "ens",    **EDGE_STYLE)
    g.edge("xgb", "ens",    **EDGE_STYLE)
    g.edge("ens", "out",    **EDGE_STYLE)

    path = str(OUTPUT_DIR / "arch_dfat")
    g.render(path, cleanup=True)
    print(f"  ✓ {path}.png")


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    draw_mfcc_rf()
    draw_ecapa()
    draw_dfat()
    print("Done.")
