#!/usr/bin/env python3
"""
Generate a fixed Train/Val/Test split manifest for ViSEC dataset.

This script creates a single source of truth for data splitting that all
training and evaluation scripts must read from, ensuring that every model
is compared on the exact same test set.

Split ratio: approx 80% Train / 10% Val / 10% Test (greedy speaker-independent split).
Random seed: 42 (used in other modules, but this split uses deterministic greedy bin-packing).

Output: split_manifest.json in the project root directory.
"""

import json
import hashlib
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path(__file__).parent.parent / "split_manifest.json"


def main():
    print("=" * 60)
    print("GENERATING FIXED SPLIT MANIFEST FOR ViSEC")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load ViSEC dataset
    # ------------------------------------------------------------------
    print("\nLoading ViSEC dataset from HuggingFace...")
    dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
    df = dataset.to_pandas()

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nEmotion distribution:")
    print(df["emotion"].value_counts().to_string())
    print(f"\nSpeaker count: {df['speaker_id'].nunique()}")

    # ------------------------------------------------------------------
    # 2. Greedy Speaker-Independent Split (80/10/10)
    # ------------------------------------------------------------------
    print("\nApplying greedy speaker-independent split...")
    target_train = int(0.8 * len(df))
    target_val = int(0.1 * len(df))
    target_test = len(df) - target_train - target_val

    train_idx, val_idx, test_idx = [], [], []
    train_counts = {e: 0 for e in df["emotion"].unique()}
    val_counts = {e: 0 for e in df["emotion"].unique()}
    test_counts = {e: 0 for e in df["emotion"].unique()}

    global_dist = df["emotion"].value_counts(normalize=True).to_dict()

    speaker_stats = []
    for spk, group in df.groupby("speaker_id"):
        speaker_stats.append(
            {
                "speaker_id": spk,
                "count": len(group),
                "indices": group.index.tolist(),
                "emotions": group["emotion"].value_counts().to_dict(),
            }
        )
    # Sort speakers by count descending (Speaker 0 is huge and goes first)
    speaker_stats.sort(key=lambda x: x["count"], reverse=True)

    def compute_cost(current_len, target_len, current_emotions, spk_emotions):
        new_len = current_len + sum(spk_emotions.values())
        if new_len > target_len:
            # Heavily penalize going over target size
            size_penalty = 1000 * (new_len - target_len) / target_len
        else:
            size_penalty = 0

        dist_loss = 0
        for e, target_prop in global_dist.items():
            new_prop = (
                (current_emotions.get(e, 0) + spk_emotions.get(e, 0)) / new_len
                if new_len > 0
                else 0
            )
            dist_loss += abs(new_prop - target_prop)
        return dist_loss + size_penalty

    for spk in speaker_stats:
        costs = [
            (
                "train",
                compute_cost(
                    len(train_idx), target_train, train_counts, spk["emotions"]
                ),
            ),
            (
                "val",
                compute_cost(len(val_idx), target_val, val_counts, spk["emotions"]),
            ),
            (
                "test",
                compute_cost(len(test_idx), target_test, test_counts, spk["emotions"]),
            ),
        ]

        best_split = min(costs, key=lambda x: x[1])[0]

        if best_split == "train":
            train_idx.extend(spk["indices"])
            for e, c in spk["emotions"].items():
                train_counts[e] = train_counts.get(e, 0) + c
        elif best_split == "val":
            val_idx.extend(spk["indices"])
            for e, c in spk["emotions"].items():
                val_counts[e] = val_counts.get(e, 0) + c
        else:
            test_idx.extend(spk["indices"])
            for e, c in spk["emotions"].items():
                test_counts[e] = test_counts.get(e, 0) + c

    # ------------------------------------------------------------------
    # 2b. Local Search Refinement
    # ------------------------------------------------------------------
    print("\nApplying local search refinement to improve class balance...")
    
    def calculate_absolute_difference(split_counts):
        total = sum(split_counts.values())
        if total == 0: return float('inf')
        diff = 0
        for e, target_prop in global_dist.items():
            prop = split_counts.get(e, 0) / total
            if prop > 0:
                # We use simple absolute difference here for symmetry and stability
                diff += abs(prop - target_prop)
            else:
                diff += target_prop
        return diff

    def split_score():
        return (
            calculate_absolute_difference(train_counts) +
            calculate_absolute_difference(val_counts) +
            calculate_absolute_difference(test_counts)
        )

    current_score = split_score()
    improved = True
    max_iters = 50
    iters = 0
    
    # We map speaker stats for quick lookup
    spk_map = {spk["speaker_id"]: spk for spk in speaker_stats}
    
    # Track which split a speaker belongs to
    spk_to_split = {}
    for spk_id in set(df.iloc[train_idx]["speaker_id"]): spk_to_split[spk_id] = "train"
    for spk_id in set(df.iloc[val_idx]["speaker_id"]): spk_to_split[spk_id] = "val"
    for spk_id in set(df.iloc[test_idx]["speaker_id"]): spk_to_split[spk_id] = "test"
    
    # Only try to swap smaller speakers to avoid blowing up the sizes
    swap_candidates = [spk for spk in speaker_stats if spk["count"] < 50]
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(swap_candidates)):
            for j in range(i + 1, len(swap_candidates)):
                spk1 = swap_candidates[i]
                spk2 = swap_candidates[j]
                split1 = spk_to_split[spk1["speaker_id"]]
                split2 = spk_to_split[spk2["speaker_id"]]
                
                if split1 == split2:
                    continue
                    
                # Try swap
                # Subtract spk1 from split1, spk2 from split2
                # Add spk2 to split1, spk1 to split2
                c1 = train_counts if split1 == "train" else (val_counts if split1 == "val" else test_counts)
                c2 = train_counts if split2 == "train" else (val_counts if split2 == "val" else test_counts)
                
                for e, c in spk1["emotions"].items():
                    c1[e] -= c
                    c2[e] += c
                for e, c in spk2["emotions"].items():
                    c2[e] -= c
                    c1[e] += c
                    
                new_score = split_score()
                
                if new_score < current_score - 0.001:  # Must improve by margin
                    current_score = new_score
                    improved = True
                    spk_to_split[spk1["speaker_id"]] = split2
                    spk_to_split[spk2["speaker_id"]] = split1
                else:
                    # Revert
                    for e, c in spk1["emotions"].items():
                        c1[e] += c
                        c2[e] -= c
                    for e, c in spk2["emotions"].items():
                        c2[e] += c
                        c1[e] -= c

    # Rebuild indices based on optimized spk_to_split
    train_idx, val_idx, test_idx = [], [], []
    for spk in speaker_stats:
        split = spk_to_split[spk["speaker_id"]]
        if split == "train":
            train_idx.extend(spk["indices"])
        elif split == "val":
            val_idx.extend(spk["indices"])
        else:
            test_idx.extend(spk["indices"])

    print(f"\nSplit sizes after refinement ({iters} iterations):")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Verify no overlap
    # ------------------------------------------------------------------
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    assert len(train_set) + len(val_set) + len(test_set) == len(
        df
    ), "Split sizes don't add up!"

    train_speakers = set(df.iloc[train_idx]["speaker_id"])
    val_speakers = set(df.iloc[val_idx]["speaker_id"])
    test_speakers = set(df.iloc[test_idx]["speaker_id"])
    assert (
        len(train_speakers & val_speakers) == 0
    ), "Train/Val speaker overlap detected!"
    assert (
        len(train_speakers & test_speakers) == 0
    ), "Train/Test speaker overlap detected!"
    assert len(val_speakers & test_speakers) == 0, "Val/Test speaker overlap detected!"

    print("\n✓ No sample or speaker overlap between splits. Integrity verified.")

    # ------------------------------------------------------------------
    # 4. Verify stratification
    # ------------------------------------------------------------------
    print("\nPer-split emotion distribution:")
    for name, idxs in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        dist = df.iloc[idxs]["emotion"].value_counts()
        total = len(idxs)
        print(f"  {name}:")
        for emotion, count in dist.items():
            print(f"    {emotion}: {count} ({count/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Save manifest
    # ------------------------------------------------------------------
    manifest = {
        "description": (
            "Fixed Train/Val/Test split for ViSEC dataset. "
            "All training and evaluation scripts MUST read from this file. "
            "Generated by generate_splits.py."
        ),
        "dataset": "hustep-lab/ViSEC",
        "total_samples": len(df),
        "split_ratio": "80/10/10 (approx)",
        "stratify_by": "speaker_independent",
        "train_indices": sorted(train_idx),
        "val_indices": sorted(val_idx),
        "test_indices": sorted(test_idx),
    }

    # Generate checksum
    manifest_string = json.dumps(manifest, sort_keys=True)
    manifest["checksum"] = hashlib.sha256(manifest_string.encode('utf-8')).hexdigest()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Manifest saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
