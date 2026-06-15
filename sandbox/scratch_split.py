from datasets import load_dataset
import pandas as pd

dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
df = dataset.to_pandas()

target_train = int(0.8 * len(df))
target_val = int(0.1 * len(df))
target_test = len(df) - target_train - target_val

train_idx, val_idx, test_idx = [], [], []
train_counts = {e: 0 for e in df["emotion"].unique()}
val_counts = {e: 0 for e in df["emotion"].unique()}
test_counts = {e: 0 for e in df["emotion"].unique()}

global_dist = df["emotion"].value_counts(normalize=True).to_dict()

speaker_stats = []
for spk, group in df.groupby('speaker_id'):
    speaker_stats.append({
        'speaker_id': spk,
        'count': len(group),
        'indices': group.index.tolist(),
        'emotions': group['emotion'].value_counts().to_dict()
    })
speaker_stats.sort(key=lambda x: x['count'], reverse=True)

def compute_cost(current_len, target_len, current_emotions, spk_emotions):
    new_len = current_len + sum(spk_emotions.values())
    if new_len > target_len:
        # Heavily penalize going over target size
        size_penalty = 1000 * (new_len - target_len) / target_len
    else:
        size_penalty = 0
        
    dist_loss = 0
    for e, target_prop in global_dist.items():
        new_prop = (current_emotions.get(e, 0) + spk_emotions.get(e, 0)) / new_len
        dist_loss += abs(new_prop - target_prop)
    return dist_loss + size_penalty

for spk in speaker_stats:
    spk_count = spk['count']
    costs = []
    
    costs.append(('train', compute_cost(len(train_idx), target_train, train_counts, spk['emotions'])))
    costs.append(('val', compute_cost(len(val_idx), target_val, val_counts, spk['emotions'])))
    costs.append(('test', compute_cost(len(test_idx), target_test, test_counts, spk['emotions'])))
    
    best_split = min(costs, key=lambda x: x[1])[0]
    
    if best_split == 'train':
        train_idx.extend(spk['indices'])
        for e, c in spk['emotions'].items(): train_counts[e] = train_counts.get(e, 0) + c
    elif best_split == 'val':
        val_idx.extend(spk['indices'])
        for e, c in spk['emotions'].items(): val_counts[e] = val_counts.get(e, 0) + c
    else:
        test_idx.extend(spk['indices'])
        for e, c in spk['emotions'].items(): test_counts[e] = test_counts.get(e, 0) + c

print(f"Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
print(f"Val: {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
print(f"Test: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

print("\nTrain dist:", {k: f"{v/len(train_idx):.2f}" for k, v in train_counts.items()})
print("Val dist:  ", {k: f"{v/len(val_idx):.2f}" for k, v in val_counts.items()})
print("Test dist: ", {k: f"{v/len(test_idx):.2f}" for k, v in test_counts.items()})
print("Global:    ", {k: f"{v:.2f}" for k, v in global_dist.items()})
