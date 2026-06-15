from datasets import load_dataset
import pandas as pd
dataset = load_dataset("hustep-lab/ViSEC", split="train", trust_remote_code=True)
df = dataset.to_pandas()
print(f"Total: {len(df)}")
counts = df['speaker_id'].value_counts()
print(counts.head(10))
print(f"Total speakers: {len(counts)}")
