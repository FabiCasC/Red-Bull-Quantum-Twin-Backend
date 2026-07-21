from datasets import load_dataset
import pandas as pd

ds = load_dataset('renumics/f1_dataset', split='train')
df = ds.to_pandas()

# Keep only scalar columns (drop Sequence1D and Embedding columns)
scalar_cols = [c for c in df.columns if df[c].dtype != object or df[c].apply(lambda x: not isinstance(x, list)).all()]
df_clean = df[scalar_cols]
df_clean.to_csv('data/f1_telemetry.csv', index=False)
print(f"Exported {len(df_clean)} rows, columns: {list(df_clean.columns)}")
