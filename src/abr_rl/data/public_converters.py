from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def convert_norway_log_file(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_csv(input_path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns in Norway log: {input_path}")
    # Columns from the dataset description:
    # 0 = unix timestamp (seconds), 4 = bytes received, 5 = elapsed milliseconds.
    unix_ts = df.iloc[:, 0].astype(int)
    bytes_received = df.iloc[:, 4].astype(float)
    elapsed_ms = df.iloc[:, 5].astype(float).clip(lower=1.0)
    mbps = (bytes_received * 8.0) / (elapsed_ms * 1000.0)
    converted = pd.DataFrame({"unix_ts": unix_ts, "bandwidth_mbps": mbps})
    per_second = converted.groupby("unix_ts", as_index=False)["bandwidth_mbps"].mean()
    return per_second



def save_project_trace(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "bandwidth_mbps" not in df.columns:
        raise ValueError("Expected a 'bandwidth_mbps' column.")
    df[["bandwidth_mbps"]].to_csv(output_path, index=False)
    return output_path
