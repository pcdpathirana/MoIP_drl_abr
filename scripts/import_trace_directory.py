from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from abr_rl.data.trace_loader import list_trace_files, load_trace_file



def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize an arbitrary trace directory into project-ready bandwidth_mbps CSV files.")
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", default="data/public_sources/converted/imported")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in list_trace_files(input_dir, recursive=args.recursive):
        values = load_trace_file(path)
        out = output_dir / f"{path.stem}.csv"
        pd.DataFrame({"bandwidth_mbps": values}).to_csv(out, index=False)
        count += 1
    print(f"Imported {count} traces into {output_dir.resolve()}")


if __name__ == "__main__":
    main()
