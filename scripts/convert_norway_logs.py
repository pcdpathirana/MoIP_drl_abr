from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abr_rl.data.public_converters import convert_norway_log_file, save_project_trace



def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Norway HSDPA raw logs into project-ready bandwidth_mbps CSV traces.")
    parser.add_argument("input_dir", help="Directory containing raw Norway log files")
    parser.add_argument("--output-dir", default="data/public_sources/converted/norway")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            df = convert_norway_log_file(path)
        except Exception:
            continue
        if df.empty:
            continue
        save_project_trace(df, output_dir / f"{path.stem}.csv")
        count += 1

    print(f"Converted {count} Norway logs into {output_dir.resolve()}")


if __name__ == "__main__":
    main()
