from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abr_rl.data.synthetic import RESEARCH_SCENARIOS, generate_research_trace, write_trace_csv
from abr_rl.data.trace_loader import load_trace_directory, save_trace_catalog



def build_research_dataset(output_root: Path, length: int, train_count: int, val_count: int, test_count: int) -> None:
    counts = {"train": train_count, "val": val_count, "test": test_count}
    for split, total_count in counts.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        per_scenario = max(1, total_count // len(RESEARCH_SCENARIOS))
        extra = total_count - per_scenario * len(RESEARCH_SCENARIOS)
        seed_base = {"train": 1000, "val": 5000, "test": 9000}[split]
        for scenario_idx, scenario in enumerate(RESEARCH_SCENARIOS):
            scenario_count = per_scenario + (1 if scenario_idx < extra else 0)
            for local_idx in range(scenario_count):
                seed = seed_base + scenario_idx * 100 + local_idx
                values = generate_research_trace(scenario=scenario, length=length, seed=seed)
                write_trace_csv(split_dir / f"{scenario}_{local_idx + 1:02d}.csv", values)
        traces = load_trace_directory(split_dir)
        save_trace_catalog(traces, output_root / f"{split}_catalog.csv")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build the richer synthetic dataset used by the professional dashboard demo.")
    parser.add_argument("--output-root", default="data/traces/research")
    parser.add_argument("--length", type=int, default=320)
    parser.add_argument("--train-count", type=int, default=72)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=24)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    build_research_dataset(output_root=output_root, length=args.length, train_count=args.train_count, val_count=args.val_count, test_count=args.test_count)
    print(f"Research dataset written to {output_root.resolve()}")


if __name__ == "__main__":
    main()
