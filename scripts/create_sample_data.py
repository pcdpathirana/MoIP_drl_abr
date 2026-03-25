from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abr_rl.data.synthetic import (
    RESEARCH_SCENARIOS,
    bursty_trace,
    generate_research_trace,
    mixed_trace,
    mobile_trace,
    stable_trace,
    step_trace,
    write_trace_csv,
)
from abr_rl.data.trace_loader import load_trace_directory, save_trace_catalog



def build_quick_demo(root: Path) -> None:
    train_dir = root / "synthetic_train"
    test_dir = root / "synthetic_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    length = 180
    train_traces = {
        "train_stable_1.csv": stable_trace(length, 6.0, 11),
        "train_stable_2.csv": stable_trace(length, 4.5, 13),
        "train_bursty_1.csv": bursty_trace(length, 5.8, 1.4, 17),
        "train_bursty_2.csv": bursty_trace(length, 3.8, 1.1, 19),
        "train_mobile_1.csv": mobile_trace(length, 23),
        "train_mixed_1.csv": mixed_trace(length, 29),
        "train_step_up.csv": step_trace(length, 1.8, 7.2, 90),
        "train_step_down.csv": step_trace(length, 7.0, 1.4, 95),
    }
    test_traces = {
        "test_stable.csv": stable_trace(length, 5.2, 31),
        "test_mobile.csv": mobile_trace(length, 37),
        "test_mixed.csv": mixed_trace(length, 41),
        "test_step.csv": step_trace(length, 2.2, 6.5, 70),
    }
    for name, values in train_traces.items():
        write_trace_csv(train_dir / name, values)
    for name, values in test_traces.items():
        write_trace_csv(test_dir / name, values)



def build_research_dataset(root: Path, length: int = 320) -> None:
    counts = {"train": 72, "val": 16, "test": 24}
    research_root = root / "research"
    for split, total_count in counts.items():
        split_dir = research_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        per_scenario = max(1, total_count // len(RESEARCH_SCENARIOS))
        extra = total_count - per_scenario * len(RESEARCH_SCENARIOS)
        seed_base = {"train": 1000, "val": 5000, "test": 9000}[split]
        item_idx = 0
        for scenario_idx, scenario in enumerate(RESEARCH_SCENARIOS):
            scenario_count = per_scenario + (1 if scenario_idx < extra else 0)
            for local_idx in range(scenario_count):
                seed = seed_base + scenario_idx * 100 + local_idx
                values = generate_research_trace(scenario=scenario, length=length, seed=seed)
                name = f"{scenario}_{local_idx + 1:02d}.csv"
                write_trace_csv(split_dir / name, values)
                item_idx += 1

        traces = load_trace_directory(split_dir)
        save_trace_catalog(traces, research_root / f"{split}_catalog.csv")

    readme = research_root / "README.md"
    readme.write_text(
        """# Research Trace Bundle

"
        "This folder contains a richer synthetic benchmark aligned to the dissertation scope:
"
        "- broadband-like stable sessions
"
        "- bursty broadband
"
        "- hotel / Wi-Fi contention
"
        "- conference peak congestion
"
        "- commuter mobile paths
"
        "- handoff / dropout events
"
        "- home Wi-Fi interference
"
        "- mixed out-of-distribution sessions

"
        "Each trace is sampled at 1-second granularity and stored as `bandwidth_mbps`.
"
        "Train/validation/test catalogs are generated automatically for the professional dashboard.
"
        """,
        encoding="utf-8",
    )



def main() -> None:
    root = Path("data/traces")
    root.mkdir(parents=True, exist_ok=True)
    build_quick_demo(root)
    build_research_dataset(root)
    print(f"Trace datasets prepared under {root.resolve()}")


if __name__ == "__main__":
    main()
