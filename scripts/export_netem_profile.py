from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abr_rl.data.trace_loader import load_trace_file
from abr_rl.emulation.netem import NetEmProfile, write_tc_netem_script



def main() -> None:
    parser = argparse.ArgumentParser(description="Export a tc/netem replay shell script from a bandwidth trace.")
    parser.add_argument("trace_file", help="Input trace CSV/TXT/JSON")
    parser.add_argument("--output", default="results/netem/replay_trace.sh")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--delay-ms", type=int, default=40)
    parser.add_argument("--jitter-ms", type=int, default=6)
    parser.add_argument("--loss-pct", type=float, default=0.2)
    args = parser.parse_args()

    trace = load_trace_file(args.trace_file)
    output = write_tc_netem_script(
        output_path=args.output,
        trace_mbps=trace,
        profile=NetEmProfile(interface=args.interface, delay_ms=args.delay_ms, jitter_ms=args.jitter_ms, loss_pct=args.loss_pct),
        trace_name=Path(args.trace_file).stem,
    )
    print(f"tc/netem replay script written to {output}")


if __name__ == "__main__":
    main()
