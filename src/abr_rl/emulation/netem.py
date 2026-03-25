from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NetEmProfile:
    interface: str = "eth0"
    delay_ms: int = 40
    jitter_ms: int = 6
    loss_pct: float = 0.2



def render_tc_netem_script(trace_mbps: np.ndarray, profile: NetEmProfile, trace_name: str = "abr_trace") -> str:
    trace_mbps = np.asarray(trace_mbps, dtype=np.float32)
    rates = np.clip(trace_mbps, 0.05, None)
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'TRACE_NAME="{trace_name}"',
        f'IFACE="${{1:-{profile.interface}}}"',
        f'DELAY_MS="{profile.delay_ms}"',
        f'JITTER_MS="{profile.jitter_ms}"',
        f'LOSS_PCT="{profile.loss_pct}"',
        'echo "Applying tc/netem replay for ${TRACE_NAME} on ${IFACE}"',
        'sudo tc qdisc del dev "${IFACE}" root 2>/dev/null || true',
        'sudo tc qdisc add dev "${IFACE}" root handle 1: htb default 10',
        f'sudo tc class add dev "${{IFACE}}" parent 1: classid 1:10 htb rate {rates[0]:.2f}mbit ceil {rates[0]:.2f}mbit',
        'sudo tc qdisc add dev "${IFACE}" parent 1:10 handle 10: netem delay ${DELAY_MS}ms ${JITTER_MS}ms loss ${LOSS_PCT}%',
        '',
        'cleanup() {',
        '  sudo tc qdisc del dev "${IFACE}" root 2>/dev/null || true',
        '}',
        'trap cleanup EXIT',
        '',
        'while true; do',
    ]
    body = []
    for value in rates:
        body.append(f'  sudo tc class change dev "${{IFACE}}" parent 1: classid 1:10 htb rate {value:.2f}mbit ceil {value:.2f}mbit')
        body.append('  sleep 1')
    footer = ['done']
    return "\n".join(header + body + footer) + "\n"



def write_tc_netem_script(output_path: str | Path, trace_mbps: np.ndarray, profile: NetEmProfile, trace_name: str = "abr_trace") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_tc_netem_script(trace_mbps, profile, trace_name=trace_name), encoding="utf-8")
    output_path.chmod(0o755)
    return output_path
