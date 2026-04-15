# Network Emulation Guide

## Goal
This project supports the emulator layer in your architecture by exporting bandwidth traces as executable `tc/netem` replay scripts.

## Export a replay script
```bash
python scripts/export_netem_profile.py data/traces/research/test/hotel_wifi_01.csv --output results/netem/hotel_wifi.sh
```

## Run it on Ubuntu
```bash
bash results/netem/hotel_wifi.sh eth0
```

## What the generated script does
- creates an `htb` root qdisc
- attaches a `netem` child qdisc
- updates the interface bandwidth once per second according to the trace
- reverts the qdisc when the script exits

## Optional parameters
```bash
python scripts/export_netem_profile.py <trace.csv>   --interface eth0   --delay-ms 40   --jitter-ms 6   --loss-pct 0.2
```

## Mininet-WiFi note
This repo focuses on the lighter-weight `tc/netem` export path because it is easier to run inside Ubuntu VMs. If you want Mininet-WiFi validation, use the same project traces as profile inputs and attach them to wireless link update logic.


## Live Simulation (Viva) integration

The dashboard includes a **Live Simulation (Viva)** page that can optionally apply the selected bandwidth/delay/loss settings using `tc/netem`.
- Install: `sudo apt-get install -y iproute2`
- If you enable the checkbox inside the dashboard, it will call `sudo tc qdisc replace ...`.
- For viva demos, run Streamlit with sudo (`sudo -E streamlit run dashboard/app.py`) or configure passwordless `tc` for the demo VM user.
