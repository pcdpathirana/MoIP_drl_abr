from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
import numpy as np
import time
import subprocess
import os

import torch


try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency at runtime
    px = None
    go = None


RESULTS_ROOT = Path("results")
DEFAULT_RUN_DIR = RESULTS_ROOT / "professional_demo"


st.set_page_config(
    page_title="DRL ABR Professional Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.dashboard-card {
    background: linear-gradient(180deg, #111827 0%, #0B0F19 100%);
    border: 1px solid #1F2937;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.dashboard-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    opacity: 0.7;
    transition: opacity 0.3s ease, background 0.3s ease;
}
.dashboard-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    border-color: #374151;
    z-index: 10;
}
.dashboard-card:hover::before {
    opacity: 1;
    background: linear-gradient(90deg, #06B6D4, #3B82F6);
}
.dashboard-card h4 {
    margin-bottom: 0.5rem;
    color: #F9FAFB;
    font-weight: 500;
    font-size: 1.15rem;
    letter-spacing: 0.02em;
}
.mini-note {
    color: #9CA3AF;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-top: 0.5rem;
}
.metric-label {
    color: #9CA3AF;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(to right, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin: 0.25rem 0;
}
.arch-box {
    background: linear-gradient(180deg, #111827 0%, #0B0F19 100%);
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 200px;
    position: relative;
    transition: all 0.3s ease;
    overflow: hidden;
}
.arch-box h4 {
    color: #F9FAFB;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 0.75rem;
}
.arch-box .mini-note {
    color: #D1D5DB;
    font-size: 0.9rem;
    line-height: 1.6;
}
.arch-box.step-1::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: #3B82F6; }
.arch-box.step-2::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: #8B5CF6; }
.arch-box.step-3::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: #EC4899; }
.arch-box.step-4::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: #10B981; }

.arch-box::after {
    content: '';
    position: absolute;
    bottom: -50px;
    right: -50px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    transition: all 0.5s ease;
    opacity: 0.15;
}
.arch-box.step-1::after { background: radial-gradient(circle, #3B82F6 0%, transparent 70%); }
.arch-box.step-2::after { background: radial-gradient(circle, #8B5CF6 0%, transparent 70%); }
.arch-box.step-3::after { background: radial-gradient(circle, #EC4899 0%, transparent 70%); }
.arch-box.step-4::after { background: radial-gradient(circle, #10B981 0%, transparent 70%); }

.arch-box:hover {
    border-color: #374151;
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
}
.arch-box:hover::after {
    transform: scale(2.5);
    opacity: 0.25;
}
/* Style Streamlit Tabs */
div[data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
}
div[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem;
    font-weight: 500;
    color: #9CA3AF !important;
    background-color: transparent !important;
    border: none !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
div[aria-selected="true"] {
    color: #F9FAFB !important;
    border-bottom: 2px solid #3B82F6 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def find_run_directories(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    runs = []
    for candidate in root.iterdir():
        if candidate.is_dir() and ((candidate / "training_history.csv").exists() or (candidate / "evaluation_summary.csv").exists()):
            runs.append(candidate)
    return sorted(runs)


@st.cache_data(show_spinner=False)
def list_session_files(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    sessions_root = run_dir / "sessions"
    if not sessions_root.exists():
        return []
    return sorted(sessions_root.rglob("*.csv"))


@st.cache_data(show_spinner=False)
def infer_algorithms(run_dir: str | Path) -> list[str]:
    files = list_session_files(run_dir)
    algos = sorted({path.parent.name for path in files})
    return algos


@st.cache_data(show_spinner=False)
def infer_traces(run_dir: str | Path) -> list[str]:
    files = list_session_files(run_dir)
    traces = sorted({path.stem for path in files})
    return traces


@st.cache_data(show_spinner=False)
def load_session(run_dir: str | Path, algorithm_folder: str, trace_name: str) -> pd.DataFrame:
    path = Path(run_dir) / "sessions" / algorithm_folder / f"{trace_name}.csv"
    return load_csv(path)


@st.cache_data(show_spinner=False)
def load_available_trace_catalogs(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    mapping = {}
    for name in ["train", "val", "eval"]:
        df = load_csv(run_dir / f"{name}_trace_catalog.csv")
        if not df.empty:
            mapping[name] = df
    return mapping



def render_metric_card(label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    delta_html = f"<div class='mini-note'>{delta}</div>" if delta else ""
    help_html = f"<div class='mini-note'>{help_text}</div>" if help_text else ""
    
    st.markdown(
f"""<div class="dashboard-card">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
{delta_html}
{help_html}
</div>""",
        unsafe_allow_html=True,
    )



def render_architecture() -> None:
    st.markdown("### Four-layer architecture")
    col1, col2, col3, col4 = st.columns(4)
    layers = [
        ("1. Network emulator", "tc/netem or Mininet-WiFi replay profiles emulate changing bandwidth, delay, jitter, and loss.", "step-1"),
        ("2. Video chunk simulator", "Variable-size segments mimic content complexity and chunk-by-chunk download behavior under adaptive streaming.", "step-2"),
        ("3. DRL agent", "PPO policy consumes throughput history, download times, buffer occupancy, previous bitrate, and next-chunk sizes.", "step-3"),
        ("4. QoE evaluator", "Computes bitrate, rebuffering, smoothness, stall ratio, throughput, and total QoE for benchmarking.", "step-4"),
    ]
    for col, (title, text, step_cls) in zip([col1, col2, col3, col4], layers):
        with col:
            st.markdown(f"<div class='arch-box {step_cls}'><h4>{title}</h4><div class='mini-note'>{text}</div></div>", unsafe_allow_html=True)



def chart_or_table_line(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart yet.")
        return
    if px is not None:
        fig = px.line(df, x=x, y=y, color=color, title=title)
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        base = df.set_index(x)
        if color and color in df.columns:
            st.dataframe(df, use_container_width=True)
        else:
            st.line_chart(base[y])



def chart_or_table_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart yet.")
        return
    if px is not None:
        fig = px.bar(df, x=x, y=y, color=color, title=title, barmode="group")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])



def render_session_explorer(run_dir: Path) -> None:
    st.markdown("### Session explorer")
    algorithms = infer_algorithms(run_dir)
    traces = infer_traces(run_dir)
    if not algorithms or not traces:
        st.info("No per-session logs were found. Run evaluation to populate `results/.../sessions/`.")
        return

    col_a, col_b = st.columns([1, 1])
    with col_a:
        chosen_trace = st.selectbox("Trace", options=traces, index=0)
    with col_b:
        chosen_algorithms = st.multiselect("Algorithms", options=algorithms, default=algorithms[: min(2, len(algorithms))])

    if not chosen_algorithms:
        st.warning("Select at least one algorithm to compare sessions.")
        return

    session_frames = []
    for algo in chosen_algorithms:
        df = load_session(run_dir, algo, chosen_trace)
        if not df.empty:
            session_frames.append(df.assign(algorithm=algo))

    if not session_frames:
        st.info("Selected session logs are empty.")
        return

    session_df = pd.concat(session_frames, ignore_index=True)
    session_df["bitrate_mbps"] = session_df["bitrate_kbps"] / 1000.0

    st.dataframe(session_df, use_container_width=True, height=240)

    left, right = st.columns(2)
    with left:
        if px is not None:
            fig = px.line(
                session_df,
                x="segment_index",
                y="observed_throughput_mbps",
                color="algorithm",
                title="Observed throughput by segment",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(session_df.pivot_table(index="segment_index", columns="algorithm", values="observed_throughput_mbps"))
    with right:
        if px is not None:
            fig = px.line(
                session_df,
                x="segment_index",
                y="bitrate_mbps",
                color="algorithm",
                title="Chosen bitrate by segment",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(session_df.pivot_table(index="segment_index", columns="algorithm", values="bitrate_mbps"))

    left2, right2 = st.columns(2)
    with left2:
        if px is not None:
            fig = px.line(
                session_df,
                x="segment_index",
                y="buffer_after_s",
                color="algorithm",
                title="Playback buffer after each download",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(session_df.pivot_table(index="segment_index", columns="algorithm", values="buffer_after_s"))
    with right2:
        if px is not None:
            fig = px.bar(
                session_df,
                x="segment_index",
                y="rebuffer_s",
                color="algorithm",
                title="Rebuffering time by segment",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(session_df.pivot_table(index="segment_index", columns="algorithm", values="rebuffer_s"))



def render_dataset_panel(run_dir: Path) -> None:
    catalogs = load_available_trace_catalogs(run_dir)
    if not catalogs:
        st.info("Dataset catalogs were not found yet. Train or evaluate once to generate them.")
        return

    split = st.selectbox("Dataset split", options=list(catalogs.keys()), index=0)
    catalog_df = catalogs[split]
    st.dataframe(catalog_df, use_container_width=True, height=260)

    col1, col2 = st.columns(2)
    with col1:
        chart_or_table_bar(
            catalog_df.groupby("scenario", as_index=False)["mean_mbps"].mean().sort_values("mean_mbps", ascending=False),
            x="scenario",
            y="mean_mbps",
            title="Average bandwidth by scenario",
        )
    with col2:
        if px is not None:
            fig = px.scatter(
                catalog_df,
                x="mean_mbps",
                y="p90_mbps",
                color="scenario",
                size="variation_ratio",
                hover_name="trace_name",
                title="Trace difficulty map",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(catalog_df[["trace_name", "scenario", "mean_mbps", "p90_mbps", "variation_ratio"]], use_container_width=True)




def _try_apply_netem(interface: str, rate_mbps: float, delay_ms: int, jitter_ms: int, loss_pct: float) -> tuple[bool, str]:
    """Best-effort tc/netem application. Requires sudo privileges."""
    cmd = [
        "sudo",
        "tc",
        "qdisc",
        "replace",
        "dev",
        interface,
        "root",
        "netem",
        "rate",
        f"{rate_mbps:.2f}mbit",
        "delay",
        f"{int(delay_ms)}ms",
        f"{int(jitter_ms)}ms",
        "loss",
        f"{float(loss_pct):.3f}%",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return False, (proc.stderr.strip() or proc.stdout.strip() or "tc returned non-zero exit code")
        return True, "Applied tc/netem successfully."
    except FileNotFoundError:
        return False, "tc not found. Install: sudo apt-get install -y iproute2"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _safe_load_ppo_policy(model_path: Path, obs_dim: int, action_dim: int, hidden_dim: int = 128):
    """Loads the PPO ActorCritic model if available."""
    try:
        from abr_rl.rl.ppo import load_model
        return load_model(model_path=model_path, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    except Exception:
        return None


def render_live_simulation() -> None:
    """Viva-friendly real-time ABR demo: side-by-side DRL vs baseline."""
    st.markdown("### 🎬 Algorithm Live Simulation")
    st.caption(
        "Use the controls below to **change bandwidth in real-time** and compare the trained DRL policy against a baseline algorithm. "
        "This demo focuses on **live behavior** (buffer, bitrate switches, stalls) rather than long training runs."
    )

    left_controls, right_controls = st.columns([1.2, 1])
    with left_controls:
        baseline_choice = st.selectbox("Baseline algorithm (right player)", ["Rate-Based", "BOLA", "FESTIVE"], index=1)
        rate_mbps = st.slider("Bandwidth (Mbps)", min_value=0.2, max_value=20.0, value=5.0, step=0.1)
        delay_ms = st.slider("Delay (ms)", min_value=0, max_value=200, value=40, step=5)
        loss_pct = st.slider("Packet loss (%)", min_value=0.0, max_value=5.0, value=0.2, step=0.1)
        jitter_ms = st.slider("Jitter (ms)", min_value=0, max_value=50, value=6, step=1)
        segments_to_run = st.slider("Segments to simulate", min_value=10, max_value=120, value=40, step=5)
        playback_speed = st.slider("Playback speed (demo)", min_value=0.0, max_value=0.35, value=0.08, step=0.01)
        st.divider()

        use_netem = st.checkbox("Apply conditions to real interface using tc/netem (requires sudo)", value=False)
        iface = st.text_input("Interface", value="eth0", disabled=not use_netem)
        apply_now = st.button("Apply tc/netem now", disabled=not use_netem)
        if apply_now and use_netem:
            ok, msg = _try_apply_netem(iface, rate_mbps, delay_ms, jitter_ms, loss_pct)
            (st.success if ok else st.error)(msg)
            st.code(" ".join(["sudo tc qdisc show dev", iface]), language="bash")

    # --- Simulation setup (two independent players, same bandwidth conditions) ---
    from abr_rl.metrics.qoe import QoEWeights
    from abr_rl.sim.video_profile import VideoProfile
    from abr_rl.sim.abr_simulator import ABRSimulator
    from abr_rl.baselines.rate_based import RateBasedController
    from abr_rl.baselines.festive import FESTIVEInspiredController
    from abr_rl.baselines.bola import BOLAInspiredController

    profile = VideoProfile.default()
    weights = QoEWeights.default()

    # Constant trace updated from slider (and re-applied per step to support interactive changes).
    trace = np.full(3600, float(rate_mbps), dtype=np.float32)

    sim_drl = ABRSimulator(trace_mbps=trace, profile=profile, reward_weights=weights)
    sim_base = ABRSimulator(trace_mbps=trace, profile=profile, reward_weights=weights)

    # Baseline controller
    chunk_sizes = profile.chunk_sizes_bytes  # shape: [segments, bitrates]
    if baseline_choice == "BOLA":
        controller = BOLAInspiredController(
            bitrate_kbps=list(profile.bitrate_kbps),
            chunk_sizes_bytes=chunk_sizes,
            max_buffer_s=float(profile.max_buffer_s),
        )
        controller_name = "BOLA (baseline)"
    elif baseline_choice == "FESTIVE":
        controller = FESTIVEInspiredController(bitrate_kbps=list(profile.bitrate_kbps))
        controller_name = "FESTIVE (baseline)"
    else:
        controller = RateBasedController(bitrate_kbps=list(profile.bitrate_kbps))
        controller_name = "Rate-Based (baseline)"

    # Load PPO checkpoint if present
    ckpt = Path("results") / "professional_demo" / "ppo_final.pt"
    obs0 = sim_drl.reset()
    obs_dim = int(len(obs0))
    action_dim = int(len(profile.bitrate_kbps))
    ppo_model = _safe_load_ppo_policy(ckpt, obs_dim=obs_dim, action_dim=action_dim)

    # Video asset (optional)
    video_path = Path("data") / "video" / "sample_hotel_stream.mp4"
    has_video = video_path.exists()

    # UI placeholders
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Player A: **Trained DRL (PPO)**")
        if has_video:
            st.video(str(video_path))
        drl_kpis = st.empty()
        drl_chart = st.empty()
    with col_right:
        st.markdown(f"#### Player B: **{controller_name}**")
        if has_video:
            st.video(str(video_path))
        base_kpis = st.empty()
        base_chart = st.empty()

    run = st.button("▶ Start Live Demo", type="primary")
    reset = st.button("↻ Reset", type="secondary")

    if reset:
        sim_drl.reset()
        sim_base.reset()
        st.success("Reset complete. Adjust bandwidth and press Start Live Demo.")

    if not run:
        st.info("Adjust bandwidth and choose a baseline, then click **Start Live Demo**.")
        return

    # --- Live loop ---
    drl_rows = []
    base_rows = []

    obs_drl = sim_drl.reset()
    obs_base = sim_base.reset()

    cum_qoe_drl = 0.0
    cum_qoe_base = 0.0

    for step_i in range(int(segments_to_run)):
        # update trace from slider for dynamic bandwidth changes during demo
        trace = np.full(3600, float(rate_mbps), dtype=np.float32)
        sim_drl.set_trace(trace)
        sim_base.set_trace(trace)

        # Optionally apply tc/netem (only when enabled; applying every step is expensive)
        if use_netem and step_i == 0:
            _try_apply_netem(iface, rate_mbps, delay_ms, jitter_ms, loss_pct)

        # --- DRL action ---
        if ppo_model is not None:
            with torch.no_grad():
                logits, _ = ppo_model(torch.tensor(obs_drl, dtype=torch.float32).unsqueeze(0))
                a_drl = int(torch.argmax(logits, dim=-1).item())
        else:
            # fallback: conservative heuristic
            a_drl = int(np.clip(np.argmax(profile.bitrate_kbps <= (sim_drl.state.throughput_history_mbps[-1] * 1000.0)), 0, action_dim - 1))

        # --- Baseline action ---
        th_hist = list(sim_base.state.throughput_history_mbps)
        if baseline_choice == "BOLA":
            a_base = int(controller.select_action(buffer_s=float(sim_base.state.buffer_s)))
        else:
            a_base = int(controller.select_action(throughput_history_mbps=th_hist, buffer_s=float(sim_base.state.buffer_s)))

        # Step both sims
        obs_drl, r_drl, done_drl, info_drl = sim_drl.step(a_drl)
        obs_base, r_base, done_base, info_base = sim_base.step(a_base)

        cum_qoe_drl += float(r_drl)
        drl_rows.append(
            dict(
                t=float(sim_drl.state.clock_s),
                throughput=float(info_drl["observed_throughput_mbps"]),
                bitrate_kbps=int(info_drl["bitrate_kbps"]),
                buffer_s=float(info_drl["buffer_s"]),
                rebuffer_s=float(info_drl["rebuffer_s"]),
                reward=float(r_drl),
                cum_qoe=cum_qoe_drl,
            )
        )
        cum_qoe_base += float(r_base)
        base_rows.append(
            dict(
                t=float(sim_base.state.clock_s),
                throughput=float(info_base["observed_throughput_mbps"]),
                bitrate_kbps=int(info_base["bitrate_kbps"]),
                buffer_s=float(info_base["buffer_s"]),
                rebuffer_s=float(info_base["rebuffer_s"]),
                reward=float(r_base),
                cum_qoe=cum_qoe_base,
            )
        )

        # Update KPIs
        drl_df = pd.DataFrame(drl_rows)
        base_df = pd.DataFrame(base_rows)

        with col_left:
            drl_kpis.markdown(
                f"""<div class='dashboard-card' style='color: white;'>
<b>Bandwidth:</b> {rate_mbps:.1f} Mbps &nbsp; | &nbsp;
<b>Current bitrate:</b> {drl_df['bitrate_kbps'].iloc[-1]} kbps &nbsp; | &nbsp;
<b>Buffer:</b> {drl_df['buffer_s'].iloc[-1]:.2f}s &nbsp; | &nbsp;
<b>Total rebuffer:</b> {drl_df['rebuffer_s'].sum():.2f}s<br/>
<b>Cumulative QoE:</b> {drl_df['cum_qoe'].iloc[-1]:.2f}
</div>""",
                unsafe_allow_html=True,
            )
        with col_right:
            base_kpis.markdown(
                f"""<div class='dashboard-card' style='color: white;'>
<b>Bandwidth:</b> {rate_mbps:.1f} Mbps &nbsp; | &nbsp;
<b>Current bitrate:</b> {base_df['bitrate_kbps'].iloc[-1]} kbps &nbsp; | &nbsp;
<b>Buffer:</b> {base_df['buffer_s'].iloc[-1]:.2f}s &nbsp; | &nbsp;
<b>Total rebuffer:</b> {base_df['rebuffer_s'].sum():.2f}s<br/>
<b>Cumulative QoE:</b> {base_df['cum_qoe'].iloc[-1]:.2f}
</div>""",
                unsafe_allow_html=True,
            )

        # Charts
        if go is not None:
            def make_fig(df: pd.DataFrame, title: str):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["t"], y=df["throughput"], name="Throughput (Mbps)"))
                fig.add_trace(go.Scatter(x=df["t"], y=df["bitrate_kbps"] / 1000.0, name="Bitrate (Mbps)", yaxis="y2"))
                fig.add_trace(go.Scatter(x=df["t"], y=df["buffer_s"], name="Buffer (s)", yaxis="y3"))
                fig.update_layout(
                    title=title,
                    margin=dict(l=30, r=30, t=40, b=20),
                    height=320,
                    legend=dict(orientation="h"),
                    yaxis=dict(title="Throughput (Mbps)"),
                    yaxis2=dict(title="Bitrate (Mbps)", overlaying="y", side="right"),
                    yaxis3=dict(title="Buffer (s)", anchor="free", overlaying="y", side="right", position=0.92),
                )
                return fig

            with col_left:
                drl_chart.plotly_chart(make_fig(drl_df, "DRL Live Timeline"), use_container_width=True)
            with col_right:
                base_chart.plotly_chart(make_fig(base_df, "Baseline Live Timeline"), use_container_width=True)
        else:
            with col_left:
                drl_chart.dataframe(drl_df.tail(10), use_container_width=True)
            with col_right:
                base_chart.dataframe(base_df.tail(10), use_container_width=True)

        if playback_speed > 0:
            time.sleep(float(playback_speed))

        if done_drl and done_base:
            break

    st.success("Live simulation finished. Change bandwidth and run again to demonstrate different conditions.")


def main() -> None:
    st.title("Professional DRL-ABR Research Dashboard")
    st.caption("PyCharm → Git → Ubuntu VM workflow with dataset intelligence, PPO training analytics, QoE benchmarking, and tc/netem deployment support.")

    available_runs = find_run_directories(RESULTS_ROOT)
    if DEFAULT_RUN_DIR.exists() and DEFAULT_RUN_DIR not in available_runs:
        available_runs = [DEFAULT_RUN_DIR] + available_runs

    with st.sidebar:
        st.header("Workspace")
        run_options = [str(path) for path in available_runs] or [str(DEFAULT_RUN_DIR)]
        chosen_run_str = st.selectbox("Results directory", options=run_options, index=0)
        run_dir = Path(chosen_run_str)
        st.markdown("---")
        st.markdown("**Quick actions**")
        st.code("python scripts/create_sample_data.py", language="bash")
        st.code("python scripts/train_ppo.py --config configs/professional_demo.yaml", language="bash")
        st.code("python scripts/evaluate_all.py --config configs/professional_demo.yaml", language="bash")
        st.code("streamlit run dashboard/app.py", language="bash")

    history_df = load_csv(run_dir / "training_history.csv")
    summary_df = load_csv(run_dir / "evaluation_summary.csv")
    grouped_df = load_csv(run_dir / "evaluation_grouped.csv")
    leaderboard_df = load_csv(run_dir / "leaderboard.csv")
    run_config = load_json(run_dir / "run_config.json")
    training_summary = load_json(run_dir / "training_summary.json")

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        render_metric_card(
            "Best algorithm",
            leaderboard_df.iloc[0]["algorithm"] if not leaderboard_df.empty else "N/A",
            delta=f"Mean QoE {leaderboard_df.iloc[0]['mean']:.2f}" if not leaderboard_df.empty else None,
            help_text="Based on the current evaluation leaderboard.",
        )
    with top2:
        render_metric_card(
            "Training traces",
            str(training_summary.get("training_traces", 0)),
            delta=f"Episodes {training_summary.get('episodes_recorded', 0)}",
            help_text="Trace repository used for PPO episode sampling.",
        )
    with top3:
        mean_rebuffer = summary_df["total_rebuffer_s"].mean() if not summary_df.empty and "total_rebuffer_s" in summary_df else 0.0
        render_metric_card(
            "Mean rebuffer",
            f"{mean_rebuffer:.2f}s",
            delta="&nbsp;",
            help_text="Across all evaluated traces and algorithms.<br>&nbsp;",
        )
    with top4:
        best_reward = history_df["episode_reward_mean"].max() if not history_df.empty and "episode_reward_mean" in history_df else 0.0
        render_metric_card(
            "Peak reward",
            f"{best_reward:.2f}",
            delta=f"Run: {run_config.get('experiment_name', run_dir.name)}",
            help_text="Best mean episode reward recorded during PPO updates.",
        )

    render_architecture()

    overview_tab, live_tab, training_tab, benchmark_tab, session_tab, dataset_tab, deploy_tab = st.tabs(
        ["Executive overview", "Live Simulation", "Training analytics", "Benchmarking", "Session explorer", "Datasets", "Deployment"]
    )

    with overview_tab:
        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Project alignment")
            st.markdown(
                """
                - **Network emulator layer:** tc/netem export is included for replaying bandwidth profiles on Ubuntu.
                - **Video chunk simulator:** variable-size chunks model scene complexity instead of fixed-size segments only.
                - **DRL agent:** PPO now resamples traces every episode, which makes the training dataset actually matter.
                - **QoE evaluator:** benchmarking now tracks bitrate, rebuffering, stall ratio, switch counts, throughput, and total QoE.
                """
            )
            if run_config:
                st.subheader("Run configuration")
                st.json(run_config, expanded=False)
        with right:
            st.subheader("Leaderboard")
            if not leaderboard_df.empty:
                st.dataframe(leaderboard_df, use_container_width=True, height=220)
                chart_or_table_bar(leaderboard_df, x="algorithm", y="mean", title="Mean QoE by algorithm")
            else:
                st.info("Evaluation leaderboard not available yet.")

    
    with live_tab:
        render_live_simulation()

    with training_tab:
        st.subheader("PPO learning curves")
        if history_df.empty:
            st.info("No training history yet. Run a training job to populate this tab.")
        else:
            metric = st.selectbox(
                "Metric",
                options=[col for col in history_df.columns if col not in {"update"} and history_df[col].dtype != object],
                index=0,
            )
            chart_or_table_line(history_df, x="total_steps", y=metric, title=f"Training curve: {metric}")
            st.dataframe(history_df, use_container_width=True, height=260)
    with benchmark_tab:
        st.subheader("Algorithm comparison")
        if summary_df.empty:
            st.info("Run evaluation first to generate algorithm comparisons.")
        else:
            st.dataframe(summary_df, use_container_width=True, height=260)
            left, right = st.columns(2)
            with left:
                chart_or_table_bar(grouped_df, x="algorithm", y="total_qoe", title="Mean total QoE")
            with right:
                if px is not None:
                    fig = px.scatter(
                        summary_df,
                        x="average_bitrate_mbps",
                        y="total_rebuffer_s",
                        color="algorithm",
                        hover_name="trace",
                        size="switch_count" if "switch_count" in summary_df.columns else None,
                        title="Bitrate vs. rebuffer trade-off",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(summary_df[["trace", "algorithm", "average_bitrate_mbps", "total_rebuffer_s"]], use_container_width=True)
    with session_tab:
        render_session_explorer(run_dir)
    with dataset_tab:
        st.subheader("Trace intelligence")
        render_dataset_panel(run_dir)
    with deploy_tab:
        st.subheader("PyCharm → Git → Ubuntu VM deployment")
        st.markdown(
            """
            1. Open the repository in **PyCharm** and create a Python 3.10 virtual environment.
            2. Commit and push the repository to GitHub/GitLab.
            3. On the Ubuntu VM, clone the repository, run `bash scripts/bootstrap_ubuntu_vm.sh`, then train/evaluate.
            4. Export a **tc/netem** replay script with `python scripts/export_netem_profile.py <trace.csv>`.
            """
        )
        st.code("python scripts/export_netem_profile.py data/traces/research/test/hotel_wifi_01.csv --output results/netem/hotel_wifi.sh", language="bash")
        st.code("bash results/netem/hotel_wifi.sh eth0", language="bash")
        workflow_doc = Path("docs/PYCHARM_TO_UBUNTU_VM_WORKFLOW.md")
        dataset_doc = Path("docs/DATASET_GUIDE.md")
        emulation_doc = Path("docs/NETWORK_EMULATION.md")
        if workflow_doc.exists():
            with st.expander("PyCharm and Git workflow guide"):
                st.markdown(load_text(workflow_doc))
        if dataset_doc.exists():
            with st.expander("Dataset guide"):
                st.markdown(load_text(dataset_doc))
        if emulation_doc.exists():
            with st.expander("Network emulation guide"):
                st.markdown(load_text(emulation_doc))


if __name__ == "__main__":
    main()
