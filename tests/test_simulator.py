import numpy as np

from abr_rl.metrics.qoe import QoEWeights
from abr_rl.sim.abr_simulator import ABRSimulator
from abr_rl.sim.video_profile import VideoProfile


def test_simulator_runs_one_step():
    trace = np.full(120, 5.0, dtype=np.float32)
    profile = VideoProfile(
        segment_duration_s=4.0,
        bitrate_kbps=[300, 750, 1200, 1850],
        total_segments=10,
        initial_buffer_s=8.0,
        max_buffer_s=24.0,
        history_length=4,
        network_overhead=1.05,
    )
    sim = ABRSimulator(trace, profile, QoEWeights())
    obs = sim.reset()
    next_obs, reward, done, info = sim.step(2)
    assert obs.shape == next_obs.shape
    assert isinstance(reward, float)
    assert info["bitrate_kbps"] == 1200
    assert done is False
