import numpy as np

from abr_rl.envs.abr_env import ABRStreamingEnv
from abr_rl.metrics.qoe import QoEWeights
from abr_rl.sim.video_profile import VideoProfile


def test_env_reset_and_step():
    trace = np.full(150, 4.5, dtype=np.float32)
    profile = VideoProfile(
        segment_duration_s=4.0,
        bitrate_kbps=[300, 750, 1200, 1850],
        total_segments=6,
        initial_buffer_s=8.0,
        max_buffer_s=24.0,
        history_length=4,
        network_overhead=1.05,
    )
    env = ABRStreamingEnv(trace, profile, QoEWeights())
    obs, info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == next_obs.shape
    assert truncated is False
    assert isinstance(terminated, bool)
