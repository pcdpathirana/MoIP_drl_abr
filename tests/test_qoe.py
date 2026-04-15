from abr_rl.metrics.qoe import QoEWeights, step_reward


def test_qoe_reward_penalizes_rebuffer_and_switch():
    reward, quality, smoothness = step_reward(
        bitrate_kbps=1200,
        previous_bitrate_kbps=750,
        rebuffer_s=1.0,
        weights=QoEWeights(rebuffer_penalty=4.3, smoothness_penalty=1.0),
    )
    assert quality == 1.2
    assert smoothness == 0.45
    assert reward < 0
