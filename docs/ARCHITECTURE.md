# Architecture

## High-level flow

```text
Trace dataset / tc-netem profile
            ↓
    Network-aware chunk simulator
            ↓
       Gym-style ABR environment
            ↓
     PPO / Baseline bitrate policy
            ↓
      QoE metrics + session logs
            ↓
    Streamlit professional dashboard
```

## State representation
The PPO agent observes:
- recent throughput history
- recent download times
- current playback buffer
- previous bitrate decision
- next-segment size ladder
- remaining video fraction

## Action space
A discrete action selects one bitrate from the configured ladder.

## Reward
```text
reward = quality - rebuffer_penalty × rebuffer_seconds - smoothness_penalty × bitrate_change
```

## Why the simulator is improved
The earlier demo used fixed chunk sizes, which makes every segment at a given bitrate look identical. This repository now uses **segment-varying chunk sizes**, which better reflects scene complexity changes in real video content.

## Why the data pipeline is improved
The earlier training setup could reuse the same trace over many episodes. The improved environment now resamples a new trace on every reset, so trace diversity influences policy learning.
