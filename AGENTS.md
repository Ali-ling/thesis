# Agent Instructions

## Start Here

Read `AGENT_HANDOFF.md` before continuing this project. It contains the current
research status, frozen experiment parameters, known negative results, data
locations, and the next milestones.

## Non-Negotiable Constraints

- Do not commit `data/`, model checkpoints, compressed trajectories, or raw
  experiment logs.
- Do not change detector thresholds or experiment windows after inspecting
  formal test outcomes without labeling the change as a new, independent
  experiment.
- Do not describe Open Bandit application experiments as successful online PPO
  adaptation. The current real-data decision metric did not pass known-change
  validation.
- Keep the finite-state, known-truth experiments as the main mechanism
  validation. Real-data and semi-synthetic results are separate evidence.
- Preserve the distinction between `Q*`, `Q^pi`, and critic `V(s)`.
- Report negative and uncertain outcomes explicitly.

## Required Checks

Run the main tests:

```bash
python -m unittest -v test_decision_detector.py test_update_scheduler.py test_ppo_arena.py test_run_arena_experiments.py test_calibrate_ppo_budget.py test_analyze_formal_results.py
```

Run the Open Bandit test separately because it can conflict with PyTorch's
OpenMP runtime when both libraries are loaded in one process:

```bash
python -m unittest -v test_open_bandit_topk.py
```

## Current Priority

Acquire or locate a trusted source for KuaiRand-Pure, audit its suitability,
and avoid further threshold tuning on Open Bandit.
