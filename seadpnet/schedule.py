"""Keep the planned LR horizon independent from a diagnostic early stop."""
import math


def run_updates(planned_updates, stop_after=None):
    actual = planned_updates if stop_after is None else stop_after
    if planned_updates < 1 or actual < 1 or actual > planned_updates:
        raise ValueError("stop-after must be within the planned training horizon")
    return actual


def lr_multiplier(step, planned_updates, warmup_updates, min_lr_ratio):
    warmup = min(warmup_updates, max(1, planned_updates // 10))
    if step < warmup:
        return max(.01, (step + 1) / warmup)
    return min_lr_ratio + (1-min_lr_ratio)*.5*(1+math.cos(
        math.pi*(step-warmup)/max(1, planned_updates-warmup)))
