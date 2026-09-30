"""Per-level thresholds chosen to minimise total cost.

cascade.tune_thresholds picks cut-offs to hit an accuracy target -- a
standard you have to name, and one flat bar at every level. This module
picks them to minimise cost instead, which is derived from the three
inputs in costing.CostModel rather than chosen.

The output is deliberately the same kind of object either way: one
threshold per level. Four numbers can go in a spec, be audited, and be
applied by another team with nothing else attached.

Two ways to get them, and the difference matters:

    implied_thresholds()       from the arithmetic, instantly, no data
    cost_optimal_thresholds()  by measuring cost on labelled data

The implied version is a closed form, but it assumes the next-shallower
answer is reliable, which is only roughly true. The measured version makes
no such assumption, and it has a property worth stating plainly: because
it minimises *observed* cost, it needs no probability calibration. It does
not care whether the model's confidence figures are honest, only that
higher means better -- the accuracy that enters the arithmetic is read off
the labels.

Note the two approaches produce differently-shaped policies. An accuracy
target applies the same bar at every level. The cost view demands more
confidence at shallow levels, because stepping from L1 to L2 saves little
staff time and so justifies little risk, while the last level saves the
most and can afford the most.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cascade import MARGIN_FMT, PROB_FMT, _depths, apply_cascade, prefix_correct
from .costing import CostModel, realised_cost


# ---------------------------------------------------------------------------
# The analytic version
# ---------------------------------------------------------------------------

def implied_thresholds(costs: CostModel, n_levels: int = 4) -> pd.DataFrame:
    """Cut-offs implied by the cost model, with no data involved.

    Comparing depth d against d-1 and assuming the shallower answer is
    reliable, depth d is worth taking when the risk it adds costs less
    than the labour it saves:

        p*_d = 1 - V_d / C,   where V_d = L(d-1) - L(d)

    Note the shape this produces: shallow depths demand *more* confidence,
    because stepping one level deeper there saves little time and so
    justifies little risk. An accuracy target applies one flat bar at
    every depth, which is a genuinely different policy, not just a
    different way of writing the same one.

    These are approximations -- the real rule compares the whole cost
    vector across depths, so rows uncertain at several depths at once can
    come out differently. Use them to understand and sanity-check, and
    use cost_optimal_thresholds() to actually set the numbers.
    """
    rows = []
    for d in range(1, n_levels + 1):
        v = costs.labour_cost(d - 1) - costs.labour_cost(d)
        p = 1.0 - v / costs.cost_of_error
        rows.append(
            {
                "depth": d,
                "labour_saved_V": v,
                "implied_threshold": float(np.clip(p, 0.0, 1.0)),
                "attainable": p <= 1.0,
            }
        )
    return pd.DataFrame(rows)


def implied_accuracy_targets(costs: CostModel, n_levels: int = 4) -> Dict[int, float]:
    """The accuracy standard each level's cost model implies, as {depth: p}.

    Answers the reverse of "what does a 95% target cost me?": given the
    money, what standard is being demanded? It is the same quantity as
    implied_thresholds, read as an accuracy rather than a cut-off --

        p*_d = 1 - V_d / C

    -- and there is one per level, not one overall, because V_d is the
    staff time saved by going one level deeper and that differs by depth.
    Shallow levels save little and so demand more confidence.

    Read it as a *marginal* standard: the accuracy the last item accepted
    must reach for accepting it to be worthwhile. The average accuracy
    across everything accepted will be higher, since most accepted items
    sit well above the margin.
    """
    imp = implied_thresholds(costs, n_levels)
    return {int(r.depth): float(r.implied_threshold) for r in imp.itertuples()}


# ---------------------------------------------------------------------------
# Fast cost evaluation for a candidate threshold vector
# ---------------------------------------------------------------------------

def _pack(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
    score: str = "probability",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull everything the search needs into plain arrays.

    Returns (scores, correct, depth_values, labour) where scores and
    correct are (n_rows, n_depths) and column j corresponds to depth j+1.
    """
    depths = _depths(prefix_df)
    fmt = PROB_FMT if score == "probability" else MARGIN_FMT

    scores = np.column_stack(
        [prefix_df[fmt.format(d=d)].to_numpy(dtype=float) for d in depths]
    )
    correct = np.column_stack(
        [prefix_correct(prefix_df, actual, level_columns, d) for d in depths]
    )
    labour = np.array([costs.labour_cost(d) for d in [0] + list(depths)], dtype=float)
    return scores, correct, np.array(depths), labour


def _assign(scores: np.ndarray, depth_values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Deepest depth whose score clears its threshold; 0 if none do.

    Deepest-passing is just the maximum depth among those passing, so this
    is one comparison and one max -- no loop over depths.
    """
    passing = scores >= t[None, :]
    return (passing * depth_values[None, :]).max(axis=1)


def _total_cost(
    scores: np.ndarray,
    correct: np.ndarray,
    depth_values: np.ndarray,
    labour: np.ndarray,
    cost_of_error: float,
    t: np.ndarray,
) -> Tuple[float, np.ndarray]:
    assigned = _assign(scores, depth_values, t)
    total = labour[assigned].sum()
    emitted = assigned > 0
    if emitted.any():
        # correct[:, d-1] is correctness of the prefix at depth d
        got = np.take_along_axis(
            correct, np.clip(assigned - 1, 0, correct.shape[1] - 1)[:, None], axis=1
        ).ravel()
        total += cost_of_error * float((emitted & ~got).sum())
    return float(total), assigned


# ---------------------------------------------------------------------------
# The measured version
# ---------------------------------------------------------------------------

def cost_optimal_thresholds(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
    score: str = "probability",
    n_grid: int = 200,
    max_sweeps: int = 12,
    verbose: bool = True,
) -> Tuple[Dict[int, float], pd.DataFrame]:
    """Search for the threshold vector with the lowest measured cost.

    Coordinate descent: hold every depth's cut-off fixed but one, try all
    candidates for that one, keep the best, move on, repeat until a full
    sweep changes nothing. Each evaluation is vectorised over rows, so a
    few thousand evaluations is quick.

    Coordinate descent is not guaranteed to find the global optimum -- the
    depths interact, since raising one cut-off pushes rows down to
    shallower depths. In practice the surface is well behaved and it
    converges in two or three sweeps. It is started from the analytic
    implied thresholds rather than an arbitrary point, which helps.

    Candidates are quantiles of the observed scores, plus infinity so that
    "never predict at this depth" is always available.

    Returns (thresholds, trace).
    """
    scores, correct, depth_values, labour = _pack(
        prefix_df, actual, level_columns, costs, score
    )
    n_depths = scores.shape[1]
    C = costs.cost_of_error

    qs = np.linspace(0.0, 1.0, n_grid)
    candidates = [
        np.unique(np.r_[np.quantile(scores[:, j], qs), np.inf]) for j in range(n_depths)
    ]

    # Start from the analytic guess, clipped into the observed range.
    imp = implied_thresholds(costs, n_depths)["implied_threshold"].to_numpy()
    t = np.array(
        [
            candidates[j][np.argmin(np.abs(candidates[j] - imp[j]))]
            for j in range(n_depths)
        ],
        dtype=float,
    )

    best, _ = _total_cost(scores, correct, depth_values, labour, C, t)
    trace = [{"sweep": 0, "depth": None, "total_cost": best,
              "cost_per_item": best / len(scores)}]

    for sweep in range(1, max_sweeps + 1):
        improved = False
        for j in range(n_depths):
            keep = t[j]
            best_j, best_val = keep, best
            for cand in candidates[j]:
                t[j] = cand
                val, _ = _total_cost(scores, correct, depth_values, labour, C, t)
                if val < best_val - 1e-9:
                    best_val, best_j = val, cand
            t[j] = best_j
            if best_val < best - 1e-9:
                best, improved = best_val, True
            trace.append(
                {
                    "sweep": sweep,
                    "depth": int(depth_values[j]),
                    "total_cost": best,
                    "cost_per_item": best / len(scores),
                }
            )
        if not improved:
            break

    if verbose:
        print(f"[cost-thresholds] converged after {sweep} sweep(s); "
              f"cost per item {best / len(scores):.4f}")
    thresholds = {int(depth_values[j]): float(t[j]) for j in range(n_depths)}
    return thresholds, pd.DataFrame(trace)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_threshold_policies(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
    named_thresholds: Optional[Dict[str, Dict[int, float]]] = None,
) -> pd.DataFrame:
    """Price several depth policies on the same rows.

    Pass whatever you want compared, e.g.
    {"accuracy target": th_acc, "cost-optimal": th_cost}. Every fixed
    depth is added automatically, because those are the choices you would
    otherwise be picking between -- and "always depth 0" is today's
    process, which is the number any of this has to beat.
    """
    rows = []

    def add(name, frame):
        rows.append({"policy": name, **realised_cost(
            frame, prefix_df, actual, level_columns, costs)})

    for name, th in (named_thresholds or {}).items():
        add(f"thresholds: {name}", apply_cascade(prefix_df, level_columns, th))

    imp = implied_thresholds(costs, len(_depths(prefix_df)))
    add("thresholds: implied (analytic)",
        apply_cascade(prefix_df, level_columns,
                      dict(zip(imp["depth"], imp["implied_threshold"]))))

    for d in [0] + _depths(prefix_df):
        fixed = pd.DataFrame({"Assigned Depth": np.full(len(prefix_df), d)})
        for c in level_columns[:d]:
            fixed[c] = prefix_df[f"L{d} Predicted {c}"].astype(str).to_numpy()
        for c in level_columns[d:]:
            fixed[c] = ""
        add("today: all manual" if d == 0 else f"fixed: always depth {d}", fixed)

    report = pd.DataFrame(rows)
    best = report["cost_per_item"].min()
    report["excess_vs_best"] = report["cost_per_item"] - best
    return report.sort_values("cost_per_item").reset_index(drop=True)


def sweep_error_cost(
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
    error_costs: tuple = (10, 25, 50, 100, 250),
    n_grid: int = 120,
) -> pd.DataFrame:
    """Re-derive the thresholds across a range of error costs.

    cost_of_error is the one input nobody can measure, so the useful move
    is not to defend a number but to show whether it matters. If mean
    depth barely moves from 25 to 100, the choice is immaterial and the
    argument is over. If it swings, the decision rests on a quantity you
    do not have -- which is an argument for the more cautious setting.
    """
    rows = []
    for ce in error_costs:
        c = CostModel(costs.minutes_by_depth, costs.hourly_rate, float(ce))
        th, _ = cost_optimal_thresholds(
            prefix_df, actual, level_columns, c, n_grid=n_grid, verbose=False
        )
        priced = realised_cost(
            apply_cascade(prefix_df, level_columns, th),
            prefix_df, actual, level_columns, c,
        )
        rows.append(
            {
                "cost_of_error": ce,
                **{f"threshold_depth_{d}": t for d, t in th.items()},
                "mean_depth": priced["mean_depth"],
                "n_errors": priced["n_errors"],
                "cost_per_item": priced["cost_per_item"],
            }
        )
    return pd.DataFrame(rows)
