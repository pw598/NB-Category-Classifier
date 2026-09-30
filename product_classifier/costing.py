"""What a depth policy costs, in money.

Three inputs, two of which are measurable:

    minutes_by_depth   how much human work a given depth leaves behind
    hourly_rate        what that work costs
    cost_of_error      what a wrong category costs downstream

From those, the cost of any policy follows. cost_thresholds.py searches
for the per-level cut-offs that make it smallest.

The model assumed here: a reviewer completes the levels beneath whatever
the model assigned, *without re-checking the levels above*. So the labour
saving is banked whether or not the model was right, and a wrong prefix
survives into the final record and is charged cost_of_error. If your
reviewers verify the whole path instead, errors get caught but
verification time lands on every product, correct ones included -- a
different model, and this one would overstate the benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .cascade import _depths, prefix_correct


@dataclass
class CostModel:
    """The three inputs to a cost-based depth decision.

    minutes_by_depth: minutes of human work still required *after* the
        model emits depth d. Key 0 is classifying from scratch -- today's
        process. The deepest key must be 0.0, since nothing is left to do.

        Supply a figure per depth rather than one flat per-level rate: the
        final distinction is usually much slower to make by hand than the
        first, so the increments should grow with depth. Something like
        {0: 8.0, 1: 7.0, 2: 5.5, 3: 3.5, 4: 0.0} has the right shape.
        Time a few people on a sample to get real numbers; this is the
        input the answer is least sensitive to.

    hourly_rate: fully-loaded labour cost per hour -- wage plus employer
        costs, benefits and overhead, not the base wage. Same currency as
        cost_of_error.

    cost_of_error: what one wrong prefix costs once it is in the records:
        time to notice and correct it, plus the downstream consequences.
        This is the only genuinely subjective input.

        Rather than pricing it directly, derive it from a standard people
        can actually state. If you would want accuracy p before accepting
        a prediction unreviewed, then

            cost_of_error = V * 1 / (1 - p)

        where V is the labour saved by the deepest step,
        (minutes_by_depth[n-1] / 60) * hourly_rate. At p = 0.95 that is
        20 x V. Then use the sensitivity sweep to check whether the exact
        figure changes the policy at all -- often it does not.

    Note the currency unit cannot bite you: scaling hourly_rate and
    cost_of_error together scales every cost equally and leaves the chosen
    policy unchanged. Only their ratio matters.
    """

    minutes_by_depth: Dict[int, float]
    hourly_rate: float
    cost_of_error: float

    def __post_init__(self) -> None:
        if not self.minutes_by_depth:
            raise ValueError("minutes_by_depth cannot be empty.")
        if 0 not in self.minutes_by_depth:
            raise KeyError(
                "minutes_by_depth needs a key 0 -- the time to classify a "
                "product from scratch, which is what abstaining costs."
            )
        deepest = max(self.minutes_by_depth)
        if self.minutes_by_depth[deepest] != 0.0:
            raise ValueError(
                f"minutes_by_depth[{deepest}] should be 0.0: at full depth "
                "there is no work left to do."
            )

    def labour_cost(self, depth: int) -> float:
        if depth not in self.minutes_by_depth:
            raise KeyError(
                f"No minutes_by_depth entry for depth {depth}. "
                f"Have: {sorted(self.minutes_by_depth)}"
            )
        return self.minutes_by_depth[depth] / 60.0 * self.hourly_rate

    def implied_error_cost(self, target_accuracy: float = 0.95) -> float:
        """cost_of_error consistent with wanting this accuracy at full depth.

        Convenience for the elicitation described above -- call it, look at
        the number, decide whether it sounds like what a bad category
        actually costs you.
        """
        depths = sorted(self.minutes_by_depth)
        v = self.labour_cost(depths[-2]) - self.labour_cost(depths[-1])
        return v / (1.0 - target_accuracy)


def realised_cost(
    assigned: pd.DataFrame,
    prefix_df: pd.DataFrame,
    actual: pd.DataFrame,
    level_columns: List[str],
    costs: CostModel,
) -> Dict[str, float]:
    """What a policy actually cost on labelled rows, decomposed.

    Splitting labour from errors matters when reading the result: two
    policies can reach the same total by very different routes, and which
    one you prefer may depend on whether the errors are visible.
    """
    depth = assigned["Assigned Depth"].to_numpy()
    n = len(assigned)
    if n == 0:
        raise ValueError("No rows to price.")

    labour = np.array([costs.labour_cost(int(d)) for d in depth])
    errors = np.zeros(n, dtype=bool)
    for d in _depths(prefix_df):
        rows = depth == d
        if rows.any():
            errors[rows] = ~prefix_correct(prefix_df, actual, level_columns, d)[rows]

    error_cost = float(errors.sum()) * costs.cost_of_error
    total = float(labour.sum()) + error_cost
    return {
        "n_items": n,
        "labour_cost": float(labour.sum()),
        "error_cost": error_cost,
        "total_cost": total,
        "cost_per_item": total / n,
        "n_errors": int(errors.sum()),
        "mean_depth": float(depth.mean()),
    }
