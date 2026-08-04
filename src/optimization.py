"""
optimization.py
Two complementary approaches for searching signal parameter spaces
efficiently:

1. Parallel grid search (multiprocessing) — exhaustive, but spread
   across CPU cores. Best when the grid is moderate-sized and full
   coverage matters (e.g. for building the kind of heatmap used
   earlier in this project).

2. Bayesian optimization (scikit-optimize) — for larger or more
   expensive search spaces, where exhaustive grid search would take
   too long. Bayesian optimization intelligently samples promising
   regions rather than testing every combination, at the cost of not
   giving full grid coverage the way parallel grid search does.
"""

import itertools
from multiprocessing import Pool, cpu_count
import pandas as pd


# ---------------------------------------------------------------------
# Parallel grid search
# ---------------------------------------------------------------------

def _evaluate_combo(args):
    """
    Internal worker function. `evaluate_fn` must be picklable (a plain
    module-level function, not a lambda or nested closure) since
    multiprocessing needs to send it to worker processes.
    """
    evaluate_fn, params = args
    result = evaluate_fn(**params)
    return {**params, **result}


def run_grid_search_parallel(evaluate_fn, param_grid, fixed_args=None, n_jobs=None):
    """
    Runs evaluate_fn across every combination in param_grid, in
    parallel across CPU cores. fixed_args is a dict of additional
    keyword arguments passed to every call (e.g. the shared price/
    returns/macro data), unchanged across the grid.
    """
    n_jobs = n_jobs or max(1, cpu_count() - 1)
    fixed_args = fixed_args or {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Running {len(combos)} combinations across {n_jobs} processes...")

    args = [(evaluate_fn, {**combo, **fixed_args}) for combo in combos]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_evaluate_combo, args)

    # Drop the large fixed_args from the result rows before building the DataFrame
    for r in results:
        for key in fixed_args:
            r.pop(key, None)

    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------------------

def run_bayesian_optimization(evaluate_fn, search_space, param_names,
                                 n_calls=50, metric="mean_ic", minimize=False,
                                 random_state=42):
    """
    Uses Bayesian optimization (Gaussian process-based) to search a
    parameter space more efficiently than exhaustive grid search,
    intelligently sampling promising regions rather than testing every
    combination.

    evaluate_fn: a function that takes the search parameters as
        keyword arguments and returns a dict of result metrics
    search_space: list of skopt dimension objects, e.g.
        [Integer(10, 250, name="beta_window"), Integer(1, 60, name="horizon")]
    param_names: list of parameter names, matching the order of
        search_space
    n_calls: number of evaluations to run (each one calls evaluate_fn
        once) — this is the main lever for how long optimization takes
    metric: which key in evaluate_fn's returned dict to optimize
    minimize: if False (default), the optimizer maximizes the metric
        (internally by minimizing its negative, since skopt only
        minimizes)
    """
    from skopt import gp_minimize
    from skopt.utils import use_named_args

    all_results = []

    @use_named_args(search_space)
    def objective(**params):
        result = evaluate_fn(**params)
        all_results.append({**params, **result})

        score = result[metric]
        return score if minimize else -score

    opt_result = gp_minimize(
        objective,
        search_space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
    )

    results_df = pd.DataFrame(all_results).sort_values(
        metric, ascending=minimize
    ).reset_index(drop=True)

    best_params = dict(zip(param_names, opt_result.x))
    print(f"\nBest parameters found: {best_params}")
    print(f"Best {metric}: {results_df.iloc[0][metric]:.4f}")

    return results_df, opt_result

def run_bayesian_optimization_parallel(evaluate_fn, search_space, param_names,
                                          fixed_args=None, n_calls=40, batch_size=6,
                                          metric="mean_ic", minimize=False,
                                          n_jobs=None, random_state=42):
    """
    Bayesian optimization with batch-parallel evaluation: at each
    iteration, asks the optimizer for `batch_size` candidate points
    (rather than one), evaluates them all simultaneously across CPU
    cores via multiprocessing, then reports the results back to the
    optimizer before the next batch. This is what lets Bayesian
    optimization actually benefit from multiprocessing, since a
    standard sequential Bayesian loop can't be parallelized directly.
    """
    from skopt import Optimizer

    n_jobs = n_jobs or max(1, cpu_count() - 1)
    fixed_args = fixed_args or {}

    optimizer = Optimizer(search_space, random_state=random_state)

    all_results = []
    n_batches = -(-n_calls // batch_size)  # ceiling division

    for batch_num in range(n_batches):
        candidate_points = optimizer.ask(n_points=batch_size)
        candidate_dicts = [dict(zip(param_names, point)) for point in candidate_points]

        print(f"Batch {batch_num + 1}/{n_batches}: evaluating {len(candidate_dicts)} candidates...")

        args = [(evaluate_fn, {**combo, **fixed_args}) for combo in candidate_dicts]
        with Pool(processes=n_jobs) as pool:
            batch_results = pool.map(_evaluate_combo, args)

        scores = []
        for combo, result in zip(candidate_dicts, batch_results):
            clean_result = {k: v for k, v in result.items() if k not in fixed_args}
            all_results.append(clean_result)
            score = result.get(metric, float("nan"))
            scores.append(score if minimize else -score)

        optimizer.tell(candidate_points, scores)

    results_df = pd.DataFrame(all_results).sort_values(metric, ascending=minimize).reset_index(drop=True)

    print(f"\nBest {metric}: {results_df.iloc[0][metric]:.4f}")
    print(f"Best parameters: {results_df.iloc[0][param_names].to_dict()}")

    return results_df