import numpy as np
from pulp import *
from utils import remove_dists, zero_init
from multivariate_categorical_distribution import MCD


def c_prune(candidates):
    """Create a convex coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    candidates (List[Dist]): A list of distributions.

    Returns:
        List[Dist]: A convex coverage set.

    """
    zero_dist = zero_init(MCD, candidates[0].num_atoms, candidates[0].v_mins, candidates[0].v_maxs)
    p_candidates = p_prune(candidates)
    ccs = [zero_dist]

    while p_candidates:
        dist = p_candidates[0]  # Get an element from the Pareto coverage set.

        weight = find_weight(dist, ccs)  # Get a weight for which the expected value improves the CCS.

        if weight is None:  # If there is none, discard the distribution.
            p_candidates = remove_dists(dist, p_candidates)
        else:  # Otherwise add the best distribution to the CCS.
            new_dist = max(p_candidates, key=lambda x: np.dot(x.expected_value, weight))
            p_candidates = remove_dists(new_dist, p_candidates)
            ccs.append(new_dist)

    ccs = remove_dists(zero_dist, ccs)
    return ccs


def pareto_dominates(a, b):
    """Check if the first distribution Pareto dominates the second distribution.

    Args:
        a (Dist): A distribution.
        b (Dist): A distribution.

    Returns:
        bool: Whether vector a dominates vector b.
    """
    ev_a = a.expected_value
    ev_b = b.expected_value
    return np.all(ev_a >= ev_b) and np.any(ev_a > ev_b)


def p_prune(candidates):
    """Create a Pareto coverage set from a set of candidate points.

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        candidates (List[Dist]): A list of distributions.

    Returns:
        List[Dist]: A list of distributions in the Pareto front.
    """
    pcs = []

    while candidates:
        dist = candidates.pop()
        to_remove = [dist]
        for alternative in candidates:
            if pareto_dominates(alternative, dist):
                dist = alternative
                to_remove.append(alternative)
            if pareto_dominates(dist, alternative):
                to_remove.append(alternative)
        candidates = remove_dists(to_remove, candidates)
        pcs.append(dist)

    return pcs


def find_weight(dist, candidates):
    """Find a weight for which a specific vector improves on a CCS [1].

    References:
        .. [1] Roijers, D. M., & Whiteson, S. (2017). Multi-objective decision making. 34, 129–129.
            https://doi.org/10.2200/S00765ED1V01Y201704AIM034

    Args:
        dist (Dist): A distribution.
        candidates (List[Dist]): The current CCS.

    Returns:
        ndarray | None: A weight array if it found one, otherwise None.
    """
    vector = dist.expected_value
    candidates = [dist.expected_value for dist in candidates]
    num_objectives = len(candidates[0])

    problem = LpProblem('findWeight', LpMaximize)
    x = LpVariable('x')
    w = []

    for obj in range(num_objectives):  # Make weight decision variables.
        w.append(LpVariable(f'w{obj}', 0, 1))

    for candidate in candidates:  # Add the constraints on the improvement of w.
        diff = list(np.subtract(vector, candidate))
        problem += lpDot(w, diff) - x >= 0

    problem += lpSum(w) == 1  # Weights should sum to one.
    problem += x  # Add x as the objective to maximise.
    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    x = problem.objective.value()  # Get the objective value.
    weight_vec = np.zeros(num_objectives)
    for var in problem.variables():  # Get the weight values.
        if var.name[0] == 'w':
            weight_idx = int(var.name[-1])
            weight_vec[weight_idx] = var.value()
    if success and x > 0:
        return weight_vec
    return None
