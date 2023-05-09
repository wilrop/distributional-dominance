import numpy as np
from pulp import *

from distrib.helpers import remove_dists


def cdd_joint_lp(dist_check, mixture_lst):
    """Check whether a mixture distribution exists which distributionally dominates the distribution to check.

    Args:
        dist_check (MultivariateCategoricalDistribution): A multivariate distribution to check.
        mixture_lst (List[MultivariateCategoricalDistribution]): A list of distributions to use in the mixture.

    Returns:
        bool, float: Success of the linear program and computed delta.
    """
    problem = LpProblem('mixtureDominanceJoint', LpMaximize)

    all_dists = mixture_lst + [dist_check]
    check_points = set()

    # Collect the points where the CDFs change.
    for dist in all_dists:
        idxs = np.argwhere(dist.dist > 0)
        check_points.update([tuple(point) for point in idxs])

    # Define weight variables.
    weight_variables = []
    for i in range(len(mixture_lst)):
        weight_variables.append(LpVariable(f'w{i}', lowBound=0, upBound=1))  # Bound weights in the interval [0, 1].
    problem += lpSum(weight_variables) == 1  # Weights should sum to one.

    # Define the joint distribution constraints.
    for idx, point in enumerate(check_points):
        s_var = LpVariable(f's{idx}', lowBound=0)
        values = [dist.get_cdf()[point] for dist in mixture_lst]
        problem += lpDot(weight_variables, values) + s_var == dist_check.get_cdf()[point]
        problem += s_var >= 0

    # Define the marginal constraints.
    l_variables = []
    for dim in range(dist_check.num_dims):
        for idx, point in enumerate(check_points):
            l_var = LpVariable(f'l{dim}{idx}', lowBound=0)
            values = [dist.get_marginal(dim).get_cdf()[point[dim]] for dist in mixture_lst]
            problem += lpDot(weight_variables, values) + l_var == dist_check.get_marginal(dim).get_cdf()[point[dim]]
            l_variables.append(l_var)

    problem += lpSum(l_variables)  # Maximise the sum of l variables.

    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    delta = problem.objective.value()  # Get the objective value.
    return success == 1, delta > 0


def cdd_marginal_lp(dist_check, mixture_lst):
    """Check whether a mixture distribution exists which distributionally dominates the distribution to check, using
        only the marginal distributions.

    Args:
        dist_check (MultivariateCategoricalDistribution): A multivariate distribution to check.
        mixture_lst (List[MultivariateCategoricalDistribution]): A list of distributions to use in the mixture.

    Returns:
        bool, float: Success of the linear program and computed delta.
    """
    problem = LpProblem('mixtureDominanceMarginals', LpMaximize)

    all_dists = mixture_lst + [dist_check]
    check_points = set()

    # Collect the points where the CDFs change.
    for dist in all_dists:
        idxs = np.argwhere(dist.dist > 0)
        check_points.update([tuple(point) for point in idxs])

    # Define weight variables.
    weight_variables = []
    for i in range(len(mixture_lst)):
        weight_variables.append(LpVariable(f'w{i}', lowBound=0, upBound=1))  # Bound weights in the interval [0, 1].
    problem += lpSum(weight_variables) == 1  # Weights should sum to one.

    # Define the joint distribution constraints from the marginals.
    s_variables = []
    for idx, point in enumerate(check_points):
        s_var = LpVariable(f's{idx}', lowBound=0)
        problem += s_var >= 0

        values = []
        for dist in mixture_lst:
            value = np.prod([dist.get_marginal(dim).get_cdf()[point[dim]] for dim in range(dist_check.num_dims)])
            values.append(value)

        dist_check_value = np.prod(
            [dist_check.get_marginal(dim).get_cdf()[point[dim]] for dim in range(dist_check.num_dims)])
        problem += lpDot(weight_variables, values) + s_var == dist_check_value
        s_variables.append(s_var)

    problem += lpSum(s_variables)  # Maximise the sum of s variables.

    success = problem.solve(solver=PULP_CBC_CMD(msg=False))  # Solve the problem.
    delta = problem.objective.value()  # Get the objective value.
    return success == 1, delta > 0


def convex_dist_dom(dist, mixture_lst, lp='joint'):
    """Whether a distribution is convex distributionally dominated by a mixture of distributions.

    Args:
        dist (MultivariateCategoricalDistribution): A multivariate distribution to check.
        mixture_lst (List[MultivariateCategoricalDistribution]): A list of distributions to use in the mixture.
        lp (str): The type of linear program to use. Either 'joint' or 'marginal'.

    Returns:
        bool: Whether the distribution is convex distributionally dominated.
    """
    if lp == 'joint':
        success, delta = cdd_joint_lp(dist, mixture_lst)
    else:
        success, delta = cdd_marginal_lp(dist, mixture_lst)

    return success and delta > 0


def cdd_prune(candidates, lp='joint'):
    """Prune a set of candidate distributions to the convex distributional dominant set.

    Args:
        candidates (List[MultivariateCategoricalDistribution]): A list of distributions to prune.
        lp (str, optional): The linear program to use. Defaults to 'joint'.

    Returns:
        List[MultivariateCategoricalDistribution]: A subset of the candidates where no distribution is dominated by a
            mixture.
    """
    cdd_set = []

    while candidates:
        dist = candidates.pop()

        mixture_lst = cdd_set + candidates
        if mixture_lst and convex_dist_dom(dist, mixture_lst, lp=lp):
            candidates = remove_dists(dist, candidates)  # Remove equals.
        else:
            cdd_set.append(dist)

    return cdd_set
