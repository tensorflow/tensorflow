# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Code for optimizing over a set of candidate solutions.

The functions in this file deal with the constrained problem:

> minimize f(w)
> s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}

Here, f(w) is the "objective function", and g_i(w) is the ith (of m) "constraint
function". Given the values of the objective and constraint functions for a set
of n "candidate solutions" {w_0,w_1,...,w_{n-1}} (for a total of n objective
function values, and n*m constraint function values), the
`find_best_candidate_distribution` function finds the best DISTRIBUTION over
these candidates, while `find_best_candidate_index' heuristically finds the
single best candidate.

Both of these functions have dependencies on `scipy`, so if you want to call
them, then you must make sure that `scipy` is available. The imports are
performed inside the functions themselves, so if they're not actually called,
then `scipy` is not needed.

For more specifics, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization".
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

The `find_best_candidate_distribution` function implements the approach
described in Lemma 3, while `find_best_candidate_index` implements the heuristic
used for hyperparameter search in the experiments of Section 5.2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


def _find_best_candidate_distribution_helper(objective_vector,
                                             constraints_matrix,
                                             maximum_violation=0.0):
  """Finds a distribution minimizing an objective subject to constraints.

  This function deals with the constrained problem:

  > minimize f(w)
  > s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}

  Here, f(w) is the "objective function", and g_i(w) is the ith (of m)
  "constraint function". Given a set of n "candidate solutions"
  {w_0,w_1,...,w_{n-1}}, this function finds a distribution over these n
  candidates that, in expectation, minimizes the objective while violating
  the constraints by no more than `maximum_violation`. If no such distribution
  exists, it returns an error (using Go-style error reporting).

  The `objective_vector` parameter should be a numpy array with shape (n,), for
  which objective_vector[i] = f(w_i). Likewise, `constraints_matrix` should be a
  numpy array with shape (m,n), for which constraints_matrix[i,j] = g_i(w_j).

  This function will return a distribution for which at most m+1 probabilities,
  and often fewer, are nonzero.

  Args:
    objective_vector: numpy array of shape (n,), where n is the number of
      "candidate solutions". Contains the objective function values.
    constraints_matrix: numpy array of shape (m,n), where m is the number of
      constraints and n is the number of "candidate solutions". Contains the
      constraint violation magnitudes.
    maximum_violation: nonnegative float, the maximum amount by which any
      constraint may be violated, in expectation.

  Returns:
    A pair (`result`, `message`), exactly one of which is None. If `message` is
      None, then the `result` contains the optimal distribution as a numpy array
      of shape (n,). If `result` is None, then `message` contains an error
      message.

  Raises:
    ValueError: If `objective_vector` and `constraints_matrix` have inconsistent
      shapes, or if `maximum_violation` is negative.
    ImportError: If we're unable to import `scipy.optimize`.
  """
  if maximum_violation < 0.0:
    raise ValueError("maximum_violation must be nonnegative")

  mm, nn = np.shape(constraints_matrix)
  if (nn,) != np.shape(objective_vector):
    raise ValueError(
        "objective_vector must have shape (n,), and constraints_matrix (m, n),"
        " where n is the number of candidates, and m is the number of "
        "constraints")

  # We import scipy inline, instead of at the top of the file, so that a scipy
  # dependency is only introduced if either find_best_candidate_distribution()
  # or find_best_candidate_index() are actually called.
  import scipy.optimize  # pylint: disable=g-import-not-at-top

  # Feasibility (within maximum_violation) constraints.
  a_ub = constraints_matrix
  b_ub = np.full((mm, 1), maximum_violation)
  # Sum-to-one constraint.
  a_eq = np.ones((1, nn))
  b_eq = np.ones((1, 1))
  # Nonnegativity constraints.
  bounds = (0, None)

  result = scipy.optimize.linprog(
      objective_vector,
      A_ub=a_ub,
      b_ub=b_ub,
      A_eq=a_eq,
      b_eq=b_eq,
      bounds=bounds)
  # Go-style error reporting. We don't raise on error, since
  # find_best_candidate_distribution() needs to handle the failure case, and we
  # shouldn't use exceptions as flow-control.
  if not result.success:
    return (None, result.message)
  else:
    return (result.x, None)


def find_best_candidate_distribution(objective_vector,
                                     constraints_matrix,
                                     epsilon=0.0):
  """Finds a distribution minimizing an objective subject to constraints.

  This function deals with the constrained problem:

  > minimize f(w)
  > s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}

  Here, f(w) is the "objective function", and g_i(w) is the ith (of m)
  "constraint function". Given a set of n "candidate solutions"
  {w_0,w_1,...,w_{n-1}}, this function finds a distribution over these n
  candidates that, in expectation, minimizes the objective while violating
  the constraints by the smallest possible amount (with the amount being found
  via bisection search).

  The `objective_vector` parameter should be a numpy array with shape (n,), for
  which objective_vector[i] = f(w_i). Likewise, `constraints_matrix` should be a
  numpy array with shape (m,n), for which constraints_matrix[i,j] = g_i(w_j).

  This function will return a distribution for which at most m+1 probabilities,
  and often fewer, are nonzero.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  This function implements the approach described in Lemma 3.

  Args:
    objective_vector: numpy array of shape (n,), where n is the number of
      "candidate solutions". Contains the objective function values.
    constraints_matrix: numpy array of shape (m,n), where m is the number of
      constraints and n is the number of "candidate solutions". Contains the
      constraint violation magnitudes.
    epsilon: nonnegative float, the threshold at which to terminate the binary
      search while searching for the minimal expected constraint violation
      magnitude.

  Returns:
    The optimal distribution, as a numpy array of shape (n,).

  Raises:
    ValueError: If `objective_vector` and `constraints_matrix` have inconsistent
      shapes, or if `epsilon` is negative.
    ImportError: If we're unable to import `scipy.optimize`.
  """
  if epsilon < 0.0:
    raise ValueError("epsilon must be nonnegative")

  # If there is a feasible solution (i.e. with maximum_violation=0), then that's
  # what we'll return.
  pp, _ = _find_best_candidate_distribution_helper(objective_vector,
                                                   constraints_matrix)
  if pp is not None:
    return pp

  # The bound is the minimum over all candidates, of the maximum per-candidate
  # constraint violation.
  lower = 0.0
  upper = np.min(np.amax(constraints_matrix, axis=0))
  best_pp, _ = _find_best_candidate_distribution_helper(
      objective_vector, constraints_matrix, maximum_violation=upper)
  assert best_pp is not None

  # Throughout this loop, a maximum_violation of "lower" is not achievable,
  # but a maximum_violation of "upper" is achievable.
  while True:
    middle = 0.5 * (lower + upper)
    if (middle - lower <= epsilon) or (upper - middle <= epsilon):
      break
    else:
      pp, _ = _find_best_candidate_distribution_helper(
          objective_vector, constraints_matrix, maximum_violation=middle)
      if pp is None:
        lower = middle
      else:
        best_pp = pp
        upper = middle

  return best_pp


def find_best_candidate_index(objective_vector,
                              constraints_matrix,
                              rank_objectives=False):
  """Heuristically finds the best candidate solution to a constrained problem.

  This function deals with the constrained problem:

  > minimize f(w)
  > s.t. g_i(w) <= 0 for all i in {0,1,...,m-1}

  Here, f(w) is the "objective function", and g_i(w) is the ith (of m)
  "constraint function". Given a set of n "candidate solutions"
  {w_0,w_1,...,w_{n-1}}, this function finds the "best" solution according
  to the following heuristic:

    1. Across all models, the ith constraint violations (i.e. max{0, g_i(0)})
       are ranked, as are the objectives (if rank_objectives=True).
    2. Each model is then associated its MAXIMUM rank across all m constraints
       (and the objective, if rank_objectives=True).
    3. The model with the minimal maximum rank is then identified. Ties are
       broken using the objective function value.
    4. The index of this "best" model is returned.

  The `objective_vector` parameter should be a numpy array with shape (n,), for
  which objective_vector[i] = f(w_i). Likewise, `constraints_matrix` should be a
  numpy array with shape (m,n), for which constraints_matrix[i,j] = g_i(w_j).

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
  > Constrained Optimization".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  This function implements the heuristic used for hyperparameter search in the
  experiments of Section 5.2.

  Args:
    objective_vector: numpy array of shape (n,), where n is the number of
      "candidate solutions". Contains the objective function values.
    constraints_matrix: numpy array of shape (m,n), where m is the number of
      constraints and n is the number of "candidate solutions". Contains the
      constraint violation magnitudes.
    rank_objectives: bool, whether the objective function values should be
      included in the initial ranking step. If True, both the objective and
      constraints will be ranked. If False, only the constraints will be ranked.
      In either case, the objective function values will be used for
      tiebreaking.

  Returns:
    The index (in {0,1,...,n-1}) of the "best" model according to the above
      heuristic.

  Raises:
    ValueError: If `objective_vector` and `constraints_matrix` have inconsistent
      shapes.
    ImportError: If we're unable to import `scipy.stats`.
  """
  mm, nn = np.shape(constraints_matrix)
  if (nn,) != np.shape(objective_vector):
    raise ValueError(
        "objective_vector must have shape (n,), and constraints_matrix (m, n),"
        " where n is the number of candidates, and m is the number of "
        "constraints")

  # We import scipy inline, instead of at the top of the file, so that a scipy
  # dependency is only introduced if either find_best_candidate_distribution()
  # or find_best_candidate_index() are actually called.
  import scipy.stats  # pylint: disable=g-import-not-at-top

  if rank_objectives:
    maximum_ranks = scipy.stats.rankdata(objective_vector, method="min")
  else:
    maximum_ranks = np.zeros(nn, dtype=np.int64)
  for ii in xrange(mm):
    # Take the maximum of the constraint functions with zero, since we want to
    # rank the magnitude of constraint *violations*. If the constraint is
    # satisfied, then we don't care how much it's satisfied by (as a result, we
    # we expect all models satisfying a constraint to be tied at rank 1).
    ranks = scipy.stats.rankdata(
        np.maximum(0.0, constraints_matrix[ii, :]), method="min")
    maximum_ranks = np.maximum(maximum_ranks, ranks)

  best_index = None
  best_rank = float("Inf")
  best_objective = float("Inf")
  for ii in xrange(nn):
    if maximum_ranks[ii] < best_rank:
      best_index = ii
      best_rank = maximum_ranks[ii]
      best_objective = objective_vector[ii]
    elif (maximum_ranks[ii] == best_rank) and (objective_vector[ii] <=
                                               best_objective):
      best_index = ii
      best_objective = objective_vector[ii]

  return best_index
