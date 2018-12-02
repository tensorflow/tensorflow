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
"""Contains helpers used by tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.constrained_optimization.python import constrained_minimization_problem

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops


class ConstantMinimizationProblem(
    constrained_minimization_problem.ConstrainedMinimizationProblem):
  """A `ConstrainedMinimizationProblem` with constant constraint violations.

  This minimization problem is intended for use in performing simple tests of
  the Lagrange multiplier (or equivalent) update in the optimizers. There is a
  one-element "dummy" model parameter, but it should be ignored.
  """

  def __init__(self, constraints):
    """Constructs a new `ConstantMinimizationProblem'.

    Args:
      constraints: 1d numpy array, the constant constraint violations.

    Returns:
      A new `ConstantMinimizationProblem'.
    """
    # We make an fake 1-parameter linear objective so that we don't get a "no
    # variables to optimize" error.
    self._objective = standard_ops.Variable(0.0, dtype=dtypes.float32)
    self._constraints = standard_ops.constant(constraints, dtype=dtypes.float32)

  @property
  def objective(self):
    """Returns the objective function."""
    return self._objective

  @property
  def constraints(self):
    """Returns the constant constraint violations."""
    return self._constraints
