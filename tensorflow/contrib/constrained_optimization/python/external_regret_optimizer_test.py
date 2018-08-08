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
"""Tests for constrained_optimization.python.external_regret_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.constrained_optimization.python import external_regret_optimizer
from tensorflow.contrib.constrained_optimization.python import test_util

from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class AdditiveExternalRegretOptimizerWrapper(
    external_regret_optimizer.AdditiveExternalRegretOptimizer):
  """Testing wrapper class around AdditiveExternalRegretOptimizer.

  This class is identical to AdditiveExternalRegretOptimizer, except that it
  caches the internal optimization state when _lagrange_multipliers() is called,
  so that we can test that the Lagrange multipliers take on their expected
  values.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               maximum_multiplier_radius=None):
    """Same as AdditiveExternalRegretOptimizer.__init__."""
    super(AdditiveExternalRegretOptimizerWrapper, self).__init__(
        optimizer=optimizer,
        constraint_optimizer=constraint_optimizer,
        maximum_multiplier_radius=maximum_multiplier_radius)
    self._cached_lagrange_multipliers = None

  @property
  def lagrange_multipliers(self):
    """Returns the cached Lagrange multipliers."""
    return self._cached_lagrange_multipliers

  def _lagrange_multipliers(self, state):
    """Caches the internal state for testing."""
    self._cached_lagrange_multipliers = super(
        AdditiveExternalRegretOptimizerWrapper,
        self)._lagrange_multipliers(state)
    return self._cached_lagrange_multipliers


class ExternalRegretOptimizerTest(test.TestCase):

  def test_project_multipliers_wrt_euclidean_norm(self):
    """Tests Euclidean projection routine on some known values."""
    multipliers1 = standard_ops.constant([-0.1, -0.6, -0.3])
    expected_projected_multipliers1 = np.array([0.0, 0.0, 0.0])

    multipliers2 = standard_ops.constant([-0.1, 0.6, 0.3])
    expected_projected_multipliers2 = np.array([0.0, 0.6, 0.3])

    multipliers3 = standard_ops.constant([0.4, 0.7, -0.2, 0.5, 0.1])
    expected_projected_multipliers3 = np.array([0.2, 0.5, 0.0, 0.3, 0.0])

    with self.test_session() as session:
      projected_multipliers1 = session.run(
          external_regret_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers1, 1.0))
      projected_multipliers2 = session.run(
          external_regret_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers2, 1.0))
      projected_multipliers3 = session.run(
          external_regret_optimizer._project_multipliers_wrt_euclidean_norm(
              multipliers3, 1.0))

    self.assertAllClose(
        expected_projected_multipliers1,
        projected_multipliers1,
        rtol=0,
        atol=1e-6)
    self.assertAllClose(
        expected_projected_multipliers2,
        projected_multipliers2,
        rtol=0,
        atol=1e-6)
    self.assertAllClose(
        expected_projected_multipliers3,
        projected_multipliers3,
        rtol=0,
        atol=1e-6)

  def test_additive_external_regret_optimizer(self):
    """Tests that the Lagrange multipliers update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    optimizer = AdditiveExternalRegretOptimizerWrapper(
        gradient_descent.GradientDescentOptimizer(1.0),
        maximum_multiplier_radius=1.0)
    train_op = optimizer.minimize_constrained(minimization_problem)

    expected_multipliers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.4]),
        np.array([0.7, 0.0, 0.3]),
        np.array([0.8, 0.0, 0.2]),
        np.array([0.9, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    multipliers = []
    with self.test_session() as session:
      session.run(standard_ops.global_variables_initializer())
      while len(multipliers) < len(expected_multipliers):
        multipliers.append(session.run(optimizer.lagrange_multipliers))
        session.run(train_op)

    for expected, actual in zip(expected_multipliers, multipliers):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)


if __name__ == "__main__":
  test.main()
