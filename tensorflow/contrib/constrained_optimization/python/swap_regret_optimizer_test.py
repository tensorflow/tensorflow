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
"""Tests for constrained_optimization.python.swap_regret_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.constrained_optimization.python import swap_regret_optimizer
from tensorflow.contrib.constrained_optimization.python import test_util

from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class AdditiveSwapRegretOptimizerWrapper(
    swap_regret_optimizer.AdditiveSwapRegretOptimizer):
  """Testing wrapper class around AdditiveSwapRegretOptimizer.

  This class is identical to AdditiveSwapRegretOptimizer, except that it caches
  the internal optimization state when _stochastic_matrix() is called, so that
  we can test that the stochastic matrices take on their expected values.
  """

  def __init__(self, optimizer, constraint_optimizer=None):
    """Same as AdditiveSwapRegretOptimizer.__init__()."""
    super(AdditiveSwapRegretOptimizerWrapper, self).__init__(
        optimizer=optimizer, constraint_optimizer=constraint_optimizer)
    self._cached_stochastic_matrix = None

  @property
  def stochastic_matrix(self):
    """Returns the cached stochastic matrix."""
    return self._cached_stochastic_matrix

  def _stochastic_matrix(self, state):
    """Caches the internal state for testing."""
    self._cached_stochastic_matrix = super(AdditiveSwapRegretOptimizerWrapper,
                                           self)._stochastic_matrix(state)
    return self._cached_stochastic_matrix


class MultiplicativeSwapRegretOptimizerWrapper(
    swap_regret_optimizer.MultiplicativeSwapRegretOptimizer):
  """Testing wrapper class around MultiplicativeSwapRegretOptimizer.

  This class is identical to MultiplicativeSwapRegretOptimizer, except that it
  caches the internal optimization state when _stochastic_matrix() is called, so
  that we can test that the stochastic matrices take on their expected values.
  """

  def __init__(self,
               optimizer,
               constraint_optimizer=None,
               minimum_multiplier_radius=None,
               initial_multiplier_radius=None):
    """Same as MultiplicativeSwapRegretOptimizer.__init__()."""
    super(MultiplicativeSwapRegretOptimizerWrapper, self).__init__(
        optimizer=optimizer,
        constraint_optimizer=constraint_optimizer,
        minimum_multiplier_radius=1e-3,
        initial_multiplier_radius=initial_multiplier_radius)
    self._cached_stochastic_matrix = None

  @property
  def stochastic_matrix(self):
    """Returns the cached stochastic matrix."""
    return self._cached_stochastic_matrix

  def _stochastic_matrix(self, state):
    """Caches the internal state for testing."""
    self._cached_stochastic_matrix = super(
        MultiplicativeSwapRegretOptimizerWrapper,
        self)._stochastic_matrix(state)
    return self._cached_stochastic_matrix


class SwapRegretOptimizerTest(test.TestCase):

  def test_maximum_eigenvector_power_method(self):
    """Tests power method routine on some known left-stochastic matrices."""
    matrix1 = np.matrix([[0.6, 0.1, 0.1], [0.0, 0.6, 0.9], [0.4, 0.3, 0.0]])
    matrix2 = np.matrix([[0.4, 0.4, 0.2], [0.2, 0.1, 0.5], [0.4, 0.5, 0.3]])

    with self.cached_session() as session:
      eigenvector1 = session.run(
          swap_regret_optimizer._maximal_eigenvector_power_method(
              standard_ops.constant(matrix1)))
      eigenvector2 = session.run(
          swap_regret_optimizer._maximal_eigenvector_power_method(
              standard_ops.constant(matrix2)))

    # Check that eigenvector1 and eigenvector2 are eigenvectors of matrix1 and
    # matrix2 (respectively) with associated eigenvalue 1.
    matrix_eigenvector1 = np.tensordot(matrix1, eigenvector1, axes=1)
    matrix_eigenvector2 = np.tensordot(matrix2, eigenvector2, axes=1)
    self.assertAllClose(eigenvector1, matrix_eigenvector1, rtol=0, atol=1e-6)
    self.assertAllClose(eigenvector2, matrix_eigenvector2, rtol=0, atol=1e-6)

  def test_project_stochastic_matrix_wrt_euclidean_norm(self):
    """Tests Euclidean projection routine on some known values."""
    matrix = standard_ops.constant([[-0.1, -0.1, 0.4], [-0.8, 0.4, 1.2],
                                    [-0.3, 0.1, 0.2]])
    expected_projected_matrix = np.array([[0.6, 0.1, 0.1], [0.0, 0.6, 0.9],
                                          [0.4, 0.3, 0.0]])

    with self.cached_session() as session:
      projected_matrix = session.run(
          swap_regret_optimizer._project_stochastic_matrix_wrt_euclidean_norm(
              matrix))

    self.assertAllClose(
        expected_projected_matrix, projected_matrix, rtol=0, atol=1e-6)

  def test_project_log_stochastic_matrix_wrt_kl_divergence(self):
    """Tests KL-divergence projection routine on some known values."""
    matrix = standard_ops.constant([[0.2, 0.8, 0.6], [0.1, 0.2, 1.5],
                                    [0.2, 1.0, 0.9]])
    expected_projected_matrix = np.array([[0.4, 0.4, 0.2], [0.2, 0.1, 0.5],
                                          [0.4, 0.5, 0.3]])

    with self.cached_session() as session:
      projected_matrix = session.run(
          standard_ops.exp(
              swap_regret_optimizer.
              _project_log_stochastic_matrix_wrt_kl_divergence(
                  standard_ops.log(matrix))))

    self.assertAllClose(
        expected_projected_matrix, projected_matrix, rtol=0, atol=1e-6)

  def test_additive_swap_regret_optimizer(self):
    """Tests that the stochastic matrices update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    optimizer = AdditiveSwapRegretOptimizerWrapper(
        gradient_descent.GradientDescentOptimizer(1.0))
    train_op = optimizer.minimize_constrained(minimization_problem)

    # Calculated using a numpy+python implementation of the algorithm.
    expected_matrices = [
        np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        np.array([[0.66666667, 1.0, 1.0, 1.0], [0.26666667, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0], [0.06666667, 0.0, 0.0, 0.0]]),
        np.array([[0.41666667, 0.93333333, 1.0,
                   0.98333333], [0.46666667, 0.05333333, 0.0,
                                 0.01333333], [0.0, 0.0, 0.0, 0.0],
                  [0.11666667, 0.01333333, 0.0, 0.00333333]]),
    ]

    matrices = []
    with self.cached_session() as session:
      session.run(standard_ops.global_variables_initializer())
      while len(matrices) < len(expected_matrices):
        matrices.append(session.run(optimizer.stochastic_matrix))
        session.run(train_op)

    for expected, actual in zip(expected_matrices, matrices):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)

  def test_multiplicative_swap_regret_optimizer(self):
    """Tests that the stochastic matrices update as expected."""
    minimization_problem = test_util.ConstantMinimizationProblem(
        np.array([0.6, -0.1, 0.4]))
    optimizer = MultiplicativeSwapRegretOptimizerWrapper(
        gradient_descent.GradientDescentOptimizer(1.0),
        initial_multiplier_radius=0.8)
    train_op = optimizer.minimize_constrained(minimization_problem)

    # Calculated using a numpy+python implementation of the algorithm.
    expected_matrices = [
        np.array([[0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2],
                  [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]),
        np.array([[0.36999014, 0.38528351, 0.38528351, 0.38528351], [
            0.23517483, 0.21720297, 0.21720297, 0.21720297
        ], [0.17774131, 0.18882719, 0.18882719, 0.18882719],
                  [0.21709373, 0.20868632, 0.20868632, 0.20868632]]),
        np.array([[0.33972109, 0.36811863, 0.37118462, 0.36906575], [
            0.27114826, 0.23738228, 0.23376693, 0.23626491
        ], [0.15712313, 0.17641793, 0.17858959, 0.17708679],
                  [0.23200752, 0.21808115, 0.21645886, 0.21758255]]),
    ]

    matrices = []
    with self.cached_session() as session:
      session.run(standard_ops.global_variables_initializer())
      while len(matrices) < len(expected_matrices):
        matrices.append(session.run(optimizer.stochastic_matrix))
        session.run(train_op)

    for expected, actual in zip(expected_matrices, matrices):
      self.assertAllClose(expected, actual, rtol=0, atol=1e-6)


if __name__ == '__main__':
  test.main()
