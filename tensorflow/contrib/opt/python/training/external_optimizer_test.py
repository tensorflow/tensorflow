# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for external_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.opt.python.training import external_optimizer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# pylint: disable=g-import-not-at-top,unused-import
try:
  import __builtin__ as builtins
except ImportError:
  import builtins


class MockOptimizerInterface(external_optimizer.ExternalOptimizerInterface):

  NUM_STEP_CALLS = 5
  NUM_LOSS_CALLS = 2

  def _minimize(self, initial_val, loss_grad_func, step_callback,
                optimizer_kwargs, **unused_kwargs):
    """Minimize (x - x0)**2 / 2 with respect to x."""
    for _ in range(self.NUM_LOSS_CALLS):
      loss_grad_func(initial_val)
    for _ in range(self.NUM_STEP_CALLS):
      step_callback(initial_val)

    _, grad = loss_grad_func(initial_val)
    return initial_val - grad


class TestCase(test.TestCase):

  def assertAllClose(self, array1, array2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    if not array1.shape:
      array1 = np.array([array1])
    if not array2.shape:
      array2 = np.array([array2])

    super(TestCase, self).assertAllClose(array1, array2, rtol=1e-5, atol=1e-5)


class ExternalOptimizerInterfaceTest(TestCase):

  def test_optimize(self):
    scalar = variables.Variable(random_ops.random_normal([]), 'scalar')
    vector = variables.Variable(random_ops.random_normal([2]), 'vector')
    matrix = variables.Variable(random_ops.random_normal([2, 3]), 'matrix')

    minimum_location = constant_op.constant(np.arange(9), dtype=dtypes.float32)

    loss = math_ops.reduce_sum(math_ops.square(vector -
                                               minimum_location[:2])) / 2.
    loss += math_ops.reduce_sum(math_ops.square(scalar - minimum_location[
        2])) / 2.
    loss += math_ops.reduce_sum(
        math_ops.square(matrix - array_ops.reshape(minimum_location[3:],
                                                   [2, 3]))) / 2.

    optimizer = MockOptimizerInterface(loss)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      optimizer.minimize(sess)

      self.assertAllClose(np.arange(2), sess.run(vector))
      self.assertAllClose(np.arange(1) + 2, sess.run(scalar))
      self.assertAllClose(np.arange(6).reshape(2, 3) + 3, sess.run(matrix))

  def test_callbacks(self):
    vector_val = np.array([7., -2.], dtype=np.float32)
    vector = variables.Variable(vector_val, 'vector')

    minimum_location_val = np.arange(2)
    minimum_location = constant_op.constant(
        minimum_location_val, dtype=dtypes.float32)

    loss = math_ops.reduce_sum(math_ops.square(vector - minimum_location)) / 2.
    loss_val = ((vector_val - minimum_location_val)**2).sum() / 2.

    optimizer = MockOptimizerInterface(loss)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      initial_vector_val = sess.run(vector)

      extra_fetches = [loss]

      step_callback = test.mock.Mock()
      loss_callback = test.mock.Mock()

      optimizer.minimize(
          sess,
          fetches=extra_fetches,
          loss_callback=loss_callback,
          step_callback=step_callback)

      call = test.mock.call(loss_val)
      loss_calls = [call] * MockOptimizerInterface.NUM_LOSS_CALLS
      loss_callback.assert_has_calls(loss_calls)

      args, _ = step_callback.call_args
      self.assertAllClose(initial_vector_val, args[0])


class ScipyOptimizerInterfaceTest(TestCase):

  def test_unconstrained(self):

    def objective(x):
      """Rosenbrock function. (Carl Edward Rasmussen, 2001-07-21).

      f(x) = sum_{i=1:D-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2

      Args:
        x: a Variable
      Returns:
        f: a tensor (objective value)
      """

      d = array_ops.size(x)
      s = math_ops.add(
          100 * math_ops.square(
              math_ops.subtract(
                  array_ops.strided_slice(x, [1], [d]),
                  math_ops.square(array_ops.strided_slice(x, [0], [d - 1])))),
          math_ops.square(
              math_ops.subtract(1.0, array_ops.strided_slice(x, [0], [d - 1]))))
      return math_ops.reduce_sum(s)

    dimension = 5
    x = variables.Variable(array_ops.zeros(dimension))
    optimizer = external_optimizer.ScipyOptimizerInterface(objective(x))

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      optimizer.minimize(sess)

      self.assertAllClose(np.ones(dimension), sess.run(x))

  def test_nonlinear_programming(self):
    vector_initial_value = [7., 7.]
    vector = variables.Variable(vector_initial_value, 'vector')

    # Make norm as small as possible.
    loss = math_ops.reduce_sum(math_ops.square(vector))
    # Ensure y = 1.
    equalities = [vector[1] - 1.]
    # Ensure x >= 1. Thus optimum should be at (1, 1).
    inequalities = [vector[0] - 1.]

    optimizer = external_optimizer.ScipyOptimizerInterface(
        loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      optimizer.minimize(sess)
      self.assertAllClose(np.ones(2), sess.run(vector))


if __name__ == '__main__':
  test.main()
