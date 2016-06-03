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
import tensorflow as tf

# pylint: disable=g-import-not-at-top,unused-import
try:
  import __builtin__ as builtins
except ImportError:
  import builtins

try:
  import mock
except ImportError:
  try:
    import unittest.mock as mock
  except ImportError:
    # At the moment TensorFlow does not have access to mock when in Python 2.7
    # mode, although mock is part of the standard Python 3 library. If mock is
    # not available, indicate this by assigning None to it.
    mock = None
# pylint: enable=g-import-not-at-top,unused-import


class MockOptimizerInterface(tf.contrib.opt.ExternalOptimizerInterface):

  NUM_STEP_CALLS = 5
  NUM_LOSS_CALLS = 2
  NUM_GRAD_CALLS = 3

  def _minimize(self, initial_val, loss_func, loss_grad_func, step_callback,
                optimizer_kwargs, **unused_kwargs):
    """Minimize (x - x0)**2 / 2 with respect to x."""
    for _ in range(self.NUM_LOSS_CALLS):
      loss_func(initial_val)
    for _ in range(self.NUM_GRAD_CALLS - 1):
      loss_grad_func(initial_val)
    for _ in range(self.NUM_STEP_CALLS):
      step_callback(initial_val)

    return initial_val - loss_grad_func(initial_val)


class TestCase(tf.test.TestCase):

  def assertAllClose(self, array1, array2):
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
    if not array1.shape:
      array1 = np.array([array1])
    if not array2.shape:
      array2 = np.array([array2])

    super(TestCase, self).assertAllClose(array1, array2, rtol=1e-5, atol=1e-5)

  def mock_import(self, module_name):
    """Causes importing a specific module to return a mock.MagicMock instance.

    Usage:
      with mock_import('scipy'):
        import scipy  # scipy is a MagicMock.
        x = scipy.blah()[7]  # x is also a MagicMock.

    Args:
      module_name: Name of module that should be mocked.

    Returns:
      A context manager for use in a with statement.
    """
    orig_import = __import__
    mocked_module = mock.MagicMock()

    def import_mock(name, *args, **kwargs):
      if name == module_name:
        return mocked_module
      return orig_import(name, *args, **kwargs)

    return mock.patch.object(builtins, '__import__', side_effect=import_mock)


class ExternalOptimizerInterfaceTest(TestCase):

  def test_optimize(self):
    scalar = tf.Variable(tf.random_normal([]), 'scalar')
    vector = tf.Variable(tf.random_normal([2]), 'vector')
    matrix = tf.Variable(tf.random_normal([2, 3]), 'matrix')

    minimum_location = tf.constant(np.arange(9), dtype=tf.float32)

    loss = tf.reduce_sum(tf.square(vector - minimum_location[:2])) / 2.
    loss += tf.reduce_sum(tf.square(scalar - minimum_location[2])) / 2.
    loss += tf.reduce_sum(tf.square(
        matrix - tf.reshape(minimum_location[3:], [2, 3]))) / 2.

    optimizer = MockOptimizerInterface(loss)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())

      optimizer.minimize(sess)

      self.assertAllClose(np.arange(2), sess.run(vector))
      self.assertAllClose(np.arange(1) + 2, sess.run(scalar))
      self.assertAllClose(np.arange(6).reshape(2, 3) + 3, sess.run(matrix))

  def test_callbacks(self):
    if mock is None:
      # This test requires mock. See comment in imports section at top.
      tf.logging.warning('This test requires mock and will not be run')
      return

    vector_val = np.array([7., -2.], dtype=np.float32)
    vector = tf.Variable(vector_val, 'vector')

    minimum_location_val = np.arange(2)
    minimum_location = tf.constant(minimum_location_val, dtype=tf.float32)

    loss = tf.reduce_sum(tf.square(vector - minimum_location)) / 2.
    loss_val = ((vector_val - minimum_location_val)**2).sum() / 2.

    optimizer = MockOptimizerInterface(loss)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())

      initial_vector_val = sess.run(vector)

      extra_fetches = [loss]

      step_callback = mock.Mock()
      loss_callback = mock.Mock()
      grad_callback = mock.Mock()

      optimizer.minimize(
          sess, fetches=extra_fetches, loss_callback=loss_callback,
          grad_callback=grad_callback, step_callback=step_callback)

      call = mock.call(loss_val)
      loss_calls = [call] * MockOptimizerInterface.NUM_LOSS_CALLS
      loss_callback.assert_has_calls(loss_calls)

      grad_calls = [call] * MockOptimizerInterface.NUM_GRAD_CALLS
      grad_callback.assert_has_calls(grad_calls)

      args, _ = step_callback.call_args
      self.assertAllClose(initial_vector_val, args[0])


class ScipyOptimizerInterfaceTest(TestCase):

  def test_unconstrained(self):
    if mock is None:
      # This test requires mock. See comment in imports section at top.
      tf.logging.warning('This test requires mock and will not be run')
      return

    vector_initial_value = [7., 7.]
    vector = tf.Variable(vector_initial_value, 'vector')

    # Make norm as small as possible.
    loss = tf.reduce_sum(tf.square(vector))

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())

      with self.mock_import('scipy.optimize'):
        import scipy.optimize  # pylint: disable=g-import-not-at-top
        # scipy.optimize is now a mock.MagicMock.
        optimized_vector = np.array([1.23, -0.1])
        scipy.optimize.minimize.return_value = {'x': optimized_vector}
        optimizer.minimize(sess)

        self.assertAllClose(optimized_vector, sess.run(vector))

        self.assertEqual(1, len(scipy.optimize.minimize.mock_calls))
        call_signature = scipy.optimize.minimize.mock_calls[0]

        args = call_signature[1]
        self.assertEqual(2, len(args))
        self.assertTrue(callable(args[0]))
        self.assertAllClose(vector_initial_value, args[1])

        kwargs = call_signature[2]
        self.assertEqual(4, len(kwargs))
        self.assertEqual('L-BFGS-B', kwargs['method'])
        self.assertTrue(callable(kwargs['jac']))
        self.assertTrue(callable(kwargs['callback']))
        self.assertEqual([], kwargs['constraints'])

  def test_nonlinear_programming(self):
    if mock is None:
      # This test requires mock. See comment in imports section at top.
      tf.logging.warning('This test requires mock and will not be run')
      return

    vector_initial_value = [7., 7.]
    vector = tf.Variable(vector_initial_value, 'vector')

    # Make norm as small as possible.
    loss = tf.reduce_sum(tf.square(vector))
    # Ensure y = 1.
    equalities = [vector[1] - 1.]
    # Ensure x >= 1. Thus optimum should be at (1, 1).
    inequalities = [vector[0] - 1.]

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss, equalities=equalities, inequalities=inequalities,
        method='SLSQP')

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())

      with self.mock_import('scipy.optimize'):
        import scipy.optimize  # pylint: disable=g-import-not-at-top
        # scipy.optimize is now a mock.MagicMock.
        optimized_vector = np.array([1.23, -0.1])
        scipy.optimize.minimize.return_value = {'x': optimized_vector}

        optimizer.minimize(sess)

        self.assertAllClose(optimized_vector, sess.run(vector))

        self.assertEqual(1, len(scipy.optimize.minimize.mock_calls))
        call_signature = scipy.optimize.minimize.mock_calls[0]

        args = call_signature[1]
        self.assertEqual(2, len(args))
        self.assertTrue(callable(args[0]))
        self.assertAllClose(vector_initial_value, args[1])

        kwargs = call_signature[2]
        self.assertEqual(3, len(kwargs))
        self.assertEqual('SLSQP', kwargs['method'])
        self.assertTrue(callable(kwargs['jac']))
        # No callback keyword arg since SLSQP doesn't support it.

        constraints = kwargs['constraints']
        self.assertEqual(2, len(constraints))

        eq_constraint = constraints[0]
        self.assertEqual(3, len(eq_constraint))
        self.assertEqual('eq', eq_constraint['type'])
        self.assertTrue(callable(eq_constraint['fun']))
        self.assertTrue(callable(eq_constraint['jac']))

        ineq_constraint = constraints[1]
        self.assertEqual(3, len(ineq_constraint))
        self.assertEqual('ineq', ineq_constraint['type'])
        self.assertTrue(callable(ineq_constraint['fun']))
        self.assertTrue(callable(ineq_constraint['jac']))


if __name__ == '__main__':
  tf.test.main()
