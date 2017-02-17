# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for imperative mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.imperative import imperative_graph
from tensorflow.contrib.imperative import imperative_mode
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.training import training

FLAGS = flags.FLAGS


class ImperativeTest(test.TestCase):

  def setUp(self):
    self._server = training.Server.create_local_server()
    self._target = self._server.target

  def testBasic(self):
    """Tests basic functionality.

    Fetching the value of `d` with `d.value` will evaluate `c` again
    in non-imperative mode. However, in imperative mode, `c` should
    have the value it had when it was first evaluated with `c.value`.
    """
    with imperative_mode.ImperativeMode(self._target):
      a = random_ops.random_normal([])
      b = random_ops.random_normal([])
      c = a + b
      c_val = c.value
      d = c + 1.0
      d_val = d.value
      self.assertAllClose(c_val + 1.0, d_val)

  def testExpGrad(self):
    """Tests gradients."""
    with imperative_mode.ImperativeMode(self._target):
      x = variables.Variable(np.random.rand(1, 3))
      x_init = x.value().value
      y = math_ops.exp(x)
      dy = gradients_impl.gradients(y, x)
      self.assertAllClose(np.exp(x_init), y.value)
      # dy/dx should be equal to y (= exp(x))
      self.assertAllClose(y.value, dy[0].value)

  def testLoopGrads(self):
    """Tests gradients in the presence of Python loops."""
    with imperative_mode.ImperativeMode(self._target):
      w = variables.Variable(np.eye(3))
      x = constant_op.constant(np.eye(3))

      for _ in range(3):
        x = math_ops.add(x, w)

      y = gradients_impl.gradients(x, w)
      self.assertAllClose(y[0].value, np.array([3.] * 9).reshape(3, 3))

  def testVariable(self):
    """Makes sure that variables can be evaluated before running initializer."""
    with imperative_mode.ImperativeMode(self._target):
      x = variables.Variable(1, name='xy')
      self.assertEqual(x.value().value, 1)
      x = x.assign_add(41)
      self.assertEqual(x.value, 1 + 41)
      y = variables.Variable(3, name='y')
      self.assertEqual(y.value().value, 3)

  def testNewStep(self):
    """Tests the `new_step` functionality."""
    with imperative_mode.ImperativeMode(self._target) as mode:
      for _ in range(4):
        with mode.new_step() as step:
          a = random_ops.random_uniform([])
          a_init = a.value
          for _ in range(4):
            with step.new_step():
              # Values coming from outside this step's scope should not
              # be changing.
              self.assertEqual(a.value, a_init)
              b = a + random_ops.random_uniform([], minval=0.1)
              self.assertGreaterEqual(b.value, a.value)

  def testGradientThroughNewStep(self):
    with imperative_mode.ImperativeMode(self._target) as mode:
      x = constant_op.constant(np.random.rand(3))
      y = math_ops.tanh(x)

      with mode.new_step():
        z = constant_op.constant(np.random.rand(3))
        w = math_ops.multiply(y, z)
        dx = gradients_impl.gradients(w, x)
        self.assertAllClose(dx[0].value, z.value * (1.0 - y.value ** 2))

  def testEscape(self):
    """Makes sure that values don't escape a `new_step` scope."""
    with imperative_mode.ImperativeMode(self._target) as mode:
      x = constant_op.constant(1)
      with mode.new_step():
        y = math_ops.add(x, constant_op.constant(3))
        self.assertEqual(y.value, 4)
      with mode.new_step():
        with imperative_graph.add_session_attr(ops.Tensor, None):
          with self.assertRaises(KeyError):
            _ = y + constant_op.constant(1)

  def testZeroSized(self):
    """Tests evaluating zero-sized tensors."""
    with imperative_mode.ImperativeMode(self._target):
      x = constant_op.constant(1)
      y = array_ops.shape(x)
      self.assertEqual(list(y.value), [])

  def testTrainingLoop(self):
    with imperative_mode.ImperativeMode(self._target) as mode:
      w = variables.Variable(np.random.rand(3))
      x = constant_op.constant(np.random.rand(3))
      y = math_ops.multiply(x, w)
      dw = gradients_impl.gradients(y, w)
      self.assertAllClose(dw[0].value, x.value)

      for _ in range(3):
        with mode.new_step():
          x = constant_op.constant(np.random.rand(3))
          y = math_ops.multiply(x, w)
          dw = gradients_impl.gradients(y, w)
          self.assertAllClose(dw[0].value, x.value)

  def testUseAfterNewStep(self):
    with imperative_mode.ImperativeMode(self._target) as mode:
      x = constant_op.constant(1)
      self.assertAllClose(x.value, 1)
      with mode.new_step():
        pass
      self.assertAllClose(x.value, 1)

  def testStringify(self):
    with imperative_mode.ImperativeMode(self._target):
      np_a = np.random.rand(2, 2)
      a = constant_op.constant(np_a)
      self.assertEqual(str(a), str(np_a))

  def testBoolCoercion(self):
    with imperative_mode.ImperativeMode(self._target):
      self.assertFalse(not constant_op.constant([1.0]))
      with self.assertRaises(ValueError) as ve:
        _ = not constant_op.constant(np.random.rand(2))
      self.assertTrue('The truth value of an array with'
                      ' more than one element is ambiguous.'
                      ' Use a.any() or a.all()' in str(ve.exception))

  def testMeanGrad(self):
    with imperative_mode.ImperativeMode(self._target):
      x = constant_op.constant([1.0, 2.0])
      y = math_ops.reduce_mean(x)
      dy = gradients_impl.gradients(y, x)[0]
      self.assertAllEqual(dy.value, [0.5, 0.5])

  def testVarUseInNewStep(self):
    with imperative_mode.ImperativeMode(self._target) as mode:
      x = variables.Variable(1.0)
      with mode.new_step():
        self.assertEqual(array_ops.identity(x).value, 1.0)

  def testVarChange(self):
    with imperative_mode.ImperativeMode(self._target) as mode:
      x = variables.Variable(constant_op.constant(1.0))
      for i in range(10):
        with mode.new_step() as step:
          step.run(state_ops.assign_sub(x, 0.1))
          self.assertAllClose(array_ops.identity(x).value, 1.0 - (i + 1) * 0.1)


if __name__ == '__main__':
  FLAGS.rpc_default_rate_acl = 'INSECURE'
  test.main()
