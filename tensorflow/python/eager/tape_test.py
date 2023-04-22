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
"""Basic tests for gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Importing nn_grad for the registration functions.
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


@custom_gradient.custom_gradient
def two_outputs(a, b):
  mm = math_ops.matmul(a, b)
  r = math_ops.reduce_sum(mm)

  def grad(dmm, dr):
    return [
        math_ops.matmul(dmm, b, transpose_b=True) +
        math_ops.matmul(array_ops.ones_like(b * dr), b, transpose_b=True),
        math_ops.matmul(a, dmm, transpose_b=True) +
        math_ops.matmul(a, array_ops.ones_like(a) * dr, transpose_b=True)
    ]

  return [mm, r], grad


@custom_gradient.custom_gradient
def gradient_is_constant(x):
  result = x * x

  def grad(dr):
    return [dr]

  return result, grad


class TapeTest(test.TestCase):

  def testMultiOutput(self):

    def fn(x, y):
      c = x + y
      # Multiple outputs from split.
      d, f = array_ops.split(c, 2)
      return d + f

    a = constant_op.constant([[1., 0.], [0., 1.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    da, db = backprop.gradients_function(fn, [0, 1])(a, b)
    with context.graph_mode(), self.cached_session():
      tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
      tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
      tf_c = tf_a + tf_b
      tf_d, tf_f = array_ops.split(tf_c, 2, axis=1)
      tf_e = tf_d + tf_f
      tf_da, tf_db = gradients_impl.gradients(tf_e, [tf_a, tf_b])

      self.assertAllEqual(da, self.evaluate(tf_da))
      self.assertAllEqual(db, self.evaluate(tf_db))

  def testBasicFunctional(self):

    def forward(a, b):
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    da, = backprop.gradients_function(forward, ['a'])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)).numpy())

  def testBasicFunctionalPositionalArg(self):

    def forward(a, b):
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    da, = backprop.gradients_function(forward, [0])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)).numpy())

  def testBasicFunctionalWithValue(self):

    def forward(a, b):
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    val, (da,) = backprop.val_and_grad_function(forward, ['a'])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)))
    self.assertAllEqual(val, forward(aa, bb))

  def testTwoOutputs(self):

    def fn(x, y):
      mm, r = two_outputs(x, y)
      return r + math_ops.reduce_sum(mm)

    a = constant_op.constant([[1., 0.], [0., 1.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    da, db = backprop.gradients_function(fn, [0, 1])(a, b)
    with context.graph_mode(), self.cached_session():
      tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
      tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
      tf_mm = math_ops.matmul(tf_a, tf_b)
      tf_rr = 2 * math_ops.reduce_sum(tf_mm)
      tf_da, tf_db = gradients_impl.gradients(tf_rr, [tf_a, tf_b])

      self.assertAllEqual(da, self.evaluate(tf_da))
      self.assertAllEqual(db, self.evaluate(tf_db))

  def testGcTwoOutputs(self):

    def fn(x, y):
      return nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                             labels=y)[0]

    labels = constant_op.constant([0])
    logits = constant_op.constant([[0.0]])
    grad, = backprop.gradients_function(fn, [0])(logits, labels)
    self.assertAllEqual(grad, [[0.0]])

  def testTfTensor(self):

    def fn(x):
      return x

    t = constant_op.constant(1.0)
    g, = backprop.gradients_function(fn, [0])(t)
    self.assertAllEqual(g, 1.0)


class VariableWatcherTest(test.TestCase):

  def testBasic(self):
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0)
    with tape.VariableWatcher() as variable_watcher:
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))

  def testNonTrainableVariables(self):
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0, trainable=False)
    with tape.VariableWatcher() as variable_watcher:
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1,))

  def testMultipleScopes(self):
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0)
    with tape.VariableWatcher() as variable_watcher1:
      var1.assign_add(1.0)
      with tape.VariableWatcher() as variable_watcher2:
        var2.assign_add(2.0)

    # variable_watcher1 should see both vars and variable_watcher2 only sees
    # var2
    self.assertAllEqual(variable_watcher1.watched_variables(), (var1, var2))
    self.assertAllEqual(variable_watcher2.watched_variables(), (var2,))

  def testCreateVariables(self):
    with tape.VariableWatcher() as variable_watcher:
      var1 = variables.Variable(0.0)
      var2 = variables.Variable(1.0)
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))


if __name__ == '__main__':
  test.main()
