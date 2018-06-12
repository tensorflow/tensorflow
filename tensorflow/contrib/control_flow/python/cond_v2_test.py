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

"""Tests for cond_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.control_flow.python import cond_v2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver


class NewCondTest(test.TestCase):

  def _testCond(self, true_fn, false_fn, train_vals):
    pred = array_ops.placeholder(dtypes.bool, name="pred")

    expected = control_flow_ops.cond(pred, true_fn, false_fn, name="expected")
    actual = cond_v2.cond_v2(pred, true_fn, false_fn, name="actual")

    expected_grad = gradients_impl.gradients(expected, train_vals)
    actual_grad = gradients_impl.gradients(actual, train_vals)

    with self.test_session() as sess:
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), {pred: True})
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), {pred: False})
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

  def testBasic(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return x * 2.0

    def false_fn():
      return y * 3.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testBasic2(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return x * y * 2.0

    def false_fn():
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testNoInputs(self):
    pred = array_ops.placeholder(dtypes.bool, name="pred")

    def true_fn():
      return constant_op.constant(1.0)

    def false_fn():
      return constant_op.constant(2.0)

    out = cond_v2.cond_v2(pred, true_fn, false_fn)

    with self.test_session() as sess:
      self.assertEqual(sess.run(out, {pred: True}), [1.0])
      self.assertEqual(sess.run(out, {pred: False}), [2.0])

  def testSecondDerivative(self):
    pred = array_ops.placeholder(dtypes.bool, name="pred")
    x = constant_op.constant(3.0, name="x")

    def true_fn():
      return math_ops.pow(x, 3)

    def false_fn():
      return x

    cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
    cond_grad = gradients_impl.gradients(cond, [x])
    cond_grad_grad = gradients_impl.gradients(cond_grad, [x])

    with self.test_session() as sess:
      # d[x^3]/dx = 3x^2
      true_val = sess.run(cond_grad, {pred: True})
      self.assertEqual(true_val, [27.0])
      # d[x]/dx = 1
      false_val = sess.run(cond_grad, {pred: False})
      self.assertEqual(false_val, [1.0])

      true_val = sess.run(cond_grad_grad, {pred: True})
      # d2[x^3]/dx2 = 6x
      self.assertEqual(true_val, [18.0])
      false_val = sess.run(cond_grad_grad, {pred: False})
      # d2[x]/dx2 = 0
      self.assertEqual(false_val, [0.0])

  def testGradientOfDeserializedCond(self):
    with ops.Graph().as_default():
      pred = array_ops.placeholder(dtypes.bool, name="pred")
      x = constant_op.constant(3.0, name="x")
      ops.add_to_collection("x", x)

      def true_fn():
        return math_ops.pow(x, 3)

      def false_fn():
        return x

      ops.add_to_collection("pred", pred)
      cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      for c in cond:
        ops.add_to_collection("cond", c)
      meta_graph = saver.export_meta_graph()

    with ops.Graph().as_default() as g:
      saver.import_meta_graph(meta_graph)
      x = ops.get_collection("x")[0]
      pred = ops.get_collection("pred")[0]
      cond = ops.get_collection("cond")
      cond_grad = gradients_impl.gradients(cond, [x], name="cond_grad")
      cond_grad_grad = gradients_impl.gradients(
          cond_grad, [x], name="cond_grad_grad")
      with self.test_session(graph=g) as sess:
        # d[x^3]/dx = 3x^2
        true_val = sess.run(cond_grad, {pred: True})
        self.assertEqual(true_val, [27.0])
        # d[x]/dx = 1
        false_val = sess.run(cond_grad, {pred: False})
        self.assertEqual(false_val, [1.0])

        true_val = sess.run(cond_grad_grad, {pred: True})
        # d2[x^3]/dx2 = 6x
        self.assertEqual(true_val, [18.0])
        false_val = sess.run(cond_grad_grad, {pred: False})
        # d2[x]/dx2 = 0
        self.assertEqual(false_val, [0.0])


if __name__ == "__main__":
  test.main()
