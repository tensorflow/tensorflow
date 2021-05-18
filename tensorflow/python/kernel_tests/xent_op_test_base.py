# Copyright 2015-2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SoftmaxCrossEntropyWithLogits op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentOpTestBase(test.TestCase):

  def _opDeterminismEnabled(self):
    deterministic_ops = os.getenv('TF_DETERMINISTIC_OPS', '0')
    return deterministic_ops == '1' or deterministic_ops == 'true'

  def _opFwdBwd(self, labels, logits, axis=-1):
    """ Runs the op-under-test forward and backwards."""
    logits = ops.convert_to_tensor(logits) # needed for the gradient tape
    with backprop.GradientTape() as tape:
      tape.watch(logits)
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, dim=axis)
    return loss, tape.gradient(loss, logits)

  def _npXent(self, labels, logits, dim=-1):
    if dim == -1:
      dim = len(logits.shape) - 1
    one_only_on_dim = list(logits.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(
        logits - np.reshape(np.amax(logits, axis=dim), one_only_on_dim))
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = (probs - labels)
    l = -np.sum(labels * np.log(probs + 1.0e-20), axis=dim)
    return l, bp

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testXent2D(self,
                  np_labels,
                  np_logits,
                  with_placeholders=False,
                  test_backprop=True):
    np_loss, np_backprop = self._npXent(labels=np_labels, logits=np_logits)
    with self.cached_session(use_gpu=True) as sess:
      if with_placeholders:
        logits_placeholder = array_ops.placeholder(np_logits.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, backprop = self._opFwdBwd(
            labels_placeholder, logits_placeholder)
        tf_loss, tf_backprop = sess.run([loss, backprop],
                                        feed_dict={
                                            labels_placeholder: np_labels,
                                            logits_placeholder: np_logits
                                        })
      else:
        loss, backprop = self._opFwdBwd(np_labels, np_logits)
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=1e-2)
    if test_backprop:
      self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testXentND(self, np_labels, np_logits, dim=-1):
    np_loss, _ = self._npXent(np_labels, np_logits, dim=dim)
    with test_util.device(use_gpu=True):
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=np_labels, logits=np_logits, dim=dim)
      tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testSingleClass(self, test_backprop=True):
    for dtype in np.float16, np.float32:
      with test_util.device(use_gpu=True):
        loss, backprop = self._opFwdBwd(
            labels=np.array([[-1.], [0.], [1.]]).astype(dtype),
            logits=np.array([[1.], [-1.], [0.]]).astype(dtype))
        tf_loss, tf_backprop = self.evaluate([loss, backprop])
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      if test_backprop:
        self.assertAllClose([[2.0], [1.0], [0.0]], tf_backprop)

  def testSingleClass(self):
    """This method is structured to be easily overridden by a child class."""
    self._testSingleClass()

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with soft targets (1, 2).
    logits = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [[0., 0., 0., 1.], [0., .5, .5, 0.]]

    # For batch 0, we expect the uniform distribution: 0.25, 0.25, 0.25, 0.25
    # With a hard target 3, the backprop is [0.25, 0.25, 0.25, -0.75]
    # The loss for this batch is -log(0.25) = 1.386
    #
    # For batch 1, we have:
    # exp(0) = 1
    # exp(1) = 2.718
    # exp(2) = 7.389
    # exp(3) = 20.085
    # SUM = 31.192
    # So we have as probabilities:
    # exp(0) / SUM = 0.032
    # exp(1) / SUM = 0.087
    # exp(2) / SUM = 0.237
    # exp(3) / SUM = 0.644
    # With a soft target (1, 2), the backprop is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_backprop = self._npXent(np.array(labels), np.array(logits))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [0.0321, -0.4129, -0.2632,
                                              0.6439]]),
        np_backprop,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 1.9401]), np_loss, rtol=1.e-3, atol=1.e-3)

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  @test_util.run_deprecated_v1
  def _testLabelsBroadcast(self, test_backprop=True):
    labels = np.array([[0., 0., 0., 1.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True,
                     test_backprop=test_backprop)
    labels = np.array([[0.], [2.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True,
                     test_backprop=test_backprop)

  def testLabelsBroadcast(self):
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsBroadcast()

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._opFwdBwd(labels=[[0., 1., 0.], [1., 0., 0.]],
                       logits=[[0., 1.], [2., 3.]])

  def testHalf(self):
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits)

  def testFloat(self):
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32)
    self._testXent2D(labels, logits)

  def testDouble(self):
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float64)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64)
    self._testXent2D(labels, logits)

  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session(use_gpu=True) as sess:
      labels = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="xent")
      err = gradient_checker.compute_gradient_error(logits, [3, 4], x, [3])

      # Check that no extra computation gets performed. When only the first
      # derivative is requested, the second derivative must not be computed.
      # So when there is no second derivative, there is no `BatchMatMul` op
      # in the graph.
      op_names = [
          op.op_def.name for op in sess.graph.get_operations() if op.op_def
      ]
      self.assertNotIn("BatchMatMul", op_names)
      self.assertNotIn("BatchMatMulV2", op_names)

    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testGradientLabelWithV2(self):
    with self.cached_session(use_gpu=True):
      labels = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits, name="xent")
      err = gradient_checker.compute_gradient_error(labels, [3, 4], x, [3])

    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testSecondGradient(self):
    with self.cached_session(use_gpu=True) as sess:
      labels = constant_op.constant(
          [
              0.0, 0.0, 1.0 / 3, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.5 / 3,
              0.0, 0.5 / 3
          ],
          shape=[12],
          dtype=dtypes.float64,
          name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[12],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="xent")
      loss = math_ops.reduce_sum(x)

      gradients = gradients_impl.gradients(loss, [logits])[0]

      err = gradient_checker.compute_gradient_error(
          logits, [12], gradients, [12])

      if not self._opDeterminismEnabled():
        # Check how second derivative is calculated.
        # (it is equivalent to a `BatchMatMul` op being in the graph because of
        # the implementation in SoftmaxCrossEntropyWithLogitsGrad)
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertIn("BatchMatMulV2", op_names)

    self.assertLess(err, 5e-8)

  def test3D(self):
    labels = np.array([[[0., 0., 0., 1.], [0., 1., 0., 0.]],
                       [[0., 0.5, 0.5, 0.], [0.5, 0.5, 0., 0.]],
                       [[0., 1., 0., 0.], [0., 0., 1., 0.]]]).astype(
                         np.float32)
    logits = np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                        [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                        [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(
                         np.float32)
    self._testXentND(labels, logits, dim=0)
    self._testXentND(labels, logits, dim=1)
    self._testXentND(labels, logits, dim=-1)

  def testZeroDimension(self):
    labels = np.zeros([0, 2, 4]).astype(np.float32)
    logits = np.zeros([0, 2, 4]).astype(np.float32)
    np_loss, _ = self._npXent(labels=labels, logits=logits)
    loss = nn_ops.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    tf_loss = self.evaluate(loss)
    self.assertAllEqual(np_loss, tf_loss)


if __name__ == "__main__":
  test.main()
