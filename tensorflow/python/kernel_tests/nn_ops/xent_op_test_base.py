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

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentOpTestBase(test.TestCase):

  def _opFwdBwd(self, labels, logits, axis=-1):
    """ Runs the op-under-test both forwards and backwards."""
    logits = ops.convert_to_tensor(logits)  # needed for the gradient tape
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
    e = np.exp(logits - np.reshape(np.amax(logits, axis=dim), one_only_on_dim))
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
                  expected_gradient=None):
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    if expected_gradient is not None:
      np_gradient = expected_gradient
    with self.cached_session() as sess:
      if with_placeholders:
        logits_placeholder = array_ops.placeholder(np_logits.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, gradient = self._opFwdBwd(labels_placeholder, logits_placeholder)
        tf_loss, tf_gradient = sess.run([loss, gradient],
                                        feed_dict={
                                            labels_placeholder: np_labels,
                                            logits_placeholder: np_logits
                                        })
      else:
        loss, gradient = self._opFwdBwd(np_labels, np_logits)
        tf_loss, tf_gradient = self.evaluate([loss, gradient])
    self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=1e-2)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def _testXentND(self, np_labels, np_logits, dim=-1):
    np_loss, _ = self._npXent(np_labels, np_logits, dim=dim)
    loss = nn_ops.softmax_cross_entropy_with_logits(
        labels=np_labels, logits=np_logits, dim=dim)
    tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testSingleClass(self, expected_gradient=[[2.0], [1.0], [0.0], [0.0]]):
    for dtype in np.float16, np.float32, dtypes.bfloat16.as_numpy_dtype:
      loss, gradient = self._opFwdBwd(
          labels=np.array([[-1.], [0.], [1.], [1.]]).astype(dtype),
          logits=np.array([[1.], [-1.], [0.], [1.]]).astype(dtype))
      self.assertAllClose([0.0, 0.0, 0.0, 0.0], loss)
      self.assertAllClose(expected_gradient, gradient)

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
    # With a hard target 3, the gradient is [0.25, 0.25, 0.25, -0.75]
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
    # With a soft target (1, 2), the gradient is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_gradient = self._npXent(np.array(labels), np.array(logits))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [0.0321, -0.4129, -0.2632,
                                              0.6439]]),
        np_gradient,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 1.9401]), np_loss, rtol=1.e-3, atol=1.e-3)

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  @test_util.run_deprecated_v1
  def _testLabelsBroadcast(self, uniform_labels_gradient):
    labels = np.array([[0., 0., 0., 1.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[1.]]).astype(np.float16)
    logits = np.array([[1.], [2.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[0.], [2.], [0.25]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.],
                       [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(
        labels,
        logits,
        with_placeholders=True,
        expected_gradient=uniform_labels_gradient)

  def testLabelsBroadcast(self):
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsBroadcast(uniform_labels_gradient=[[
        0.25, 0.25, 0.25, 0.25
    ], [-1.968, -1.913, -1.763, -1.355], [-0.218, -0.163, -0.013, 0.394]])

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._opFwdBwd(
            labels=[[0., 1., 0.], [1., 0., 0.]], logits=[[0., 1.], [2., 3.]])

  def testHalf(self):
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits)

  def testBfloat16(self):
    labels = np.array([[0., 0., 0., 1.],
                       [0., .5, .5, 0.]]).astype(dtypes.bfloat16.as_numpy_dtype)
    logits = np.array([[1., 1., 1., 1.],
                       [1., 2., 3., 4.]]).astype(dtypes.bfloat16.as_numpy_dtype)
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
    with self.cached_session() as sess:
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
    with self.cached_session():
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
    with self.cached_session() as sess:
      labels = constant_op.constant([
          0.0, 0.0, 1.0 / 3, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.5 / 3, 0.0,
          0.5 / 3
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

      err = gradient_checker.compute_gradient_error(logits, [12], gradients,
                                                    [12])

      if not config.is_op_determinism_enabled():
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
                       [[0., 1., 0., 0.], [0., 0., 1., 0.]]]).astype(np.float32)
    logits = np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                       [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                       [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32)
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
