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
"""Tests for SparseSoftmaxCrossEntropyWithLogits op."""

import numpy as np

from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseXentOpTestBase(test.TestCase):

  def _opFwdBwd(self, labels, logits):
    """Runs the op-under-test both forwards and backwards"""
    logits = ops_lib.convert_to_tensor(logits)  # needed for the gradient tape
    with backprop_lib.GradientTape() as tape:
      tape.watch(logits)
      loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits)
    return loss, tape.gradient(loss, logits)

  def _npXent(self, labels, logits):
    logits = np.reshape(logits, [-1, logits.shape[-1]])
    labels = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = logits.shape[batch_dim]
    e = np.exp(logits -
               np.reshape(np.amax(logits, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels] = 1.0
    gradient = (probs - labels_mat)
    loss = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    return loss, gradient

  def _testXent(self, np_labels, np_logits):
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    tf_loss, tf_gradient = self._opFwdBwd(labels=np_labels, logits=np_logits)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def testSingleClass(self):
    for label_dtype in np.int32, np.int64:
      tf_loss, tf_gradient = self._opFwdBwd(
          labels=np.array([0, 0, 0]).astype(label_dtype),
          logits=np.array([[1.], [-1.], [0.]]).astype(np.float32))
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[0.0], [0.0], [0.0]], tf_gradient)

  @test_util.run_gpu_only()
  def _testInvalidLabelGPU(self, invalid_label_gradient=np.nan):
    labels = [4, 3, 0, -1]
    logits = [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 2., 3., 4.],
              [1., 2., 3., 4.]]
    loss, gradient = self._opFwdBwd(labels=labels, logits=logits)
    self.assertAllClose([np.nan, 1.3862, 3.4420, np.nan],
                        loss,
                        rtol=1e-3,
                        atol=1e-3)
    self.assertAllClose(
        [[invalid_label_gradient] * 4, [0.25, 0.25, 0.25, -0.75],
         [-0.968, 0.087, 0.237, 0.6439], [invalid_label_gradient] * 4],
        gradient,
        rtol=1e-3,
        atol=1e-3)

  def testInvalidLabelGPU(self):
    """This method is structured to be easily overridden by a child class."""
    self._testInvalidLabelGPU()

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  @test_util.disable_xla("XLA cannot assert inside of a kernel.")
  def _testInvalidLabelCPU(self, expected_regex="Received a label value of"):
    labels = [4, 3, 0, -1]
    logits = [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 2., 3., 4.],
              [1., 2., 3., 4.]]
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, errors_impl.UnknownError),
        expected_regex):
      self.evaluate(
          nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
              labels=labels, logits=logits))

  def testInvalidLabelCPU(self):
    """This method is structured to be easily overridden by a child class."""
    self._testInvalidLabelCPU()

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with target 0.
    labels = [3, 0]
    logits = [[1., 1., 1., 1.], [1., 2., 3., 4.]]

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
    # With a hard 1, the gradient is [0.032 - 1.0 = -0.968, 0.087, 0.237, 0.644]
    # The loss for this batch is [1.0 * -log(0.25), 1.0 * -log(0.032)]
    # = [1.3862, 3.4420]
    np_loss, np_gradient = self._npXent(
        labels=np.array(labels), logits=np.array(logits))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [-0.968, 0.087, 0.237, 0.6439]]),
        np_gradient,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 3.4420]), np_loss, rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    with self.assertRaisesRegex(
        ValueError, "`labels.shape.rank` must equal `logits.shape.rank - 1`"):
      nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=[[0, 2]], logits=[[0., 1.], [2., 3.], [2., 3.]])

  def testScalar(self):
    with self.assertRaisesRegex(ValueError, "`logits` cannot be a scalar"):
      nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=constant_op.constant(0), logits=constant_op.constant(1.0))

  def _testLabelsPlaceholderScalar(self, expected_error_message):
    with ops_lib.Graph().as_default(), self.session():
      labels = array_ops.placeholder(np.int32)
      y = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=[[7.]])
      with self.assertRaisesOpError(expected_error_message):
        y.eval(feed_dict={labels: 0})

  def testLabelsPlaceholderScalar(self):
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsPlaceholderScalar(
        expected_error_message="labels must be 1-D")

  def testVector(self):
    loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=constant_op.constant(0), logits=constant_op.constant([1.0]))
    self.assertAllClose(0.0, loss)

  def testFloat(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float32))

  def testDouble(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([0, 3]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float64))

  def testHalf(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float16))

  def testBfloat16(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.],
                              [1., 2., 3.,
                               4.]]).astype(dtypes.bfloat16.as_numpy_dtype))

  def testEmpty(self):
    self._testXent(
        np_labels=np.zeros((0,), dtype=np.int32), np_logits=np.zeros((0, 3)))

  @test_util.run_in_graph_and_eager_modes()
  def testGradient(self):
    with self.session() as sess:
      labels = constant_op.constant([3, 0, 1], name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")

      def xent(logits):
        # gradient_checker_v2.computee_gradient doesn't take int32/int64.
        # labels must be of type int32/int64, so passing them separately here.
        return nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits, name="xent")

      analytical, numerical = gradient_checker_v2.compute_gradient(
          xent, [logits])

      if not context.executing_eagerly():
        # Check that no extra computation performed. When only first derivative
        # is requested, second derivative must not be computed. So when there is
        # no second derivative, there is no `BatchMatMul` op in the graph.
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertNotIn("BatchMatMul", op_names)
        self.assertNotIn("BatchMatMulV2", op_names)

    tol = 5e-8
    self.assertAllClose(analytical, numerical, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes()
  def testSecondGradient(self):
    with self.session() as sess:
      labels = constant_op.constant([3, 0, 1], name="labels")
      logits = constant_op.constant(
          [0.3, 0.4, 0.1, 1.2, 0.1, 1.9, 0.1, 0.7, 0.8, 0.2, 1.3, 1.3],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")

      def xent_grad(logits):
        with backprop_lib.GradientTape() as tape:
          tape.watch(logits)
          return tape.gradient(
              nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                  labels=labels, logits=logits, name="xent"), [logits])[0]

      analytical, numerical = gradient_checker_v2.compute_gradient(
          xent_grad, [logits])

      if (not context.executing_eagerly() and
          not config.is_op_determinism_enabled()):
        # Check that second derivative is calculated.
        # (it is equivalent to being `BatchMatMul` op in the graph because of
        # implementation of xentropy grad)
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertIn("BatchMatMulV2", op_names)

    tol = 5e-8
    self.assertAllClose(analytical, numerical, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes()
  def _testHighDim(self, labels, logits):
    np_loss, np_gradient = self._npXent(
        labels=np.array(labels), logits=np.array(logits))
    # manually reshape loss
    np_loss = np.reshape(np_loss, np.array(labels).shape)
    tf_loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    with backprop_lib.GradientTape() as tape:
      logits = constant_op.constant(logits)
      tape.watch(logits)
      tf_gradient = tape.gradient(
          nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
              labels=labels, logits=logits), [logits])[0]
      tf_gradient = array_ops.reshape(tf_gradient, np_gradient.shape)

    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def testHighDim(self):
    labels = [[3], [0]]
    logits = [[[1., 1., 1., 1.]], [[1., 2., 3., 4.]]]
    self._testHighDim(labels, logits)

  def testHighDim2(self):
    labels = [[3, 2], [0, 3]]
    logits = [[[1., 1., 1., 1.], [2., 2., 2., 2.]],
              [[1., 2., 3., 4.], [5., 6., 7., 8.]]]
    self._testHighDim(labels, logits)

  def _testScalarHandling(self, expected_regex):
    with ops_lib.Graph().as_default(), self.session(use_gpu=False) as sess:
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  expected_regex):
        labels = array_ops.placeholder(dtypes.int32, shape=[None, 1])
        logits = array_ops.placeholder(dtypes.float32, shape=[None, 3])
        ce = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
            labels=array_ops.squeeze(labels), logits=logits)
        labels_v2 = np.zeros((1, 1), dtype=np.int32)
        logits_v2 = np.random.randn(1, 3)
        sess.run([ce], feed_dict={labels: labels_v2, logits: logits_v2})

  def testScalarHandling(self):
    """This method is structured to be easily overridden by a child class."""
    self._testScalarHandling(expected_regex=".*labels must be 1-D.*")
