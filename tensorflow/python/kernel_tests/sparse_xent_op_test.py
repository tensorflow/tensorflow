# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import app
from tensorflow.python.platform import test


class SparseXentTest(test.TestCase):

  def _npXent(self, features, labels):
    features = np.reshape(features, [-1, features.shape[-1]])
    labels = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels] = 1.0
    bp = (probs - labels_mat)
    l = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    return l, bp

  def _testXent(self, np_features, np_labels):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.test_session(use_gpu=True) as sess:
      loss, backprop = gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
          np_features, np_labels)
      tf_loss, tf_backprop = sess.run([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def testSingleClass(self):
    for label_dtype in np.int32, np.int64:
      with self.test_session(use_gpu=True) as sess:
        loss, backprop = gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
            np.array([[1.], [-1.], [0.]]).astype(np.float32),
            np.array([0, 0, 0]).astype(label_dtype))
        tf_loss, tf_backprop = sess.run([loss, backprop])
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[0.0], [0.0], [0.0]], tf_backprop)

  def testInvalidLabel(self):
    features = [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 2., 3., 4.],
                [1., 2., 3., 4.]]
    labels = [4, 3, 0, -1]

    if test.is_built_with_cuda() and test.is_gpu_available():
      with self.test_session(use_gpu=True) as sess:
        loss, backprop = (gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
            features, labels))
        tf_loss, tf_backprop = sess.run([loss, backprop])
        self.assertAllClose(
            [[np.nan] * 4, [0.25, 0.25, 0.25, -0.75],
             [-0.968, 0.087, 0.237, 0.6439], [np.nan] * 4],
            tf_backprop,
            rtol=1e-3,
            atol=1e-3)
        self.assertAllClose(
            [np.nan, 1.3862, 3.4420, np.nan], tf_loss, rtol=1e-3, atol=1e-3)

    with self.test_session(use_gpu=False) as sess:
      loss, backprop = (gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
          features, labels))
      with self.assertRaisesOpError("Received a label value of"):
        sess.run([loss, backprop])

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with target 0.
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [3, 0]

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
    # With a hard 1, the backprop is [0.032 - 1.0 = -0.968, 0.087, 0.237, 0.644]
    # The loss for this batch is [1.0 * -log(0.25), 1.0 * -log(0.032)]
    # = [1.3862, 3.4420]
    np_loss, np_backprop = self._npXent(np.array(features), np.array(labels))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [-0.968, 0.087, 0.237, 0.6439]]),
        np_backprop,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 3.4420]), np_loss, rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    with self.test_session(use_gpu=True):
      with self.assertRaisesRegexp(ValueError, ".*Rank mismatch:*"):
        nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=[[0, 2]], logits=[[0., 1.], [2., 3.], [2., 3.]])

  def testScalar(self):
    with self.test_session(use_gpu=True):
      with self.assertRaisesRegexp(ValueError, ".*Logits cannot be scalars*"):
        nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=constant_op.constant(0), logits=constant_op.constant(1.0))

  def testLabelsPlaceholderScalar(self):
    with self.test_session(use_gpu=True):
      labels = array_ops.placeholder(np.int32)
      y = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=[[7.]])
      with self.assertRaisesOpError("labels must be 1-D"):
        y.eval(feed_dict={labels: 0})

  def testVector(self):
    with self.test_session(use_gpu=True):
      loss = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=constant_op.constant(0), logits=constant_op.constant([1.0]))
      self.assertAllClose(0.0, loss.eval())

  def testFloat(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
          np.array([3, 0]).astype(label_dtype))

  def testDouble(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
          np.array([0, 3]).astype(label_dtype))

  def testHalf(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
          np.array([3, 0]).astype(label_dtype))

  def testEmpty(self):
    self._testXent(np.zeros((0, 3)), np.zeros((0,), dtype=np.int32))

  def testGradient(self):
    with self.test_session(use_gpu=True):
      l = constant_op.constant([3, 0, 1], name="l")
      f = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="f")
      x = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=l, logits=f, name="xent")
      err = gradient_checker.compute_gradient_error(f, [3, 4], x, [3])
    print("cross entropy gradient err = ", err)
    self.assertLess(err, 5e-8)

  def _testHighDim(self, features, labels):
    np_loss, np_backprop = self._npXent(np.array(features), np.array(labels))
    # manually reshape loss
    np_loss = np.reshape(np_loss, np.array(labels).shape)
    with self.test_session(use_gpu=True) as sess:
      loss = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=features)
      backprop = loss.op.inputs[0].op.outputs[1]
      tf_loss, tf_backprop = sess.run([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def testHighDim(self):
    features = [[[1., 1., 1., 1.]], [[1., 2., 3., 4.]]]
    labels = [[3], [0]]
    self._testHighDim(features, labels)

  def testHighDim2(self):
    features = [[[1., 1., 1., 1.], [2., 2., 2., 2.]],
                [[1., 2., 3., 4.], [5., 6., 7., 8.]]]
    labels = [[3, 2], [0, 3]]
    self._testHighDim(features, labels)

  def testScalarHandling(self):
    with self.test_session(use_gpu=False) as sess:
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   ".*labels must be 1-D.*"):
        labels = array_ops.placeholder(dtypes.int32, shape=[None, 1])
        logits = array_ops.placeholder(dtypes.float32, shape=[None, 3])
        ce = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=array_ops.squeeze(labels), logits=logits)
        labels_v2 = np.zeros((1, 1), dtype=np.int32)
        logits_v2 = np.random.randn(1, 3)
        sess.run([ce], feed_dict={labels: labels_v2, logits: logits_v2})


def _sparse_vs_dense_xent_benchmark_dense(labels, logits):
  labels = array_ops.identity(labels)
  logits = array_ops.identity(logits)
  with ops_lib.device("/cpu:0"):  # Sparse-to-dense must be on CPU
    batch_size = array_ops.shape(logits)[0]
    num_entries = array_ops.shape(logits)[1]
    length = batch_size * num_entries
    labels += num_entries * math_ops.range(batch_size)
    target = sparse_ops.sparse_to_dense(labels,
                                        array_ops.stack([length]), 1.0, 0.0)
  target = array_ops.reshape(target, array_ops.stack([-1, num_entries]))
  crossent = nn_ops.softmax_cross_entropy_with_logits(
      labels=target, logits=logits, name="SequenceLoss/CrossEntropy")
  crossent_sum = math_ops.reduce_sum(crossent)
  grads = gradients_impl.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def _sparse_vs_dense_xent_benchmark_sparse(labels, logits):
  # Using sparse_softmax_cross_entropy_with_logits
  labels = labels.astype(np.int64)
  labels = array_ops.identity(labels)
  logits = array_ops.identity(logits)
  crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name="SequenceLoss/CrossEntropy")
  crossent_sum = math_ops.reduce_sum(crossent)
  grads = gradients_impl.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def sparse_vs_dense_xent_benchmark(batch_size, num_entries, use_gpu):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.3
  labels = np.random.randint(num_entries, size=batch_size).astype(np.int32)
  logits = np.random.randn(batch_size, num_entries).astype(np.float32)

  def _timer(sess, ops):
    # Warm in
    for _ in range(20):
      sess.run(ops)

    # Timing run
    start = time.time()
    for _ in range(20):
      sess.run(ops)
    end = time.time()

    return (end - start) / 20.0  # Average runtime per iteration

  # Using sparse_to_dense and softmax_cross_entropy_with_logits
  with session.Session(config=config) as sess:
    if not use_gpu:
      with ops_lib.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    delta_dense = _timer(sess, ops)

  # Using sparse_softmax_cross_entropy_with_logits
  with session.Session(config=config) as sess:
    if not use_gpu:
      with ops_lib.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    delta_sparse = _timer(sess, ops)

  print("%d \t %d \t %s \t %f \t %f \t %f" % (batch_size, num_entries, use_gpu,
                                              delta_dense, delta_sparse,
                                              delta_sparse / delta_dense))


def main(_):
  print("Sparse Xent vs. SparseToDense + Xent")
  print("batch \t depth \t gpu \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")
  for use_gpu in (False, True):
    for batch_size in (32, 64, 128):
      for num_entries in (100, 1000, 10000):
        sparse_vs_dense_xent_benchmark(batch_size, num_entries, use_gpu)
    sparse_vs_dense_xent_benchmark(32, 100000, use_gpu)
    sparse_vs_dense_xent_benchmark(8, 1000000, use_gpu)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    app.run()
  else:
    test.main()
