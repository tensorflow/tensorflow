# Copyright 2015 Google Inc. All Rights Reserved.
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
import tensorflow as tf

from tensorflow.python.ops import sparse_ops


class SparseXentTest(tf.test.TestCase):

  def _npXent(self, features, labels):
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features -
               np.reshape(np.amax(features, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels] = 1.0
    bp = (probs - labels_mat)
    l = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    return l, bp

  def _testXent(self, np_features, np_labels, use_gpu=False):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.test_session(use_gpu=use_gpu) as sess:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          np_features, np_labels)
      backprop = loss.op.outputs[1]
      tf_loss, tf_backprop = sess.run([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testAll(self, features, labels):
    self._testXent(features, labels, use_gpu=False)
    self._testXent(features, labels, use_gpu=True)

  def _testSingleClass(self, use_gpu=False):
    for label_dtype in np.int32, np.int64:
      with self.test_session(use_gpu=use_gpu) as sess:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            np.array([[1.], [-1.], [0.]]).astype(np.float32),
            np.array([0, 0, 0]).astype(label_dtype))
        backprop = loss.op.outputs[1]
        tf_loss, tf_backprop = sess.run([loss, backprop])
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[0.0], [0.0], [0.0]], tf_backprop)

  def testSingleClass(self):
    self._testSingleClass(use_gpu=True)
    self._testSingleClass(use_gpu=False)

  def testRankTooLarge(self):
    np_features = np.array(
        [[[1., 1., 1., 1.]], [[1., 2., 3., 4.]]]).astype(np.float32)
    np_labels = np.array([1, 2])
    self.assertRaisesRegexp(
        ValueError, "must have rank 2",
        tf.nn.sparse_softmax_cross_entropy_with_logits, np_features, np_labels)

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
    self.assertAllClose(np.array([[0.25, 0.25, 0.25, -0.75],
                                  [-0.968, 0.087, 0.237, 0.6439]]),
                        np_backprop,
                        rtol=1.e-3, atol=1.e-3)
    self.assertAllClose(np.array([1.3862, 3.4420]), np_loss,
                        rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            [[0., 1.], [2., 3.]], [[0, 2]])

  def testNotMatrix(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            [0., 1., 2., 3.], [0, 2])

  def testFloat(self):
    for label_dtype in np.int32, np.int64:
      self._testAll(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
          np.array([3, 0]).astype(label_dtype))

  def testDouble(self):
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
          np.array([0, 3]).astype(label_dtype),
          use_gpu=False)

  def testHalf(self):
    for label_dtype in np.int32, np.int64:
      self._testAll(
          np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
          np.array([3, 0]).astype(label_dtype))

  def testGradient(self):
    with self.test_session():
      l = tf.constant([3, 0, 1], name="l")
      f = tf.constant([0.1, 0.2, 0.3, 0.4,
                       0.1, 0.4, 0.9, 1.6,
                       0.1, 0.8, 2.7, 6.4], shape=[3, 4],
                      dtype=tf.float64, name="f")
      x = tf.nn.sparse_softmax_cross_entropy_with_logits(f, l, name="xent")
      err = tf.test.compute_gradient_error(f, [3, 4], x, [3])
    print("cross entropy gradient err = ", err)
    self.assertLess(err, 5e-8)


def _sparse_vs_dense_xent_benchmark_dense(labels, logits):
  labels = tf.identity(labels)
  logits = tf.identity(logits)
  with tf.device("/cpu:0"):  # Sparse-to-dense must be on CPU
    batch_size = tf.shape(logits)[0]
    num_entries = tf.shape(logits)[1]
    length = batch_size * num_entries
    labels += num_entries * tf.range(batch_size)
    target = sparse_ops.sparse_to_dense(
        labels, tf.pack([length]), 1.0, 0.0)
  target = tf.reshape(target, tf.pack([-1, num_entries]))
  crossent = tf.nn.softmax_cross_entropy_with_logits(
      logits, target, name="SequenceLoss/CrossEntropy")
  crossent_sum = tf.reduce_sum(crossent)
  grads = tf.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def _sparse_vs_dense_xent_benchmark_sparse(labels, logits):
  # Using sparse_softmax_cross_entropy_with_logits
  labels = labels.astype(np.int64)
  labels = tf.identity(labels)
  logits = tf.identity(logits)
  crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name="SequenceLoss/CrossEntropy")
  crossent_sum = tf.reduce_sum(crossent)
  grads = tf.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def sparse_vs_dense_xent_benchmark(batch_size, num_entries, use_gpu):
  config = tf.ConfigProto()
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

    return (end - start)/20.0  # Average runtime per iteration

  # Using sparse_to_dense and softmax_cross_entropy_with_logits
  with tf.Session(config=config) as sess:
    if not use_gpu:
      with tf.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    delta_dense = _timer(sess, ops)

  # Using sparse_softmax_cross_entropy_with_logits
  with tf.Session(config=config) as sess:
    if not use_gpu:
      with tf.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    delta_sparse = _timer(sess, ops)

  print(
      "%d \t %d \t %s \t %f \t %f \t %f"
      % (batch_size, num_entries, use_gpu, delta_dense, delta_sparse,
         delta_sparse/delta_dense))


def main(_):
  print("Sparse Xent vs. SparseToDense + Xent")
  print("batch \t depth \t gpu \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")
  for use_gpu in (False, True):
    for batch_size in (32, 64, 128):
      for num_entries in (100, 1000, 10000):
        sparse_vs_dense_xent_benchmark(
            batch_size, num_entries, use_gpu)
    sparse_vs_dense_xent_benchmark(
        32, 100000, use_gpu)
    sparse_vs_dense_xent_benchmark(
        8, 1000000, use_gpu)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    tf.app.run()
  else:
    tf.test.main()
