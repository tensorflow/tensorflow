# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse training operations."""

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class SparseApplyOpsTest(test.TestCase):
  """Tests for sparse training operations."""

  def _test_mismatched_dims(self, op_fn, **kwargs):
    """Verifies that the op raises InvalidArgumentError for mismatched dims."""
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "var and grad must have the same number of dimensions"):
      op_fn(**kwargs)

  @test_util.run_in_graph_and_eager_modes
  def testSparseApplyAdadeltaInvalidDimensions(self):
    """Tests ResourceSparseApplyAdadelta with mismatched dimensions."""
    var = tf.Variable(tf.random.uniform([10, 10], dtype=tf.float32))
    accum = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))
    accum_update = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))
    lr = tf.constant(0.1, dtype=tf.float32)
    rho = tf.constant(0.95, dtype=tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    grad = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)  # shape [3, 1]
    indices = tf.constant([0, 2, 4], dtype=tf.int32)

    self._test_mismatched_dims(
        tf.raw_ops.ResourceSparseApplyAdadelta,
        var=var.handle,
        accum=accum.handle,
        accum_update=accum_update.handle,
        lr=lr,
        rho=rho,
        epsilon=epsilon,
        grad=grad,
        indices=indices,
        use_locking=False)

  @test_util.run_in_graph_and_eager_modes
  def testSparseApplyProximalGradientDescentInvalidDimensions(self):
    """Tests ResourceSparseApplyProximalGradientDescent with mismatched dims."""
    var = tf.Variable(tf.random.uniform([10, 10], dtype=tf.float32))
    alpha = tf.constant(0.1, dtype=tf.float32)
    l1 = tf.constant(0.01, dtype=tf.float32)
    l2 = tf.constant(0.01, dtype=tf.float32)
    grad = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)  # shape [3, 1]
    indices = tf.constant([0, 2, 4], dtype=tf.int32)

    self._test_mismatched_dims(
        tf.raw_ops.ResourceSparseApplyProximalGradientDescent,
        var=var.handle,
        alpha=alpha,
        l1=l1,
        l2=l2,
        grad=grad,
        indices=indices,
        use_locking=False)

  @test_util.run_in_graph_and_eager_modes
  def testSparseApplyFtrlInvalidDimensions(self):
    """Tests ResourceSparseApplyFtrl with mismatched dimensions."""
    var = tf.Variable(tf.random.uniform([10, 10], dtype=tf.float32))
    accum = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))
    linear = tf.Variable(tf.zeros([10, 10], dtype=tf.float32))
    lr = tf.constant(0.1, dtype=tf.float32)
    l1 = tf.constant(0.01, dtype=tf.float32)
    l2 = tf.constant(0.01, dtype=tf.float32)
    lr_power = tf.constant(-0.5, dtype=tf.float32)
    grad = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)  # shape [3, 1]
    indices = tf.constant([0, 2, 4], dtype=tf.int32)

    self._test_mismatched_dims(
        tf.raw_ops.ResourceSparseApplyFtrl,
        var=var.handle,
        accum=accum.handle,
        linear=linear.handle,
        lr=lr,
        l1=l1,
        l2=l2,
        lr_power=lr_power,
        grad=grad,
        indices=indices,
        use_locking=False)


if __name__ == "__main__":
  test.main()
