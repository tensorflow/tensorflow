# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (i.e., "LICENSE");
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
"""Tests for fused_linear_cross_entropy op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class FusedLinearCrossEntropyOpTest(test_util.TensorFlowTestCase):

  def _unfused_linear_cross_entropy(self, x, weights, labels):
    """Reference implementation using standard TF operations."""
    logits = math_ops.matmul(x, weights)
    return nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

  @test_util.run_in_graph_and_eager_modes
  def testFusedLinearCrossEntropyForward(self):
    """Test forward pass against reference unfused computation."""
    np.random.seed(42)
    batch_size = 4
    in_features = 8
    num_classes = 5

    x_val = np.random.randn(batch_size, in_features).astype(np.float32)
    w_val = np.random.randn(in_features, num_classes).astype(np.float32)
    
    # Generate one-hot or probability distribution labels
    labels_val = np.random.dirichlet(np.ones(num_classes), size=batch_size).astype(np.float32)

    x = constant_op.constant(x_val, dtype=dtypes.float32)
    w = constant_op.constant(w_val, dtype=dtypes.float32)
    labels = constant_op.constant(labels_val, dtype=dtypes.float32)

    expected_loss = self._unfused_linear_cross_entropy(x, w, labels)
    actual_loss = nn_ops.fused_linear_cross_entropy(x, w, labels)

    self.assertAllClose(expected_loss, actual_loss, rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def testFusedLinearCrossEntropyGradients(self):
    """Test backward pass gradients using tf.test.compute_gradient."""
    batch_size = 2
    in_features = 4
    num_classes = 3

    x_val = np.random.randn(batch_size, in_features).astype(np.float32)
    w_val = np.random.randn(in_features, num_classes).astype(np.float32)
    labels_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    x = constant_op.constant(x_val, dtype=dtypes.float32)
    w = constant_op.constant(w_val, dtype=dtypes.float32)
    labels = constant_op.constant(labels_val, dtype=dtypes.float32)

    def forward_fn(inputs, weights):
      return nn_ops.fused_linear_cross_entropy(inputs, weights, labels)

    # Verify gradients with respect to input X and weight matrix W
    err_x = test_util.compute_gradient_error(
        lambda x_in: forward_fn(x_in, w), [x]
    )
    err_w = test_util.compute_gradient_error(
        lambda w_in: forward_fn(x, w_in), [w]
    )

    self.assertLess(err_x, 1e-3)
    self.assertLess(err_w, 1e-3)


if __name__ == "__main__":
  test.main()
