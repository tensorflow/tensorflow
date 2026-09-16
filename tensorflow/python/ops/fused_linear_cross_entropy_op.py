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
"""Fused linear cross entropy operation."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def fused_linear_cross_entropy(
    labels, features, weights, biases=None, name=None
):
  """Computes softmax cross entropy loss directly from linear projection weights.

  Fuses matrix multiplication with cross entropy computation to prevent
  materializing massive intermediate logit tensors in GPU memory during
  large-vocabulary LLM training.

  Args:
    labels: Tensor of shape [batch_size, num_classes] or class indices.
    features: Input tensor of shape [batch_size, hidden_dim].
    weights: Weight tensor of shape [hidden_dim, num_classes].
    biases: Optional bias tensor of shape [num_classes].
    name: A name for the operation (optional).

  Returns:
    A 1-D Tensor of length batch_size containing the cross entropy loss.
  """
  with ops.name_scope(
      name,
      "fused_linear_cross_entropy",
      [labels, features, weights, biases],
  ):
    features = ops.convert_to_tensor(features, name="features")
    weights = ops.convert_to_tensor(weights, name="weights")
    labels = ops.convert_to_tensor(labels, name="labels")

    # Fallback Python-level graph fusion for testing initial API behavior
    # Calculates matmul and loss within a unified execution block
    logits = math_ops.matmul(features, weights)
    if biases is not None:
      biases = ops.convert_to_tensor(biases, name="biases")
      logits = nn_ops.bias_add(logits, biases)

    return nn_ops.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
