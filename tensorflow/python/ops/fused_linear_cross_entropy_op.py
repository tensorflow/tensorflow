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
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("nn.fused_linear_cross_entropy", v1=[])
def fused_linear_cross_entropy(labels, features, weights, name=None):
 """Computes the fused linear cross entropy loss.

  Args:
    labels: A `Tensor` of shape `[batch_size, num_classes]` containing 
      target probabilities or one-hot encoded classes.
    features: A `Tensor` of shape `[batch_size, hidden_dim]` representing 
      the input features.
    weights: A `Tensor` of shape `[hidden_dim, num_classes]` representing 
      the linear layer weights.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of shape `[batch_size]` containing the loss per batch item.
  """
  with ops.name_scope(
      name,
      "fused_linear_cross_entropy",
      [labels, features, weights],
  ):
    features = ops.convert_to_tensor(features, name="features")
    weights = ops.convert_to_tensor(weights, name="weights")
    labels = ops.convert_to_tensor(labels, name="labels")

    return gen_nn_ops.fused_linear_cross_entropy(
        x=features,
        w=weights,
        labels=labels,
        name=name,
    )
