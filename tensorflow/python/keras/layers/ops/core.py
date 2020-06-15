# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Stateless ops for core Keras layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops


# TODO(b/157913406): Expose this publicly.
def dense(inputs, kernel, bias=None, activation=None, dtype=None):
  """Densely connected NN layer op.

  Arguments:
    inputs: `tf.Tensor` or `tf.SparseTensor`. Inputs to operation.
    kernel: `tf.Variable`. Matrix kernel.
    bias: (Optional) `tf.Variable`. Bias to add to outputs.
    activation: (Optional) 1-argument callable. Activation function to apply to
      outputs.
    dtype: (Optional) `tf.DType`. Dtype to cast `inputs` to.

  Returns:
    `tf.Tensor`. Output of dense connection.
  """
  if dtype:
    if inputs.dtype.base_dtype != dtype.base_dtype:
      inputs = math_ops.cast(inputs, dtype=dtype)

  rank = inputs.shape.rank
  if rank == 2 or rank is None:
    if isinstance(inputs, sparse_tensor.SparseTensor):
      outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
    else:
      outputs = gen_math_ops.mat_mul(inputs, kernel)
  # Broadcast kernel to inputs.
  else:
    outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
    # Reshape the output back to the original ndim of the input.
    if not context.executing_eagerly():
      shape = inputs.shape.as_list()
      output_shape = shape[:-1] + [kernel.shape[-1]]
      outputs.set_shape(output_shape)

  if bias is not None:
    outputs = nn_ops.bias_add(outputs, bias)

  if activation is not None:
    outputs = activation(outputs)

  return outputs
