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
"""Quantized Dense module."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


class QuantizedDense(module.Module):
  """A densely-connected layer with weight quantization.

  This module acts like a standard Dense layer but simulates
  4-bit or 8-bit weight quantization using fake quantization nodes.
  """

  def __init__(self, units, bits=8, use_bias=True, name=None):
    super(QuantizedDense, self).__init__(name=name)
    self.units = int(units)
    self.bits = int(bits)
    if self.bits not in [4, 8]:
      raise ValueError("Only 4-bit and 8-bit quantization are supported.")
    self.use_bias = use_bias
    self.kernel = None
    self.bias = None

  def __call__(self, inputs):
    inputs = ops.convert_to_tensor(inputs)
    if self.kernel is None:
      last_dim = inputs.shape[-1]
      if last_dim is None:
        raise ValueError(
            "The last dimension of the inputs to `QuantizedDense` should be"
            " defined."
        )
      last_dim = int(last_dim)
      # Initialize weights with glorot uniform
      limit = math_ops.sqrt(6.0 / (last_dim + self.units))
      self.kernel = variables.Variable(
          initial_value=random_ops.random_uniform(
              [last_dim, self.units],
              minval=-limit,
              maxval=limit,
              dtype=inputs.dtype,
          ),
          name="kernel",
          trainable=True,
      )
      if self.use_bias:
        self.bias = variables.Variable(
            initial_value=array_ops.zeros(
                [
                    self.units,
                ],
                dtype=inputs.dtype,
            ),
            name="bias",
            trainable=True,
        )

    kernel = math_ops.cast(self.kernel, dtypes.float32)
    min_val = math_ops.reduce_min(kernel)
    max_val = math_ops.reduce_max(kernel)
    max_val = math_ops.maximum(max_val, min_val + 1e-5)

    quantized_kernel = array_ops.fake_quant_with_min_max_vars(
        kernel, min_val, max_val, num_bits=self.bits, narrow_range=True
    )
    quantized_kernel = math_ops.cast(quantized_kernel, inputs.dtype)

    rank = inputs.shape.rank
    if rank is not None and rank <= 2:
      outputs = math_ops.matmul(a=inputs, b=quantized_kernel)
    else:
      outputs = math_ops.tensordot(
          inputs, quantized_kernel, [[rank - 1 if rank else -1], [0]]
      )

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    return outputs
