# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""ModRelu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def modrelu(input, biases, name=None):
  """Mod ReLU.

  ModReLU is a modified ReLU that applied to complex number.
  ModReLU applies ReLU to the magnitude of complex numbers and keep the phases.
  Source: [Tunable Efficient Unitary Neural Networks (EUNN) and their
  application to RNNs. L. Jing, et al.](https://arxiv.org/abs/1612.05231)

  Args:
    input: A `Tensor` with type `complex64` or `complex128`.
    biases: A 1-D `Tensor` with size matching the last dimension of `inputs`.
      Must have type `float` or `double`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `input`.
  """
  with ops.name_scope(name, "modrelu", [input]):
    input = ops.convert_to_tensor(input, name="input")
    input_re = math_ops.real(input)
    input_im = math_ops.imag(input)
    input_norm = math_ops.sqrt(input_re * input_re + input_im * input_im)
    input_norm = input_norm + 0.000001
    zero_im = array_ops.zeros_like(input_norm)
    magnitudes = nn_ops.relu(nn_ops.bias_add(input_norm, biases))
    phases = input / math_ops.complex(input_norm, zero_im)
    return math_ops.complex(magnitudes, zero_im) * phases
