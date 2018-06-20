# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental library that exposes XLA operations directly in TensorFlow.

It is sometimes useful to be able to build HLO programs directly from
TensorFlow. This file provides Tensorflow operators that map as closely as
possible to HLO operators.

There is no promise of backward or forward compatibility for operators defined
in this module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tf2xla.ops import gen_xla_ops

# TODO(phawkins): provide wrappers for all XLA operators.

dynamic_update_slice = gen_xla_ops.xla_dynamic_update_slice


def reduce_window(operand,
                  init,
                  reducer,
                  window_dimensions,
                  window_strides=None,
                  padding=None,
                  name=None):
  """Wraps the XLA ReduceWindow operator.

  ReduceWindow is documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    operand: the input tensor
    init: a scalar tensor representing the initial value for the reduction
    reducer: a reduction function that combines a pair of scalars.
    window_dimensions: shape of the window, as a list of integers
    window_strides: inter-window strides, as a list of integers. Optional;
      if omitted, defaults to strides of 1.
    padding: padding to apply to 'operand'. List of (low, high) pairs of
      integers that specify the padding to apply before and after each
      dimension. Optional; if omitted, defaults to no padding.
    name: the operator name, or None.
  Returns:
    A tensor that represents the output of the reduce_window operator.
  """
  window_strides = window_strides or [1] * len(window_dimensions)
  padding = padding or [(0, 0)] * len(window_dimensions)
  padding_low = [x for (x, _) in padding]
  padding_high = [y for (_, y) in padding]
  return gen_xla_ops.xla_reduce_window(
      operand,
      init,
      reducer,
      window_dimensions,
      window_strides,
      padding_low,
      padding_high,
      name=name)


recv = gen_xla_ops.xla_recv
send = gen_xla_ops.xla_send

while_loop = gen_xla_ops.xla_while
