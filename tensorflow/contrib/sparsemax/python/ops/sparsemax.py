# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Sparsemax op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__all__ = ["sparsemax"]


def sparsemax(logits, name=None):
  """Computes sparsemax activations [1].

  For each batch `i` and class `j` we have
    $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$

  [1]: https://arxiv.org/abs/1602.02068

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  """

  with ops.name_scope(name, "sparsemax", [logits]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    obs = array_ops.shape(logits)[0]
    dims = array_ops.shape(logits)[1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    z = logits

    # sort z
    z_sorted, _ = nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = math_ops.cumsum(z_sorted, axis=1)
    k = math_ops.range(
        1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = math_ops.maximum(k_z, 1)
    indices = array_ops.stack([math_ops.range(0, obs), k_z_safe - 1], axis=1)
    tau_sum = array_ops.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

    # calculate p
    p = math_ops.maximum(
        math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = array_ops.where(
        math_ops.logical_or(
            math_ops.equal(k_z, 0), math_ops.is_nan(z_cumsum[:, -1])),
        array_ops.fill([obs, dims], math_ops.cast(float("nan"), logits.dtype)),
        p)

    return p_safe
