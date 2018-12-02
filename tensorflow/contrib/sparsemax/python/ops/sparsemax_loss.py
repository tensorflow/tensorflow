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
"""Sparsemax Loss op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

__all__ = ["sparsemax_loss"]


def sparsemax_loss(logits, sparsemax, labels, name=None):
  """Computes sparsemax loss function [1].

  [1]: https://arxiv.org/abs/1602.02068

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    sparsemax: A `Tensor`. Must have the same type as `logits`.
    labels: A `Tensor`. Must have the same type as `logits`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  """

  with ops.name_scope(name, "sparsemax_loss",
                      [logits, sparsemax, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    sparsemax = ops.convert_to_tensor(sparsemax, name="sparsemax")
    labels = ops.convert_to_tensor(labels, name="labels")

    # In the paper, they call the logits z.
    # A constant can be substracted from logits to make the algorithm
    # more numerically stable in theory. However, there are really no major
    # source numerical instability in this algorithm.
    z = logits

    # sum over support
    # Use a conditional where instead of a multiplication to support z = -inf.
    # If z = -inf, and there is no support (sparsemax = 0), a multiplication
    # would cause 0 * -inf = nan, which is not correct in this case.
    sum_s = array_ops.where(
        math_ops.logical_or(sparsemax > 0, math_ops.is_nan(sparsemax)),
        sparsemax * (z - 0.5 * sparsemax), array_ops.zeros_like(sparsemax))

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - z)
    # Fix the case where labels = 0 and z = -inf, where q_part would
    # otherwise be 0 * -inf = nan. But since the lables = 0, no cost for
    # z = -inf should be consideredself.
    # The code below also coveres the case where z = inf. Howeverm in this
    # caose the sparsemax will be nan, which means the sum_s will also be nan,
    # therefor this case doesn't need addtional special treatment.
    q_part_safe = array_ops.where(
        math_ops.logical_and(math_ops.equal(labels, 0), math_ops.is_inf(z)),
        array_ops.zeros_like(z), q_part)

    return math_ops.reduce_sum(sum_s + q_part_safe, axis=1)
