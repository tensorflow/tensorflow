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

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
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

    shifted_logits = logits - \
        math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

    # sum over support
    support = math_ops.cast(sparsemax > 0, sparsemax.dtype)
    sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - shifted_logits)

    return math_ops.reduce_sum(sum_s + q_part, axis=1)
