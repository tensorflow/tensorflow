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
"""Implementation of `head.Head` with custom loss and link function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class CustomLossHead(head_lib._RegressionHead):  # pylint: disable=protected-access
  """A Head object with custom loss function and link function."""

  def __init__(self,
               loss_fn,
               link_fn,
               logit_dimension,
               head_name=None,
               weight_column_name=None,
               metrics_fn=None):
    """`Head` for specifying arbitrary loss function.

    Args:
      loss_fn: Loss function.
      link_fn: Function that converts logits to prediction.
      logit_dimension: Number of dimensions for the logits.
      head_name: name of the head. Predictions, summary, metrics keys are
        suffixed by `"/" + head_name` and the default variable scope is
        `head_name`.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      metrics_fn: a function that takes predictions dict, labels and weights and
        returns a dictionary of metrics to be calculated.
    """

    def loss_wrapper(labels, logits, weight_tensor):
      if weight_tensor is None:
        weight_tensor = array_ops.ones(
            shape=[array_ops.shape(labels)[0], 1], dtype=dtypes.float32)
      weighted_loss, _ = loss_fn(labels, weight_tensor, logits)
      average_loss = math_ops.reduce_mean(weighted_loss)
      return average_loss, average_loss / math_ops.reduce_mean(weight_tensor)

    super(CustomLossHead, self).__init__(
        loss_fn=loss_wrapper,
        link_fn=link_fn,
        head_name=head_name,
        weight_column_name=weight_column_name,
        enable_centered_bias=False,
        label_dimension=logit_dimension)

    self._metrics_fn = metrics_fn

  def _metrics(self, eval_loss, predictions, labels, weights):
    if self._metrics_fn is not None:
      return self._metrics_fn(predictions, labels, weights)
