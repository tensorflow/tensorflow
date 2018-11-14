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
"""Utill functions for distillation loss.

The distillation loss_fn will be called with the following:

Args:
  dnn_logits: Tensor of logits from the dnn, treated as the "target". This will
    be the output of a call to tf.stop_gradient().
  tree_logits: Tensor of logits from the tree, treated as the "predictions".
  example_weights: Tensor of example weights, or a single scalar.

Returns:
  A scalar indicating the reduced loss for that batch of examples.

Note: we calls the loss_fn defined in contrib head, which is computing two
losses, first one for training and second one for reporting. We only take the
first one here.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def _logits_to_label_for_tree(logits, n_classes):
  if n_classes == 2:
    return math_ops.sigmoid(logits)
  else:
    return nn.softmax(logits)


def create_dnn_to_tree_squared_loss_fn(n_classes):
  """Returns a squared loss function for dnn to tree distillation."""

  def _dnn_to_tree_squared_loss(dnn_logits, tree_logits, example_weights):
    return head_lib._mean_squared_loss(  # pylint: disable=protected-access
        labels=_logits_to_label_for_tree(dnn_logits, n_classes),
        logits=_logits_to_label_for_tree(tree_logits, n_classes),
        weights=example_weights)[0]

  return _dnn_to_tree_squared_loss


def create_dnn_to_tree_cross_entropy_loss_fn(n_classes):
  """Returns a cross entropy loss function for dnn to tree distillation."""

  def _dnn_to_tree_cross_entropy_loss(dnn_logits, tree_logits, example_weights):
    if n_classes == 2:
      return head_lib._log_loss_with_two_classes(  # pylint: disable=protected-access
          labels=_logits_to_label_for_tree(dnn_logits, n_classes),
          logits=tree_logits,
          weights=example_weights)[0]
    else:
      return head_lib._softmax_cross_entropy_loss(  # pylint: disable=protected-access
          labels=_logits_to_label_for_tree(dnn_logits, n_classes),
          logits=tree_logits,
          weights=example_weights)[0]

  return _dnn_to_tree_cross_entropy_loss
