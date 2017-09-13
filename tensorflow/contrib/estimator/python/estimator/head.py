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
"""Abstractions for the head(s) of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.canned import head as head_lib


def multi_class_head(n_classes,
                     weight_column=None,
                     label_vocabulary=None,
                     head_name=None):
  """Creates a `_Head` for multi class classification.

  Uses `sparse_softmax_cross_entropy` loss.

  This head expects to be fed integer labels specifying the class index.

  Args:
    n_classes: Number of classes, must be greater than 2 (for 2 classes, use
      `_BinaryLogisticHeadWithSigmoidCrossEntropyLoss`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_vocabulary: A list of strings represents possible label values. If it
      is not given, that means labels are already encoded as integer within
      [0, n_classes). If given, labels must be string type and have any value in
      `label_vocabulary`. Also there will be errors if vocabulary is not
      provided and labels are string.
    head_name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + head_name`.

  Returns:
    An instance of `_Head` for multi class classification.

  Raises:
    ValueError: if `n_classes`, `metric_class_ids` or `label_keys` is invalid.
  """
  return head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint:disable=protected-access
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      head_name=head_name)


def binary_classification_head(
    weight_column=None, thresholds=None, label_vocabulary=None, head_name=None):
  """Creates a `_Head` for single label binary classification.

  This head uses `sigmoid_cross_entropy_with_logits` loss.

  This head expects to be fed float labels of shape `(batch_size, 1)`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    thresholds: Iterable of floats in the range `(0, 1)`. For binary
      classification metrics such as precision and recall, an eval metric is
      generated for each threshold value. This threshold is applied to the
      logistic values to determine the binary classification (i.e., above the
      threshold is `true`, below is `false`.
    label_vocabulary: A list of strings represents possible label values. If it
      is not given, that means labels are already encoded within [0, 1]. If
      given, labels must be string type and have any value in
      `label_vocabulary`. Also there will be errors if vocabulary is not
      provided and labels are string.
    head_name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + head_name`.

  Returns:
    An instance of `_Head` for binary classification.

  Raises:
    ValueError: if `thresholds` contains a value outside of `(0, 1)`.
  """
  return head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint:disable=protected-access
      weight_column=weight_column,
      thresholds=thresholds,
      label_vocabulary=label_vocabulary,
      head_name=head_name)


def regression_head(weight_column=None,
                    label_dimension=1,
                    head_name=None):
  """Creates a `_Head` for regression using the mean squared loss.

  Uses `mean_squared_error` loss.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    head_name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + head_name`.

  Returns:
    An instance of `_Head` for linear regression.
  """
  return head_lib._regression_head_with_mean_squared_error_loss(  # pylint:disable=protected-access
      weight_column=weight_column,
      label_dimension=label_dimension,
      head_name=head_name)
