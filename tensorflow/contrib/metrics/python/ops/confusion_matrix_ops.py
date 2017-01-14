# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Confusion matrix related metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.metrics.python.ops import metric_ops_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def confusion_matrix(predictions, labels, num_classes=None,
                     dtype=dtypes.int32, name=None):
  """Computes the confusion matrix from predictions and labels.

  Calculate the Confusion Matrix for a pair of prediction and
  label 1-D int arrays.

  Considering a prediction array such as: `[1, 2, 3]`
  And a label array such as: `[2, 2, 3]`

  The confusion matrix returned would be the following one:
      [[0, 0, 0]
       [0, 1, 0]
       [0, 1, 0]
       [0, 0, 1]]

  Where the matrix rows represent the prediction labels and the columns
  represents the real labels. The confusion matrix is always a 2-D array
  of shape [n, n], where n is the number of valid labels for a given
  classification task. Both prediction and labels must be 1-D arrays of
  the same shape in order for this function to work.

  Args:
    predictions: A 1-D array represeting the predictions for a given
                 classification.
    labels: A 1-D represeting the real labels for the classification task.
    num_classes: The possible number of labels the classification task can
                 have. If this value is not provided, it will be calculated
                 using both predictions and labels array.
    dtype: Data type of the confusion matrix.
    name: Scope name.

  Returns:
    A l X l matrix represeting the confusion matrix, where l in the number of
    possible labels in the classification task.

  Raises:
    ValueError: If both predictions and labels are not 1-D vectors and do not
                have the same size.
  """
  with ops.op_scope([predictions, labels, num_classes], name,
                    'confusion_matrix') as name:
    predictions, labels = metric_ops_util.remove_squeezable_dimensions(
        ops.convert_to_tensor(
            predictions, name='predictions', dtype=dtypes.int64),
        ops.convert_to_tensor(labels, name='labels', dtype=dtypes.int64))

    if num_classes is None:
      num_classes = math_ops.maximum(math_ops.reduce_max(predictions),
                                     math_ops.reduce_max(labels)) + 1

    shape = array_ops.pack([num_classes, num_classes])
    indices = array_ops.transpose(array_ops.pack([predictions, labels]))
    values = array_ops.ones_like(predictions, dtype)
    cm_sparse = ops.SparseTensor(
        indices=indices, values=values, shape=shape)
    zero_matrix = array_ops.zeros(math_ops.to_int32(shape), dtype)

    return sparse_ops.sparse_add(zero_matrix, cm_sparse)
