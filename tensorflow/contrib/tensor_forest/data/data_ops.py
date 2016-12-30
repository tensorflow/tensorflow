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
"""Ops for preprocessing data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.python import constants
from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def ParseDataTensorOrDict(data):
  """Return a tensor to use for input data.

  The incoming features can be a dict where keys are the string names of the
  columns, which we turn into a single 2-D tensor.

  Args:
    data: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for input to tensor_forest, a keys tensor for the
    tf.Examples if they exist, and a list of the type of each column
    (e.g. continuous float, categorical).
  """
  if isinstance(data, dict):
    # If there's at least one sparse tensor, everything has to be sparse.
    is_sparse = False
    for v in data.values():
      if isinstance(v, sparse_tensor.SparseTensor):
        is_sparse = True
        break

    categorical_types = (dtypes.string, dtypes.int32, dtypes.int64)
    data_spec = [constants.DATA_CATEGORICAL if
                 data[k].dtype in categorical_types else
                 constants.DATA_FLOAT for k in sorted(data.keys())]
    data_spec = [constants.DATA_FLOAT] + data_spec
    features = []
    for k in sorted(data.keys()):
      if data[k].dtype == dtypes.string:
        features.append(tensor_forest_ops.reinterpret_string_to_float(data[k]))
      elif data[k].dtype.is_integer:
        features.append(math_ops.to_float(data[k]))
      else:
        features.append(data[k])

    if is_sparse:
      return sparse_ops.sparse_concat(1, features), data_spec
    else:
      return array_ops.concat_v2(features, 1), data_spec
  else:
    return (data, [constants.DATA_FLOAT])


def ParseLabelTensorOrDict(labels):
  """Return a tensor to use for input labels to tensor_forest.

  The incoming targets can be a dict where keys are the string names of the
  columns, which we turn into a single 1-D tensor for classification or
  2-D tensor for regression.

  Converts sparse tensors to dense ones.

  Args:
    labels: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for labels/outputs.
  """
  if isinstance(labels, dict):
    return math_ops.to_float(
        array_ops.concat_v2(
            [
                sparse_ops.sparse_tensor_to_dense(
                    labels[k], default_value=-1) if isinstance(
                        labels, sparse_tensor.SparseTensor) else labels[k]
                for k in sorted(labels.keys())
            ],
            1))
  else:
    if isinstance(labels, sparse_tensor.SparseTensor):
      return math_ops.to_float(sparse_ops.sparse_tensor_to_dense(
          labels, default_value=-1))
    else:
      return math_ops.to_float(labels)
