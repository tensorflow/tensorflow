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

from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import tf_logging as logging

# Data column types for indicating categorical or other non-float values.
DATA_FLOAT = 0
DATA_CATEGORICAL = 1

DTYPE_TO_FTYPE = {
    dtypes.string: DATA_CATEGORICAL,
    dtypes.int32: DATA_CATEGORICAL,
    dtypes.int64: DATA_CATEGORICAL,
    dtypes.float32: DATA_FLOAT,
    dtypes.float64: DATA_FLOAT
}


def CastToFloat(tensor):
  if tensor.dtype == dtypes.string:
    return tensor_forest_ops.reinterpret_string_to_float(tensor)
  elif tensor.dtype.is_integer:
    return math_ops.to_float(tensor)
  else:
    return tensor


# TODO(gilberth): If protos are ever allowed in dynamically loaded custom
# op libraries, convert this to a proto like a sane person.
class TensorForestDataSpec(object):

  def __init__(self):
    self.sparse = DataColumnCollection()
    self.dense = DataColumnCollection()
    self.dense_features_size = 0

  def SerializeToString(self):
    return 'dense_features_size: %d dense: [%s] sparse: [%s]' % (
        self.dense_features_size, self.dense.SerializeToString(),
        self.sparse.SerializeToString())


class DataColumnCollection(object):
  """Collection of DataColumns, meant to mimic a proto repeated field."""

  def __init__(self):
    self.cols = []

  def add(self):  # pylint: disable=invalid-name
    self.cols.append(DataColumn())
    return self.cols[-1]

  def size(self):  # pylint: disable=invalid-name
    return len(self.cols)

  def SerializeToString(self):
    ret = ''
    for c in self.cols:
      ret += '{%s}' % c.SerializeToString()
    return ret


class DataColumn(object):

  def __init__(self):
    self.name = ''
    self.original_type = ''
    self.size = 0

  def SerializeToString(self):
    return 'name: {0} original_type: {1} size: {2}'.format(self.name,
                                                           self.original_type,
                                                           self.size)


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
  data_spec = TensorForestDataSpec()
  if isinstance(data, dict):
    dense_features_size = 0
    dense_features = []
    sparse_features = []
    for k in sorted(data.keys()):
      is_sparse = isinstance(data[k], sparse_tensor.SparseTensor)
      if is_sparse:
        # TODO(gilberth): support sparse categorical.
        if data[k].dtype == dtypes.string:
          logging.info('TensorForest does not support sparse categorical. '
                       'Transform it into a number with hash buckets.')
          continue
        elif data_spec.sparse.size() == 0:
          col_spec = data_spec.sparse.add()
          col_spec.original_type = DATA_FLOAT
          col_spec.name = 'all_sparse'
          col_spec.size = -1
        sparse_features.append(
            sparse_tensor.SparseTensor(data[
                k].indices, CastToFloat(data[k].values), data[k].dense_shape))
      else:
        col_spec = data_spec.dense.add()

        col_spec.original_type = DTYPE_TO_FTYPE[data[k].dtype]
        col_spec.name = k
        # the second dimension of get_shape should always be known.
        col_spec.size = data[k].get_shape()[1].value

        dense_features_size += col_spec.size
        dense_features.append(CastToFloat(data[k]))

    processed_dense_features = None
    processed_sparse_features = None
    if dense_features:
      processed_dense_features = array_ops.concat(dense_features, 1)
      data_spec.dense_features_size = dense_features_size
    if sparse_features:
      processed_sparse_features = sparse_ops.sparse_concat(1, sparse_features)
    logging.info(data_spec.SerializeToString())
    return processed_dense_features, processed_sparse_features, data_spec
  elif isinstance(data, sparse_tensor.SparseTensor):
    col_spec = data_spec.sparse.add()
    col_spec.name = 'sparse_features'
    col_spec.original_type = DTYPE_TO_FTYPE[data.dtype]
    col_spec.size = -1
    data_spec.dense_features_size = 0
    return None, data, data_spec
  else:
    data = ops.convert_to_tensor(data)
    col_spec = data_spec.dense.add()
    col_spec.name = 'dense_features'
    col_spec.original_type = DTYPE_TO_FTYPE[data.dtype]
    col_spec.size = data.get_shape()[1]
    data_spec.dense_features_size = col_spec.size
    return data, None, data_spec


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
        array_ops.concat(
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
