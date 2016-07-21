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

import threading

from tensorflow.contrib.tensor_forest.python import constants

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

DATA_OPS_FILE = '_data_ops.so'

_data_ops = None
_ops_lock = threading.Lock()


ops.NoGradient('StringToFloat')


@ops.RegisterShape('StringToFloat')
def StringToFloatShape(op):
  """Shape function for StringToFloat Op."""
  return [op.inputs[0].get_shape()]


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load the data ops library and return the loaded module."""
  with _ops_lock:
    global _data_ops
    if not _data_ops:
      ops_path = resource_loader.get_path_to_datafile(DATA_OPS_FILE)
      logging.info('data path: %s', ops_path)
      _data_ops = load_library.load_op_library(ops_path)

      assert _data_ops, 'Could not load _data_ops.so'
  return _data_ops


def ParseDataTensorOrDict(data):
  """Return a tensor to use for input data.

  The incoming features can be a dict where keys are the string names of the
  columns, which we turn into a single 2-D tensor.

  Args:
    data: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for input to tensor_forest and a 1-D tensor of the
      type of each column (e.g. continuous float, categorical).
  """
  convert_ops = Load()
  if isinstance(data, dict):
    data_spec = [constants.DATA_CATEGORICAL if data[k].dtype == dtypes.string
                 else constants.DATA_FLOAT
                 for k in sorted(data.keys())]
    return array_ops.concat(1, [
        convert_ops.string_to_float(data[k])
        if data[k].dtype == dtypes.string else data[k]
        for k in sorted(data.keys())]), data_spec
  else:
    return data, [constants.DATA_FLOAT] * data.get_shape().as_list()[1]


def ParseLabelTensorOrDict(labels):
  """Return a tensor to use for input labels to tensor_forest.

  The incoming targets can be a dict where keys are the string names of the
  columns, which we turn into a single 1-D tensor for classification or
  2-D tensor for regression.

  Args:
    labels: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for labels/outputs.
  """
  if isinstance(labels, dict):
    return math_ops.to_float(array_ops.concat(
        1, [labels[k] for k in sorted(labels.keys())]))
  else:
    return math_ops.to_float(labels)
