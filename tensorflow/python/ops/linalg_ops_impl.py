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
"""Operations for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

# Names below are lower_case.
# pylint: disable=invalid-name


def eye(num_rows,
        num_columns=None,
        batch_shape=None,
        dtype=dtypes.float32,
        name=None):
  """Construct an identity matrix, or a batch of matrices.

  See `linalg_ops.eye`.
  """
  with ops.name_scope(
      name, default_name='eye', values=[num_rows, num_columns, batch_shape]):
    is_square = num_columns is None
    batch_shape = [] if batch_shape is None else batch_shape
    num_columns = num_rows if num_columns is None else num_columns
    if isinstance(num_rows, ops.Tensor) or isinstance(
        num_columns, ops.Tensor) or isinstance(batch_shape, ops.Tensor):
      batch_shape = ops.convert_to_tensor(
          batch_shape, name='shape', dtype=dtypes.int32)
      diag_size = math_ops.minimum(num_rows, num_columns)
      diag_shape = array_ops.concat((batch_shape, [diag_size]), 0)
      if not is_square:
        shape = array_ops.concat((batch_shape, [num_rows, num_columns]), 0)
    else:
      if not isinstance(num_rows, compat.integral_types) or not isinstance(
          num_columns, compat.integral_types):
        raise TypeError(
            'num_rows and num_columns must be positive integer values.')
      batch_shape = [dim for dim in batch_shape]
      is_square = num_rows == num_columns
      diag_shape = batch_shape + [np.minimum(num_rows, num_columns)]
      if not is_square:
        shape = batch_shape + [num_rows, num_columns]

    diag_ones = array_ops.ones(diag_shape, dtype=dtype)
    if is_square:
      return array_ops.matrix_diag(diag_ones)
    else:
      zero_matrix = array_ops.zeros(shape, dtype=dtype)
      return array_ops.matrix_set_diag(zero_matrix, diag_ones)

# pylint: enable=invalid-name,redefined-builtin
