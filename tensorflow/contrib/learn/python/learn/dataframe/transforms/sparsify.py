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

"""Transforms Dense to Sparse Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe import transform

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class Sparsify(transform.TensorFlowTransform):
  """Transforms Dense to Sparse Tensor."""

  def __init__(self, strip_value):
    super(Sparsify, self).__init__()
    self._strip_value = strip_value

  @transform.parameter
  def strip_value(self):
    return self._strip_value

  @property
  def name(self):
    return "Sparsify"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    """Applies the transformation to the `transform_input`.

    Args:
      input_tensors: a list of Tensors representing the input to
        the Transform.
      **kwargs: Additional keyword arguments, unused here.

    Returns:
        A namedtuple of Tensors representing the transformed output.
    """
    d = input_tensors[0]

    if self.strip_value is np.nan:
      strip_hot = math_ops.is_nan(d)
    else:
      strip_hot = math_ops.equal(d,
                                 array_ops.constant([self.strip_value],
                                                    dtype=d.dtype))
    keep_hot = math_ops.logical_not(strip_hot)

    length = array_ops.reshape(array_ops.shape(d), [])
    indices = array_ops.boolean_mask(math_ops.range(length), keep_hot)
    values = array_ops.boolean_mask(d, keep_hot)

    sparse_indices = array_ops.reshape(
        math_ops.cast(indices, dtypes.int64), [-1, 1])
    shape = math_ops.cast(array_ops.shape(d), dtypes.int64)

    # pylint: disable=not-callable
    return self.return_type(ops.SparseTensor(sparse_indices, values, shape))
