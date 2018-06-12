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


"""
This module provides sparse operators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.layers.ops import gen_sparse_tile_like_op
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework.sparse_tensor import SparseTensor

_sparse_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_sparse_tile_like_op.so"))


def sparse_tile_like(sp_a, sp_b, axes):
  """Tile a matrix like another higher-dimensional matrix.

  This `Op` is used for tile a `1D matrix` to `2D matrix` or `2D matrix`
  to a `3D matrix`.
  The result matrix will be the same size as `sparsetensor` `b`.
  And filled with the values in the sparsetensor `a` at the right positions.
  See sparse_tile_like_kernel.cc for more details.

  Args:
    sp_a: A SparseTensor `a` which has a dimension of 1 or 2.
    sp_b: A SparseTensor `b` for `a` to tile, whose dimension should be equal
      to the dimension of `a` plus 1.
    axes: The axes to be tile.

  Returns:
    A SparseTensor with the same shape as `sp_b` whose values is filled by
      `sp_a`.
  """

  r_indices, r_values, r_shape = gen_sparse_tile_like_op.sparse_tile_like(
      sp_a.indices,
      sp_a.values,
      sp_a.dense_shape,
      sp_b.indices,
      sp_b.values,
      sp_b.dense_shape,
      axes,
  )
  return SparseTensor(r_indices, r_values, r_shape)
