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

"""A `Transform` that performs subtraction on two `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import ops
from tensorflow.python.ops import sparse_ops


def _negate_sparse(sparse_tensor):
  return ops.SparseTensor(indices=sparse_tensor.indices,
                          values=-sparse_tensor.values,
                          shape=sparse_tensor.shape)


@series.Series.register_binary_op("__sub__")
class Difference(transform.Transform):
  """Subtracts one 'Series` from another."""

  def __init__(self):
    super(Difference, self).__init__()

  @property
  def name(self):
    return "difference"

  @property
  def input_valency(self):
    return 2

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    pair_sparsity = (isinstance(input_tensors[0], ops.SparseTensor),
                     isinstance(input_tensors[1], ops.SparseTensor))

    if pair_sparsity == (False, False):
      result = input_tensors[0] - input_tensors[1]
    # note tf.sparse_add accepts the mixed cases,
    # so long as at least one input is sparse.
    elif not pair_sparsity[1]:
      result = sparse_ops.sparse_add(input_tensors[0], - input_tensors[1])
    else:
      result = sparse_ops.sparse_add(input_tensors[0],
                                     _negate_sparse(input_tensors[1]))
    # pylint: disable=not-callable
    return self.return_type(result)
