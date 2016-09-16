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

"""Masks one `Series` based on the content of another `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def sparse_boolean_mask(sparse_tensor, mask, name="sparse_boolean_mask"):
  """Boolean mask for `SparseTensor`s.

  Args:
    sparse_tensor: a `SparseTensor`.
    mask: a 1D boolean dense`Tensor` whose length is equal to the 0th dimension
      of `sparse_tensor`.
    name: optional name for this operation.
  Returns:
    A `SparseTensor` that contains row `k` of `sparse_tensor` iff `mask[k]` is
    `True`.
  """
  # TODO(jamieas): consider mask dimension > 1 for symmetry with `boolean_mask`.
  with ops.name_scope(name, values=[sparse_tensor, mask]):
    mask = ops.convert_to_tensor(mask)
    mask_rows = array_ops.where(mask)
    first_indices = array_ops.squeeze(array_ops.slice(sparse_tensor.indices,
                                                      [0, 0], [-1, 1]))

    # Identify indices corresponding to the rows identified by mask_rows.
    sparse_entry_matches = functional_ops.map_fn(
        lambda x: math_ops.equal(first_indices, x),
        mask_rows,
        dtype=dtypes.bool)
    # Combine the rows of index_matches to form a mask for the sparse indices
    # and values.
    to_retain = array_ops.reshape(
        functional_ops.foldl(math_ops.logical_or, sparse_entry_matches), [-1])

    return sparse_ops.sparse_retain(sparse_tensor, to_retain)


@series.Series.register_binary_op("select_rows")
class BooleanMask(transform.TensorFlowTransform):
  """Apply a boolean mask to a `Series`."""

  @property
  def name(self):
    return "BooleanMask"

  @property
  def input_valency(self):
    return 2

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
    input_tensor = input_tensors[0]
    mask = input_tensors[1]
    if mask.get_shape().ndims > 1:
      mask = array_ops.squeeze(mask)

    if isinstance(input_tensor, ops.SparseTensor):
      mask_fn = sparse_boolean_mask
    else:
      mask_fn = array_ops.boolean_mask

    # pylint: disable=not-callable
    return self.return_type(mask_fn(input_tensor, mask))
