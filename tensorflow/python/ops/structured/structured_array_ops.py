# Lint as python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""StructuredTensor array ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_types(array_ops.expand_dims, StructuredTensor)
@deprecation.deprecated_args(None, 'Use the `axis` argument instead', 'dim')
def expand_dims(input, axis=None, name=None, dim=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    input: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.
    dim: deprecated: use axis.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  axis = deprecation.deprecated_argument_lookup('axis', axis, 'dim', dim)
  return _expand_dims_impl(input, axis, name=name)


@dispatch.dispatch_for_types(array_ops.expand_dims_v2, StructuredTensor)
def expand_dims_v2(input, axis, name=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    input: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  return _expand_dims_impl(input, axis, name=name)


def _expand_dims_impl(st, axis, name=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    st: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  axis = array_ops.get_positive_axis(
      axis, st.rank + 1, axis_name='axis', ndims_name='rank(st)')
  with ops.name_scope(name, 'ExpandDims', [st, axis]):
    new_fields = {
        k: array_ops.expand_dims(v, axis)
        for (k, v) in st._fields.items()
    }
    new_shape = st.shape[:axis] + (1,) + st.shape[axis:]
    new_row_partitions = _expand_st_row_partitions(st, axis)
    new_nrows = st.nrows() if (axis > 0) else 1
    return StructuredTensor.from_fields(
        new_fields,
        shape=new_shape,
        row_partitions=new_row_partitions,
        nrows=new_nrows)


def _expand_st_row_partitions(st, axis):
  """Create the row_partitions for expand_dims."""
  if axis == 0:
    if st.shape.rank == 0:
      return ()
    nvals = st.nrows()
    new_partition = RowPartition.from_uniform_row_length(
        nvals, nvals, nrows=1, validate=False)
    return (new_partition,) + st.row_partitions
  elif axis == st.rank:
    nvals = (
        st.row_partitions[axis - 2].nvals() if (axis - 2 >= 0) else st.nrows())
    return st.row_partitions + (RowPartition.from_uniform_row_length(
        1, nvals, nrows=nvals, validate=False),)
  else:
    nvals = (
        st.row_partitions[axis - 1].nrows() if (axis - 1 >= 0) else st.nrows())
    return st.row_partitions[:axis - 1] + (RowPartition.from_uniform_row_length(
        1, nvals, nrows=nvals, validate=False),) + st.row_partitions[axis - 1:]
