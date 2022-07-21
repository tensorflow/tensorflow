# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Code for creating a dataset out of a NumPy array."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def init_var_from_numpy(input_var, numpy_input, session):
  """Initialize `input_var` to `numpy_input` using `session` in graph mode."""
  with ops.init_scope():
    if context.executing_eagerly():
      input_var.assign(numpy_input)
      return

    assert session is not None
    session.run(input_var.initializer)

    start_placeholder = array_ops.placeholder(dtypes.int64, ())
    end_placeholder = array_ops.placeholder(dtypes.int64, ())
    slice_placeholder = array_ops.placeholder(input_var.dtype)
    assign_slice_op = input_var[start_placeholder:end_placeholder].assign(
        slice_placeholder)

    # If each batch element is > 64 MB, then we copy each batch element
    # individually. Otherwise, the slices will be < 128 MB. There might be
    # padding which might mean that the slices are 128 MB even if the size of
    # the tensor allocated is less than 128 MB.  This formula gives slices with
    # size: ceil(64 MB / byte size per batch element) bytes.  Using ceil()
    # guarantees we get a number >= 1.

    # Calculate the size of each batch element.
    byte_size_per_batch_element = (
        np.prod(numpy_input.shape[1:]) * input_var.dtype.size)

    # Calculate number of elements we want to copy per slice.
    batch_size_per_slice = int(
        np.ceil((64 << 20) / byte_size_per_batch_element))

    # Copy slices of the above size starting at 0, except the last slice will be
    # smaller.
    start = 0
    limit = numpy_input.shape[0]
    while start < limit:
      end = min(start + batch_size_per_slice, limit)
      session.run(assign_slice_op, feed_dict={
          start_placeholder: start,
          end_placeholder: end,
          slice_placeholder: numpy_input[start:end]})
      start = end


def one_host_numpy_dataset(numpy_input, colocate_with, session):
  """Create a dataset on `colocate_with` from `numpy_input`."""

  def create_colocated_variable(next_creator, **kwargs):
    kwargs["colocate_with"] = colocate_with
    return next_creator(**kwargs)

  numpy_flat = nest.flatten(numpy_input)
  with variable_scope.variable_creator_scope(create_colocated_variable):
    vars_flat = tuple(variable_scope.variable(array_ops.zeros(i.shape, i.dtype),
                                              trainable=False)
                      for i in numpy_flat)
  for v, i in zip(vars_flat, numpy_flat):
    init_var_from_numpy(v, i, session)
  vars_nested = nest.pack_sequence_as(numpy_input, vars_flat)
  return dataset_ops.Dataset.from_tensor_slices(vars_nested)


class SingleDevice(object):
  """Used with `colocate_with` to create a non-mirrored variable."""

  def __init__(self, device):
    self.device = device
