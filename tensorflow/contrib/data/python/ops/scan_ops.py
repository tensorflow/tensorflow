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
"""Scan dataset transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_dataset_ops


class _ScanDataset(dataset_ops.UnaryDataset):
  """A dataset that scans a function across its input."""

  def __init__(self, input_dataset, initial_state, scan_func):
    """See `scan()` for details."""
    super(_ScanDataset, self).__init__(input_dataset)
    self._input_dataset = input_dataset

    with ops.name_scope("initial_state"):
      # Convert any `SparseTensorValue`s to `SparseTensor`s and all other
      # values to tensors.
      self._initial_state = nest.pack_sequence_as(initial_state, [
          sparse_tensor.SparseTensor.from_value(t)
          if sparse_tensor.is_sparse(t) else ops.convert_to_tensor(
              t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(initial_state))
      ])

    # Compute initial values for the state classes, shapes and types based on
    # the initial state. The shapes may be refined by running `tf_scan_func` one
    # or more times below.
    self._state_classes = sparse.get_classes(self._initial_state)
    self._state_shapes = nest.pack_sequence_as(
        self._initial_state,
        [t.get_shape() for t in nest.flatten(self._initial_state)])
    self._state_types = nest.pack_sequence_as(
        self._initial_state,
        [t.dtype for t in nest.flatten(self._initial_state)])

    # Will be populated by calling `tf_scan_func`.
    self._output_classes = None
    self._output_shapes = None
    self._output_types = None

    # Iteratively rerun the scan function until reaching a fixed point on
    # `self._state_shapes`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = dataset_ops.StructuredFunctionWrapper(
          scan_func, "tf.contrib.data.scan()",
          input_classes=(self._state_classes, input_dataset.output_classes),
          input_shapes=(self._state_shapes, input_dataset.output_shapes),
          input_types=(self._state_types, input_dataset.output_types),
          add_to_graph=False)
      if not (
          isinstance(wrapped_func.output_types, collections.Sequence) and
          len(wrapped_func.output_types) == 2):
        raise TypeError("The scan function must return a pair comprising the "
                        "new state and the output value.")

      new_state_classes, self._output_classes = wrapped_func.output_classes

      # Extract and validate class information from the returned values.
      for new_state_class, state_class in zip(
          nest.flatten(new_state_classes),
          nest.flatten(self._state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._state_classes, new_state_classes))

      # Extract and validate type information from the returned values.
      new_state_types, self._output_types = wrapped_func.output_types
      for new_state_type, state_type in zip(
          nest.flatten(new_state_types), nest.flatten(self._state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._state_types, new_state_types))

      # Extract shape information from the returned values.
      new_state_shapes, self._output_shapes = wrapped_func.output_shapes

      flat_state_shapes = nest.flatten(self._state_shapes)
      flat_new_state_shapes = nest.flatten(new_state_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        self._state_shapes = nest.pack_sequence_as(self._state_shapes,
                                                   weakened_state_shapes)

    self._scan_func = wrapped_func.function
    self._scan_func.add_to_graph(ops.get_default_graph())

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    return gen_dataset_ops.scan_dataset(
        input_t,
        nest.flatten(sparse.serialize_sparse_tensors(self._initial_state)),
        self._scan_func.captured_inputs,
        f=self._scan_func,
        **dataset_ops.flat_structure(self))

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


def scan(initial_state, scan_func):
  """A transformation that scans a function across an input dataset.

  This transformation is a stateful relative of `tf.data.Dataset.map`.
  In addition to mapping `scan_func` across the elements of the input dataset,
  `scan()` accumulates one or more state tensors, whose initial values are
  `initial_state`.

  Args:
    initial_state: A nested structure of tensors, representing the initial state
      of the accumulator.
    scan_func: A function that maps `(old_state, input_element)` to
      `(new_state, output_element). It must take two arguments and return a
      pair of nested structures of tensors. The `new_state` must match the
      structure of `initial_state`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return _ScanDataset(dataset, initial_state, scan_func)

  return _apply_fn
