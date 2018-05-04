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
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


class _ScanDataset(dataset_ops.Dataset):
  """A dataset that scans a function across its input."""

  def __init__(self, input_dataset, initial_state, scan_func):
    """See `scan()` for details."""
    super(_ScanDataset, self).__init__()
    self._input_dataset = input_dataset

    with ops.name_scope("initial_state"):
      self._initial_state = nest.pack_sequence_as(initial_state, [
          ops.convert_to_tensor(t, name="component_%d" % i)
          for i, t in enumerate(nest.flatten(initial_state))
      ])

    # Compute initial values for the state shapes and types based on
    # the initial state. These will be refined by running
    # `tf_scan_func` one or more times below.
    # TODO(b/68937811): Allow the initial state to be a tf.SparseTensor.
    self._state_shapes = nest.pack_sequence_as(
        self._initial_state,
        [t.shape for t in nest.flatten(self._initial_state)])
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

      flat_state_shapes = nest.flatten(self._state_shapes)
      flat_state_types = nest.flatten(self._state_types)

      # Create a list in which `tf_scan_func` will store the s
      flat_new_state_shapes = []

      @function.Defun(*(flat_state_types + nest.flatten(
          sparse.as_dense_types(input_dataset.output_types,
                                input_dataset.output_classes))))
      def tf_scan_func(*args):
        """A wrapper for Defun that facilitates shape inference."""
        # Pass in shape information from the state and input_dataset.
        # TODO(b/69424092): Check that neither inputs nor outputs are sparse.
        dense_shapes = sparse.as_dense_shapes(input_dataset.output_shapes,
                                              input_dataset.output_classes)
        for arg, shape in zip(args,
                              flat_state_shapes + nest.flatten(dense_shapes)):
          arg.set_shape(shape)

        pivot = len(flat_state_shapes)
        old_state = nest.pack_sequence_as(self._initial_state, args[:pivot])
        input_value = nest.pack_sequence_as(input_dataset.output_types,
                                            args[pivot:])

        ret = scan_func(old_state, input_value)
        if not isinstance(ret, collections.Sequence) or len(ret) != 2:
          raise TypeError("The scan function must return a pair comprising the "
                          "new state and the output value.")
        new_state, output_value = ret

        flat_new_state = [
            ops.convert_to_tensor(t) for t in nest.flatten(new_state)
        ]
        flat_output_value = [
            ops.convert_to_tensor(t) for t in nest.flatten(output_value)
        ]

        # Extract shape information from the returned values.
        flat_new_state_shapes.extend([t.shape for t in flat_new_state])
        self._output_shapes = nest.pack_sequence_as(
            output_value, [t.shape for t in flat_output_value])

        # Extract and validate type information from the returned values.
        for t, dtype in zip(flat_new_state, flat_state_types):
          if t.dtype != dtype:
            raise TypeError(
                "The element types for the new state must match the initial "
                "state. Expected %s; got %s." %
                (self._state_types, nest.pack_sequence_as(
                    self._state_types, [t.dtype for t in flat_new_state])))
        self._output_classes = nest.pack_sequence_as(
            output_value, [ops.Tensor for _ in flat_output_value])
        self._output_types = nest.pack_sequence_as(
            output_value, [t.dtype for t in flat_output_value])

        return flat_new_state + flat_output_value

      # Use the private method that will execute `tf_scan_func` but delay
      # adding it to the graph in case we need to rerun the function.
      tf_scan_func._create_definition_if_needed()  # pylint: disable=protected-access

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
        # NOTE(mrry): `self._output_shapes` will be overwritten when we rerun
        # `tf_scan_func`.
        self._state_shapes = nest.pack_sequence_as(self._state_shapes,
                                                   weakened_state_shapes)

    self._scan_func = tf_scan_func

  def _as_variant_tensor(self):
    input_t = self._input_dataset._as_variant_tensor()  # pylint: disable=protected-access
    return gen_dataset_ops.scan_dataset(
        input_t,
        nest.flatten(self._initial_state),
        self._scan_func.captured_inputs,
        f=self._scan_func,
        output_types=nest.flatten(
            sparse.as_dense_types(self.output_types, self.output_classes)),
        output_shapes=nest.flatten(
            sparse.as_dense_shapes(self.output_shapes, self.output_classes)))

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

  This transformation is a stateful relative of @{tf.data.Dataset.map}.
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
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return _ScanDataset(dataset, initial_state, scan_func)

  return _apply_fn
