# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""The implementation of `tf.data.Dataset.shuffle`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.compat import collections_abc


def _scan(input_dataset,
          initial_state,
          scan_func,
          use_default_device=None,
          name=None):
  return _ScanDataset(
      input_dataset, initial_state, scan_func, use_default_device, name=name)


class _ScanDataset(dataset_ops.UnaryDataset):
  """A dataset that scans a function across its input."""

  def __init__(self,
               input_dataset,
               initial_state,
               scan_func,
               use_default_device=None,
               name=None):
    """See `scan()` for details."""
    self._input_dataset = input_dataset
    self._initial_state = structure.normalize_element(initial_state)

    # Compute initial values for the state classes, shapes and types based on
    # the initial state. The shapes may be refined by running `tf_scan_func` one
    # or more times below.
    self._state_structure = structure.type_spec_from_value(self._initial_state)

    # Iteratively rerun the scan function until reaching a fixed point on
    # `self._state_shapes`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = structured_function.StructuredFunctionWrapper(
          scan_func,
          self._transformation_name(),
          input_structure=(self._state_structure, input_dataset.element_spec),
          add_to_graph=False)
      if not (isinstance(wrapped_func.output_types, collections_abc.Sequence)
              and len(wrapped_func.output_types) == 2):
        raise TypeError(f"Invalid `scan_func`. `scan_func` should return a "
                        f"pair consisting of new state and the output value "
                        f"but its return type is "
                        f"{wrapped_func.output_structure}.")

      new_state_classes, self._output_classes = wrapped_func.output_classes

      # Extract and validate class information from the returned values.
      new_state_classes, output_classes = wrapped_func.output_classes
      old_state_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
          self._state_structure)
      for new_state_class, old_state_class in zip(
          nest.flatten(new_state_classes), nest.flatten(old_state_classes)):
        if not issubclass(new_state_class, old_state_class):
          raise TypeError(f"Invalid `scan_func`. The element classes for the "
                          f"new state must match the initial state. Expected "
                          f"{old_state_classes}, got {new_state_classes}.")

      # Extract and validate type information from the returned values.
      new_state_types, output_types = wrapped_func.output_types
      old_state_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
          self._state_structure)
      for new_state_type, old_state_type in zip(
          nest.flatten(new_state_types), nest.flatten(old_state_types)):
        if new_state_type != old_state_type:
          raise TypeError(f"Invalid `scan_func`. The element types for the "
                          f"new state must match the initial state. Expected "
                          f"{old_state_types}, got {new_state_types}.")

      # Extract shape information from the returned values.
      new_state_shapes, output_shapes = wrapped_func.output_shapes
      old_state_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
          self._state_structure)
      self._element_spec = structure.convert_legacy_structure(
          output_types, output_shapes, output_classes)

      flat_state_shapes = nest.flatten(old_state_shapes)
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
        # TODO(b/110122868): Support a "most specific compatible structure"
        # method for combining structures, to avoid using legacy structures
        # in this method.
        self._state_structure = structure.convert_legacy_structure(
            old_state_types,
            nest.pack_sequence_as(old_state_shapes, weakened_state_shapes),
            old_state_classes)

    self._scan_func = wrapped_func
    self._scan_func.function.add_to_graph(ops.get_default_graph())

    self._name = name
    # pylint: disable=protected-access
    if use_default_device is not None:
      variant_tensor = ged_ops.scan_dataset(
          self._input_dataset._variant_tensor,
          structure.to_tensor_list(self._state_structure, self._initial_state),
          self._scan_func.function.captured_inputs,
          f=self._scan_func.function,
          preserve_cardinality=True,
          use_default_device=use_default_device,
          **self._common_args)
    else:
      variant_tensor = ged_ops.scan_dataset(
          self._input_dataset._variant_tensor,
          structure.to_tensor_list(self._state_structure, self._initial_state),
          self._scan_func.function.captured_inputs,
          f=self._scan_func.function,
          preserve_cardinality=True,
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._scan_func]

  @property
  def element_spec(self):
    return self._element_spec

  def _transformation_name(self):
    return "Dataset.scan()"
