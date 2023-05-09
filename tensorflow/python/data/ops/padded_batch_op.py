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
"""The implementation of `tf.data.Dataset.padded_batch`."""

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops


def _padded_batch(input_dataset,
                  batch_size,
                  padded_shapes=None,
                  padding_values=None,
                  drop_remainder=False,
                  name=None):
  """See `tf.data.Dataset.padded_batch` for details."""
  if padded_shapes is None:
    padded_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
    for i, shape in enumerate(nest.flatten(padded_shapes)):
      # A `tf.TensorShape` is only false if its *rank* is unknown.
      if not shape:
        raise ValueError(f"You must provide `padded_shapes` argument because "
                         f"component {i} has unknown rank.")
  return _PaddedBatchDataset(
      input_dataset,
      batch_size,
      padded_shapes,
      padding_values,
      drop_remainder,
      name=name)


def _is_padded_shape_compatible_with(padded_shape, input_component_shape):
  """Returns `True` if `input_component_shape` can be padded to `padded_shape`.

  Args:
    padded_shape: A `tf.TensorShape`.
    input_component_shape: A `tf.TensorShape`.

  Returns:
    `True` if `input_component_shape` can be padded to `padded_shape`, otherwise
    `False`.
  """

  if padded_shape.dims is None or input_component_shape.dims is None:
    return True
  if len(padded_shape.dims) != len(input_component_shape.dims):
    return False
  for padded_dim, input_dim in zip(padded_shape.dims,
                                   input_component_shape.dims):
    if (padded_dim.value is not None and input_dim.value is not None and
        padded_dim.value < input_dim.value):
      return False
  return True


def _padded_shape_to_tensor(padded_shape, input_component_shape):
  """Converts `padded_shape` to a `tf.Tensor` representing that shape.

  Args:
    padded_shape: A shape-like object, which may be a `tf.TensorShape`, a Python
      sequence, or a 1-D `tf.Tensor` of `tf.int64` elements.
    input_component_shape: A `tf.TensorShape`, with which `padded_shape` must be
      compatible.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements, representing `padded_shape`.

  Raises:
    ValueError: If `padded_shape` is not a shape or not compatible with
      `input_component_shape`.
    TypeError: If `padded_shape` is not convertible to a `tf.int64` tensor.
  """
  try:
    # Try to convert the `padded_shape` to a `tf.TensorShape`
    padded_shape_as_shape = tensor_shape.as_shape(padded_shape)
    # We will return the "canonical" tensor representation, which uses
    # `-1` in place of `None`.
    ret = ops.convert_to_tensor([
        dim if dim is not None else -1
        for dim in padded_shape_as_shape.as_list()
    ],
                                dtype=dtypes.int64)
  except (TypeError, ValueError) as e:
    # The argument was not trivially convertible to a
    # `tf.TensorShape`, so fall back on the conversion to tensor
    # machinery.
    ret = ops.convert_to_tensor(padded_shape, preferred_dtype=dtypes.int64)
    if ret.shape.dims is not None and len(ret.shape.dims) != 1:
      raise ValueError(
          f"Padded shape {padded_shape} must be a `tf.int64` vector tensor, "
          f"but its shape was {ret.shape}.") from e
    if ret.dtype != dtypes.int64:
      raise TypeError(
          f"Padded shape {padded_shape} must be a `tf.int64` vector "
          f"tensor, but its element type was {ret.dtype.name}.") from e
    padded_shape_as_shape = tensor_util.constant_value_as_shape(ret)

  if not _is_padded_shape_compatible_with(padded_shape_as_shape,
                                          input_component_shape):
    raise ValueError(f"The padded shape {padded_shape_as_shape} is not "
                     f"compatible with the shape {input_component_shape} of "
                     f"the corresponding input component.")

  return ret


def _padding_values_or_default(padding_values, input_dataset):
  """Returns padding values with None elements replaced with default values."""

  def make_zero(t):
    if t.base_dtype == dtypes.string:
      return ""
    elif t.base_dtype == dtypes.variant:
      raise TypeError("Unable to create default padding value for a component "
                      "of type 'variant'.")
    elif t.base_dtype == dtypes.bfloat16:
      # Special case `bfloat16` because it is not supported by NumPy.
      return constant_op.constant(0, dtype=dtypes.bfloat16)
    else:
      return np.zeros_like(t.as_numpy_dtype())

  def value_or_default(value, default):
    return default if value is None else value

  default_padding = nest.map_structure(
      make_zero, dataset_ops.get_legacy_output_types(input_dataset))
  return nest.map_structure_up_to(padding_values, value_or_default,
                                  padding_values, default_padding)


def _padding_value_to_tensor(value, output_type):
  """Converts the padding value to a tensor.

  Args:
    value: The padding value.
    output_type: Its expected dtype.

  Returns:
    A scalar `Tensor`.

  Raises:
    ValueError: if the padding value is not a scalar.
    TypeError: if the padding value's type does not match `output_type`.
  """
  value = ops.convert_to_tensor(value, name="padding_value")
  if not value.shape.is_compatible_with(tensor_shape.TensorShape([])):
    raise ValueError(f"Invalid `padding_values`. `padding_values` values "
                     f"should be scalars, but got {value.shape}.")
  if value.dtype != output_type:
    raise TypeError(f"Invalid `padding_values`. `padding_values` values "
                    f"type {value.dtype} does not match type {output_type} "
                    f"of the corresponding input component.")
  return value


class _PaddedBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self,
               input_dataset,
               batch_size,
               padded_shapes,
               padding_values,
               drop_remainder,
               name=None):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset

    def check_types(component_spec):
      if not isinstance(component_spec, tensor_spec.TensorSpec):
        if isinstance(component_spec, dataset_ops.DatasetSpec):
          raise TypeError(
              "`padded_batch` is not supported for datasets of datasets")
        raise TypeError(f"`padded_batch` is only supported for datasets that "
                        f"produce tensor elements but type spec of elements in "
                        f"the input dataset is not a subclass of TensorSpec: "
                        f"`{component_spec}`.")

    nest.map_structure(check_types, input_dataset.element_spec)
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    padding_values = _padding_values_or_default(padding_values, input_dataset)

    input_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
    flat_padded_shapes = nest.flatten_up_to(input_shapes, padded_shapes)

    flat_padded_shapes_as_tensors = []

    for input_component_shape, padded_shape in zip(
        nest.flatten(input_shapes), flat_padded_shapes):
      flat_padded_shapes_as_tensors.append(
          _padded_shape_to_tensor(padded_shape, input_component_shape))

    self._padded_shapes = nest.pack_sequence_as(input_shapes,
                                                flat_padded_shapes_as_tensors)

    # If padding_values is a single element and input_shapes is a structure,
    # "broadcast" padding_values to the same structure as input_shapes.
    if nest.is_nested(input_shapes) and not nest.is_nested(padding_values):
      padding_values = nest.map_structure(lambda _: padding_values,
                                          input_shapes)

    self._padding_values = nest.map_structure_up_to(
        input_shapes, _padding_value_to_tensor, padding_values,
        dataset_ops.get_legacy_output_types(input_dataset))
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    def _padded_shape_to_batch_shape(s):
      return tensor_shape.TensorShape([
          tensor_util.constant_value(self._batch_size)
          if smart_cond.smart_constant_value(self._drop_remainder) else None
      ]).concatenate(tensor_util.constant_value_as_shape(s))

    output_shapes = nest.map_structure(_padded_shape_to_batch_shape,
                                       self._padded_shapes)
    self._structure = structure.convert_legacy_structure(
        dataset_ops.get_legacy_output_types(self._input_dataset), output_shapes,
        dataset_ops.get_legacy_output_classes(self._input_dataset))

    self._name = name
    # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        batch_size=self._batch_size,
        padded_shapes=[
            ops.convert_to_tensor(s, dtype=dtypes.int64)
            for s in nest.flatten(self._padded_shapes)
        ],
        padding_values=nest.flatten(self._padding_values),
        drop_remainder=self._drop_remainder,
        output_shapes=structure.get_flat_tensor_shapes(self._structure),
        metadata=self._metadata.SerializeToString())
    super().__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure
