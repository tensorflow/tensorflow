# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for testing of tf.data."""
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops


def assert_next(transformations):
  """A transformation that asserts which transformations happen next.

  Transformations should be referred to by their base name, not including
  version suffix. For example, use "Batch" instead of "BatchV2". "Batch" will
  match any of "Batch", "BatchV1", "BatchV2", etc.

  Args:
    transformations: A `tf.string` vector `tf.Tensor` identifying the
      transformations that are expected to happen next.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _AssertNextDataset(dataset, transformations)

  return _apply_fn


def assert_prev(transformations):
  r"""Asserts which transformations, with which attributes, happened previously.

    Each transformation is repesented as a tuple in the input.

    The first element is the base op name of the transformation, not including
    version suffix.  For example, use "BatchDataset" instead of
    "BatchDatasetV2".  "BatchDataset" will match any of "BatchDataset",
    "BatchDatasetV1", "BatchDatasetV2", etc.

    The second element is a dict of attribute name-value pairs.  Attributes
    values must be of type bool, int, or string.

    Example usage:

    >>> dataset_ops.Dataset.from_tensors(0) \
    ... .map(lambda x: x) \
    ... .batch(1, deterministic=True, num_parallel_calls=8) \
    ... .assert_prev([("ParallelBatchDataset", {"deterministic": True}), \
    ...               ("MapDataset", {})])

  Args:
    transformations: A list of tuples identifying the (required) transformation
      name, with (optional) attribute name-value pairs, that are expected to
      have happened previously.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _AssertPrevDataset(dataset, transformations)

  return _apply_fn


def non_serializable():
  """A non-serializable identity transformation.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _NonSerializableDataset(dataset)

  return _apply_fn


def sleep(sleep_microseconds):
  """Sleeps for `sleep_microseconds` before producing each input element.

  Args:
    sleep_microseconds: The number of microseconds to sleep before producing an
      input element.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SleepDataset(dataset, sleep_microseconds)

  return _apply_fn


class _AssertNextDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that asserts which transformations happen next."""

  def __init__(self, input_dataset, transformations):
    """See `assert_next()` for details."""
    self._input_dataset = input_dataset
    if transformations is None:
      raise ValueError(
          "Invalid `transformations`. `transformations` should not be empty.")

    self._transformations = ops.convert_to_tensor(
        transformations, dtype=dtypes.string, name="transformations")
    variant_tensor = (
        gen_experimental_dataset_ops.experimental_assert_next_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._transformations,
            **self._flat_structure))
    super(_AssertNextDataset, self).__init__(input_dataset, variant_tensor)


class _AssertPrevDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that asserts which transformations happened previously."""

  def __init__(self, input_dataset, transformations):
    """See `assert_prev()` for details."""
    self._input_dataset = input_dataset
    if transformations is None:
      raise ValueError("`transformations` cannot be empty")

    def serialize_transformation(op_name, attributes):
      proto = attr_value_pb2.NameAttrList(name=op_name)
      if attributes is None or isinstance(attributes, set):
        attributes = dict()
      for (name, value) in attributes.items():
        if isinstance(value, bool):
          proto.attr[name].b = value
        elif isinstance(value, int):
          proto.attr[name].i = value
        elif isinstance(value, str):
          proto.attr[name].s = value.encode()
        else:
          raise ValueError(
              f"attribute value type ({type(value)}) must be bool, int, or str")
      return text_format.MessageToString(proto)

    self._transformations = ops.convert_to_tensor(
        [serialize_transformation(*x) for x in transformations],
        dtype=dtypes.string,
        name="transformations")
    variant_tensor = (
        gen_experimental_dataset_ops.assert_prev_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            self._transformations,
            **self._flat_structure))
    super(_AssertPrevDataset, self).__init__(input_dataset, variant_tensor)


class _NonSerializableDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that performs non-serializable identity transformation."""

  def __init__(self, input_dataset):
    """See `non_serializable()` for details."""
    self._input_dataset = input_dataset
    variant_tensor = (
        gen_experimental_dataset_ops.experimental_non_serializable_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            **self._flat_structure))
    super(_NonSerializableDataset, self).__init__(input_dataset, variant_tensor)


class _SleepDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that sleeps before producing each upstream element."""

  def __init__(self, input_dataset, sleep_microseconds):
    self._input_dataset = input_dataset
    self._sleep_microseconds = sleep_microseconds
    variant_tensor = gen_experimental_dataset_ops.sleep_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._sleep_microseconds,
        **self._flat_structure)
    super(_SleepDataset, self).__init__(input_dataset, variant_tensor)
