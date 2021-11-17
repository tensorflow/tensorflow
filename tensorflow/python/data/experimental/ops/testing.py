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
