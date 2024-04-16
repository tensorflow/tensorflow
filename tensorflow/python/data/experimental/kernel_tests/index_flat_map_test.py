# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the index flat map dataset."""

from typing import Any, Callable, Union

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import index_flat_map_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test

_IndexType = index_flat_map_op._IndexType


class IndexFlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for global shuffling of index flat map datasets."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(use_tensors=[True, False])))
  def test_split_strings(self, use_tensors: bool):
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    # The metadata is [(0, 2, 0), (2, 6, 1), (6, 8, 2), (8, 9, 3)].
    metadata = _get_metadata(input_data)

    def _index_map_func(index: _IndexType) -> tuple[_IndexType, _IndexType]:
      index = _maybe_convert_to_tensor(index)
      element_index, offset = _get_index_map_func(metadata)(index)
      return (_maybe_convert_to_tensor(element_index),
              _maybe_convert_to_tensor(offset))

    def _maybe_convert_to_tensor(value: Any) -> _IndexType:
      return math_ops.cast(value, dtypes.int64) if use_tensors else value

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(dataset, _split, _index_map_func)
    output = self.getDatasetOutput(dataset)
    self.assertEqual(output,
                     [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"])

  @combinations.generate(test_base.default_test_combinations())
  def test_cache(self):
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = dataset.cache()
    dataset = index_flat_map_op.index_flat_map(
        dataset, _split, _get_index_map_func(metadata))
    output = self.getDatasetOutput(dataset)
    self.assertEqual(output,
                     [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"])

  @combinations.generate(test_base.default_test_combinations())
  def test_global_shuffle(self):
    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _split, _get_index_map_func(metadata))
    dataset = dataset.apply(cardinality_lib.assert_cardinality(9))
    dataset = global_shuffle_op._global_shuffle(dataset)

    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    expected = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8"]
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(dataset_range=[0, 10])))
  def test_identity_map(self, dataset_range: int):

    def _map_func(element: Any) -> Any:
      return element

    def _index_map_func(index: int) -> tuple[int, int]:
      return (index, 0)

    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _map_func, _index_map_func)
    self.assertDatasetProduces(dataset, list(range(dataset_range)))

  @combinations.generate(test_base.default_test_combinations())
  def test_invalid_map_fn(self):

    def _index_map_func(_) -> str:
      # Expected to return two integers.
      return "Hello"

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
    dataset = index_flat_map_op.index_flat_map(
        dataset, _split, _index_map_func)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "expected to return two int values"):
      self.getDatasetOutput(dataset)


class IndexFlatMapCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  # TODO(b/325112575): Support the graph mode.
  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[True, False])))
  def test_index_flat_map(
      self,
      verify_fn: Callable[..., None],
      symbolic_checkpoint: bool):

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
      dataset = index_flat_map_op.index_flat_map(
          dataset, _split, _get_index_map_func(metadata))
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(self, _build_dataset, num_outputs=9)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def test_global_shuffle(
      self,
      verify_fn: Callable[..., None],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):

    input_data = ["0 1", "2 3 4 5", "6 7", "8"]
    metadata = _get_metadata(input_data)

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.from_tensor_slices(input_data)
      dataset = index_flat_map_op.index_flat_map(
          dataset, _split, _get_index_map_func(metadata))
      dataset = dataset.apply(cardinality_lib.assert_cardinality(9))
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)

      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=9,
        assert_items_equal=reshuffle_each_iteration)


def _split(element: str) -> tensor.Tensor:
  return ragged_string_ops.string_split_v2(element, " ")


def _get_metadata(input_data: list[str]) -> tensor.Tensor:
  """Given a list of strings, creates a metadata matrix."""

  metadata = []
  for i, data in enumerate(input_data):
    split_data = data.split()
    last_index = metadata[-1][1] if metadata else 0
    metadata.append((last_index, last_index + len(split_data), i))
  return constant_op.constant(metadata, dtype=dtypes.int64)


def _get_index_map_func(
    metadata: tensor.Tensor) -> Callable[[int], tuple[int, int]]:
  """Turns a `metadata` Tensor into an index map function."""

  def _index_map_func(index: Union[int, tensor.Tensor]) -> tuple[int, int]:
    element_index = 0
    while (element_index < metadata.shape[0] and
           index >= array_ops.gather_nd(metadata, [element_index, 1])):
      element_index += 1
    offset = (
        index - array_ops.gather_nd(metadata, [element_index, 0])
        if element_index < metadata.shape[0]
        else constant_op.constant(0, dtype=dtypes.int64))
    return (element_index, offset)

  return _index_map_func


if __name__ == "__main__":
  test.main()
