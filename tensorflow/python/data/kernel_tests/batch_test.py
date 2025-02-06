# -*- coding: utf-8 -*-
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
"""Tests for `tf.data.Dataset.batch()`."""

import time
from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class BatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              count=[0, 28],
              batch_size=[14, 15],
              drop_remainder=[True, False],
              num_parallel_calls=[None, 1, 2, 4])))
  def testBasic(self, count, batch_size, drop_remainder, num_parallel_calls):
    """Tests the batch dataset logic for various input configurations.

    Args:
      count: the number of input elements
      batch_size: the batch size
      drop_remainder: whether a smaller batch size should be produced if batch
        size does not divide number of inputs evenly
      num_parallel_calls: the number batches to process asynchronously in
        parallel
    """

    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count) -> BatchDataset(batch_size).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        _map_fn).repeat(count).batch(batch_size, drop_remainder,
                                     num_parallel_calls)
    get_next = self.getNext(dataset)

    if drop_remainder:
      dim0 = batch_size
    else:
      dim0 = None
    self.assertEqual(
        [ts.as_list() for ts in nest.flatten(
            dataset_ops.get_legacy_output_shapes(dataset))],
        [[dim0] + list(c.shape[1:]) for c in components])

    num_full_batches = (count * 7) // batch_size
    for i in range(num_full_batches):
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range(batch_size):
          self.assertAllEqual(component[(i * batch_size + j) % 7]**2,
                              result_component[j])
    if not drop_remainder and (count * 7) % batch_size > 0:
      result = self.evaluate(get_next())
      for component, result_component in zip(components, result):
        for j in range((count * 7) % batch_size):
          self.assertAllEqual(
              component[(num_full_batches * batch_size + j) % 7]**2,
              result_component[j])
    with self.assertRaises(errors.OutOfRangeError):
      result = self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidBatchSize(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = (dataset_ops.Dataset.range(10).batch(0))
      self.evaluate(dataset._variant_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testDataset(self):

    def map_fn(i):
      return dataset_ops.Dataset.from_tensors(i)

    dataset = dataset_ops.Dataset.range(10).map(map_fn).batch(5)
    dataset = dataset.map(lambda x: x)
    dataset = dataset.unbatch().flat_map(lambda x: x)
    self.assertDatasetProduces(dataset, expected_output=range(10))

  def testSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5)
    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
            dense_shape=[5, 1]) for i in range(2)
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testSparseWithDifferentDenseShapes(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=array_ops.expand_dims(
              math_ops.range(i, dtype=dtypes.int64), 1),
          values=array_ops.fill([math_ops.cast(i, dtypes.int32)], i),
          dense_shape=[i])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5)
    expected_output = []
    for i in range(2):
      expected_indices = []
      expected_outputs = []
      for j in range(5):
        for k in range(i * 5 + j):
          expected_indices.append([j, k])
          expected_outputs.append(i * 5 + j)
      expected_output.append(
          sparse_tensor.SparseTensorValue(
              indices=expected_indices,
              values=expected_outputs,
              dense_shape=[5, (i + 1) * 5 - 1]))
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testSparseNested(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).map(_sparse).batch(5).batch(2)
    expected_output = [
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0],
                     [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0]],
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dense_shape=[2, 5, 1])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testShapeError(self):

    def generator():
      yield [1.0, 2.0, 3.0]
      yield [4.0, 5.0, 6.0]
      yield [7.0, 8.0, 9.0, 10.0]

    dataset = (
        dataset_ops.Dataset.from_generator(
            generator, dtypes.float32, output_shapes=[None]).batch(3))
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            r"Cannot batch tensors with different shapes in component 0. First "
            r"element had shape \[3\] and element 2 had shape \[4\]."))

  @combinations.generate(test_base.default_test_combinations())
  def testRagged(self):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(10).map(_ragged).batch(5)
    expected_output = [
        ragged_factory_ops.constant([[[0]], [[1]], [[2]], [[3]], [[4]]]),
        ragged_factory_ops.constant([[[5]], [[6]], [[7]], [[8]], [[9]]])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedWithDifferentShapes(self):
    dataset = dataset_ops.Dataset.range(10).map(ragged_math_ops.range).batch(5)
    expected_output = [
        ragged_concat_ops.stack([ragged_math_ops.range(i) for i in range(5)]),
        ragged_concat_ops.stack(
            [ragged_math_ops.range(i) for i in range(5, 10)])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedNested(self):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(10).map(_ragged).batch(5).batch(2)
    expected_output = [
        ragged_factory_ops.constant([[[[0]], [[1]], [[2]], [[3]], [[4]]],
                                     [[[5]], [[6]], [[7]], [[8]], [[9]]]])
    ]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneComponent(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: (x, None)).batch(
        10).map(lambda x, y: x)
    self.assertDatasetProduces(dataset, expected_output=[list(range(10))])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def testDeterminismConfiguration(self, local_determinism, global_determinism):
    expect_determinism = local_determinism or (local_determinism is None and
                                               global_determinism)
    elements = list(range(100))

    def dataset_fn(delay_ms):

      def sleep(x):
        time.sleep(delay_ms / 1000)
        return x

      def map_function(x):
        if math_ops.equal(x, 0):
          return script_ops.py_func(sleep, [x], x.dtype)
        else:
          return x

      dataset = dataset_ops.Dataset.from_tensor_slices(elements)
      dataset = dataset.map(
          map_function, num_parallel_calls=2, deterministic=local_determinism)
      dataset = dataset.batch(
          batch_size=6, num_parallel_calls=2,
          deterministic=local_determinism).unbatch()
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)
      return dataset

    self.checkDeterminism(dataset_fn, expect_determinism, elements)

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeBatches(self):
    # Batches of size 512M
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.ones((64, 1024, 1024), dtype=dtypes.float32)).repeat()
    dataset = dataset.batch(2, num_parallel_calls=5)
    iterator = iter(dataset)
    next(iterator)  # request an element to fill the buffer
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testName(self, num_parallel_calls):
    dataset = dataset_ops.Dataset.range(5).batch(
        5, num_parallel_calls=num_parallel_calls, name="batch")
    self.assertDatasetProduces(dataset, [list(range(5))])


class BatchCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):

  def _build_dataset(self,
                     multiplier=15.0,
                     tensor_slice_len=2,
                     batch_size=2,
                     num_parallel_calls=None,
                     options=None):
    components = (np.arange(tensor_slice_len), np.array([[1, 2, 3]]) *
                  np.arange(tensor_slice_len)[:, np.newaxis],
                  np.array(multiplier) * np.arange(tensor_slice_len))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.batch(batch_size, num_parallel_calls=num_parallel_calls)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              symbolic_checkpoint=[False, True], num_parallel_calls=[None, 4]
          ),
      )
  )
  def test(self, verify_fn, symbolic_checkpoint, num_parallel_calls):
    tensor_slice_len = 8
    batch_size = 2
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    num_outputs = tensor_slice_len // batch_size
    verify_fn(
        self,
        lambda: self._build_dataset(
            15.0, tensor_slice_len, batch_size, num_parallel_calls, options
        ),
        num_outputs,
    )

  def _sparse(self, i):
    return sparse_tensor.SparseTensorValue(
        indices=[[0]], values=(i * [1]), dense_shape=[1])

  def _build_dataset_sparse(self, batch_size=5):
    return dataset_ops.Dataset.range(10).map(self._sparse).batch(batch_size)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    verify_fn(self, self._build_dataset_sparse, num_outputs=2)

  def _build_dataset_nested_sparse(self):
    return dataset_ops.Dataset.range(10).map(self._sparse).batch(5).batch(2)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testNestedSparse(self, verify_fn):
    verify_fn(self, self._build_dataset_nested_sparse, num_outputs=1)


class BatchRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3, 4])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4]).batch(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).batch(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              count=[0, 10, 20, 30, 40, 50],
              batch_size=[1, 3, 5, 7, 10, 20],
              drop_remainder=[True, False])))
  def testBasic(self, count, batch_size, drop_remainder):
    """Tests the batch dataset logic for various input configurations.

    Args:
      count: the number of input elements
      batch_size: the batch size
      drop_remainder: whether a smaller batch size should be produced if batch
        size does not divide number of inputs evenly
    """
    dataset = dataset_ops.Dataset.from_tensor_slices(list(range(count))).batch(
        batch_size=batch_size, drop_remainder=drop_remainder)
    num_full_batches = count // batch_size
    for i in range(num_full_batches):
      expected_batch = np.arange(
          i * batch_size, (i * batch_size + batch_size), 1, dtype=np.int32)
      self.assertAllEqual(expected_batch,
                          self.evaluate(random_access.at(dataset, i)))
    has_remainder = (not drop_remainder) and (count % batch_size != 0)
    if has_remainder:
      expected_batch = np.arange(batch_size * num_full_batches, count, 1)
      self.assertAllEqual(
          expected_batch,
          self.evaluate(random_access.at(dataset, num_full_batches)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(
          random_access.at(
              dataset, index=num_full_batches + (1 if has_remainder else 0)))

  @combinations.generate(test_base.default_test_combinations())
  def testRandomAccessBatchWithShuffle(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7])
    shuffle_dataset = dataset.shuffle(buffer_size=10, seed=2)
    batch_dataset = shuffle_dataset.batch(2)

    expected_output = [
        np.array([5, 2], dtype=np.int32),
        np.array([4, 7], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([6], dtype=np.int32)
    ]
    for i in range(4):
      self.assertAllEqual(expected_output[i],
                          self.evaluate(random_access.at(batch_dataset, i)))

    # Checks the order is consistent with shuffle dataset.
    for i in range(3):
      self.assertAllEqual(
          expected_output[i][0],
          self.evaluate(random_access.at(shuffle_dataset, i * 2)))
      self.assertAllEqual(
          expected_output[i][1],
          self.evaluate(random_access.at(shuffle_dataset, (i * 2) + 1)))

    # Checks the remainder is the last element in shuffled dataset.
    self.assertAllEqual(expected_output[3][0],
                        self.evaluate(random_access.at(shuffle_dataset, 6)))


class BatchGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              batch_size=[2, 7])))
  def testBatch(
      self, dataset_range: int, batch_size: int):
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(dataset)
    dataset = dataset.unbatch()

    expected = list(range(0, (dataset_range // batch_size) * batch_size))
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              batch_size=[2, 7],
              reshuffle=[True, False],
              seed=[None, 42])))
  def testReshuffleRepeatEpochs(
      self,
      dataset_range: int,
      batch_size: int,
      reshuffle: bool,
      seed: Optional[int]):
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle)
    dataset = dataset.repeat(2)
    dataset = dataset.unbatch()

    expected = list(range(0, (dataset_range // batch_size) * batch_size))
    len_per_iteration = len(expected)
    expected *= 2

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, expected)
    output_per_iteration = [
        output[i : i + len_per_iteration]
        for i in range(0, len(output), len_per_iteration)]
    if reshuffle:
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
    else:
      self.assertEqual(output_per_iteration[0], output_per_iteration[1])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              batch_size=[2, 7])))
  def testNoDropRemainder(
      self, dataset_range: int, batch_size: int):
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "does not support global shuffling with `drop_remainder=False`."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class BatchGlobalShuffleCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              batch_size=[2, 3],
              symbolic_checkpoint=[True, False])))
  def testBatch(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      batch_size: int,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(dataset, seed=42)
      dataset = dataset.unbatch()
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=(dataset_range // batch_size) * batch_size,
        assert_items_equal=True)

  # Creating multiple iterators with the same seed is only supported in v2 API.
  @combinations.generate(
      combinations.times(
          combinations.combine(tf_api_version=2, mode="eager"),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              batch_size=[2, 3],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testReshuffleEachIteration(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      batch_size: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      dataset = dataset.unbatch()
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=(dataset_range // batch_size) * batch_size,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
