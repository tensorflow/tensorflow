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
"""Tests for training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import multiprocessing.pool
import time

from absl.testing import parameterized
import numpy as np


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class ModelInputsTest(test.TestCase):

  def test_single_thing(self):
    a = np.ones(10)
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['input_1'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tf_type(vals))
    vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.assertEqual(1, len(vals))
    self.assertTrue(tensor_util.is_tf_type(vals[0]))
    self.assertEqual(backend.floatx(), vals[0].dtype)

  def test_single_thing_eager(self):
    if not context.executing_eagerly():
      self.skipTest('Run in eager mode only.')
    a = np.ones(10, dtype=np.int32)
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['input_1'], model_inputs.get_input_names())
    val = model_inputs.get_symbolic_inputs()
    self.assertIsInstance(val, keras_tensor.KerasTensor)
    vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.assertEqual(1, len(vals))
    self.assertIsInstance(vals[0], keras_tensor.KerasTensor)
    self.assertEqual(dtypes.int32, vals[0].dtype)

  def test_list(self):
    a = [np.ones(10), np.ones(20)]
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tf_type(vals[0]))
    self.assertTrue(tensor_util.is_tf_type(vals[1]))

  def test_list_eager(self):
    if not context.executing_eagerly():
      self.skipTest('Run in eager mode only.')
    a = [np.ones(10), np.ones(20)]
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertIsInstance(vals[0], keras_tensor.KerasTensor)
    self.assertIsInstance(vals[1], keras_tensor.KerasTensor)

  def test_dict(self):
    a = {'b': np.ones(10), 'a': np.ones(20)}
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['a', 'b'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tf_type(vals['a']))
    self.assertTrue(tensor_util.is_tf_type(vals['b']))

  def test_dict_eager(self):
    if not context.executing_eagerly():
      self.skipTest('Run in eager mode only.')
    a = {'b': np.ones(10), 'a': np.ones(20)}
    model_inputs = training_utils_v1.ModelInputs(a)
    self.assertEqual(['a', 'b'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertIsInstance(vals['a'], keras_tensor.KerasTensor)
    self.assertIsInstance(vals['b'], keras_tensor.KerasTensor)


class DatasetUtilsTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # pylint: disable=g-long-lambda
      ('Batch', lambda: dataset_ops.Dataset.range(5).batch(2)),
      ('Cache', lambda: dataset_ops.Dataset.range(5).cache()),
      ('Concatenate', lambda: dataset_ops.Dataset.range(5).concatenate(
          dataset_ops.Dataset.range(5))),
      ('FlatMap', lambda: dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensors(0))),
      ('FlatMap_Shuffle', lambda: dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensors(0).shuffle(1)), True),
      ('Filter', lambda: dataset_ops.Dataset.range(5).filter(lambda _: True)),
      ('FixedLengthRecordDatasetV2',
       lambda: readers.FixedLengthRecordDatasetV2([], 42)),
      ('FromTensors', lambda: dataset_ops.Dataset.from_tensors(0)),
      ('FromTensorSlices',
       lambda: dataset_ops.Dataset.from_tensor_slices([0, 0, 0])),
      ('Interleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0), cycle_length=1)),
      ('Interleave_Shuffle', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0).shuffle(1),
          cycle_length=1), True),
      ('Map', lambda: dataset_ops.Dataset.range(5).map(lambda x: x)),
      ('Options',
       lambda: dataset_ops.Dataset.range(5).with_options(dataset_ops.Options())
      ),
      ('PaddedBatch', lambda: dataset_ops.Dataset.range(5).padded_batch(2, [])),
      ('ParallelInterleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0),
          cycle_length=1,
          num_parallel_calls=1)),
      ('ParallelMap', lambda: dataset_ops.Dataset.range(5).map(
          lambda x: x, num_parallel_calls=1)),
      ('Prefetch', lambda: dataset_ops.Dataset.range(5).prefetch(1)),
      ('Range', lambda: dataset_ops.Dataset.range(0)),
      ('Repeat', lambda: dataset_ops.Dataset.range(0).repeat(0)),
      ('Shuffle', lambda: dataset_ops.Dataset.range(5).shuffle(1), True),
      ('Skip', lambda: dataset_ops.Dataset.range(5).skip(2)),
      ('Take', lambda: dataset_ops.Dataset.range(5).take(2)),
      ('TextLineDataset', lambda: readers.TextLineDatasetV2([])),
      ('TFRecordDataset', lambda: readers.TFRecordDatasetV2([])),
      ('Window', lambda: dataset_ops.Dataset.range(5).window(2)),
      ('Zip', lambda: dataset_ops.Dataset.zip(dataset_ops.Dataset.range(5))),
      # pylint: enable=g-long-lambda
  )
  def test_verify_dataset_shuffled(self, dataset_fn, expect_shuffled=False):
    dataset = dataset_fn()

    if not expect_shuffled:
      with test.mock.patch.object(logging, 'warning') as mock_log:
        shuffled = training_utils_v1.verify_dataset_shuffled(dataset)
        self.assertRegex(
            str(mock_log.call_args), 'input dataset `x` is not shuffled.')
        self.assertFalse(shuffled)
    else:
      self.assertTrue(training_utils_v1.verify_dataset_shuffled(dataset))


class StandardizeWeightsTest(keras_parameterized.TestCase):

  def test_sample_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    sample_weights = np.array([0.5, 1., 1., 0., 2.])
    weights = training_utils_v1.standardize_weights(y, sample_weights)
    self.assertAllClose(weights, sample_weights)

  def test_class_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    class_weights = {0: 0.5, 1: 1., 2: 1.5}
    weights = training_utils_v1.standardize_weights(
        y, class_weight=class_weights)
    self.assertAllClose(weights, np.array([0.5, 1., 0.5, 0.5, 1.5]))

  def test_sample_weights_and_class_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    sample_weights = np.array([0.5, 1., 1., 0., 2.])
    class_weights = {0: 0.5, 1: 1., 2: 1.5}
    weights = training_utils_v1.standardize_weights(y, sample_weights,
                                                    class_weights)
    expected = sample_weights * np.array([0.5, 1., 0.5, 0.5, 1.5])
    self.assertAllClose(weights, expected)

  def test_dataset_with_class_weight(self):
    model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
    model.compile('rmsprop', 'mse')

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    class_weight_np = np.array([0.25, 0.25, 0.25, 0.25])
    class_weight = dict(enumerate(class_weight_np))

    model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=2,
        verbose=1,
        class_weight=class_weight)


class MonitoredPool(multiprocessing.pool.ThreadPool):

  def __init__(self, *args, **kwargs):
    self._apply_counter = 0
    self._func_wrapper = None
    super(MonitoredPool, self).__init__(*args, **kwargs)

  def apply_async(self, func, *args, **kwargs):
    self._apply_counter += 1
    if self._func_wrapper:
      func = self._func_wrapper(func)  # pylint: disable=not-callable
    return super(MonitoredPool, self).apply_async(func, *args, **kwargs)


def add_sleep(f):
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    time.sleep(1.)
    return f(*args, **kwargs)
  return wrapped


def cause_error(f):
  @functools.wraps(f)
  def wrapped(batch_element, batch_start, batch_end, is_finished):  # pylint: disable=unused-argument
    # Induce a TypeError during assignment.
    return f(None, None, None, is_finished)
  return wrapped


_TEST_DATA = np.array((
    (3, 1, 3, 1, 2, 0, 3, 3, 1, 2),
    (0, 1, 2, 1, 3, 0, 0, 1, 3, 0),
    (3, 2, 1, 1, 1, 1, 1, 3, 2, 3),
    (2, 2, 0, 1, 0, 3, 3, 2, 1, 1),
    (3, 0, 3, 3, 3, 2, 1, 0, 0, 1),
    (1, 0, 3, 3, 3, 2, 1, 2, 3, 1),))


class AggregationTest(keras_parameterized.TestCase):

  def setUp(self):
    super(AggregationTest, self).setUp()
    self._old_pool = training_utils_v1._COPY_POOL
    self._old_threshold = (
        training_utils_v1.SliceAggregator._BINARY_SIZE_THRESHOLD)
    self._old_timeout = training_utils_v1.SliceAggregator._MAX_COPY_SECONDS
    training_utils_v1._COPY_POOL = MonitoredPool(
        training_utils_v1._COPY_THREADS)

  def tearDown(self):
    super(AggregationTest, self).tearDown()
    training_utils_v1._COPY_POOL = self._old_pool
    training_utils_v1.SliceAggregator._BINARY_SIZE_THRESHOLD = (
        self._old_threshold)
    training_utils_v1.SliceAggregator._MAX_COPY_SECONDS = self._old_timeout

  def _run_with_steps(self):
    aggregator = training_utils_v1.OutputsAggregator(use_steps=True)
    for i, batch in enumerate(np.array_split(_TEST_DATA, 4)):
      if i == 0:
        aggregator.create(batch)
      aggregator.aggregate(batch)

    assert len(aggregator.results) == 1
    assert isinstance(aggregator.results[0], training_utils_v1.ConcatAggregator)

    aggregator.finalize()
    return aggregator.results

  def _run_without_steps(self):
    aggregator = training_utils_v1.OutputsAggregator(
        use_steps=False, num_samples=6)

    batch_start = 0
    for i, batch in enumerate(np.array_split(_TEST_DATA, 4)):
      if i == 0:
        aggregator.create(batch)

      batch_end = batch_start + batch.shape[0]
      aggregator.aggregate(batch, batch_start, batch_end)
      batch_start = batch_end

    assert len(aggregator.results) == 1
    assert isinstance(aggregator.results[0], training_utils_v1.SliceAggregator)

    aggregator.finalize()
    return aggregator.results

  def test_with_steps(self):
    self.assertAllEqual(self._run_with_steps(), _TEST_DATA)

  def test_without_steps(self):
    self.assertAllEqual(self._run_without_steps(), _TEST_DATA)

  def test_nested_aggregation(self):
    aggregator = training_utils_v1.OutputsAggregator(
        use_steps=False, num_samples=6)

    batches = np.array_split(_TEST_DATA, 4)
    batch_start = 0
    for i, batch in enumerate(zip(batches, batches)):
      if i == 0:
        aggregator.create(batch)

      batch_end = batch_start + batch[0].shape[0]
      aggregator.aggregate(batch, batch_start, batch_end)
      batch_start = batch_end

    assert len(aggregator.results) == 2
    aggregator.finalize()
    self.assertAllEqual(aggregator.results, (_TEST_DATA, _TEST_DATA))

  def test_concat_single_batch(self):
    aggregator = training_utils_v1.OutputsAggregator(use_steps=True)
    data = _TEST_DATA.copy()
    aggregator.create(data)
    assert len(aggregator.results) == 1
    assert isinstance(aggregator.results[0], training_utils_v1.ConcatAggregator)

    aggregator.aggregate(data)
    aggregator.finalize()
    assert aggregator.results is data  # No copy.

  def test_slice_single_batch(self):
    aggregator = training_utils_v1.OutputsAggregator(
        use_steps=False, num_samples=6)
    data = _TEST_DATA.copy()
    aggregator.create(data)
    assert len(aggregator.results) == 1
    assert isinstance(aggregator.results[0], training_utils_v1.SliceAggregator)

    aggregator.aggregate(data, 0, 6)
    aggregator.finalize()
    assert aggregator.results is data  # No copy.

  def test_async_copy(self):
    training_utils_v1.SliceAggregator._BINARY_SIZE_THRESHOLD = 15
    self.assertAllEqual(self._run_without_steps(), _TEST_DATA)

    # Two of the four batches will have 20 elements and two will have 10.
    self.assertEqual(training_utils_v1._COPY_POOL._apply_counter, 2)

  def test_async_copy_timeout(self):
    training_utils_v1.SliceAggregator._BINARY_SIZE_THRESHOLD = 15
    training_utils_v1.SliceAggregator._MAX_COPY_SECONDS = 0.1
    training_utils_v1._COPY_POOL._func_wrapper = add_sleep
    with self.assertRaisesRegex(ValueError, 'Timed out waiting for copy'):
      self._run_without_steps()

  def test_async_copy_reraise(self):
    training_utils_v1.SliceAggregator._BINARY_SIZE_THRESHOLD = 15
    training_utils_v1.SliceAggregator._MAX_COPY_SECONDS = 1.
    training_utils_v1._COPY_POOL._func_wrapper = cause_error
    with self.assertRaisesRegex(TypeError, 'NoneType'):
      self._run_without_steps()


class CompositeTensorTestUtils(keras_parameterized.TestCase):

  def test_is_composite(self):
    # Validate that all composite tensor and value types return true.
    self.assertTrue(
        training_utils_v1.is_composite_or_composite_value(
            sparse_tensor.SparseTensor([[0, 0]], [1], [1, 1])))
    self.assertTrue(
        training_utils_v1.is_composite_or_composite_value(
            sparse_tensor.SparseTensorValue([[0, 0]], [1], [1, 1])))
    self.assertTrue(
        training_utils_v1.is_composite_or_composite_value(
            ragged_tensor.RaggedTensor.from_row_splits(
                np.array([0, 1, 2]), np.array([0, 1, 3], dtype=np.int64))))
    self.assertTrue(
        training_utils_v1.is_composite_or_composite_value(
            ragged_tensor_value.RaggedTensorValue(
                np.array([0, 1, 2]), np.array([0, 1, 3], dtype=np.int64))))

    # Test that numpy arrays and tensors return false.
    self.assertFalse(
        training_utils_v1.is_composite_or_composite_value(np.ndarray([0, 1])))
    self.assertFalse(
        training_utils_v1.is_composite_or_composite_value(
            ops.convert_to_tensor_v2_with_dispatch([3, 1])))

  def test_sparse_concatenation(self):
    tensor_1 = sparse_tensor.SparseTensor([[0, 0]], [1], [1, 1])
    tensor_2 = sparse_tensor.SparseTensor([[0, 0]], [2], [1, 1])
    concatenated_tensor = training_utils_v1._append_composite_tensor(
        tensor_1, tensor_2)
    evaluated_tensor = self.evaluate(concatenated_tensor)
    self.assertAllEqual(evaluated_tensor.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(evaluated_tensor.values, [1, 2])
    self.assertAllEqual(evaluated_tensor.dense_shape, [2, 1])

  def test_sparse_value_concatenation(self):
    tensor_1 = sparse_tensor.SparseTensorValue([[0, 0]], [1], [1, 1])
    tensor_2 = sparse_tensor.SparseTensorValue([[0, 0]], [2], [1, 1])
    concatenated_tensor = training_utils_v1._append_composite_tensor(
        tensor_1, tensor_2)
    self.assertAllEqual(concatenated_tensor.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(concatenated_tensor.values, [1, 2])
    self.assertAllEqual(concatenated_tensor.dense_shape, [2, 1])

  def test_ragged_concatenation(self):
    tensor_1 = ragged_tensor.RaggedTensor.from_row_splits(
        np.array([0, 1, 2]), np.array([0, 1, 3], dtype=np.int64))
    tensor_2 = ragged_tensor.RaggedTensor.from_row_splits(
        np.array([3, 4, 5]), np.array([0, 2, 3], dtype=np.int64))
    concatenated_tensor = training_utils_v1._append_composite_tensor(
        tensor_1, tensor_2)
    evaluated_tensor = self.evaluate(concatenated_tensor)

    self.assertAllEqual(evaluated_tensor.values, [0, 1, 2, 3, 4, 5])
    self.assertAllEqual(evaluated_tensor.row_splits, [0, 1, 3, 5, 6])

  def test_ragged_value_concatenation(self):
    tensor_1 = ragged_tensor_value.RaggedTensorValue(
        np.array([0, 1, 2]), np.array([0, 1, 3], dtype=np.int64))
    tensor_2 = ragged_tensor_value.RaggedTensorValue(
        np.array([3, 4, 5]), np.array([0, 2, 3], dtype=np.int64))
    concatenated_tensor = training_utils_v1._append_composite_tensor(
        tensor_1, tensor_2)

    self.assertAllEqual(concatenated_tensor.values, [0, 1, 2, 3, 4, 5])
    self.assertAllEqual(concatenated_tensor.row_splits, [0, 1, 3, 5, 6])


if __name__ == '__main__':
  test.main()
