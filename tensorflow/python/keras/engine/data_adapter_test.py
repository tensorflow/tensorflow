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
"""DataAdapter tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DummyArrayLike(object):
  """Dummy array-like object."""

  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, key):
    return self.data[key]

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype


def fail_on_convert(x, **kwargs):
  _ = x
  _ = kwargs
  raise TypeError('Cannot convert DummyArrayLike to a tensor')
ops.register_tensor_conversion_function(DummyArrayLike, fail_on_convert)


class DataAdapterTestBase(keras_parameterized.TestCase):

  def setUp(self):
    super(DataAdapterTestBase, self).setUp()
    self.batch_size = 5
    self.numpy_input = np.zeros((50, 10))
    self.numpy_target = np.ones(50)
    self.tensor_input = constant_op.constant(2.0, shape=(50, 10))
    self.tensor_target = array_ops.ones((50,))
    self.arraylike_input = DummyArrayLike(self.numpy_input)
    self.arraylike_target = DummyArrayLike(self.numpy_target)
    self.dataset_input = dataset_ops.DatasetV2.from_tensor_slices(
        (self.numpy_input, self.numpy_target)).shuffle(50).batch(
            self.batch_size)

    def generator():
      while True:
        yield (np.zeros((self.batch_size, 10)), np.ones(self.batch_size))
    self.generator_input = generator()
    self.iterator_input = data_utils.threadsafe_generator(generator)()
    self.sequence_input = TestSequence(batch_size=self.batch_size,
                                       feature_shape=10)
    self.model = keras.models.Sequential(
        [keras.layers.Dense(8, input_shape=(10,), activation='softmax')])


class TestSequence(data_utils.Sequence):

  def __init__(self, batch_size, feature_shape):
    self.batch_size = batch_size
    self.feature_shape = feature_shape

  def __getitem__(self, item):
    return (np.zeros((self.batch_size, self.feature_shape)),
            np.ones((self.batch_size,)))

  def __len__(self):
    return 10


class TensorLikeDataAdapterTest(DataAdapterTestBase):

  def setUp(self):
    super(TensorLikeDataAdapterTest, self).setUp()
    self.adapter_cls = data_adapter.TensorLikeDataAdapter

  def test_can_handle_numpy(self):
    self.assertTrue(self.adapter_cls.can_handle(self.numpy_input))
    self.assertTrue(
        self.adapter_cls.can_handle(self.numpy_input, self.numpy_target))

    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  def test_iterator_expect_batch_size_numpy(self):
    with self.assertRaisesRegexp(
        ValueError, r'`batch_size` or `steps` is required'):
      self.adapter_cls(self.numpy_input, self.numpy_target)

  def test_size_numpy(self):
    adapter = self.adapter_cls(
        self.numpy_input, self.numpy_target, batch_size=5)
    self.assertEqual(adapter.get_size(), 10)
    self.assertFalse(adapter.has_partial_batch())

  def test_batch_size_numpy(self):
    adapter = self.adapter_cls(
        self.numpy_input, self.numpy_target, batch_size=5)
    self.assertEqual(adapter.batch_size(), 5)

  def test_partial_batch_numpy(self):
    adapter = self.adapter_cls(
        self.numpy_input, self.numpy_target, batch_size=4)
    self.assertEqual(adapter.get_size(), 13)   # 50/4
    self.assertTrue(adapter.has_partial_batch())
    self.assertEqual(adapter.partial_batch_size(), 2)

  def test_epochs(self):
    num_epochs = 3
    adapter = self.adapter_cls(
        self.numpy_input, self.numpy_target, batch_size=5, epochs=num_epochs)
    ds_iter = iter(adapter.get_dataset())
    num_batches_per_epoch = self.numpy_input.shape[0] // 5
    for _ in range(num_batches_per_epoch * num_epochs):
      next(ds_iter)
    with self.assertRaises(StopIteration):
      next(ds_iter)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training_numpy(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.numpy_input, self.numpy_target, batch_size=5)

  def test_can_handle(self):
    self.assertTrue(self.adapter_cls.can_handle(self.tensor_input))
    self.assertTrue(
        self.adapter_cls.can_handle(self.tensor_input, self.tensor_target))

    self.assertFalse(self.adapter_cls.can_handle(self.arraylike_input))
    self.assertFalse(
        self.adapter_cls.can_handle(self.arraylike_input,
                                    self.arraylike_target))
    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.tensor_input, self.tensor_target, batch_size=5)

  def test_size(self):
    adapter = self.adapter_cls(
        self.tensor_input, self.tensor_target, batch_size=5)
    self.assertEqual(adapter.get_size(), 10)
    self.assertFalse(adapter.has_partial_batch())

  def test_shuffle_correctness(self):
    with context.eager_mode():
      num_samples = 100
      batch_size = 32
      x = np.arange(num_samples)
      np.random.seed(99)
      adapter = self.adapter_cls(
          x, y=None, batch_size=batch_size, shuffle=True, epochs=2)

      def _get_epoch(ds_iter):
        ds_data = []
        for _ in range(int(math.ceil(num_samples / batch_size))):
          ds_data.append(next(ds_iter)[0].numpy())
        return np.concatenate(ds_data)

      ds_iter = iter(adapter.get_dataset())

      # First epoch.
      epoch_data = _get_epoch(ds_iter)
      # Check that shuffling occurred.
      self.assertNotAllClose(x, epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(epoch_data))

      # Second epoch.
      second_epoch_data = _get_epoch(ds_iter)
      # Check that shuffling occurred.
      self.assertNotAllClose(x, second_epoch_data)
      # Check that shuffling is different across epochs.
      self.assertNotAllClose(epoch_data, second_epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(second_epoch_data))

  def test_batch_shuffle_correctness(self):
    with context.eager_mode():
      num_samples = 100
      batch_size = 6
      x = np.arange(num_samples)
      np.random.seed(99)
      adapter = self.adapter_cls(
          x, y=None, batch_size=batch_size, shuffle='batch', epochs=2)

      def _get_epoch_batches(ds_iter):
        ds_data = []
        for _ in range(int(math.ceil(num_samples / batch_size))):
          ds_data.append(next(ds_iter)[0].numpy())
        return ds_data

      ds_iter = iter(adapter.get_dataset())

      # First epoch.
      epoch_batch_data = _get_epoch_batches(ds_iter)
      epoch_data = np.concatenate(epoch_batch_data)

      def _verify_batch(batch):
        # Verify that a batch contains only contiguous data, and that it has
        # been shuffled.
        shuffled_batch = np.sort(batch)
        self.assertNotAllClose(batch, shuffled_batch)
        for i in range(1, len(batch)):
          self.assertEqual(shuffled_batch[i-1] + 1, shuffled_batch[i])

      # Assert that the data within each batch remains contiguous
      for batch in epoch_batch_data:
        _verify_batch(batch)

      # Check that individual batches are unshuffled
      # Check that shuffling occurred.
      self.assertNotAllClose(x, epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(epoch_data))

      # Second epoch.
      second_epoch_batch_data = _get_epoch_batches(ds_iter)
      second_epoch_data = np.concatenate(second_epoch_batch_data)

      # Assert that the data within each batch remains contiguous
      for batch in second_epoch_batch_data:
        _verify_batch(batch)

      # Check that shuffling occurred.
      self.assertNotAllClose(x, second_epoch_data)
      # Check that shuffling is different across epochs.
      self.assertNotAllClose(epoch_data, second_epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(second_epoch_data))

  @parameterized.named_parameters(
      ('batch_size_5', 5, None, 5),
      ('batch_size_50', 50, 4, 50),  # Sanity check: batch_size takes precedence
      ('steps_1', None, 1, 50),
      ('steps_4', None, 4, 13),
      )
  def test_batch_size(self, batch_size_in, steps, batch_size_out):
    adapter = self.adapter_cls(
        self.tensor_input, self.tensor_target, batch_size=batch_size_in,
        steps=steps)
    self.assertEqual(adapter.batch_size(), batch_size_out)

  @parameterized.named_parameters(
      ('batch_size_5', 5, None, 10, 0),
      ('batch_size_4', 4, None, 13, 2),
      ('steps_1', None, 1, 1, 0),
      ('steps_5', None, 5, 5, 0),
      ('steps_4', None, 4, 4, 11),
      )
  def test_partial_batch(
      self, batch_size_in, steps, size, partial_batch_size):
    adapter = self.adapter_cls(
        self.tensor_input, self.tensor_target, batch_size=batch_size_in,
        steps=steps)
    self.assertEqual(adapter.get_size(), size)   # 50/steps
    self.assertEqual(adapter.has_partial_batch(), bool(partial_batch_size))
    self.assertEqual(adapter.partial_batch_size(), partial_batch_size or None)


class GenericArrayLikeDataAdapterTest(DataAdapterTestBase):

  def setUp(self):
    super(GenericArrayLikeDataAdapterTest, self).setUp()
    self.adapter_cls = data_adapter.GenericArrayLikeDataAdapter

  def test_can_handle_some_numpy(self):
    self.assertTrue(self.adapter_cls.can_handle(
        self.arraylike_input))
    self.assertTrue(
        self.adapter_cls.can_handle(self.arraylike_input,
                                    self.arraylike_target))

    # Because adapters are mutually exclusive, don't handle cases
    # where all the data is numpy or an eagertensor
    self.assertFalse(self.adapter_cls.can_handle(self.numpy_input))
    self.assertFalse(
        self.adapter_cls.can_handle(self.numpy_input,
                                    self.numpy_target))
    self.assertFalse(self.adapter_cls.can_handle(self.tensor_input))
    self.assertFalse(
        self.adapter_cls.can_handle(self.tensor_input, self.tensor_target))

    # But do handle mixes that include generic arraylike data
    self.assertTrue(
        self.adapter_cls.can_handle(self.numpy_input,
                                    self.arraylike_target))
    self.assertTrue(
        self.adapter_cls.can_handle(self.arraylike_input,
                                    self.numpy_target))
    self.assertTrue(
        self.adapter_cls.can_handle(self.arraylike_input,
                                    self.tensor_target))
    self.assertTrue(
        self.adapter_cls.can_handle(self.tensor_input,
                                    self.arraylike_target))

    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  def test_iterator_expect_batch_size_generic_arraylike(self):
    with self.assertRaisesRegexp(
        ValueError, r'`batch_size` or `steps` is required'):
      self.adapter_cls(self.arraylike_input,
                       self.arraylike_target)

  def test_size(self):
    adapter = self.adapter_cls(
        self.arraylike_input,
        self.arraylike_target, batch_size=5)
    self.assertEqual(adapter.get_size(), 10)
    self.assertFalse(adapter.has_partial_batch())

  def test_epochs(self):
    num_epochs = 3
    adapter = self.adapter_cls(
        self.arraylike_input,
        self.numpy_target, batch_size=5, epochs=num_epochs)
    ds_iter = iter(adapter.get_dataset())
    num_batches_per_epoch = self.numpy_input.shape[0] // 5
    for _ in range(num_batches_per_epoch * num_epochs):
      next(ds_iter)
    with self.assertRaises(StopIteration):
      next(ds_iter)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training(self):
    # First verify that DummyArrayLike can't be converted to a Tensor
    with self.assertRaises(TypeError):
      ops.convert_to_tensor(self.arraylike_input)

    # Then train on the array like.
    # It should not be converted to a tensor directly (which would force it into
    # memory), only the sliced data should be converted.
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.arraylike_input,
                   self.arraylike_target, batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.arraylike_target,
                   shuffle=True, batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.arraylike_target,
                   shuffle='batch', batch_size=5)
    self.model.evaluate(self.arraylike_input,
                        self.arraylike_target, batch_size=5)
    self.model.predict(self.arraylike_input, batch_size=5)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training_numpy_target(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.arraylike_input,
                   self.numpy_target, batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.numpy_target, shuffle=True,
                   batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.numpy_target, shuffle='batch',
                   batch_size=5)
    self.model.evaluate(self.arraylike_input,
                        self.numpy_target, batch_size=5)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training_tensor_target(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.arraylike_input,
                   self.tensor_target, batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.tensor_target, shuffle=True,
                   batch_size=5)
    self.model.fit(self.arraylike_input,
                   self.tensor_target, shuffle='batch',
                   batch_size=5)
    self.model.evaluate(self.arraylike_input,
                        self.tensor_target, batch_size=5)

  def test_shuffle_correctness(self):
    with context.eager_mode():
      num_samples = 100
      batch_size = 32
      x = DummyArrayLike(np.arange(num_samples))
      np.random.seed(99)
      adapter = self.adapter_cls(
          x, y=None, batch_size=batch_size, shuffle=True, epochs=2)

      def _get_epoch(ds_iter):
        ds_data = []
        for _ in range(int(math.ceil(num_samples / batch_size))):
          ds_data.append(next(ds_iter)[0].numpy())
        return np.concatenate(ds_data)

      ds_iter = iter(adapter.get_dataset())

      # First epoch.
      epoch_data = _get_epoch(ds_iter)
      # Check that shuffling occurred.
      self.assertNotAllClose(x, epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(epoch_data))

      # Second epoch.
      second_epoch_data = _get_epoch(ds_iter)
      # Check that shuffling occurred.
      self.assertNotAllClose(x, second_epoch_data)
      # Check that shuffling is different across epochs.
      self.assertNotAllClose(epoch_data, second_epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(second_epoch_data))

  def test_batch_shuffle_correctness(self):
    with context.eager_mode():
      num_samples = 100
      batch_size = 6
      x = DummyArrayLike(np.arange(num_samples))
      np.random.seed(99)
      adapter = self.adapter_cls(
          x, y=None, batch_size=batch_size, shuffle='batch', epochs=2)

      def _get_epoch_batches(ds_iter):
        ds_data = []
        for _ in range(int(math.ceil(num_samples / batch_size))):
          ds_data.append(next(ds_iter)[0].numpy())
        return ds_data

      ds_iter = iter(adapter.get_dataset())

      # First epoch.
      epoch_batch_data = _get_epoch_batches(ds_iter)
      epoch_data = np.concatenate(epoch_batch_data)

      def _verify_batch(batch):
        # Verify that a batch contains only contiguous data, but that it has
        # been shuffled.
        shuffled_batch = np.sort(batch)
        self.assertNotAllClose(batch, shuffled_batch)
        for i in range(1, len(batch)):
          self.assertEqual(shuffled_batch[i-1] + 1, shuffled_batch[i])

      # Assert that the data within each batch is shuffled contiguous data
      for batch in epoch_batch_data:
        _verify_batch(batch)

      # Check that individual batches are unshuffled
      # Check that shuffling occurred.
      self.assertNotAllClose(x, epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(epoch_data))

      # Second epoch.
      second_epoch_batch_data = _get_epoch_batches(ds_iter)
      second_epoch_data = np.concatenate(second_epoch_batch_data)

      # Assert that the data within each batch remains contiguous
      for batch in second_epoch_batch_data:
        _verify_batch(batch)

      # Check that shuffling occurred.
      self.assertNotAllClose(x, second_epoch_data)
      # Check that shuffling is different across epochs.
      self.assertNotAllClose(epoch_data, second_epoch_data)
      # Check that each elements appears, and only once.
      self.assertAllClose(x, np.sort(second_epoch_data))

  @parameterized.named_parameters(
      ('batch_size_5', 5, None, 5),
      ('batch_size_50', 50, 4, 50),  # Sanity check: batch_size takes precedence
      ('steps_1', None, 1, 50),
      ('steps_4', None, 4, 13),
  )
  def test_batch_size(self, batch_size_in, steps, batch_size_out):
    adapter = self.adapter_cls(
        self.arraylike_input,
        self.arraylike_target, batch_size=batch_size_in,
        steps=steps)
    self.assertEqual(adapter.batch_size(), batch_size_out)

  @parameterized.named_parameters(
      ('batch_size_5', 5, None, 10, 0),
      ('batch_size_4', 4, None, 13, 2),
      ('steps_1', None, 1, 1, 0),
      ('steps_5', None, 5, 5, 0),
      ('steps_4', None, 4, 4, 11),
  )
  def test_partial_batch(
      self, batch_size_in, steps, size, partial_batch_size):
    adapter = self.adapter_cls(
        self.arraylike_input, self.arraylike_target,
        batch_size=batch_size_in,
        steps=steps)
    self.assertEqual(adapter.get_size(), size)   # 50/steps
    self.assertEqual(adapter.has_partial_batch(), bool(partial_batch_size))
    self.assertEqual(adapter.partial_batch_size(), partial_batch_size or None)


class DatasetAdapterTest(DataAdapterTestBase):

  def setUp(self):
    super(DatasetAdapterTest, self).setUp()
    self.adapter_cls = data_adapter.DatasetAdapter

  def test_can_handle(self):
    self.assertFalse(self.adapter_cls.can_handle(self.numpy_input))
    self.assertFalse(self.adapter_cls.can_handle(self.tensor_input))
    self.assertTrue(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training(self):
    dataset = self.adapter_cls(self.dataset_input).get_dataset()
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(dataset)

  def test_size(self):
    adapter = self.adapter_cls(self.dataset_input)
    self.assertIsNone(adapter.get_size())

  def test_batch_size(self):
    adapter = self.adapter_cls(self.dataset_input)
    self.assertIsNone(adapter.batch_size())

  def test_partial_batch(self):
    adapter = self.adapter_cls(self.dataset_input)
    self.assertFalse(adapter.has_partial_batch())
    self.assertIsNone(adapter.partial_batch_size())

  def test_invalid_targets_argument(self):
    with self.assertRaisesRegexp(ValueError, r'`y` argument is not supported'):
      self.adapter_cls(self.dataset_input, y=self.dataset_input)

  def test_invalid_sample_weights_argument(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'`sample_weight` argument is not supported'):
      self.adapter_cls(self.dataset_input, sample_weights=self.dataset_input)


class GeneratorDataAdapterTest(DataAdapterTestBase):

  def setUp(self):
    super(GeneratorDataAdapterTest, self).setUp()
    self.adapter_cls = data_adapter.GeneratorDataAdapter

  def test_can_handle(self):
    self.assertFalse(self.adapter_cls.can_handle(self.numpy_input))
    self.assertFalse(self.adapter_cls.can_handle(self.tensor_input))
    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertTrue(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.generator_input, steps_per_epoch=10)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @test_util.run_v2_only
  @data_utils.dont_use_multiprocessing_pool
  def test_with_multiprocessing_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.iterator_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)
    # Fit twice to ensure there isn't any duplication that prevent the worker
    # from starting.
    self.model.fit(self.iterator_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)

  def test_size(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertIsNone(adapter.get_size())

  def test_batch_size(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertEqual(adapter.batch_size(), None)
    self.assertEqual(adapter.representative_batch_size(), 5)

  def test_partial_batch(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertFalse(adapter.has_partial_batch())
    self.assertIsNone(adapter.partial_batch_size())

  def test_invalid_targets_argument(self):
    with self.assertRaisesRegexp(ValueError, r'`y` argument is not supported'):
      self.adapter_cls(self.generator_input, y=self.generator_input)

  def test_invalid_sample_weights_argument(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'`sample_weight` argument is not supported'):
      self.adapter_cls(
          self.generator_input, sample_weights=self.generator_input)


class KerasSequenceAdapterTest(DataAdapterTestBase):

  def setUp(self):
    super(KerasSequenceAdapterTest, self).setUp()
    self.adapter_cls = data_adapter.KerasSequenceAdapter

  def test_can_handle(self):
    self.assertFalse(self.adapter_cls.can_handle(self.numpy_input))
    self.assertFalse(self.adapter_cls.can_handle(self.tensor_input))
    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertTrue(self.adapter_cls.can_handle(self.sequence_input))

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.sequence_input)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @test_util.run_v2_only
  @data_utils.dont_use_multiprocessing_pool
  def test_with_multiprocessing_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd',
                       run_eagerly=testing_utils.should_run_eagerly())
    self.model.fit(self.sequence_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)
    # Fit twice to ensure there isn't any duplication that prevent the worker
    # from starting.
    self.model.fit(self.sequence_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)

  def test_size(self):
    adapter = self.adapter_cls(self.sequence_input)
    self.assertEqual(adapter.get_size(), 10)

  def test_batch_size(self):
    adapter = self.adapter_cls(self.sequence_input)
    self.assertEqual(adapter.batch_size(), None)
    self.assertEqual(adapter.representative_batch_size(), 5)

  def test_partial_batch(self):
    adapter = self.adapter_cls(self.sequence_input)
    self.assertFalse(adapter.has_partial_batch())
    self.assertIsNone(adapter.partial_batch_size())

  def test_invalid_targets_argument(self):
    with self.assertRaisesRegexp(ValueError, r'`y` argument is not supported'):
      self.adapter_cls(self.sequence_input, y=self.sequence_input)

  def test_invalid_sample_weights_argument(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'`sample_weight` argument is not supported'):
      self.adapter_cls(self.sequence_input, sample_weights=self.sequence_input)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
