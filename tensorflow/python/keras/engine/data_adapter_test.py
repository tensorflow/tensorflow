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
import os
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class DataAdapterTestBase(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DataAdapterTestBase, self).setUp()
    self.batch_size = 5
    self.numpy_input = np.zeros((50, 10))
    self.numpy_target = np.ones(50)
    self.tensor_input = constant_op.constant(2.0, shape=(50, 10))
    self.tensor_target = array_ops.ones((50,))
    self.dataset_input = dataset_ops.DatasetV2.from_tensor_slices(
        (self.numpy_input, self.numpy_target)).shuffle(50).batch(
            self.batch_size)

    def generator():
      while True:
        yield (np.zeros((self.batch_size, 10)), np.ones(self.batch_size))
    self.generator_input = generator()
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

  @test_util.run_in_graph_and_eager_modes
  def test_training_numpy(self):
    if not context.executing_eagerly():
      return  # Only test in eager.

    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    self.model.fit(self.numpy_input, self.numpy_target, batch_size=5)

  def test_can_handle(self):
    self.assertTrue(self.adapter_cls.can_handle(self.tensor_input))
    self.assertTrue(
        self.adapter_cls.can_handle(self.tensor_input, self.tensor_target))

    self.assertFalse(self.adapter_cls.can_handle(self.dataset_input))
    self.assertFalse(self.adapter_cls.can_handle(self.generator_input))
    self.assertFalse(self.adapter_cls.can_handle(self.sequence_input))

  @test_util.run_in_graph_and_eager_modes
  def test_training(self):
    if not context.executing_eagerly():
      return  # Only test EagerTensors.

    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
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

  def test_training(self):
    dataset = self.adapter_cls(self.dataset_input).get_dataset()
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
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

  def test_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    self.model.fit(self.generator_input, steps_per_epoch=10)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @test_util.run_v2_only
  def test_with_multiprocessing_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    self.model.fit(self.generator_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)
    # Fit twice to ensure there isn't any duplication that prevent the worker
    # from starting.
    self.model.fit(self.generator_input, workers=1, use_multiprocessing=True,
                   max_queue_size=10, steps_per_epoch=10)

  def test_size(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertIsNone(adapter.get_size())

  def test_batch_size(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertEqual(adapter.batch_size(), 5)

  def test_partial_batch(self):
    adapter = self.adapter_cls(self.generator_input)
    self.assertFalse(adapter.has_partial_batch())
    self.assertIsNone(adapter.partial_batch_size())


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

  def test_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    self.model.fit(self.sequence_input)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @test_util.run_v2_only
  def test_with_multiprocessing_training(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
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
    self.assertEqual(adapter.batch_size(), 5)

  def test_partial_batch(self):
    adapter = self.adapter_cls(self.sequence_input)
    self.assertFalse(adapter.has_partial_batch())
    self.assertIsNone(adapter.partial_batch_size())


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
