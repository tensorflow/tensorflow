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
"""Tests for training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_v2_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TestSequence(data_utils.Sequence):

  def __init__(self, batch_size, feature_shape):
    self.batch_size = batch_size
    self.feature_shape = feature_shape

  def __getitem__(self, item):
    return (np.zeros((self.batch_size, self.feature_shape)),
            np.ones((self.batch_size,)))

  def __len__(self):
    return 10


class CallbackFallbackTest(test.TestCase):

  def setUp(self):
    super(CallbackFallbackTest, self).setUp()
    self.batch_size = 5
    self.numpy_input = np.zeros((50, 10))
    self.numpy_target = np.ones(50)
    self.tensor_input = constant_op.constant(2.0, shape=(50, 10))
    self.tensor_target = array_ops.ones((50,))
    self.dataset_input = dataset_ops.DatasetV2.from_tensor_slices(
        (self.numpy_input, self.numpy_target)).shuffle(50).batch(
            self.batch_size)

    def generator():
      yield (np.zeros((self.batch_size, 10)), np.ones(self.batch_size))
    self.generator_input = generator()
    self.sequence_input = TestSequence(batch_size=self.batch_size,
                                       feature_shape=10)

    self.fallback_ckeckpoint_cb = cbks.ModelCheckpoint(
        self.get_temp_dir(), save_freq=10)
    self.normal_checkpoint_cb = cbks.ModelCheckpoint(
        self.get_temp_dir(), save_freq='epoch')
    self.fallback_tensorboard_cb = cbks.TensorBoard(update_freq=10)
    self.normal_tensorboard_cb = cbks.TensorBoard(update_freq='batch')
    self.unaffected_cb = cbks.CSVLogger(self.get_temp_dir())

  def test_not_fallback_based_on_input(self):
    callback_list = [self.fallback_ckeckpoint_cb]

    test_cases = [
        [(self.numpy_input, self.numpy_target), False],
        [[self.tensor_input, self.tensor_target], False],
        [self.sequence_input, False],
        [self.dataset_input, True],
        [self.generator_input, True],
    ]

    for case in test_cases:
      inputs, expected_result = case
      self.assertEqual(training_v2_utils.should_fallback_to_v1_for_callback(
          inputs, callback_list), expected_result)

  def test_fallback_based_on_callbacks(self):
    inputs = self.dataset_input
    test_cases = [
        [[self.fallback_ckeckpoint_cb], True],
        [[self.normal_checkpoint_cb], False],
        [[self.fallback_ckeckpoint_cb, self.normal_checkpoint_cb], True],
        [[self.fallback_tensorboard_cb], True],
        [[self.normal_tensorboard_cb], False],
        [[self.unaffected_cb], False],
    ]

    for case in test_cases:
      callbacks, expected_result = case
      self.assertEqual(training_v2_utils.should_fallback_to_v1_for_callback(
          inputs, callbacks), expected_result)

if __name__ == '__main__':
  test.main()
