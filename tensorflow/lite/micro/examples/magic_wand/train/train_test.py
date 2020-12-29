# Lint as: python3
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

"""Test for train.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf
from train import build_cnn
from train import build_lstm
from train import load_data
from train import reshape_function


class TestTrain(unittest.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self.seq_length = 128
    self.train_len, self.train_data, self.valid_len, self.valid_data, \
        self.test_len, self.test_data = \
        load_data("./data/train", "./data/valid", "./data/test",
                  self.seq_length)

  def test_load_data(self):
    self.assertIsInstance(self.train_data, tf.data.Dataset)
    self.assertIsInstance(self.valid_data, tf.data.Dataset)
    self.assertIsInstance(self.test_data, tf.data.Dataset)

  def test_build_net(self):
    cnn, cnn_path = build_cnn(self.seq_length)
    lstm, lstm_path = build_lstm(self.seq_length)
    cnn_data = np.random.rand(60, 128, 3, 1)
    lstm_data = np.random.rand(60, 128, 3)
    cnn_prob = cnn(tf.constant(cnn_data, dtype="float32")).numpy()
    lstm_prob = lstm(tf.constant(lstm_data, dtype="float32")).numpy()
    self.assertIsInstance(cnn, tf.keras.Sequential)
    self.assertIsInstance(lstm, tf.keras.Sequential)
    self.assertEqual(cnn_path, "./netmodels/CNN")
    self.assertEqual(lstm_path, "./netmodels/LSTM")
    self.assertEqual(cnn_prob.shape, (60, 4))
    self.assertEqual(lstm_prob.shape, (60, 4))

  def test_reshape_function(self):
    for data, label in self.train_data:
      original_data_shape = data.numpy().shape
      original_label_shape = label.numpy().shape
      break
    self.train_data = self.train_data.map(reshape_function)
    for data, label in self.train_data:
      reshaped_data_shape = data.numpy().shape
      reshaped_label_shape = label.numpy().shape
      break
    self.assertEqual(
        reshaped_data_shape,
        (int(original_data_shape[0] * original_data_shape[1] / 3), 3, 1))
    self.assertEqual(reshaped_label_shape, original_label_shape)


if __name__ == "__main__":
  unittest.main()
