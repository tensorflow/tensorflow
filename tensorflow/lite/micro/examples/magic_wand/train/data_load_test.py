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
# pylint: disable=g-bad-import-order

"""Test for data_load.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from data_load import DataLoader

import tensorflow as tf


class TestLoad(unittest.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    self.loader = DataLoader(
        "./data/train", "./data/valid", "./data/test", seq_length=512)

  def test_get_data(self):
    self.assertIsInstance(self.loader.train_data, list)
    self.assertIsInstance(self.loader.train_label, list)
    self.assertIsInstance(self.loader.valid_data, list)
    self.assertIsInstance(self.loader.valid_label, list)
    self.assertIsInstance(self.loader.test_data, list)
    self.assertIsInstance(self.loader.test_label, list)
    self.assertEqual(self.loader.train_len, len(self.loader.train_data))
    self.assertEqual(self.loader.train_len, len(self.loader.train_label))
    self.assertEqual(self.loader.valid_len, len(self.loader.valid_data))
    self.assertEqual(self.loader.valid_len, len(self.loader.valid_label))
    self.assertEqual(self.loader.test_len, len(self.loader.test_data))
    self.assertEqual(self.loader.test_len, len(self.loader.test_label))

  def test_pad(self):
    original_data1 = [[2, 3], [1, 1]]
    expected_data1_0 = [[2, 3], [2, 3], [2, 3], [2, 3], [1, 1]]
    expected_data1_1 = [[2, 3], [1, 1], [1, 1], [1, 1], [1, 1]]
    original_data2 = [[-2, 3], [-77, -681], [5, 6], [9, -7], [22, 3333],
                      [9, 99], [-100, 0]]
    expected_data2 = [[-2, 3], [-77, -681], [5, 6], [9, -7], [22, 3333]]
    padding_data1 = self.loader.pad(original_data1, seq_length=5, dim=2)
    padding_data2 = self.loader.pad(original_data2, seq_length=5, dim=2)
    for i in range(len(padding_data1[0])):
      for j in range(len(padding_data1[0].tolist()[0])):
        self.assertLess(
            abs(padding_data1[0].tolist()[i][j] - expected_data1_0[i][j]),
            10.001)
    for i in range(len(padding_data1[1])):
      for j in range(len(padding_data1[1].tolist()[0])):
        self.assertLess(
            abs(padding_data1[1].tolist()[i][j] - expected_data1_1[i][j]),
            10.001)
    self.assertEqual(padding_data2[0].tolist(), expected_data2)
    self.assertEqual(padding_data2[1].tolist(), expected_data2)

  def test_format(self):
    self.loader.format()
    expected_train_label = int(self.loader.label2id[self.loader.train_label[0]])
    expected_valid_label = int(self.loader.label2id[self.loader.valid_label[0]])
    expected_test_label = int(self.loader.label2id[self.loader.test_label[0]])
    for feature, label in self.loader.train_data:  # pylint: disable=unused-variable
      format_train_label = label.numpy()
      break
    for feature, label in self.loader.valid_data:
      format_valid_label = label.numpy()
      break
    for feature, label in self.loader.test_data:
      format_test_label = label.numpy()
      break
    self.assertEqual(expected_train_label, format_train_label)
    self.assertEqual(expected_valid_label, format_valid_label)
    self.assertEqual(expected_test_label, format_test_label)
    self.assertIsInstance(self.loader.train_data, tf.data.Dataset)
    self.assertIsInstance(self.loader.valid_data, tf.data.Dataset)
    self.assertIsInstance(self.loader.test_data, tf.data.Dataset)


if __name__ == "__main__":
  unittest.main()
