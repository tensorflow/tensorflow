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
"""Tests for creating different number of masks in rnn_cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import rnn_cells
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell as tf_rnn_cells
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RnnCellsTest(test.TestCase):

  def setUp(self):
    super(RnnCellsTest, self).setUp()
    self.batch_size = 8
    self.dim = 10

  def testMaskedBasicLSTMCell(self):
    expected_num_masks = 1
    expected_num_rows = 2 * self.dim
    expected_num_cols = 4 * self.dim
    with self.cached_session():
      inputs = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      c = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      h = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      state = tf_rnn_cells.LSTMStateTuple(c, h)
      lstm_cell = rnn_cells.MaskedBasicLSTMCell(self.dim)
      lstm_cell(inputs, state)
      self.assertEqual(len(pruning.get_masks()), expected_num_masks)
      self.assertEqual(len(pruning.get_masked_weights()), expected_num_masks)
      self.assertEqual(len(pruning.get_thresholds()), expected_num_masks)
      self.assertEqual(len(pruning.get_weights()), expected_num_masks)

      for mask in pruning.get_masks():
        self.assertEqual(mask.shape, (expected_num_rows, expected_num_cols))
      for weight in pruning.get_weights():
        self.assertEqual(weight.shape, (expected_num_rows, expected_num_cols))

  def testMaskedLSTMCell(self):
    expected_num_masks = 1
    expected_num_rows = 2 * self.dim
    expected_num_cols = 4 * self.dim
    with self.cached_session():
      inputs = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      c = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      h = variables.Variable(
          random_ops.random_normal([self.batch_size, self.dim]))
      state = tf_rnn_cells.LSTMStateTuple(c, h)
      lstm_cell = rnn_cells.MaskedLSTMCell(self.dim)
      lstm_cell(inputs, state)
      self.assertEqual(len(pruning.get_masks()), expected_num_masks)
      self.assertEqual(len(pruning.get_masked_weights()), expected_num_masks)
      self.assertEqual(len(pruning.get_thresholds()), expected_num_masks)
      self.assertEqual(len(pruning.get_weights()), expected_num_masks)

      for mask in pruning.get_masks():
        self.assertEqual(mask.shape, (expected_num_rows, expected_num_cols))
      for weight in pruning.get_weights():
        self.assertEqual(weight.shape, (expected_num_rows, expected_num_cols))

if __name__ == '__main__':
  test.main()
