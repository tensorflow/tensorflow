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
"""Tests dense attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class BaseDenseAttentionTest(test.TestCase):

  def test_one_dim_with_mask(self):
    # Scores tensor of shape [1, 1, 1]
    scores = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 1, 1]
    v = np.array([[[1.6]]], dtype=np.float32)
    # Value mask tensor of shape [1, 1]
    v_mask = np.array([[True]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention().apply_attention_scores(
        scores=scores, value=v, value_mask=v_mask)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    expected = np.array([[[1.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_one_dim_no_mask(self):
    # Scores tensor of shape [1, 1, 1]
    scores = np.array([[[1.1]]], dtype=np.float32)
    # Value tensor of shape [1, 1, 1]
    v = np.array([[[1.6]]], dtype=np.float32)
    actual = dense_attention.BaseDenseAttention().apply_attention_scores(
        scores=scores, value=v)

    # Expected tensor of shape [1, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    expected = np.array([[[1.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_with_mask(self):
    # Scores tensor of shape [1, 1, 3]
    scores = np.array([[[1., 0., 1.]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    # Value mask tensor of shape [1, 3]
    v_mask = np.array([[True, True, False]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention().apply_attention_scores(
        scores=scores, value=v, value_mask=v_mask)

    # Expected attention distribution = softmax(scores) with zeros in
    # positions where v_mask == False.
    # => attention_distribution000 = exp(1)/(exp(1) + exp(0)) = 0.73105857863
    #    attention_distribution001 = exp(0)/(exp(1) + exp(0)) = 0.26894142137
    #    attention_distribution002 = 0
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.73105857863 * 1.6 + 0.26894142137 * 0.7 - 0 * 0.8
    #             = 1.35795272077
    expected = np.array([[[1.35795272077]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_multi_dim_no_mask(self):
    # Scores tensor of shape [1, 1, 3]
    scores = np.array([[[1., 0., 1.]]], dtype=np.float32)
    # Value tensor of shape [1, 3, 1]
    v = np.array([[[1.6], [0.7], [-0.8]]], dtype=np.float32)
    actual = dense_attention.BaseDenseAttention().apply_attention_scores(
        scores=scores, value=v)

    # Expected attention distribution = softmax(scores).
    # => attention_distribution000 = exp(1)/(exp(1) + exp(0) + exp(1))
    #                              = 0.42231879825
    #    attention_distribution001 = exp(0)/(exp(1) + exp(0) + exp(1))
    #                              = 0.15536240349
    #    attention_distribution002 = exp(1)/(exp(1) + exp(0) + exp(1))
    #                              = 0.42231879825
    #
    # Expected tensor of shape [1, 1, 1].
    # expected000 = 0.42231879825 * 1.6 + 0.15536240349 * 0.7
    #               - 0.42231879825 * 0.8
    #             = 0.44660872104
    expected = np.array([[[0.44660872104]]], dtype=np.float32)
    self.assertAllClose(expected, actual)

  def test_one_dim_batch_size_two(self):
    # Scores tensor of shape [2, 1, 1]
    scores = np.array([[[1.1]], [[2.1]]], dtype=np.float32)
    # Value tensor of shape [2, 1, 1]
    v = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    # Value mask tensor of shape [2, 1]
    v_mask = np.array([[True], [True]], dtype=np.bool_)
    actual = dense_attention.BaseDenseAttention().apply_attention_scores(
        scores=scores, value=v, value_mask=v_mask)

    # Expected tensor of shape [2, 1, 1].
    # expected000 = softmax(scores)[0, 0] * 1.6 = 1.6
    # expected100 = softmax(scores)[1, 0] * 2.6 = 2.6
    expected = np.array([[[1.6]], [[2.6]]], dtype=np.float32)
    self.assertAllClose(expected, actual)


if __name__ == '__main__':
  test.main()
