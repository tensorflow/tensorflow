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
"""Tests for contrib.seq2seq.python.ops.attention_wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class AttentionMechanismTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(AttentionMechanismTest, self).setUp()
    self.batch = 10
    self.timestep = 5
    self.memory_size = 6
    self.units = 8

    self.memory = ops.convert_to_tensor(
        np.random.random((self.batch, self.timestep, self.memory_size)),
        dtype=np.float32)
    self.query = ops.convert_to_tensor(
        np.random.random((self.batch, self.units)), dtype=np.float32)
    self.state = ops.convert_to_tensor(
        np.random.random((self.batch, self.timestep)), dtype=np.float32)

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_attention_shape_inference(self, attention_cls):
    attention = attention_cls(self.units)
    attention_score = attention([self.query, self.state, self.memory])
    self.assertLen(attention_score, 2)
    self.assertEqual(attention_score[0].shape, (self.batch, self.timestep))
    self.assertEqual(attention_score[1].shape, (self.batch, self.timestep))

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_get_config(self, attention_cls):
    attention = attention_cls(self.units)
    config = attention.get_config()

    attention_from_config = attention_cls.from_config(config)
    config_from_clone = attention_from_config.get_config()

    self.assertDictEqual(config, config_from_clone)

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_layer_output(self, attention_cls):
    attention = attention_cls(self.units)

    score = attention([self.query, self.state, self.memory])
    self.evaluate(variables.variables_initializer(attention.variables))

    score_val = self.evaluate(score)
    self.assertLen(score_val, 2)
    self.assertEqual(score_val[0].shape, (self.batch, self.timestep))
    self.assertEqual(score_val[1].shape, (self.batch, self.timestep))

if __name__ == "__main__":
  test.main()
