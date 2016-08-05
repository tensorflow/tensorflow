# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for rnn_cell"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.ops.init_ops import random_normal_initializer
import tensorflow as tf


class ESNCellTest(tf.test.TestCase):

  def test_esn_dynamics(self):
    """ A simple smoke test """

    input_size = 4
    input_length = 4
    batch_size = 2
    n_units = 4

    cell = rnn_cell.ESNCell(n_units)
    inputs = tf.get_variable("inputs", [input_length, batch_size, input_size], initializer=random_normal_initializer())

    state = cell.zero_state(batch_size, inputs.dtype)
    for i in range(input_length):
      if i == 1 : tf.get_variable_scope().reuse_variables()
      state, _ = cell(inputs[i, :, :], state)

    with self.test_session() as S:
      S.run(tf.initialize_all_variables())
      final_states = S.run([state])

    expected_final_states = [[[0.56619251, 0.90794069, -0.99339789, -0.38217574],
                              [0.97225142, 0.98931664, -0.65432256, -0.66463804]]]

    self.assertAllClose(final_states, expected_final_states)