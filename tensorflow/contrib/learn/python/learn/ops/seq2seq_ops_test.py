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
"""Sequence-to-sequence tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import ops


class Seq2SeqOpsTest(tf.test.TestCase):
  """Sequence-to-sequence tests."""

  def test_sequence_classifier(self):
    with self.test_session() as session:
      decoding = [tf.placeholder(tf.float32, [2, 2]) for _ in range(3)]
      labels = [tf.placeholder(tf.float32, [2, 2]) for _ in range(3)]
      sampling_decoding = [tf.placeholder(tf.float32, [2, 2]) for _ in range(3)]
      predictions, loss = ops.sequence_classifier(decoding, labels,
                                                  sampling_decoding)
      pred, cost = session.run(
          [predictions, loss],
          feed_dict={
              decoding[0].name: [[0.1, 0.9], [0.7, 0.3]],
              decoding[1].name: [[0.9, 0.1], [0.8, 0.2]],
              decoding[2].name: [[0.5, 0.5], [0.4, 0.6]],
              labels[0].name: [[1, 0], [0, 1]],
              labels[1].name: [[1, 0], [0, 1]],
              labels[2].name: [[1, 0], [0, 1]],
              sampling_decoding[0].name: [[0.1, 0.9], [0.7, 0.3]],
              sampling_decoding[1].name: [[0.9, 0.1], [0.8, 0.2]],
              sampling_decoding[2].name: [[0.5, 0.5], [0.4, 0.6]],
          })
    self.assertAllEqual(pred.argmax(axis=2), [[1, 0, 0], [0, 0, 1]])
    self.assertAllClose(cost, 4.7839908599)

  def test_seq2seq_inputs(self):
    inp = np.array([[[1, 0], [0, 1], [1, 0]], [[0, 1], [1, 0], [0, 1]]])
    out = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]])
    with self.test_session() as session:
      x = tf.placeholder(tf.float32, [2, 3, 2])
      y = tf.placeholder(tf.float32, [2, 2, 3])
      in_x, in_y, out_y = ops.seq2seq_inputs(x, y, 3, 2)
      enc_inp = session.run(in_x, feed_dict={x.name: inp})
      dec_inp = session.run(in_y, feed_dict={x.name: inp, y.name: out})
      dec_out = session.run(out_y, feed_dict={x.name: inp, y.name: out})
    # Swaps from batch x len x height to list of len of batch x height.
    self.assertAllEqual(enc_inp, np.swapaxes(inp, 0, 1))
    self.assertAllEqual(dec_inp, [[[0, 0, 0], [0, 0, 0]],
                                  [[0, 1, 0], [1, 0, 0]],
                                  [[1, 0, 0], [0, 1, 0]]])
    self.assertAllEqual(dec_out, [[[0, 1, 0], [1, 0, 0]],
                                  [[1, 0, 0], [0, 1, 0]],
                                  [[0, 0, 0], [0, 0, 0]]])

  def test_rnn_decoder(self):
    with self.test_session():
      decoder_inputs = [tf.placeholder(tf.float32, [2, 2]) for _ in range(3)]
      encoding = tf.placeholder(tf.float32, [2, 2])
      cell = tf.contrib.rnn.GRUCell(2)
      outputs, states, sampling_outputs, sampling_states = (
          ops.rnn_decoder(decoder_inputs, encoding, cell))
      self.assertEqual(len(outputs), 3)
      self.assertEqual(outputs[0].get_shape(), [2, 2])
      self.assertEqual(len(states), 4)
      self.assertEqual(states[0].get_shape(), [2, 2])
      self.assertEqual(len(sampling_outputs), 3)
      self.assertEqual(sampling_outputs[0].get_shape(), [2, 2])
      self.assertEqual(len(sampling_states), 4)
      self.assertEqual(sampling_states[0].get_shape(), [2, 2])


if __name__ == '__main__':
  tf.test.main()
