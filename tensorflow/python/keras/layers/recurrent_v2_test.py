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
"""Tests for recurrent v2 layers functionality other than GRU, LSTM.

See also: lstm_v2_test.py, gru_v2_test.py.
"""

import os
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class RNNV2Test(keras_parameterized.TestCase):

  @parameterized.parameters([rnn_v2.LSTM, rnn_v2.GRU])
  def test_device_placement(self, layer):
    if not test.is_gpu_available():
      self.skipTest('Need GPU for testing.')
    vocab_size = 20
    embedding_dim = 10
    batch_size = 8
    timestep = 12
    units = 5
    x = np.random.randint(0, vocab_size, size=(batch_size, timestep))
    y = np.random.randint(0, vocab_size, size=(batch_size, timestep))

    # Test when GPU is available but not used, the graph should be properly
    # created with CPU ops.
    with testing_utils.device(should_use_gpu=False):
      model = keras.Sequential([
          keras.layers.Embedding(vocab_size, embedding_dim,
                                 batch_input_shape=[batch_size, timestep]),
          layer(units, return_sequences=True, stateful=True),
          keras.layers.Dense(vocab_size)
      ])
      model.compile(
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          run_eagerly=testing_utils.should_run_eagerly())
      model.fit(x, y, epochs=1, shuffle=False)

  @parameterized.parameters([rnn_v2.LSTM, rnn_v2.GRU])
  def test_reset_dropout_mask_between_batch(self, layer):
    # See https://github.com/tensorflow/tensorflow/issues/29187 for more details
    batch_size = 8
    timestep = 12
    embedding_dim = 10
    units = 5
    layer = layer(units, dropout=0.5, recurrent_dropout=0.5)

    inputs = np.random.random((batch_size, timestep, embedding_dim)).astype(
        np.float32)
    previous_dropout, previous_recurrent_dropout = None, None

    for _ in range(5):
      layer(inputs, training=True)
      dropout = layer.cell.get_dropout_mask_for_cell(inputs, training=True)
      recurrent_dropout = layer.cell.get_recurrent_dropout_mask_for_cell(
          inputs, training=True)
      if previous_dropout is not None:
        self.assertNotAllClose(self.evaluate(previous_dropout),
                               self.evaluate(dropout))
        previous_dropout = dropout
      if previous_recurrent_dropout is not None:
        self.assertNotAllClose(self.evaluate(previous_recurrent_dropout),
                               self.evaluate(recurrent_dropout))
        previous_recurrent_dropout = recurrent_dropout

  @parameterized.parameters([rnn_v2.LSTM, rnn_v2.GRU])
  def test_recurrent_dropout_with_stateful_RNN(self, layer):
    # See https://github.com/tensorflow/tensorflow/issues/27829 for details.
    # The issue was caused by using inplace mul for a variable, which was a
    # warning for RefVariable, but an error for ResourceVariable in 2.0
    keras.models.Sequential([
        layer(128, stateful=True, return_sequences=True, dropout=0.2,
              batch_input_shape=[32, None, 5], recurrent_dropout=0.2)
    ])

  def test_recurrent_dropout_saved_model(self):
    if not context.executing_eagerly():
      self.skipTest('v2-only test')
    inputs = keras.Input(shape=(784, 3), name='digits')
    x = keras.layers.GRU(64, activation='relu', name='GRU', dropout=0.1)(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense')(x)
    outputs = keras.layers.Dense(
        10, activation='softmax', name='predictions')(
            x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer')
    model.save(os.path.join(self.get_temp_dir(), 'model'), save_format='tf')

  @parameterized.parameters([rnn_v2.LSTM, rnn_v2.GRU])
  def test_ragged(self, layer):
    vocab_size = 100
    inputs = ragged_factory_ops.constant(
        np.random.RandomState(0).randint(0, vocab_size, [128, 25]))
    embedder = embeddings.Embedding(input_dim=vocab_size, output_dim=16)
    embedded_inputs = embedder(inputs)
    lstm = layer(32)
    lstm(embedded_inputs)


if __name__ == '__main__':
  test.main()
