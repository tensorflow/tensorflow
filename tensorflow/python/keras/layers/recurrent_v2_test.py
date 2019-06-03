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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
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
    with test_util.device(use_gpu=False):
      model = keras.Sequential([
          keras.layers.Embedding(vocab_size, embedding_dim,
                                 batch_input_shape=[batch_size, timestep]),
          layer(units, return_sequences=True, stateful=True),
          keras.layers.Dense(vocab_size)
      ])
      model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    run_eagerly=testing_utils.should_run_eagerly())
      model.fit(x, y, epochs=1, shuffle=False)


if __name__ == '__main__':
  test.main()
