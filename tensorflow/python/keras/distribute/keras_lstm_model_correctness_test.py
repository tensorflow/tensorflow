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
"""Correctness tests for tf.keras LSTM model using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.distribute import combinations
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.layers import recurrent as rnn_v1
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras


class DistributionStrategyLstmModelCorrectnessTest(
    keras_correctness_test_base
    .TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def get_model(self,
                max_words=10,
                initial_weights=None,
                distribution=None,
                experimental_run_tf_function=None,
                input_shapes=None):
    del input_shapes

    if tf2.enabled():
      if not context.executing_eagerly():
        self.skipTest("LSTM v2 and legacy graph mode don't work together.")
      lstm = rnn_v2.LSTM
    else:
      lstm = rnn_v1.LSTM

    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words')
      word_embed = keras.layers.Embedding(input_dim=20, output_dim=10)(word_ids)
      lstm_embed = lstm(units=4, return_sequences=False)(
          word_embed)

      preds = keras.layers.Dense(2, activation='softmax')(lstm_embed)
      model = keras.Model(inputs=[word_ids], outputs=[preds])

      if initial_weights:
        model.set_weights(initial_weights)

      optimizer_fn = gradient_descent_keras.SGD

      model.compile(
          optimizer=optimizer_fn(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'],
          experimental_run_tf_function=experimental_run_tf_function)
    return model

  @combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model())
  def test_lstm_model_correctness(self, distribution, use_numpy,
                                  use_validation_data,
                                  experimental_run_tf_function):
    self.run_correctness_test(distribution, use_numpy, use_validation_data,
                              experimental_run_tf_function)


if __name__ == '__main__':
  test.main()
