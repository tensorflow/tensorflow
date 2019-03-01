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

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import keras_correctness_test_base
from tensorflow.python import keras
from tensorflow.python.eager import test
from tensorflow.python.training import gradient_descent


class DistributionStrategyLstmModelCorrectnessTest(
    keras_correctness_test_base.
    TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def get_model(self, max_words=10, initial_weights=None, distribution=None):
    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words')
      word_embed = keras.layers.Embedding(input_dim=20,
                                          output_dim=10)(word_ids)
      lstm_embed = keras.layers.LSTM(units=4,
                                     return_sequences=False)(word_embed)

      preds = keras.layers.Dense(2, activation='softmax')(lstm_embed)
      model = keras.Model(inputs=[word_ids], outputs=[preds])

      if initial_weights:
        model.set_weights(initial_weights)

      model.compile(
          optimizer=gradient_descent.GradientDescentOptimizer(
              learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
    return model

  @combinations.generate(keras_correctness_test_base.
                         test_combinations_for_embedding_model())
  def test_lstm_model_correctness(self,
                                  distribution,
                                  use_numpy,
                                  use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)


if __name__ == '__main__':
  test.main()
