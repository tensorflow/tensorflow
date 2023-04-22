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
"""Tests for stateful tf.keras LSTM models using DistributionStrategy."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.platform import test


def strategies_for_stateful_embedding_model():
  """Returns TPUStrategy with single core device assignment."""

  return [
      strategy_combinations.tpu_strategy_one_core,
  ]


def test_combinations_for_stateful_embedding_model():
  return (combinations.combine(
      distribution=strategies_for_stateful_embedding_model(),
      mode='graph',
      use_numpy=False,
      use_validation_data=False))


class DistributionStrategyStatefulLstmModelCorrectnessTest(
    keras_correctness_test_base
    .TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def get_model(self,
                max_words=10,
                initial_weights=None,
                distribution=None,
                input_shapes=None):
    del input_shapes
    batch_size = keras_correctness_test_base._GLOBAL_BATCH_SIZE

    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids = keras.layers.Input(
          shape=(max_words,),
          batch_size=batch_size,
          dtype=np.int32,
          name='words')
      word_embed = keras.layers.Embedding(input_dim=20, output_dim=10)(word_ids)
      lstm_embed = keras.layers.LSTM(
          units=4, return_sequences=False, stateful=True)(
              word_embed)

      preds = keras.layers.Dense(2, activation='softmax')(lstm_embed)
      model = keras.Model(inputs=[word_ids], outputs=[preds])

      if initial_weights:
        model.set_weights(initial_weights)

      optimizer_fn = gradient_descent_keras.SGD

      model.compile(
          optimizer=optimizer_fn(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
    return model

  # TODO(jhseu): Disabled to fix b/130808953. Need to investigate why it
  # doesn't work and enable for DistributionStrategy more generally.
  @ds_combinations.generate(test_combinations_for_stateful_embedding_model())
  def disabled_test_stateful_lstm_model_correctness(
      self, distribution, use_numpy, use_validation_data):
    self.run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        is_stateful_model=True)

  @ds_combinations.generate(
      combinations.times(
          keras_correctness_test_base
          .test_combinations_with_tpu_strategies_graph()))
  def test_incorrectly_use_multiple_cores_for_stateful_lstm_model(
      self, distribution, use_numpy, use_validation_data):
    with self.assertRaisesRegex(
        ValueError, 'RNNs with stateful=True not yet supported with '
        'tf.distribute.Strategy.'):
      self.run_correctness_test(
          distribution,
          use_numpy,
          use_validation_data,
          is_stateful_model=True)


if __name__ == '__main__':
  test.main()
