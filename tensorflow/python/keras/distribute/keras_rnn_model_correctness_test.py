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
"""Correctness tests for tf.keras RNN models using DistributionStrategy."""

import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.layers import recurrent as rnn_v1
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras


class _DistributionStrategyRnnModelCorrectnessTest(
    keras_correctness_test_base
    .TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def _get_layer_class(self):
    raise NotImplementedError

  def get_model(self,
                max_words=10,
                initial_weights=None,
                distribution=None,
                input_shapes=None):
    del input_shapes
    rnn_cls = self._get_layer_class()

    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words')
      word_embed = keras.layers.Embedding(input_dim=20, output_dim=10)(word_ids)
      rnn_embed = rnn_cls(units=4, return_sequences=False)(word_embed)

      dense_output = keras.layers.Dense(2)(rnn_embed)
      preds = keras.layers.Softmax(dtype='float32')(dense_output)
      model = keras.Model(inputs=[word_ids], outputs=[preds])

      if initial_weights:
        model.set_weights(initial_weights)

      optimizer_fn = gradient_descent_keras.SGD

      model.compile(
          optimizer=optimizer_fn(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
    return model


@testing_utils.run_all_without_tensor_float_32(
    'Uses Dense layers, which call matmul')
class DistributionStrategyGruModelCorrectnessTest(
    _DistributionStrategyRnnModelCorrectnessTest):

  def _get_layer_class(self):
    if tf2.enabled():
      if not context.executing_eagerly():
        self.skipTest("GRU v2 and legacy graph mode don't work together.")
      return rnn_v2.GRU
    else:
      return rnn_v1.GRU

  @ds_combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  def test_gru_model_correctness(self, distribution, use_numpy,
                                 use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)


@testing_utils.run_all_without_tensor_float_32(
    'Uses Dense layers, which call matmul')
class DistributionStrategyLstmModelCorrectnessTest(
    _DistributionStrategyRnnModelCorrectnessTest):

  def _get_layer_class(self):
    if tf2.enabled():
      if not context.executing_eagerly():
        self.skipTest("LSTM v2 and legacy graph mode don't work together.")
      return rnn_v2.LSTM
    else:
      return rnn_v1.LSTM

  @ds_combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  def test_lstm_model_correctness(self, distribution, use_numpy,
                                  use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)

  @ds_combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  @testing_utils.enable_v2_dtype_behavior
  def test_lstm_model_correctness_mixed_precision(self, distribution, use_numpy,
                                                  use_validation_data):
    if isinstance(distribution,
                  (central_storage_strategy.CentralStorageStrategy,
                   central_storage_strategy.CentralStorageStrategyV1)):
      self.skipTest('CentralStorageStrategy is not supported by '
                    'mixed precision.')
    if isinstance(distribution,
                  (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)):
      policy_name = 'mixed_bfloat16'
    else:
      policy_name = 'mixed_float16'

    with policy.policy_scope(policy_name):
      self.run_correctness_test(distribution, use_numpy, use_validation_data)


if __name__ == '__main__':
  multi_process_runner.test_main()
