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
"""Correctness test for tf.keras Embedding models using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.distribute import combinations
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.platform import test


class DistributionStrategyEmbeddingModelCorrectnessTest(
    keras_correctness_test_base
    .TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def get_model(self,
                max_words=10,
                initial_weights=None,
                distribution=None,
                input_shapes=None):
    del input_shapes
    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words')
      word_embed = keras.layers.Embedding(input_dim=20, output_dim=10)(word_ids)
      if self.use_distributed_dense:
        word_embed = keras.layers.TimeDistributed(keras.layers.Dense(4))(
            word_embed)
      avg = keras.layers.GlobalAveragePooling1D()(word_embed)
      preds = keras.layers.Dense(2, activation='softmax')(avg)
      model = keras.Model(inputs=[word_ids], outputs=[preds])

      if initial_weights:
        model.set_weights(initial_weights)

      model.compile(
          optimizer=gradient_descent_keras.SGD(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
    return model

  @combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model())
  def test_embedding_model_correctness(self, distribution, use_numpy,
                                       use_validation_data):

    self.use_distributed_dense = False
    self.run_correctness_test(distribution, use_numpy, use_validation_data)

  @combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model())
  def test_embedding_time_distributed_model_correctness(
      self, distribution, use_numpy, use_validation_data):
    self.use_distributed_dense = True
    self.run_correctness_test(distribution, use_numpy, use_validation_data)


class DistributionStrategySiameseEmbeddingModelCorrectnessTest(
    keras_correctness_test_base
    .TestDistributionStrategyEmbeddingModelCorrectnessBase):

  def get_model(self,
                max_words=10,
                initial_weights=None,
                distribution=None,
                input_shapes=None):
    del input_shapes
    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      word_ids_a = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words_a')
      word_ids_b = keras.layers.Input(
          shape=(max_words,), dtype=np.int32, name='words_b')

      def submodel(embedding, word_ids):
        word_embed = embedding(word_ids)
        rep = keras.layers.GlobalAveragePooling1D()(word_embed)
        return keras.Model(inputs=[word_ids], outputs=[rep])

      word_embed = keras.layers.Embedding(
          input_dim=20,
          output_dim=10,
          input_length=max_words,
          embeddings_initializer=keras.initializers.RandomUniform(0, 1))

      a_rep = submodel(word_embed, word_ids_a).outputs[0]
      b_rep = submodel(word_embed, word_ids_b).outputs[0]
      sim = keras.layers.Dot(axes=1, normalize=True)([a_rep, b_rep])

      model = keras.Model(inputs=[word_ids_a, word_ids_b], outputs=[sim])

      if initial_weights:
        model.set_weights(initial_weights)

      # TODO(b/130808953): Switch back to the V1 optimizer after global_step
      # is made mirrored.
      model.compile(
          optimizer=gradient_descent_keras.SGD(learning_rate=0.1),
          loss='mse',
          metrics=['mse'])
    return model

  def get_data(self,
               count=(keras_correctness_test_base._GLOBAL_BATCH_SIZE *
                      keras_correctness_test_base._EVAL_STEPS),
               min_words=5,
               max_words=10,
               max_word_id=19,
               num_classes=2):
    features_a, labels_a, _ = (
        super(DistributionStrategySiameseEmbeddingModelCorrectnessTest,
              self).get_data(count, min_words, max_words, max_word_id,
                             num_classes))

    features_b, labels_b, _ = (
        super(DistributionStrategySiameseEmbeddingModelCorrectnessTest,
              self).get_data(count, min_words, max_words, max_word_id,
                             num_classes))

    y_train = np.zeros((count, 1), dtype=np.float32)
    y_train[labels_a == labels_b] = 1.0
    y_train[labels_a != labels_b] = -1.0
    # TODO(b/123360757): Add tests for using list as inputs for multi-input
    # models.
    x_train = {
        'words_a': features_a,
        'words_b': features_b,
    }
    x_predict = x_train

    return x_train, y_train, x_predict

  @combinations.generate(
      keras_correctness_test_base.test_combinations_for_embedding_model())
  def test_siamese_embedding_model_correctness(self, distribution, use_numpy,
                                               use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)


if __name__ == '__main__':
  test.main()
