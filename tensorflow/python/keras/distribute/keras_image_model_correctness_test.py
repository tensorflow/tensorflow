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
"""Correctness tests for tf.keras CNN models using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.eager import context
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.distribute import keras_correctness_test_base
from tensorflow.python.keras.optimizer_v2 import gradient_descent


@testing_utils.run_all_without_tensor_float_32(
    'Uses Dense layers, which call matmul. Even if Dense layers run in '
    'float64, the test sometimes fails with TensorFloat-32 enabled for unknown '
    'reasons')
class DistributionStrategyCnnCorrectnessTest(
    keras_correctness_test_base.TestDistributionStrategyCorrectnessBase):

  def get_model(self,
                initial_weights=None,
                distribution=None,
                input_shapes=None):
    del input_shapes
    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      image = keras.layers.Input(shape=(28, 28, 3), name='image')
      c1 = keras.layers.Conv2D(
          name='conv1',
          filters=16,
          kernel_size=(3, 3),
          strides=(4, 4),
          kernel_regularizer=keras.regularizers.l2(1e-4))(
              image)
      if self.with_batch_norm == 'regular':
        c1 = keras.layers.BatchNormalization(name='bn1')(c1)
      elif self.with_batch_norm == 'sync':
        # Test with parallel batch norms to verify all-reduce works OK.
        bn1 = keras.layers.SyncBatchNormalization(name='bn1')(c1)
        bn2 = keras.layers.SyncBatchNormalization(name='bn2')(c1)
        c1 = keras.layers.Add()([bn1, bn2])
      c1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)
      logits = keras.layers.Dense(
          10, activation='softmax', name='pred')(
              keras.layers.Flatten()(c1))
      model = keras.Model(inputs=[image], outputs=[logits])

      if initial_weights:
        model.set_weights(initial_weights)

      model.compile(
          optimizer=gradient_descent.SGD(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])

    return model

  def _get_data(self, count, shape=(28, 28, 3), num_classes=10):
    centers = np.random.randn(num_classes, *shape)

    features = []
    labels = []
    for _ in range(count):
      label = np.random.randint(0, num_classes, size=1)[0]
      offset = np.random.normal(loc=0, scale=0.1, size=np.prod(shape))
      offset = offset.reshape(shape)
      labels.append(label)
      features.append(centers[label] + offset)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape((count, 1))
    return x, y

  def get_data(self):
    x_train, y_train = self._get_data(
        count=keras_correctness_test_base._GLOBAL_BATCH_SIZE *
        keras_correctness_test_base._EVAL_STEPS)
    x_predict = x_train
    return x_train, y_train, x_predict

  def get_data_with_partial_last_batch_eval(self):
    x_train, y_train = self._get_data(count=1280)
    x_eval, y_eval = self._get_data(count=1000)
    return x_train, y_train, x_eval, y_eval, x_eval

  @ds_combinations.generate(
      keras_correctness_test_base.all_strategy_and_input_config_combinations() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  def test_cnn_correctness(self, distribution, use_numpy, use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)

  @ds_combinations.generate(
      keras_correctness_test_base.all_strategy_and_input_config_combinations() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  def test_cnn_with_batch_norm_correctness(self, distribution, use_numpy,
                                           use_validation_data):
    self.run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        with_batch_norm='regular')

  @ds_combinations.generate(
      keras_correctness_test_base.all_strategy_and_input_config_combinations() +
      keras_correctness_test_base.multi_worker_mirrored_eager())
  def test_cnn_with_sync_batch_norm_correctness(self, distribution, use_numpy,
                                                use_validation_data):
    if not context.executing_eagerly():
      self.skipTest('SyncBatchNorm is not enabled in graph mode.')

    self.run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        with_batch_norm='sync')

  @ds_combinations.generate(
      keras_correctness_test_base
      .all_strategy_and_input_config_combinations_eager() +
      keras_correctness_test_base.multi_worker_mirrored_eager() +
      keras_correctness_test_base.test_combinations_with_tpu_strategies_graph())
  def test_cnn_correctness_with_partial_last_batch_eval(self, distribution,
                                                        use_numpy,
                                                        use_validation_data):
    self.run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        partial_last_batch=True,
        training_epochs=1)

  @ds_combinations.generate(
      keras_correctness_test_base.
      all_strategy_and_input_config_combinations_eager() +
      keras_correctness_test_base.multi_worker_mirrored_eager() +
      keras_correctness_test_base.test_combinations_with_tpu_strategies_graph())
  def test_cnn_with_batch_norm_correctness_and_partial_last_batch_eval(
      self, distribution, use_numpy, use_validation_data):
    self.run_correctness_test(
        distribution,
        use_numpy,
        use_validation_data,
        with_batch_norm='regular',
        partial_last_batch=True)


if __name__ == '__main__':
  multi_process_runner.test_main()
