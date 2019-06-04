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
"""Tests for Keras regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


DATA_DIM = 5
NUM_CLASSES = 2


class KerasRegularizersTest(keras_parameterized.TestCase):

  def create_model(self, kernel_regularizer=None, activity_regularizer=None):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(NUM_CLASSES,
                                 kernel_regularizer=kernel_regularizer,
                                 activity_regularizer=activity_regularizer,
                                 input_shape=(DATA_DIM,)))
    return model

  def get_data(self):
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=10,
        test_samples=10,
        input_shape=(DATA_DIM,),
        num_classes=NUM_CLASSES)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)

  @parameterized.named_parameters([
      ('l1', keras.regularizers.l1()),
      ('l2', keras.regularizers.l2()),
      ('l1_l2', keras.regularizers.l1_l2()),
  ])
  def test_kernel_regularization(self, regularizer):
    with self.cached_session():
      (x_train, y_train), _ = self.get_data()
      model = self.create_model(kernel_regularizer=regularizer)
      model.compile(loss='categorical_crossentropy', optimizer='sgd')
      assert len(model.losses) == 1
      model.fit(x_train, y_train, batch_size=10,
                epochs=1, verbose=0)

  @parameterized.named_parameters([
      ('l1', keras.regularizers.l1()),
      ('l2', keras.regularizers.l2()),
      ('l2_zero', keras.regularizers.l2(0.)),
  ])
  @test_util.deprecated_graph_mode_only
  def test_activity_regularization(self, regularizer):
    with self.cached_session():
      (x_train, y_train), _ = self.get_data()
      model = self.create_model(activity_regularizer=regularizer)
      model.compile(loss='categorical_crossentropy', optimizer='sgd')
      assert len(model.losses) == 1
      model.fit(x_train, y_train, batch_size=10,
                epochs=1, verbose=0)

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_zero_regularization(self):
    # Verifies that training with zero regularization works.
    x, y = np.ones((10, 10)), np.ones((10, 3))
    model = testing_utils.get_model_from_layers(
        [keras.layers.Dense(3, kernel_regularizer=keras.regularizers.l2(0))],
        input_shape=(10,))
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=5, epochs=1)

  def test_custom_regularizer_saving(self):

    def my_regularizer(weights):
      return math_ops.reduce_sum(math_ops.abs(weights))

    inputs = keras.Input((10,))
    outputs = keras.layers.Dense(1, kernel_regularizer=my_regularizer)(inputs)
    model = keras.Model(inputs, outputs)
    model2 = model.from_config(
        model.get_config(), custom_objects={'my_regularizer': my_regularizer})
    self.assertEqual(model2.layers[1].kernel_regularizer, my_regularizer)


if __name__ == '__main__':
  test.main()
