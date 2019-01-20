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

from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


DATA_DIM = 5
NUM_CLASSES = 2


def get_data():
  (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
      train_samples=10,
      test_samples=10,
      input_shape=(DATA_DIM,),
      num_classes=NUM_CLASSES)
  y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
  return (x_train, y_train), (x_test, y_test)


def create_model(kernel_regularizer=None, activity_regularizer=None):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(NUM_CLASSES,
                               kernel_regularizer=kernel_regularizer,
                               activity_regularizer=activity_regularizer,
                               input_shape=(DATA_DIM,)))
  return model


class KerasRegularizersTest(test.TestCase):

  def test_kernel_regularization(self):
    with self.cached_session():
      (x_train, y_train), _ = get_data()
      for reg in [keras.regularizers.l1(),
                  keras.regularizers.l2(),
                  keras.regularizers.l1_l2()]:
        model = create_model(kernel_regularizer=reg)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        assert len(model.losses) == 1
        model.fit(x_train, y_train, batch_size=10,
                  epochs=1, verbose=0)

  @test_util.run_deprecated_v1
  def test_activity_regularization(self):
    with self.cached_session():
      (x_train, y_train), _ = get_data()
      for reg in [keras.regularizers.l1(), keras.regularizers.l2()]:
        model = create_model(activity_regularizer=reg)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        assert len(model.losses) == 1
        model.fit(x_train, y_train, batch_size=10,
                  epochs=1, verbose=0)

  def test_zero_regularization(self):
    inputs = keras.backend.ones(shape=(10, 10))
    layer = keras.layers.Dense(3, kernel_regularizer=keras.regularizers.l2(0))
    layer(inputs)


if __name__ == '__main__':
  test.main()
