# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests specific to Feature Columns integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class TestDNNModel(keras.models.Model):

  def __init__(self, feature_columns, units, name=None, **kwargs):
    super(TestDNNModel, self).__init__(name=name, **kwargs)
    self._input_layer = fc.DenseFeatures(feature_columns, name='input_layer')
    self._dense_layer = keras.layers.Dense(units, name='dense_layer')

  def call(self, features):
    net = self._input_layer(features)
    net = self._dense_layer(net)
    return net


class FeatureColumnsIntegrationTest(keras_parameterized.TestCase):
  """Most Sequential model API tests are covered in `training_test.py`.

  """

  @keras_parameterized.run_all_keras_modes
  def test_sequential_model(self):
    columns = [fc.numeric_column('a')]
    model = keras.models.Sequential([
        fc.DenseFeatures(columns),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(20, activation='softmax')
    ])
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())

    x = {'a': np.random.random((10, 1))}
    y = np.random.randint(20, size=(10, 1))
    y = keras.utils.to_categorical(y, num_classes=20)
    model.fit(x, y, epochs=1, batch_size=5)
    model.fit(x, y, epochs=1, batch_size=5)
    model.evaluate(x, y, batch_size=5)
    model.predict(x, batch_size=5)

  @keras_parameterized.run_all_keras_modes
  def test_sequential_model_with_ds_input(self):
    columns = [fc.numeric_column('a')]
    model = keras.models.Sequential([
        fc.DenseFeatures(columns),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(20, activation='softmax')
    ])
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())

    y = np.random.randint(20, size=(100, 1))
    y = keras.utils.to_categorical(y, num_classes=20)
    x = {'a': np.random.random((100, 1))}
    ds1 = dataset_ops.Dataset.from_tensor_slices(x)
    ds2 = dataset_ops.Dataset.from_tensor_slices(y)
    ds = dataset_ops.Dataset.zip((ds1, ds2)).batch(5)
    model.fit(ds, steps_per_epoch=1)
    model.fit(ds, steps_per_epoch=1)
    model.evaluate(ds, steps=1)
    model.predict(ds, steps=1)

  @keras_parameterized.run_all_keras_modes
  def test_subclassed_model_with_feature_columns(self):
    col_a = fc.numeric_column('a')
    col_b = fc.numeric_column('b')

    dnn_model = TestDNNModel([col_a, col_b], 20)

    dnn_model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())

    x = {'a': np.random.random((10, 1)), 'b': np.random.random((10, 1))}
    y = np.random.randint(20, size=(10, 1))
    y = keras.utils.to_categorical(y, num_classes=20)
    dnn_model.fit(x=x, y=y, epochs=1, batch_size=5)
    dnn_model.fit(x=x, y=y, epochs=1, batch_size=5)
    dnn_model.evaluate(x=x, y=y, batch_size=5)
    dnn_model.predict(x=x, batch_size=5)

  @keras_parameterized.run_all_keras_modes
  def test_subclassed_model_with_feature_columns_with_ds_input(self):
    col_a = fc.numeric_column('a')
    col_b = fc.numeric_column('b')

    dnn_model = TestDNNModel([col_a, col_b], 20)

    dnn_model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())

    y = np.random.randint(20, size=(100, 1))
    y = keras.utils.to_categorical(y, num_classes=20)
    x = {'a': np.random.random((100, 1)), 'b': np.random.random((100, 1))}
    ds1 = dataset_ops.Dataset.from_tensor_slices(x)
    ds2 = dataset_ops.Dataset.from_tensor_slices(y)
    ds = dataset_ops.Dataset.zip((ds1, ds2)).batch(5)
    dnn_model.fit(ds, steps_per_epoch=1)
    dnn_model.fit(ds, steps_per_epoch=1)
    dnn_model.evaluate(ds, steps=1)
    dnn_model.predict(ds, steps=1)

  # TODO(kaftan) seems to throw an error when enabled.
  @keras_parameterized.run_all_keras_modes
  def DISABLED_test_function_model_feature_layer_input(self):
    col_a = fc.numeric_column('a')
    col_b = fc.numeric_column('b')

    feature_layer = fc.DenseFeatures([col_a, col_b], name='fc')
    dense = keras.layers.Dense(4)

    # This seems problematic.... We probably need something for DenseFeatures
    # the way Input is for InputLayer.
    output = dense(feature_layer)

    model = keras.models.Model([feature_layer], [output])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights)

    data = ({'a': np.arange(10), 'b': np.arange(10)}, np.arange(10, 20))
    print(model.fit(*data, epochs=1))

  # TODO(kaftan) seems to throw an error when enabled.
  @keras_parameterized.run_all_keras_modes
  def DISABLED_test_function_model_multiple_feature_layer_inputs(self):
    col_a = fc.numeric_column('a')
    col_b = fc.numeric_column('b')
    col_c = fc.numeric_column('c')

    fc1 = fc.DenseFeatures([col_a, col_b], name='fc1')
    fc2 = fc.DenseFeatures([col_b, col_c], name='fc2')
    dense = keras.layers.Dense(4)

    # This seems problematic.... We probably need something for DenseFeatures
    # the way Input is for InputLayer.
    output = dense(fc1) + dense(fc2)

    model = keras.models.Model([fc1, fc2], [output])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights)

    data_list = ([{
        'a': np.arange(10),
        'b': np.arange(10)
    }, {
        'b': np.arange(10),
        'c': np.arange(10)
    }], np.arange(10, 100))
    print(model.fit(*data_list, epochs=1))

    data_bloated_list = ([{
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10)
    }, {
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10)
    }], np.arange(10, 100))
    print(model.fit(*data_bloated_list, epochs=1))

    data_dict = ({
        'fc1': {
            'a': np.arange(10),
            'b': np.arange(10)
        },
        'fc2': {
            'b': np.arange(10),
            'c': np.arange(10)
        }
    }, np.arange(10, 100))
    print(model.fit(*data_dict, epochs=1))

    data_bloated_dict = ({
        'fc1': {
            'a': np.arange(10),
            'b': np.arange(10),
            'c': np.arange(10)
        },
        'fc2': {
            'a': np.arange(10),
            'b': np.arange(10),
            'c': np.arange(10)
        }
    }, np.arange(10, 100))
    print(model.fit(*data_bloated_dict, epochs=1))


if __name__ == '__main__':
  test.main()
