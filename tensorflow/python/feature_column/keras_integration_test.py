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
"""Tests specific to Feature Columns and Keras integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import dense_features_v2
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.premade import linear
from tensorflow.python.keras.premade import wide_deep
from tensorflow.python.keras.utils import np_utils
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
    y = np_utils.to_categorical(y, num_classes=20)
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
    y = np_utils.to_categorical(y, num_classes=20)
    x = {'a': np.random.random((100, 1))}
    ds1 = dataset_ops.Dataset.from_tensor_slices(x)
    ds2 = dataset_ops.Dataset.from_tensor_slices(y)
    ds = dataset_ops.Dataset.zip((ds1, ds2)).batch(5)
    model.fit(ds, steps_per_epoch=1)
    model.fit(ds, steps_per_epoch=1)
    model.evaluate(ds, steps=1)
    model.predict(ds, steps=1)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_sequential_model_with_crossed_column(self):
    feature_columns = []
    age_buckets = fc.bucketized_column(
        fc.numeric_column('age'),
        boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # indicator cols
    thal = fc.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])

    crossed_feature = fc.crossed_column([age_buckets, thal],
                                        hash_bucket_size=1000)
    crossed_feature = fc.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    feature_layer = fc.DenseFeatures(feature_columns)

    model = keras.models.Sequential([
        feature_layer,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    age_data = np.random.randint(10, 100, size=100)
    thal_data = np.random.choice(['fixed', 'normal', 'reversible'], size=100)
    inp_x = {'age': age_data, 'thal': thal_data}
    inp_y = np.random.randint(0, 1, size=100)
    ds = dataset_ops.Dataset.from_tensor_slices((inp_x, inp_y)).batch(5)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],)
    model.fit(ds, epochs=1)
    model.fit(ds, epochs=1)
    model.evaluate(ds)
    model.predict(ds)

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
    y = np_utils.to_categorical(y, num_classes=20)
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
    y = np_utils.to_categorical(y, num_classes=20)
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
    model.fit(*data, epochs=1)

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
    model.fit(*data_list, epochs=1)

    data_bloated_list = ([{
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10)
    }, {
        'a': np.arange(10),
        'b': np.arange(10),
        'c': np.arange(10)
    }], np.arange(10, 100))
    model.fit(*data_bloated_list, epochs=1)

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
    model.fit(*data_dict, epochs=1)

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
    model.fit(*data_bloated_dict, epochs=1)

  @keras_parameterized.run_all_keras_modes
  def test_string_input(self):
    x = {'age': np.random.random((1024, 1)),
         'cabin': np.array(['a'] * 1024)}
    y = np.random.randint(2, size=(1024, 1))
    ds1 = dataset_ops.Dataset.from_tensor_slices(x)
    ds2 = dataset_ops.Dataset.from_tensor_slices(y)
    dataset = dataset_ops.Dataset.zip((ds1, ds2)).batch(4)
    categorical_cols = [fc.categorical_column_with_hash_bucket('cabin', 10)]
    feature_cols = ([fc.numeric_column('age')]
                    + [fc.indicator_column(cc) for cc in categorical_cols])
    layers = [fc.DenseFeatures(feature_cols),
              keras.layers.Dense(128),
              keras.layers.Dense(1)]

    model = keras.models.Sequential(layers)
    model.compile(optimizer='sgd',
                  loss=keras.losses.BinaryCrossentropy())
    model.fit(dataset)

  def test_serialization_dense_features(self):
    dense_feature = fc.DenseFeatures([fc.numeric_column('a')])
    config = keras.layers.serialize(dense_feature)
    self.assertEqual(config['class_name'], 'DenseFeatures')

    revived = keras.layers.deserialize(config)
    if tf2.enabled():
      self.assertIsInstance(revived, dense_features_v2.DenseFeatures)
    else:
      self.assertIsInstance(revived, fc.DenseFeatures)
      self.assertNotIsInstance(revived, dense_features_v2.DenseFeatures)

  def test_serialization_sequence_features(self):
    rating = fc.sequence_numeric_column('rating')
    sequence_feature = fc.SequenceFeatures([rating])
    config = keras.layers.serialize(sequence_feature)

    revived = keras.layers.deserialize(config)
    self.assertIsInstance(revived, fc.SequenceFeatures)

  # This test is an example for a regression on categorical inputs, i.e.,
  # the output is 0.4, 0.6, 0.9 when input is 'alpha', 'beta', 'gamma'
  # separately.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_linear_model_with_feature_column(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = feature_column_v2.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = feature_column_v2.indicator_column(cat_column)
    dense_feature_layer = dense_features_v2.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        use_bias=False, kernel_initializer='zeros')
    combined = keras.Sequential([dense_feature_layer, linear_model])
    opt = gradient_descent.SGD(learning_rate=0.1)
    combined.compile(opt, 'mse', [])
    combined.fit(x={'symbol': data}, y=y, batch_size=32, epochs=10)
    self.assertAllClose([[0.4], [0.6], [0.9]],
                        combined.layers[1].dense_layers[0].kernel.numpy(),
                        atol=0.01)

  # This test is an example for cases where linear and dnn model accepts
  # same raw input and same transformed inputs, i.e., the raw input is
  # categorical, and both linear and dnn model accept one hot encoding.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_wide_deep_model_with_single_feature_column(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = feature_column_v2.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = feature_column_v2.indicator_column(cat_column)
    dense_feature_layer = dense_features_v2.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        use_bias=False, kernel_initializer='zeros')
    dnn_model = keras.Sequential([keras.layers.Dense(units=1)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    combined = keras.Sequential([dense_feature_layer, wide_deep_model])
    opt = gradient_descent.SGD(learning_rate=0.1)
    combined.compile(
        opt,
        'mse', [],
        run_eagerly=testing_utils.should_run_eagerly())
    combined.fit(x={'symbol': data}, y=y, batch_size=32, epochs=10)

  # This test is an example for cases where linear and dnn model accepts
  # same raw input but different transformed inputs, i.e,. the raw input is
  # categorical, and linear model accepts one hot encoding, while dnn model
  # accepts embedding encoding.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_wide_deep_model_with_two_feature_columns(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = feature_column_v2.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = feature_column_v2.indicator_column(cat_column)
    emb_column = feature_column_v2.embedding_column(cat_column, dimension=5)
    linear_feature_layer = dense_features_v2.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        use_bias=False, kernel_initializer='zeros')
    combined_linear = keras.Sequential(
        [linear_feature_layer, linear_model])
    dnn_model = keras.Sequential([keras.layers.Dense(units=1)])
    dnn_feature_layer = dense_features_v2.DenseFeatures([emb_column])
    combined_dnn = keras.Sequential([dnn_feature_layer, dnn_model])
    wide_deep_model = wide_deep.WideDeepModel(combined_linear, combined_dnn)
    opt = gradient_descent.SGD(learning_rate=0.1)
    wide_deep_model.compile(
        opt,
        'mse', [],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(x={'symbol': data}, y=y, batch_size=32, epochs=10)


if __name__ == '__main__':
  test.main()
