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
"""Tests for Keras Premade WideNDeep models."""

import numpy as np

from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.feature_column import dense_features_v2
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.premade import linear
from tensorflow.python.keras.premade import wide_deep
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class WideDeepModelTest(keras_parameterized.TestCase):

  def test_wide_deep_model(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    linear_inp = np.random.uniform(low=-5, high=5, size=(64, 2))
    dnn_inp = np.random.uniform(low=-5, high=5, size=(64, 3))
    inputs = [linear_inp, dnn_inp]
    output = .3 * linear_inp[:, 0] + .2 * dnn_inp[:, 1]
    wide_deep_model.compile(
        optimizer=['sgd', 'adam'],
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(inputs, output, epochs=5)
    self.assertTrue(wide_deep_model.built)

  def test_wide_deep_model_backprop(self):
    with self.cached_session():
      linear_model = linear.LinearModel(units=1, kernel_initializer='zeros')
      dnn_model = sequential.Sequential(
          [core.Dense(units=1, kernel_initializer='zeros')])
      wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
      linear_inp = np.array([1.])
      dnn_inp = np.array([1.])
      inputs = [linear_inp, dnn_inp]
      output = linear_inp + 2 * dnn_inp
      linear_opt = gradient_descent.SGD(learning_rate=.1)
      dnn_opt = gradient_descent.SGD(learning_rate=.3)
      wide_deep_model.compile(
          optimizer=[linear_opt, dnn_opt],
          loss='mse',
          metrics=[],
          run_eagerly=testing_utils.should_run_eagerly())
      self.evaluate(variables.global_variables_initializer())
      wide_deep_model.fit(inputs, output, epochs=1)
      self.assertAllClose(
          [[0.6]],
          self.evaluate(wide_deep_model.linear_model.dense_layers[0].kernel))
      self.assertAllClose([[1.8]],
                          self.evaluate(
                              wide_deep_model.dnn_model.layers[0].kernel))

  def test_wide_deep_model_with_single_input(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    inputs = np.random.uniform(low=-5, high=5, size=(64, 3))
    output = .3 * inputs[:, 0]
    wide_deep_model.compile(
        optimizer=['sgd', 'adam'],
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(inputs, output, epochs=5)

  def test_wide_deep_model_with_multi_outputs(self):
    inp = input_layer.Input(shape=(1,), name='linear')
    l = linear.LinearModel(units=2, use_bias=False)(inp)
    l1, l2 = array_ops.split(l, num_or_size_splits=2, axis=1)
    linear_model = training.Model(inp, [l1, l2])
    linear_model.set_weights([np.asarray([[0.5, 0.3]])])
    h = core.Dense(units=2, use_bias=False)(inp)
    h1, h2 = array_ops.split(h, num_or_size_splits=2, axis=1)
    dnn_model = training.Model(inp, [h1, h2])
    dnn_model.set_weights([np.asarray([[0.1, -0.5]])])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    inp_np = np.asarray([[1.]])
    out1, out2 = wide_deep_model(inp_np)
    # output should be (0.5 + 0.1), and (0.3 - 0.5)
    self.assertAllClose([[0.6]], out1)
    self.assertAllClose([[-0.2]], out2)

    wide_deep_model = wide_deep.WideDeepModel(
        linear_model, dnn_model, activation='relu')
    out1, out2 = wide_deep_model(inp_np)
    # output should be relu((0.5 + 0.1)), and relu((0.3 - 0.5))
    self.assertAllClose([[0.6]], out1)
    self.assertAllClose([[0.]], out2)

  def test_wide_deep_model_with_single_optimizer(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    linear_inp = np.random.uniform(low=-5, high=5, size=(64, 2))
    dnn_inp = np.random.uniform(low=-5, high=5, size=(64, 3))
    inputs = [linear_inp, dnn_inp]
    output = .3 * linear_inp[:, 0] + .2 * dnn_inp[:, 1]
    wide_deep_model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(inputs, output, epochs=5)
    self.assertTrue(wide_deep_model.built)

  def test_wide_deep_model_as_layer(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1)])
    linear_input = input_layer.Input(shape=(3,), name='linear')
    dnn_input = input_layer.Input(shape=(5,), name='dnn')
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    wide_deep_output = wide_deep_model((linear_input, dnn_input))
    input_b = input_layer.Input(shape=(1,), name='b')
    output_b = core.Dense(units=1)(input_b)
    model = training.Model(
        inputs=[linear_input, dnn_input, input_b],
        outputs=[wide_deep_output + output_b])
    linear_input_np = np.random.uniform(low=-5, high=5, size=(64, 3))
    dnn_input_np = np.random.uniform(low=-5, high=5, size=(64, 5))
    input_b_np = np.random.uniform(low=-5, high=5, size=(64,))
    output_np = linear_input_np[:, 0] + .2 * dnn_input_np[:, 1] + input_b_np
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit([linear_input_np, dnn_input_np, input_b_np], output_np, epochs=5)

  def test_wide_deep_model_with_sub_model_trained(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(
        linear.LinearModel(units=1),
        sequential.Sequential([core.Dense(units=1, input_dim=3)]))
    linear_inp = np.random.uniform(low=-5, high=5, size=(64, 2))
    dnn_inp = np.random.uniform(low=-5, high=5, size=(64, 3))
    inputs = [linear_inp, dnn_inp]
    output = .3 * linear_inp[:, 0] + .2 * dnn_inp[:, 1]
    linear_model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    dnn_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    linear_model.fit(linear_inp, output, epochs=50)
    dnn_model.fit(dnn_inp, output, epochs=50)
    wide_deep_model.compile(
        optimizer=['sgd', 'adam'],
        loss='mse',
        metrics=[],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(inputs, output, epochs=50)

  # This test is an example for cases where linear and dnn model accepts
  # same raw input and same transformed inputs, i.e., the raw input is
  # categorical, and both linear and dnn model accept one hot encoding.
  def test_wide_deep_model_with_single_feature_column(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = fc.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = fc.indicator_column(cat_column)
    dense_feature_layer = dense_features_v2.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        use_bias=False, kernel_initializer='zeros')
    dnn_model = sequential.Sequential([core.Dense(units=1)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    combined = sequential.Sequential([dense_feature_layer, wide_deep_model])
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
  def test_wide_deep_model_with_two_feature_columns(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = fc.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = fc.indicator_column(cat_column)
    emb_column = fc.embedding_column(cat_column, dimension=5)
    linear_feature_layer = dense_features_v2.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        use_bias=False, kernel_initializer='zeros')
    combined_linear = sequential.Sequential(
        [linear_feature_layer, linear_model])
    dnn_model = sequential.Sequential([core.Dense(units=1)])
    dnn_feature_layer = dense_features_v2.DenseFeatures([emb_column])
    combined_dnn = sequential.Sequential([dnn_feature_layer, dnn_model])
    wide_deep_model = wide_deep.WideDeepModel(combined_linear, combined_dnn)
    opt = gradient_descent.SGD(learning_rate=0.1)
    wide_deep_model.compile(
        opt,
        'mse', [],
        run_eagerly=testing_utils.should_run_eagerly())
    wide_deep_model.fit(x={'symbol': data}, y=y, batch_size=32, epochs=10)

  def test_config(self):
    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
    config = wide_deep_model.get_config()
    cloned_wide_deep_model = wide_deep.WideDeepModel.from_config(config)
    self.assertEqual(linear_model.units,
                     cloned_wide_deep_model.linear_model.units)
    self.assertEqual(dnn_model.layers[0].units,
                     cloned_wide_deep_model.dnn_model.layers[0].units)

  def test_config_with_custom_objects(self):

    def my_activation(x):
      return x

    linear_model = linear.LinearModel(units=1)
    dnn_model = sequential.Sequential([core.Dense(units=1, input_dim=3)])
    wide_deep_model = wide_deep.WideDeepModel(
        linear_model, dnn_model, activation=my_activation)
    config = wide_deep_model.get_config()
    cloned_wide_deep_model = wide_deep.WideDeepModel.from_config(
        config, custom_objects={'my_activation': my_activation})
    self.assertEqual(cloned_wide_deep_model.activation, my_activation)


if __name__ == '__main__':
  test.main()
