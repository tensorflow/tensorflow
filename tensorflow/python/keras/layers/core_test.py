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
"""Tests for Keras core layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class DropoutLayersTest(keras_parameterized.TestCase):

  def test_dropout(self):
    testing_utils.layer_test(
        keras.layers.Dropout, kwargs={'rate': 0.5}, input_shape=(3, 2))

    testing_utils.layer_test(
        keras.layers.Dropout,
        kwargs={'rate': 0.5,
                'noise_shape': [3, 1]},
        input_shape=(3, 2))

  def test_dropout_supports_masking(self):
    dropout = keras.layers.Dropout(0.5)
    self.assertEqual(True, dropout.supports_masking)

  def test_spatial_dropout_1d(self):
    testing_utils.layer_test(
        keras.layers.SpatialDropout1D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4))

  def test_spatial_dropout_2d(self):
    testing_utils.layer_test(
        keras.layers.SpatialDropout2D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4, 5))

    testing_utils.layer_test(
        keras.layers.SpatialDropout2D,
        kwargs={'rate': 0.5, 'data_format': 'channels_first'},
        input_shape=(2, 3, 4, 5))

  def test_spatial_dropout_3d(self):
    testing_utils.layer_test(
        keras.layers.SpatialDropout3D,
        kwargs={'rate': 0.5},
        input_shape=(2, 3, 4, 4, 5))

    testing_utils.layer_test(
        keras.layers.SpatialDropout3D,
        kwargs={'rate': 0.5, 'data_format': 'channels_first'},
        input_shape=(2, 3, 4, 4, 5))


@keras_parameterized.run_all_keras_modes
class LambdaLayerTest(keras_parameterized.TestCase):

  def test_lambda(self):
    testing_utils.layer_test(
        keras.layers.Lambda,
        kwargs={'function': lambda x: x + 1},
        input_shape=(3, 2))

    testing_utils.layer_test(
        keras.layers.Lambda,
        kwargs={
            'function': lambda x, a, b: x * a + b,
            'arguments': {
                'a': 0.6,
                'b': 0.4
            }
        },
        input_shape=(3, 2))

    # test serialization with function
    def f(x):
      return x + 1

    ld = keras.layers.Lambda(f)
    config = ld.get_config()
    ld = keras.layers.deserialize({
        'class_name': 'Lambda',
        'config': config
    })

    # test with lambda
    ld = keras.layers.Lambda(
        lambda x: keras.backend.concatenate([math_ops.square(x), x]))
    config = ld.get_config()
    ld = keras.layers.Lambda.from_config(config)

  def test_lambda_multiple_inputs(self):
    ld = keras.layers.Lambda(lambda x: x[0], output_shape=lambda x: x[0])
    x1 = np.ones([3, 2], np.float32)
    x2 = np.ones([3, 5], np.float32)
    out = ld([x1, x2])
    self.assertAllEqual(out.shape, [3, 2])

  def test_lambda_output_shape(self):
    l = keras.layers.Lambda(lambda x: x + 1, output_shape=(1, 1))
    l(keras.backend.variable(np.ones((1, 1))))
    self.assertEqual((1, 1), l.get_config()['output_shape'])

  def test_lambda_output_shape_function(self):
    def get_output_shape(input_shape):
      return 1 * input_shape

    l = keras.layers.Lambda(lambda x: x + 1, output_shape=get_output_shape)
    l(keras.backend.variable(np.ones((1, 1))))
    self.assertEqual('lambda', l.get_config()['output_shape_type'])

  def test_lambda_output_shape_autocalculate_multiple_inputs(self):

    def lambda_fn(x):
      return math_ops.matmul(x[0], x[1])

    l = keras.layers.Lambda(lambda_fn)
    output_shape = l.compute_output_shape([(10, 10), (10, 20)])
    self.assertAllEqual((10, 20), output_shape)

  def test_lambda_output_shape_list_multiple_outputs(self):

    def lambda_fn(x):
      return x

    l = keras.layers.Lambda(lambda_fn, output_shape=[(10,), (20,)])
    output_shape = l.compute_output_shape([(10, 10), (10, 20)])
    self.assertAllEqual([(10, 10), (10, 20)], output_shape)

  def test_lambda_output_shape_tuple_with_none(self):

    def lambda_fn(x):
      return x

    l = keras.layers.Lambda(lambda_fn, output_shape=(None, 10))
    output_shape = l.compute_output_shape((5, 10, 20))
    self.assertAllEqual([5, None, 10], output_shape.as_list())

  def test_lambda_output_shape_function_multiple_outputs(self):

    def lambda_fn(x):
      return x

    def output_shape_fn(input_shape):
      return input_shape

    l = keras.layers.Lambda(lambda_fn, output_shape=output_shape_fn)
    output_shape = l.compute_output_shape([(10, 10), (10, 20)])
    self.assertAllEqual([(10, 10), (10, 20)], output_shape)

  def test_lambda_output_shape_nested(self):

    def lambda_fn(inputs):
      return (inputs[1]['a'], {'b': inputs[0]})

    l = keras.layers.Lambda(lambda_fn)
    output_shape = l.compute_output_shape(((10, 20), {'a': (10, 5)}))
    self.assertAllEqual(((10, 5), {'b': (10, 20)}), output_shape)

  def test_lambda_config_serialization(self):
    # Test serialization with output_shape and output_shape_type
    layer = keras.layers.Lambda(lambda x: x + 1, output_shape=(1, 1))
    layer(keras.backend.variable(np.ones((1, 1))))
    config = layer.get_config()
    layer = keras.layers.deserialize({
        'class_name': 'Lambda',
        'config': config
    })
    layer = keras.layers.Lambda.from_config(config)

  def test_lambda_with_variable(self):

    def fn(x):
      return x * variables.Variable(2., name='multiplier')

    layer = keras.layers.Lambda(fn)
    for _ in range(10):
      layer(np.ones((10, 10), 'float32'))
    self.assertLen(layer.trainable_weights, 1)
    self.assertEqual(layer.trainable_weights[0].name, 'lambda/multiplier:0')

  def test_lambda_with_training_arg(self):

    def fn(x, training=True):
      return keras.backend.in_train_phase(x, 2 * x, training=training)

    layer = keras.layers.Lambda(fn)
    x = keras.backend.ones(())
    train_out = layer(x, training=True)
    eval_out = layer(x, training=False)

    self.assertEqual(keras.backend.get_value(train_out), 1.)
    self.assertEqual(keras.backend.get_value(eval_out), 2.)


class TestStatefulLambda(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_lambda_with_variable_in_model(self):

    def lambda_fn(x):
      # Variable will only get created once.
      v = variables.Variable(1., trainable=True)
      return x * v

    model = testing_utils.get_model_from_layers(
        [keras.layers.Lambda(lambda_fn)], input_shape=(10,))
    model.compile(
        keras.optimizer_v2.gradient_descent.SGD(0.1),
        'mae',
        run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10), 'float32'), 2 * np.ones((10, 10), 'float32')
    model.fit(x, y, batch_size=2, epochs=2, validation_data=(x, y))
    self.assertLen(model.trainable_weights, 1)
    self.assertAllClose(keras.backend.get_value(model.trainable_weights[0]), 2.)


@keras_parameterized.run_all_keras_modes
class CoreLayersTest(keras_parameterized.TestCase):

  def test_masking(self):
    testing_utils.layer_test(
        keras.layers.Masking, kwargs={}, input_shape=(3, 2, 3))

  def test_keras_mask(self):
    x = np.ones((10, 10))
    y = keras.layers.Masking(1.)(x)
    self.assertTrue(hasattr(y, '_keras_mask'))
    self.assertTrue(y._keras_mask is not None)
    self.assertAllClose(self.evaluate(y._keras_mask), np.zeros((10,)))

  def test_activation(self):
    # with string argument
    testing_utils.layer_test(
        keras.layers.Activation,
        kwargs={'activation': 'relu'},
        input_shape=(3, 2))

    # with function argument
    testing_utils.layer_test(
        keras.layers.Activation,
        kwargs={'activation': keras.backend.relu},
        input_shape=(3, 2))

  def test_reshape(self):
    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (8, 1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (-1, 1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (1, -1)},
        input_shape=(3, 2, 4))

    testing_utils.layer_test(
        keras.layers.Reshape,
        kwargs={'target_shape': (-1, 1)},
        input_shape=(None, None, 2))

  def test_permute(self):
    testing_utils.layer_test(
        keras.layers.Permute, kwargs={'dims': (2, 1)}, input_shape=(3, 2, 4))

  def test_permute_errors_on_invalid_starting_dims_index(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid permutation .*dims.*'):
      testing_utils.layer_test(
          keras.layers.Permute,
          kwargs={'dims': (0, 1, 2)}, input_shape=(3, 2, 4))

  def test_permute_errors_on_invalid_set_of_dims_indices(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid permutation .*dims.*'):
      testing_utils.layer_test(
          keras.layers.Permute,
          kwargs={'dims': (1, 4, 2)}, input_shape=(3, 2, 4))

  def test_flatten(self):
    testing_utils.layer_test(
        keras.layers.Flatten, kwargs={}, input_shape=(3, 2, 4))

    # Test channels_first
    inputs = np.random.random((10, 3, 5, 5)).astype('float32')
    outputs = testing_utils.layer_test(
        keras.layers.Flatten,
        kwargs={'data_format': 'channels_first'},
        input_data=inputs)
    target_outputs = np.reshape(
        np.transpose(inputs, (0, 2, 3, 1)), (-1, 5 * 5 * 3))
    self.assertAllClose(outputs, target_outputs)

  def test_flatten_scalar_channels(self):
    testing_utils.layer_test(
        keras.layers.Flatten, kwargs={}, input_shape=(3,))

    # Test channels_first
    inputs = np.random.random((10,)).astype('float32')
    outputs = testing_utils.layer_test(
        keras.layers.Flatten,
        kwargs={'data_format': 'channels_first'},
        input_data=inputs)
    target_outputs = np.expand_dims(inputs, -1)
    self.assertAllClose(outputs, target_outputs)

  def test_repeat_vector(self):
    testing_utils.layer_test(
        keras.layers.RepeatVector, kwargs={'n': 3}, input_shape=(3, 2))

  def test_dense(self):
    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 4, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(None, None, 2))

    testing_utils.layer_test(
        keras.layers.Dense, kwargs={'units': 3}, input_shape=(3, 4, 5, 2))

  def test_dense_dtype(self):
    inputs = ops.convert_to_tensor(
        np.random.randint(low=0, high=7, size=(2, 2)))
    layer = keras.layers.Dense(5, dtype='float32')
    outputs = layer(inputs)
    self.assertEqual(outputs.dtype, 'float32')

  def test_dense_with_policy(self):
    inputs = ops.convert_to_tensor(
        np.random.randint(low=0, high=7, size=(2, 2)), dtype='float16')
    layer = keras.layers.Dense(5, dtype=policy.Policy('infer_float32_vars'))
    outputs = layer(inputs)
    self.assertEqual(outputs.dtype, 'float16')
    self.assertEqual(layer.kernel.dtype, 'float32')

  def test_dense_regularization(self):
    layer = keras.layers.Dense(
        3,
        kernel_regularizer=keras.regularizers.l1(0.01),
        bias_regularizer='l1',
        activity_regularizer='l2',
        name='dense_reg')
    layer(keras.backend.variable(np.ones((2, 4))))
    self.assertEqual(3, len(layer.losses))

  def test_dense_constraints(self):
    k_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    layer = keras.layers.Dense(
        3, kernel_constraint=k_constraint, bias_constraint=b_constraint)
    layer(keras.backend.variable(np.ones((2, 4))))
    self.assertEqual(layer.kernel.constraint, k_constraint)
    self.assertEqual(layer.bias.constraint, b_constraint)

  def test_activity_regularization(self):
    layer = keras.layers.ActivityRegularization(l1=0.1)
    layer(keras.backend.variable(np.ones((2, 4))))
    self.assertEqual(1, len(layer.losses))
    config = layer.get_config()
    self.assertEqual(config.pop('l1'), 0.1)

  def test_numpy_inputs(self):
    if context.executing_eagerly():
      layer = keras.layers.RepeatVector(2)
      x = np.ones((10, 10))
      self.assertAllEqual(np.ones((10, 2, 10)), layer(x))

      layer = keras.layers.Concatenate()
      x, y = np.ones((10, 10)), np.ones((10, 10))
      self.assertAllEqual(np.ones((10, 20)), layer([x, y]))


if __name__ == '__main__':
  test.main()
