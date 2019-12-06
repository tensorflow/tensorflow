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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import sys

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.rmsprop import RMSPropOptimizer

try:
  import scipy.sparse as scipy_sparse  # pylint: disable=g-import-not-at-top
except ImportError:
  scipy_sparse = None


class CompileTest(keras_parameterized.TestCase):

  def _get_multi_output_model(self):
    input_a = keras.layers.Input(shape=(3,), name='input_a')
    output_a = keras.layers.Dense(1, name='dense_1')(input_a)
    output_b = keras.layers.Dense(1, name='dense_2')(input_a)
    return keras.models.Model(input_a, [output_a, output_b])

  def _do_test_compile_with_model_and_single_loss(self, model, loss):
    model.compile(
        optimizer='adam',
        loss=loss,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    self.assertEqual(model.loss, loss)

    loss = losses.get(loss)
    if not isinstance(loss, list):
      loss_list = [loss] * len(model.outputs)

    self.assertEqual(len(model.loss_functions), len(loss_list))
    for i in range(len(loss_list)):
      self.assertIsInstance(model.loss_functions[i], losses.LossFunctionWrapper)
      if not isinstance(loss_list[i], losses.LossFunctionWrapper):
        self.assertEqual(model.loss_functions[i].fn, loss_list[i])
    self.assertAllEqual(model._loss_weights_list, [1.] * len(loss_list))

  def test_respect_run_functions_eagerly(self):
    with context.eager_mode():
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=10, num_classes=2, input_dim=3)
      model.compile('sgd', 'mse')
      def_function.run_functions_eagerly(True)
      self.assertTrue(model.run_eagerly)
      def_function.run_functions_eagerly(False)
      self.assertFalse(model.run_eagerly)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(('loss_string', 'mse'),
                                  ('loss_function', losses.mean_squared_error),
                                  ('loss_instance', losses.MeanSquaredError()))
  def test_compile_with_single_output(self, loss):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    self._do_test_compile_with_model_and_single_loss(model, loss)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(('loss_string', 'mse'),
                                  ('loss_function', losses.mean_squared_error),
                                  ('loss_instance', losses.MeanSquaredError()))
  def test_compile_with_multi_output(self, loss):
    model = self._get_multi_output_model()
    self._do_test_compile_with_model_and_single_loss(model, loss)

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_multi_output_and_multi_loss(self):
    model = self._get_multi_output_model()
    # Test loss is a list.
    loss = ['mse', 'mae']
    model.compile(
        optimizer='adam',
        loss=loss,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    self.assertEqual(model.loss_functions[0].fn, losses.mean_squared_error)
    self.assertEqual(model.loss_functions[1].fn, losses.mean_absolute_error)
    self.assertAllEqual(model._loss_weights_list, [1., 1.])

    # Test loss is a dict.
    loss = {'dense_1': 'mae', 'dense_2': 'mse'}
    model.compile(
        optimizer='adam',
        loss=loss,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    self.assertEqual(model.loss_functions[0].fn, losses.mean_absolute_error)
    self.assertEqual(model.loss_functions[1].fn, losses.mean_squared_error)
    self.assertAllEqual(model._loss_weights_list, [1., 1.])

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_multi_output_and_loss_weights_list(self):
    model = self._get_multi_output_model()
    loss_weights = [1., 2.]
    model.compile(
        optimizer='adam',
        loss='mse',
        loss_weights=loss_weights,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    self.assertAllEqual(model._loss_weights_list, [1., 2.])

  def test_compile_with_multi_output_and_loss_weights_dict(self):
    with context.graph_mode():
      model = self._get_multi_output_model()
      loss_weights = {'dense_1': 1., 'dense_2': 2.}
      model.compile(optimizer='adam', loss='mse', loss_weights=loss_weights)
      self.assertAllEqual(model._loss_weights_list, [1., 2.])

      input_np = np.random.random((10, 3))
      output_a_np = np.random.random((10, 1))
      output_b_np = np.random.random((10, 1))

      with self.cached_session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        total_loss, y_preds = sess.run(
            [model.total_loss, model.outputs],
            feed_dict={
                'input_a:0': input_np,
                'dense_1_target:0': output_a_np,
                'dense_2_target:0': output_b_np
            })
        self.assertAllClose(
            total_loss,
            np.mean(
                np.add((output_a_np - y_preds[0])**2,
                       2 * (output_b_np - y_preds[1])**2)))

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_incorrect_loss_size(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    with self.assertRaisesRegexp(ValueError, 'The model has 1 outputs'):
      model.compile(
          optimizer='adam',
          loss=['mse', 'mae'],
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_incorrect_loss_key(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    with self.assertRaisesRegexp(
        ValueError,
        r'Unknown entries in loss dictionary: \[\'unknown_output\'\]. '
        r'Only expected following keys: \[\'dense_1\'\]'):
      model.compile(
          optimizer='adam',
          loss={'unknown_output': 'mse'},
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_incorrect_loss_weights_size(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    with self.assertRaisesRegexp(ValueError,
                                 'it should have one entry per model output'):
      model.compile(
          optimizer='adam',
          loss='mse',
          loss_weights=[1., 2.],
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_incorrect_loss_weights_key(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    with self.assertRaisesRegexp(
        ValueError,
        r'Unknown entries in loss_weights dictionary: \[\'unknown_output\'\]. '
        r'Only expected following keys: \[\'dense_1\'\]'):
      model.compile(
          optimizer='adam',
          loss='mse',
          loss_weights={'unknown_output': 1.},
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_incorrect_sample_weight_mode(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    with self.assertRaisesRegexp(
        ValueError,
        r'Unknown entries in sample_weight_mode dictionary: \[\'unknown\'\]. '
        r'Only expected following keys: \[\'dense_1\'\]'):
      model.compile(
          optimizer='adam',
          loss='mse',
          sample_weight_mode={'unknown': 'temporal'},
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @tf_test_util.run_deprecated_v1
  def test_compile_with_session_kwargs(self):
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)

    # Test that unknown arguments are not accepted
    with self.assertRaisesRegexp(
        TypeError,
        r'Invalid keyword argument'):
      model.compile(
          optimizer='adam',
          loss='mse',
          foo=True)


class TrainingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_fit_training_arg(self):

    class ReturnTraining(keras.layers.Layer):

      def call(self, inputs, training):
        if training:
          return inputs + array_ops.constant([100], 'float32')
        else:
          return inputs + array_ops.constant([0], 'float32')

    model = keras.Sequential([ReturnTraining()])
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    hist = model.fit(x=np.array([0.]), y=np.array([0.]))
    self.assertAllClose(hist.history['loss'][0], 10000)

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_learning_phase(self):

    class ReturnTraining(keras.layers.Layer):

      def call(self, inputs):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs))

    model = keras.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(
        train_dataset, epochs=2, verbose=1, validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_training_arg(self):

    class ReturnTraining(keras.layers.Layer):

      def call(self, inputs, training=None):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs),
            training=training)

    model = keras.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(
        train_dataset, epochs=2, verbose=1, validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_target_dtype_matches_output(self):

    def _loss_fn(labels, preds):
      self.assertEqual(labels.dtype, preds.dtype)
      return labels - preds

    layers = [keras.layers.Dense(10, dtype=np.float64),
              keras.layers.Dense(10, dtype=np.float64)]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    inputs = np.ones(10, dtype=np.float64)
    targets = np.ones(10, dtype=np.float64)
    model.compile(
        'sgd',
        loss=_loss_fn,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    self.assertEqual(model.predict(inputs).dtype, np.float64)

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_nested_training_arg(self):

    class NestedReturnTraining(keras.layers.Layer):

      def call(self, inputs, training=None):
        return keras.backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs),
            training=training)

    class ReturnTraining(keras.layers.Layer):

      def __init__(self, input_shape=None, **kwargs):
        super(ReturnTraining, self).__init__(input_shape=input_shape, **kwargs)
        self._nested_layer = None

      def build(self, input_shape):
        self._nested_layer = NestedReturnTraining()
        self.built = True

      def call(self, inputs):
        return self._nested_layer(inputs)

    model = keras.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    inputs = np.ones((40, 2), dtype=np.float32)
    targets = np.ones((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(
        train_dataset, epochs=2, verbose=1, validation_data=val_dataset)

    # The training loss should be 0.0
    self.assertAllClose(history.history['loss'][0], 0.0)
    # The validation loss should be 1.0.
    self.assertAllClose(history.history['val_loss'][0], 1.0)

  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  @keras_parameterized.run_all_keras_modes
  def test_fit_on_arrays(self):
    input_a = keras.layers.Input(shape=(3,), name='input_a')
    input_b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    dropout = keras.layers.Dropout(0.5, name='dropout')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    # Test fit at different verbosity
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=1)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=2,
        batch_size=5,
        verbose=2)
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

    # Test with validation data
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        validation_data=([input_a_np, input_b_np], [output_d_np,
                                                    output_e_np]),
        epochs=1,
        batch_size=5,
        verbose=0)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        validation_data=([input_a_np, input_b_np], [output_d_np,
                                                    output_e_np]),
        epochs=2,
        batch_size=5,
        verbose=1)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        validation_data=([input_a_np, input_b_np], [output_d_np,
                                                    output_e_np]),
        epochs=2,
        batch_size=5,
        verbose=2)
    # Test with validation split
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=2,
        batch_size=5,
        verbose=0,
        validation_split=0.2)

    if testing_utils.get_model_type() == 'functional':
      # Test with dictionary inputs
      model.fit(
          {
              'input_a': input_a_np,
              'input_b': input_b_np
          }, {
              'dense': output_d_np,
              'dropout': output_e_np
          },
          epochs=1,
          batch_size=5,
          verbose=0)
      model.fit(
          {
              'input_a': input_a_np,
              'input_b': input_b_np
          }, {
              'dense': output_d_np,
              'dropout': output_e_np
          },
          epochs=1,
          batch_size=5,
          verbose=1)
      model.fit(
          {
              'input_a': input_a_np,
              'input_b': input_b_np
          }, {
              'dense': output_d_np,
              'dropout': output_e_np
          },
          validation_data=({
              'input_a': input_a_np,
              'input_b': input_b_np
          }, {
              'dense': output_d_np,
              'dropout': output_e_np
          }),
          epochs=1,
          batch_size=5,
          verbose=0)
      model.train_on_batch({
          'input_a': input_a_np,
          'input_b': input_b_np
      }, {
          'dense': output_d_np,
          'dropout': output_e_np
      })

    # Test with lists for loss, metrics
    loss = ['mae', 'mse']
    model.compile(
        optimizer,
        loss,
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Test with dictionaries for loss, metrics, loss weights
    if testing_utils.get_model_type() == 'functional':
      loss = {'dense': 'mse', 'dropout': 'mae'}
      loss_weights = {'dense': 1., 'dropout': 0.5}
      metrics = {
          'dense': 'mse',
          'dropout': metrics_module.CategoricalAccuracy()
      }
      model.compile(
          optimizer,
          loss,
          metrics=metrics,
          loss_weights=loss_weights,
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Invalid use cases
    with self.assertRaises(ValueError):
      model.train_on_batch({'input_a': input_a_np},
                           [output_d_np, output_e_np])
    with self.assertRaises(ValueError):
      model.fit(
          [input_a_np, input_b_np], [output_d_np, output_e_np],
          epochs=1,
          validation_data=([input_a_np, input_b_np], 0, 0),
          verbose=0)
    with self.assertRaises(ValueError):
      model.train_on_batch([input_a_np], [output_d_np, output_e_np])
    with self.assertRaises(ValueError):
      model.train_on_batch(1, [output_d_np, output_e_np])
    with self.assertRaises(ValueError):
      model.train_on_batch(input_a_np, [output_d_np, output_e_np])
    with self.assertRaises(ValueError):
      bad_input = np.random.random((11, 3))
      model.train_on_batch([bad_input, input_b_np],
                           [output_d_np, output_e_np])
    with self.assertRaises(ValueError):
      bad_target = np.random.random((11, 4))
      model.train_on_batch([input_a_np, input_b_np],
                           [bad_target, output_e_np])

    # Build single-input model
    x = keras.layers.Input(shape=(3,), name='input_a')
    y = keras.layers.Dense(4)(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer,
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    # This will work
    model.fit([input_a_np], output_d_np, epochs=1)
    # TODO(gsundeep) Test only works in eager, file ticket
    if testing_utils.should_run_eagerly() and context.executing_eagerly():
      with self.assertRaises(ValueError):
        model.fit([input_a_np, input_a_np], output_d_np, epochs=1)

    # Test model on a list of floats
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 4))

    # Test execution on inputs that are lists of scalars.
    # TF2 and TF1 have slightly different semantics:
    if (testing_utils.should_run_tf_function()
        or testing_utils.should_run_eagerly()):
      # In TF2 to avoid any ambiguity when there are nested lists
      # the entire input gets converted to a
      # single numpy array (& it only works in the case of a single io model)
      model.fit(np.ndarray.tolist(input_a_np),
                np.ndarray.tolist(input_b_np),
                epochs=2,
                batch_size=5,
                verbose=2)
    else:
      # In TF1 there was logic to try disambiguating between the individual
      # inputs when lists are nested. This allowed multi-io functional models
      # to support lists of scalars as input, but it caused ambiguity issues
      # for subclass models & made it trickier to pass multi-dimensional inputs
      # as lists of scalars to single io models. This was an excessive amount
      # of complexity for what boiled down to a convenience method we were
      # mainly just using for writing tests.
      model.fit([np.ndarray.tolist(input_a_np)],
                [np.ndarray.tolist(input_b_np)],
                epochs=2,
                batch_size=5,
                verbose=2)

  @keras_parameterized.run_all_keras_modes
  def test_evaluate_predict_on_arrays(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        loss_weights=loss_weights,
        sample_weight_mode=None,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    # Test evaluate at different verbosity
    out = model.evaluate(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        batch_size=5,
        verbose=0)
    self.assertEqual(len(out), 7)
    out = model.evaluate(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        batch_size=5,
        verbose=1)
    self.assertEqual(len(out), 7)
    out = model.evaluate(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        batch_size=5,
        verbose=2)
    self.assertEqual(len(out), 7)
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_d_np, output_e_np])
    self.assertEqual(len(out), 7)

    # Test evaluate with dictionary inputs
    model.evaluate(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {
            'dense': output_d_np,
            'dropout': output_e_np
        },
        batch_size=5,
        verbose=0)
    model.evaluate(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {
            'dense': output_d_np,
            'dropout': output_e_np
        },
        batch_size=5,
        verbose=1)

    # Test predict
    out = model.predict([input_a_np, input_b_np], batch_size=5)
    self.assertEqual(len(out), 2)
    out = model.predict({'input_a': input_a_np, 'input_b': input_b_np})
    self.assertEqual(len(out), 2)
    out = model.predict_on_batch({
        'input_a': input_a_np,
        'input_b': input_b_np
    })
    self.assertEqual(len(out), 2)

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_activity_regularizer_fit(self):
    loss = {}
    for reg in [None, 'l2']:
      layers = [
          keras.layers.Dense(
              10, activation='relu', activity_regularizer=reg,
              kernel_initializer='ones', use_bias=False),
          keras.layers.Dense(
              1, activation='sigmoid', kernel_initializer='ones',
              use_bias=False),
      ]

      model = testing_utils.get_model_from_layers(
          layers, input_shape=(10,))

      x = np.ones((10, 10), 'float32')
      y = np.ones((10, 1), 'float32')

      optimizer = RMSPropOptimizer(learning_rate=0.001)
      model.compile(
          optimizer,
          'binary_crossentropy',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, batch_size=2, epochs=5)
      loss[reg] = model.evaluate(x, y)
    self.assertLess(loss[None], loss['l2'])

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_activity_regularizer_loss_value(self):
    layer = keras.layers.Dense(
        1, kernel_initializer=keras.initializers.zeros(),
        bias_initializer=keras.initializers.ones(), activity_regularizer='l2')

    model = testing_utils.get_model_from_layers([layer], input_shape=(10,))

    x = np.ones((10, 10), 'float32')
    y = np.ones((10, 1), 'float32')
    optimizer = RMSPropOptimizer(learning_rate=0.001)
    model.compile(
        optimizer,
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    loss = model.test_on_batch(x, y)
    self.assertAlmostEqual(0.01, loss, places=4)

  @keras_parameterized.run_all_keras_modes
  def test_activity_regularizer_batch_independent(self):
    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(
        10, activation='relu', activity_regularizer='l2')(
            inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    model.compile(
        optimizer,
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((10, 10), 'float32')
    y = np.ones((10, 1), 'float32')
    loss_small_batch = model.test_on_batch(x, y)

    x2 = np.ones((20, 10), 'float32')
    y2 = np.ones((20, 1), 'float32')
    loss_big_batch = model.test_on_batch(x2, y2)

    self.assertAlmostEqual(loss_small_batch, loss_big_batch, places=4)

  @keras_parameterized.run_all_keras_modes
  def test_activity_regularizer_in_model_call(self):

    class MyModel(keras.Model):

      def call(self, inputs):
        self.add_loss(inputs)
        return inputs

    x = ops.convert_to_tensor(1.)
    model = MyModel()
    _ = model(x)
    self.assertEqual(1, len(model.losses))

  @keras_parameterized.run_all_keras_modes
  def test_custom_mapping_in_config(self):

    class MyModel(keras.Model):

      def call(self, inputs):
        return inputs

      def get_config(self):
        self.a = {}
        return {'a': self.a}

    model = MyModel()
    self.assertIn('{"a": {}}', model.to_json())

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training_on_sparse_data_with_dense_placeholders(self):
    if scipy_sparse is None:
      return

    test_inputs = [
        scipy_sparse.random(6, 3, density=0.25).tocsr() for _ in range(2)
    ]
    test_outputs = [
        scipy_sparse.random(6, i, density=0.25).tocsr() for i in range(3, 5)
    ]
    in1 = keras.layers.Input(shape=(3,))
    in2 = keras.layers.Input(shape=(3,))
    out1 = keras.layers.Dropout(0.5, name='dropout')(in1)
    out2 = keras.layers.Dense(4, name='dense_1')(in2)
    model = keras.Model([in1, in2], [out1, out2])
    model.experimental_run_tf_function = testing_utils.should_run_tf_function()

    with self.assertRaisesRegexp(ValueError, 'Please densify'):
      model.predict(test_inputs, batch_size=2)
    optimizer = 'rmsprop'
    model.compile(
        optimizer,
        'mse',
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    with self.assertRaisesRegexp(ValueError, 'Please densify'):
      model.fit(test_inputs, test_outputs,
                epochs=1, batch_size=2)

    with self.assertRaisesRegexp(ValueError, 'Please densify'):
      model.evaluate(test_inputs, test_outputs, batch_size=2)

  @tf_test_util.run_deprecated_v1
  def test_training_on_sparse_data_with_dense_placeholders_v1(self):
    if scipy_sparse is None:
      return

    test_inputs = [
        scipy_sparse.random(6, 3, density=0.25).tocsr() for _ in range(2)
    ]
    test_outputs = [
        scipy_sparse.random(6, i, density=0.25).tocsr() for i in range(3, 5)
    ]
    in1 = keras.layers.Input(shape=(3,))
    in2 = keras.layers.Input(shape=(3,))
    out1 = keras.layers.Dropout(0.5, name='dropout')(in1)
    out2 = keras.layers.Dense(4, name='dense_1')(in2)
    model = keras.Model([in1, in2], [out1, out2])
    model.predict(test_inputs, batch_size=2)
    optimizer = 'rmsprop'
    model.compile(
        optimizer,
        'mse',
        metrics=['mae', metrics_module.CategoricalAccuracy()])
    model.fit(test_inputs, test_outputs,
              epochs=1, batch_size=2, validation_split=0.5)
    model.evaluate(test_inputs, test_outputs, batch_size=2)

  @keras_parameterized.run_all_keras_modes
  def test_compile_with_sparse_placeholders(self):
    input_layer = keras.layers.Input(shape=(10,), sparse=True)
    weights = variables_lib.Variable(
        np.ones((10, 1)).astype(np.float32), name='weights')
    weights_mult = lambda x: sparse_ops.sparse_tensor_dense_matmul(x, weights)
    output_layer = keras.layers.Lambda(weights_mult)(input_layer)
    model = keras.Model([input_layer], output_layer)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_that_trainable_disables_updates(self):
    val_a = np.random.random((10, 4))
    val_out = np.random.random((10, 4))

    a = keras.layers.Input(shape=(4,))
    layer = keras.layers.BatchNormalization(input_shape=(4,))
    b = layer(a)
    model = keras.Model(a, b)

    model.trainable = False
    assert not model.updates

    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    assert not model.updates

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    self.assertAllClose(x1, x2, atol=1e-7)

    model.trainable = True
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    assert model.updates

    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert np.abs(np.sum(x1 - x2)) > 1e-5

    layer.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    assert not model.updates

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    self.assertAllClose(x1, x2, atol=1e-7)

  def test_weight_deduplication_in_methods(self):
    inp = keras.layers.Input(shape=(1,))
    bn = keras.layers.BatchNormalization()
    d = keras.layers.Dense(1)

    m0 = keras.models.Model(inp, d(bn(inp)))
    m1 = keras.models.Model(inp, d(bn(inp)))

    x0 = m0(inp)
    x1 = m1(inp)
    x = keras.layers.Add()([x0, x1])

    model = keras.models.Model(inp, x)
    self.assertLen(model.trainable_weights, 4)
    self.assertLen(model.non_trainable_weights, 2)
    self.assertLen(model.weights, 6)

  @keras_parameterized.run_all_keras_modes
  def test_weight_deduplication(self):
    class WatchingLayer(keras.layers.Layer):

      def __init__(self, dense_to_track):
        # This will cause the kernel and bias to be double counted, effectively
        # doubling the learning rate if weights are not deduped.
        self._kernel = dense_to_track.kernel
        self._bias = dense_to_track.bias
        super(WatchingLayer, self).__init__()

    inp = keras.layers.Input(shape=(1,))
    dense_layer = keras.layers.Dense(1)
    dense_output = dense_layer(inp)  # This will build the dense kernel

    # Deterministically set weights to make the test repeatable.
    dense_layer.set_weights([np.ones((1, 1)), np.zeros((1,))])
    output = WatchingLayer(dense_layer)(dense_output)

    model = keras.models.Model(inp, output)

    # 0.25 is the edge of the radius of convergence for the double apply case.
    # At lr=0.24, the double apply case will very slowly descend while the
    # correct case will drop very quickly.
    model.compile(loss='mse', optimizer=gradient_descent.SGD(0.24),
                  run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones((64 * 2,))
    y = 4.5 * x - 3.

    history = model.fit(x, y, batch_size=64, epochs=2, verbose=2)

    # If the gradient apply is duplicated then the loss after 2 epochs will
    # be ~0.15, compared to the correct answer of O(1e-7).
    self.assertLess(history.history['loss'][-1], 1e-6)

  def test_logs_passed_to_callbacks(self):
    with self.cached_session():
      input_dim = 5
      num_classes = 1

      class TestCallback(Callback):

        def __init__(self):
          super(TestCallback, self).__init__()
          self.epoch_end_logs = None
          self.batch_end_logs = None
          self.epoch_end_call_count = 0
          self.batch_end_call_count = 0

        def on_epoch_end(self, epoch, logs=None):
          self.epoch_end_logs = logs
          self.epoch_end_call_count += 1

        def on_batch_end(self, batch, logs=None):
          self.batch_end_logs = logs
          self.batch_end_call_count += 1

      model = testing_utils.get_small_sequential_mlp(
          num_hidden=10, num_classes=num_classes, input_dim=input_dim)
      model.compile(
          loss='binary_crossentropy',
          metrics=['acc'],
          weighted_metrics=['mae'],
          optimizer=RMSPropOptimizer(learning_rate=0.01))

      np.random.seed(1337)
      (x_train, y_train), (_, _) = testing_utils.get_test_data(
          train_samples=10,
          test_samples=10,
          input_shape=(input_dim,),
          num_classes=num_classes)

      test_callback = TestCallback()
      model.fit(
          x_train,
          y_train,
          batch_size=2,
          epochs=2,
          verbose=0,
          callbacks=[test_callback],
          validation_data=(x_train, y_train))
      self.assertEqual(test_callback.batch_end_call_count, 10)
      self.assertEqual(test_callback.epoch_end_call_count, 2)

      weighted_metric = ('mae'
                         if tf2.enabled() else 'weighted_mean_absolute_error')
      self.assertSetEqual(
          set(test_callback.batch_end_logs.keys()),
          set(['batch', 'size', 'acc', 'loss', weighted_metric]))
      self.assertSetEqual(
          set(test_callback.epoch_end_logs.keys()),
          set([
              'acc', 'loss', weighted_metric, 'val_acc', 'val_loss',
              'val_' + weighted_metric
          ]))

  @keras_parameterized.run_all_keras_modes
  def test_mismatched_output_shape_and_target_shape(self):
    model = keras.Sequential([
        keras.layers.Dense(2, input_shape=(3, 4)),
        keras.layers.Dense(5),
    ])
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    # Test with Numpy data
    x_train = np.random.random((10, 3, 4))
    y_train = np.random.randint(0, 5, size=(10, 3))
    model.fit(x_train, y_train, batch_size=5, epochs=1)

    # Test with iterator
    dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(10)
    model.fit(dataset, epochs=1, steps_per_epoch=2)

    if context.executing_eagerly():
      # Test with eager execution
      model.compile(RMSPropOptimizer(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    run_eagerly=True)
      model.fit(x_train, y_train, batch_size=5, epochs=1)

      # Test with eager execution and iterator
      model.fit(dataset, epochs=1, steps_per_epoch=2)

  def test_losses_in_defun(self):
    with context.eager_mode():
      layer = keras.layers.Dense(1, kernel_regularizer='l1')
      layer(array_ops.ones([1, 10]))

      @function.defun
      def get_losses():
        return layer.losses

      self.assertAllEqual(
          self.evaluate(layer.losses), self.evaluate(get_losses()))

  @keras_parameterized.run_all_keras_modes
  def test_logging(self):
    mock_stdout = io.BytesIO() if six.PY2 else io.StringIO()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      model.fit(
          np.ones((10, 10), 'float32'), np.ones((10, 1), 'float32'), epochs=10)
    self.assertTrue('Epoch 5/10' in mock_stdout.getvalue())

  @tf_test_util.run_in_graph_and_eager_modes
  def test_training_with_loss_instance(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])
    loss_weights = [1., 0.5]
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_static_batch_in_input_layer(self):

    class Counter(keras.callbacks.Callback):

      def __init__(self):
        self.batches = 0

      def on_batch_end(self, batch, logs=None):
        self.batches += 1

    x, y = np.ones((64, 10), 'float32'), np.ones((64, 1), 'float32')

    for batch_size, expected_batches in [(None, 2), (4, 16)]:
      inputs = keras.Input(batch_size=batch_size, shape=(10,))
      outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
      model = keras.Model(inputs, outputs)

      model.compile(keras.optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
      counter = Counter()
      model.fit(x, y, callbacks=[counter])
      self.assertEqual(counter.batches, expected_batches)

      model = keras.Sequential(
          [keras.layers.Dense(1, batch_input_shape=(batch_size, 10))])
      model.compile(keras.optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
      counter = Counter()
      model.fit(x, y, callbacks=[counter])
      self.assertEqual(counter.batches, expected_batches)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_static_batch_in_input_layer_consistency_checks(self):
    x, y = np.ones((64, 10), 'float32'), np.ones((64, 1), 'float32')

    inputs = keras.Input(batch_size=2, shape=(10,))
    outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(keras.optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
    with self.assertRaisesRegexp(ValueError,
                                 'incompatible with the specified batch size'):
      model.fit(x, y, batch_size=4)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_compatible_batch_size_functional_model(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs):
        return array_ops.concat(inputs, axis=0)

    input1 = keras.Input(batch_size=2, shape=(10,))
    input2 = keras.Input(batch_size=3, shape=(10,))
    outputs = MyLayer()([input1, input2])
    with self.assertRaisesRegexp(ValueError,
                                 'specified batch sizes of the Input Layers'):
      keras.Model([input1, input2], outputs)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_calling_subclass_model_on_different_datasets(self):

    class SubclassedModel(keras.models.Model):

      def call(self, inputs):
        return inputs * 2

    model = SubclassedModel()
    dataset_one = dataset_ops.Dataset.range(2).batch(2)
    dataset_two = dataset_ops.Dataset.range(3, 10).batch(2)
    self.assertAllEqual([[0], [2]], model.predict(dataset_one, steps=1))
    self.assertAllEqual([[6], [8], [10], [12]],
                        model.predict(dataset_two, steps=2))

  def test_training_on_sparse_categorical_crossentropy_loss_with_softmax(self):
    with context.eager_mode():
      np.random.seed(1337)
      train_x = np.ones((100, 4))
      train_y = np.random.randint(0, 1, size=(100, 1))

      reference_model = testing_utils.get_small_sequential_mlp(16, 2,
                                                               input_dim=4)
      reference_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=RMSPropOptimizer(learning_rate=0.001),
                              run_eagerly=True)
      fixed_weights = reference_model.get_weights()
      reference_model_loss = reference_model.train_on_batch(train_x, train_y)

      test_model = testing_utils.get_small_sequential_mlp(16, 2, input_dim=4)
      test_model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=RMSPropOptimizer(learning_rate=0.001),
                         run_eagerly=False)
      test_model.set_weights(fixed_weights)
      test_model_loss = test_model.train_on_batch(train_x, train_y)
      self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

  def test_training_on_categorical_crossentropy_loss_with_softmax(self):
    with context.eager_mode():
      np.random.seed(1337)
      train_x = np.ones((100, 4))
      train_y = np_utils.to_categorical(
          np.random.randint(0, 1, size=(100, 1)), 2)

      reference_model = testing_utils.get_small_sequential_mlp(16, 2,
                                                               input_dim=4)
      reference_model.compile(loss='categorical_crossentropy',
                              optimizer=RMSPropOptimizer(learning_rate=0.001),
                              run_eagerly=True)
      fixed_weights = reference_model.get_weights()
      reference_model_loss = reference_model.train_on_batch(train_x, train_y)

      test_model = testing_utils.get_small_sequential_mlp(16, 2, input_dim=4)
      test_model.compile(loss='categorical_crossentropy',
                         optimizer=RMSPropOptimizer(learning_rate=0.001),
                         run_eagerly=False)
      test_model.set_weights(fixed_weights)
      test_model_loss = test_model.train_on_batch(train_x, train_y)
      self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

  def test_training_on_binary_crossentropy_loss(self):
    with context.eager_mode():
      train_x = np.ones((100, 4), dtype=np.float32)
      train_y = np.ones((100, 1), dtype=np.float32)
      reference_model = testing_utils.get_small_sequential_mlp(16, 1,
                                                               input_dim=4)
      reference_model.compile(loss='binary_crossentropy',
                              optimizer=RMSPropOptimizer(learning_rate=0.001),
                              run_eagerly=True)
      fixed_weights = reference_model.get_weights()
      reference_model_loss = reference_model.train_on_batch(train_x, train_y)

      test_model = testing_utils.get_small_sequential_mlp(16, 1, input_dim=4)
      test_model.compile(loss='binary_crossentropy',
                         optimizer=RMSPropOptimizer(learning_rate=0.001),
                         run_eagerly=False)
      test_model.set_weights(fixed_weights)
      test_model_loss = test_model.train_on_batch(train_x, train_y)
      self.assertAlmostEqual(test_model_loss, reference_model_loss, places=4)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      ('default', 1, 4), ('integer_two', 2, 2), ('integer_four', 4, 1),
      ('simple_list', [1, 3, 4], 3), ('duplicated_list', [4, 2, 2], 2))
  def test_validation_freq(self, validation_freq, expected_runs):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_small_mlp(2, 1, 10)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    class ValCounter(keras.callbacks.Callback):

      def __init__(self):
        self.val_runs = 0

      def on_test_begin(self, logs=None):
        self.val_runs += 1

    val_counter = ValCounter()
    model.fit(
        x,
        y,
        epochs=4,
        validation_data=(x, y),
        validation_freq=validation_freq,
        callbacks=[val_counter])
    self.assertEqual(val_counter.val_runs, expected_runs)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_validation_steps_without_data(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_small_mlp(2, 1, 10)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    with self.assertRaisesRegexp(
        ValueError, '`validation_steps` should not be specified if '
        '`validation_data` is None.'):
      model.fit(x, y, epochs=4, validation_data=None, validation_steps=3)

  @keras_parameterized.run_all_keras_modes
  def test_add_loss_correctness(self):
    class Bias(keras.layers.Layer):

      def build(self, input_shape):
        self.bias = self.add_variable('bias', (1,), initializer='zeros')

      def call(self, inputs):
        return inputs + self.bias

    inputs = keras.Input(shape=(1,))
    targets = keras.Input(shape=(1,))
    outputs = Bias()(inputs)
    model = keras.Model([inputs, targets], outputs)

    model.add_loss(2 * math_ops.reduce_mean(
        keras.losses.mean_absolute_error(targets, outputs)))

    model.add_loss(keras.losses.MeanAbsoluteError()(targets, outputs))

    model.compile(
        keras.optimizer_v2.gradient_descent.SGD(0.025),
        loss=keras.losses.MeanAbsoluteError(),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    history = model.fit([x, y], y, batch_size=3, epochs=5)
    self.assertAllClose(history.history['loss'], [4., 3.6, 3.2, 2.8, 2.4], 1e-3)

  @keras_parameterized.run_all_keras_modes
  def test_unconditional_add_loss_correctness(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None):
        # Reachable from the inputs but marked as unconditional.
        self.add_loss(math_ops.reduce_sum(inputs))
        return inputs

    inputs = keras.Input((3,))
    layer = MyLayer()
    outputs = layer(inputs)
    model = keras.Model(inputs, outputs)
    self.assertEqual(len(model.losses), 1)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    loss = model.train_on_batch(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual(loss, 2 * 3)

  @keras_parameterized.run_all_keras_modes
  def test_clear_losses(self):

    class LayerWithSharedNestedLossLayer(keras.layers.Layer):

      def __init__(self):
        super(LayerWithSharedNestedLossLayer, self).__init__()
        self.loss_layer = keras.layers.ActivityRegularization(l2=0.001)
        self.add_weight(shape=(1,), regularizer='l2')

      def call(self, x):
        x = self.loss_layer(x)
        return self.loss_layer(x)

    inputs = keras.Input(shape=(1,))
    outputs = LayerWithSharedNestedLossLayer()(inputs)
    model = keras.Model(inputs, outputs)
    # Weight loss + 2 activity losses.
    self.assertEqual(len(model.losses), 3)

    x = array_ops.ones((1, 1))
    model(x)
    y = array_ops.ones((1, 1))
    model(y)
    if context.executing_eagerly():
      # Eager losses are cleared every `__call__`.
      self.assertEqual(len(model.losses), 3)
    else:
      self.assertEqual(len(model.get_losses_for(x)), 2)
      self.assertEqual(len(model.get_losses_for(y)), 2)
      self.assertEqual(len(model.get_losses_for(None)), 1)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_layer_with_variable_output(self):

    class VariableOutputLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.v = self.add_weight('output_var', shape=(2, 5), initializer='ones')

      def call(self, inputs):
        return self.v

    model = testing_utils.get_model_from_layers(
        [VariableOutputLayer(), keras.layers.Dense(1)], input_shape=(10,))
    # TODO(omalleyt): Make this work with `run_eagerly=True`.
    model.compile('sgd', 'mse', run_eagerly=False)
    model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2, epochs=5)

    self.assertLen(model.trainable_variables, 3)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_model_dtype(self):

    class AssertTypeLayer(keras.layers.Layer):

      def call(self, inputs):
        assert inputs.dtype.name == self.dtype, (
            'Input tensor has type %s which does not match assert type %s' %
            (inputs.dtype.name, self.assert_type))
        return inputs + 1.

    for dtype in ('float16', 'float32', 'float64'):
      model = testing_utils.get_model_from_layers(
          [AssertTypeLayer(dtype=dtype)], input_shape=(10,))
      model.compile(
          'sgd',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      x = np.ones((10, 10))
      y = np.ones((10, 10))
      model.fit(x, y)
      model.test_on_batch(x, y)
      model(x)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_model_input_dtype(self):
    model = testing_utils.get_small_mlp(1, 10, 10)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x = np.ones((10, 10)).astype(np.float64)
    y = np.ones((10, 10)).astype(np.float64)
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
    model.fit(dataset)
    self.assertEqual(model._compute_dtype, 'float32')
    # Input dtype should match the model dtype, even if the inputs passed to the
    # model have a different dtype.
    self.assertEqual(model.inputs[0].dtype, 'float32')

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_subclassed_model_with_training_arg(self):
    class LayerWithTrainingArg(keras.layers.Layer):

      def call(self, inputs, training=None):
        self.training = training
        return inputs

    class ModelWithTrainingArg(keras.Model):

      def __init__(self):
        super(ModelWithTrainingArg, self).__init__()
        self.l1 = LayerWithTrainingArg()

      def call(self, inputs, training=None):
        self.training = training
        inputs = self.l1(inputs, training=training)
        return inputs

    x = np.zeros((1, 2))
    model = ModelWithTrainingArg()
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, x, epochs=1)

    if (testing_utils.should_run_eagerly() or
        testing_utils.should_run_tf_function()):
      expected_training_arg = True
    else:
      expected_training_arg = keras.backend.symbolic_learning_phase()

    self.assertIs(model.training, expected_training_arg)
    self.assertIs(model.l1.training, expected_training_arg)

  @keras_parameterized.run_all_keras_modes
  def test_error_when_model_is_not_compiled(self):
    inputs = keras.Input(shape=(1,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.ones((1, 1)), np.ones((1, 1)))

    class MyModel(keras.Model):

      def call(self, x):
        self.add_loss(math_ops.reduce_sum(x))
        return x

    model = MyModel()
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.random.random((32, 1)), epochs=2)


class TestExceptionsAndWarnings(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_invalid_batch_dimension(self):

    def custom_reshape(inputs):
      return keras.backend.reshape(inputs, (-1, 8, 8, 3))

    layer_1 = keras.layers.Lambda(custom_reshape)
    layer_2 = keras.layers.Conv2D(32, (3, 3))

    model = testing_utils.get_model_from_layers([layer_1, layer_2],
                                                input_shape=(8, 8, 6))
    model.compile('sgd', loss='mse')

    with self.assertRaisesRegex(
        ValueError,
        'Mismatch between expected batch size and model output batch size. '
        r'Output shape = \(20, 6, 6, 32\), expected output shape = '
        r'shape \(10, 6, 6, 32\)'):
      model.predict(np.ones((10, 8, 8, 6)), batch_size=10)

  @keras_parameterized.run_all_keras_modes
  def test_invalid_loss(self):
    num_classes = 5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5

    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=num_classes, input_dim=input_dim)
    optimizer = RMSPropOptimizer(learning_rate=0.001)
    model.compile(optimizer, loss='categorical_crossentropy')
    np.random.seed(1337)
    (x_train, y_train), (_, _) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)

    with self.assertRaises(ValueError):
      model.fit(x_train, np.concatenate([y_train, y_train], axis=-1))

    if not context.executing_eagerly():
      # TODO(psv): Investigate these use cases in eager mode.
      with self.assertRaises(ValueError):
        model.fit(x_train, y_train)

      with self.assertRaises(ValueError):
        model.compile(
            optimizer,
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_compile_warning_for_loss_missing_output(self):
    with self.cached_session():
      inp = keras.layers.Input(shape=(16,), name='input_a')
      out_1 = keras.layers.Dense(8, name='dense_1')(inp)
      out_2 = keras.layers.Dense(3, activation='softmax', name='dense_2')(out_1)
      model = keras.models.Model(inputs=[inp], outputs=[out_1, out_2])
      optimizer = RMSPropOptimizer(learning_rate=0.001)

      with test.mock.patch.object(logging, 'warning') as mock_log:
        model.compile(
            optimizer,
            loss={
                'dense_2': 'categorical_crossentropy',
            },
            metrics={
                'dense_2': 'categorical_accuracy',
                'dense_1': metrics_module.CategoricalAccuracy(),
            },
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())
        msg = ('Output dense_1 missing from loss dictionary. We assume this '
               'was done on purpose. The fit and evaluate APIs will not be '
               'expecting any data to be passed to dense_1.')
        self.assertRegexpMatches(str(mock_log.call_args), msg)

  @keras_parameterized.run_all_keras_modes
  def test_invalid_steps_per_epoch_usage(self):
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(1)(x)

    model = keras.Model(x, y)
    model.compile(
        'sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=False)
    err_msg = 'When passing input data as arrays, do not specify'

    with test.mock.patch.object(logging, 'warning') as mock_log:
      model._standardize_user_data(
          np.zeros((100, 1)), np.ones((100, 1)), check_steps=True, steps=4)
      self.assertRegexpMatches(str(mock_log.call_args), err_msg)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_invalid_batch_size_argument_with_sequence_input(self):

    class DummySequence(data_utils.Sequence):

      def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10, 4])

      def __len__(self):
        return 10

    model = testing_utils.get_small_mlp(
        num_hidden=10, num_classes=1, input_dim=10)

    model.compile(
        'adam',
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    with self.assertRaisesRegexp(
        ValueError, 'The `batch_size` argument must not be specified'):
      model.fit(DummySequence(), batch_size=2, epochs=2)
    with self.assertRaisesRegexp(
        ValueError, 'The `batch_size` argument must not be specified'):
      model.evaluate(DummySequence(), batch_size=2)

    with self.assertRaisesRegexp(
        ValueError, 'The `batch_size` argument must not be specified'):
      model.predict(DummySequence(), batch_size=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_non_returning_sequence(self):
    if not testing_utils.should_run_tf_function():
      self.skipTest('This case is only handled in the new execution path.')

    class DummySequence(data_utils.Sequence):

      def __getitem__(self, idx):
        return

      def __len__(self):
        return 10

    model = testing_utils.get_small_mlp(
        num_hidden=10, num_classes=1, input_dim=10)

    model.compile(
        'adam',
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    with self.assertRaisesRegexp(IndexError, 'Could not infer batch size'):
      model.fit(DummySequence(), epochs=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_sparse_op_with_op_layer(self):
    inputs = keras.layers.Input(shape=(2,), sparse=True, name='sparse_tensor')
    output = sparse_ops.sparse_minimum(inputs, inputs)
    with self.assertRaisesRegexp(
        ValueError,
        'Sparse ops are not supported with functional models with built-in '
        'layer wrapping'
    ):
      keras.Model([inputs], output)


class LossWeightingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = 10.
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    learning_rate = 0.001

    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=num_classes, input_dim=input_dim)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['acc', metrics_module.CategoricalAccuracy()],
        weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
        optimizer=RMSPropOptimizer(learning_rate=learning_rate),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_test = y_test.copy()
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    test_ids = np.where(int_y_test == np.array(weighted_class))[0]

    class_weight = dict([(i, 1.) for i in range(num_classes)])
    class_weight[weighted_class] = weight

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 3,
        verbose=0,
        class_weight=class_weight,
        validation_data=(x_train, y_train))
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 2,
        verbose=0,
        class_weight=class_weight)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 2,
        verbose=0,
        class_weight=class_weight,
        validation_split=0.1)

    model.train_on_batch(
        x_train[:batch_size], y_train[:batch_size], class_weight=class_weight)
    ref_score = model.evaluate(x_test, y_test, verbose=0)
    score = model.evaluate(
        x_test[test_ids, :], y_test[test_ids, :], verbose=0)
    self.assertLess(score[0], ref_score[0])

  @keras_parameterized.run_all_keras_modes
  def test_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = 10.
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    learning_rate = 0.001

    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=num_classes, input_dim=input_dim)
    model.compile(
        RMSPropOptimizer(learning_rate=learning_rate),
        metrics=['acc', metrics_module.CategoricalAccuracy()],
        weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
        loss='categorical_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    np.random.seed(43)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_test = y_test.copy()
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    test_ids = np.where(int_y_test == np.array(weighted_class))[0]

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = weight

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 3,
        verbose=0,
        sample_weight=sample_weight)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 3,
        verbose=0,
        sample_weight=sample_weight,
        validation_split=0.1)

    model.train_on_batch(
        x_train[:batch_size],
        y_train[:batch_size],
        sample_weight=sample_weight[:batch_size])
    model.test_on_batch(
        x_train[:batch_size],
        y_train[:batch_size],
        sample_weight=sample_weight[:batch_size])
    ref_score = model.evaluate(
        x_test, y_test, verbose=0, sample_weight=sample_weight)
    score = model.evaluate(
        x_test[test_ids, :],
        y_test[test_ids, :],
        verbose=0,
        sample_weight=sample_weight[test_ids])
    self.assertLess(score[0], ref_score[0])

  @keras_parameterized.run_all_keras_modes
  def test_temporal_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = 10.
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3
    learning_rate = 0.001

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(num_classes),
              input_shape=(timesteps, input_dim)))
      model.add(keras.layers.Activation('softmax'))

      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=test_samples,
          input_shape=(input_dim,),
          num_classes=num_classes)
      int_y_test = y_test.copy()
      int_y_train = y_train.copy()
      # convert class vectors to binary class matrices
      y_train = np_utils.to_categorical(y_train, num_classes)
      y_test = np_utils.to_categorical(y_test, num_classes)
      test_ids = np.where(int_y_test == np.array(weighted_class))[0]

      sample_weight = np.ones((y_train.shape[0]))
      sample_weight[int_y_train == weighted_class] = weight

      temporal_x_train = np.reshape(x_train, (len(x_train), 1,
                                              x_train.shape[1]))
      temporal_x_train = np.repeat(temporal_x_train, timesteps, axis=1)
      temporal_x_test = np.reshape(x_test, (len(x_test), 1, x_test.shape[1]))
      temporal_x_test = np.repeat(temporal_x_test, timesteps, axis=1)

      temporal_y_train = np.reshape(y_train, (len(y_train), 1,
                                              y_train.shape[1]))
      temporal_y_train = np.repeat(temporal_y_train, timesteps, axis=1)
      temporal_y_test = np.reshape(y_test, (len(y_test), 1, y_test.shape[1]))
      temporal_y_test = np.repeat(temporal_y_test, timesteps, axis=1)

      temporal_sample_weight = np.reshape(sample_weight, (len(sample_weight),
                                                          1))
      temporal_sample_weight = np.repeat(
          temporal_sample_weight, timesteps, axis=1)

      model.compile(
          RMSPropOptimizer(learning_rate=learning_rate),
          loss='categorical_crossentropy',
          metrics=['acc', metrics_module.CategoricalAccuracy()],
          weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
          sample_weight_mode='temporal',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      model.fit(
          temporal_x_train,
          temporal_y_train,
          batch_size=batch_size,
          epochs=epochs // 3,
          verbose=0,
          sample_weight=temporal_sample_weight)
      model.fit(
          temporal_x_train,
          temporal_y_train,
          batch_size=batch_size,
          epochs=epochs // 3,
          verbose=0,
          sample_weight=temporal_sample_weight,
          validation_split=0.1)

      model.train_on_batch(
          temporal_x_train[:batch_size],
          temporal_y_train[:batch_size],
          sample_weight=temporal_sample_weight[:batch_size])
      model.test_on_batch(
          temporal_x_train[:batch_size],
          temporal_y_train[:batch_size],
          sample_weight=temporal_sample_weight[:batch_size])
      ref_score = model.evaluate(temporal_x_test, temporal_y_test, verbose=0)
      if not context.executing_eagerly():
        score = model.evaluate(
            temporal_x_test[test_ids], temporal_y_test[test_ids], verbose=0)
        self.assertLess(score[0], ref_score[0])

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  def test_fit_with_incorrect_weights(self):
    input_a = keras.layers.Input(shape=(3,), name='input_a')
    input_b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(2, name='output_1')
    dropout = keras.layers.Dropout(0.5, name='output_2')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)
    model.compile(
        optimizer='adam',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x = np.random.random((10, 3))
    y = np.random.random((10, 2))

    with self.assertRaisesRegexp(
        ValueError,
        r'Unknown entries in sample_weight dictionary: \[\'unknown\'\]. '
        r'Only expected following keys: \[\'output_1\', \'output_2\'\]'):
      model.fit([x, x], [y, y],
                epochs=1,
                sample_weight={'unknown': 'something'})

    with self.assertRaisesRegexp(
        ValueError,
        r'Unknown entries in class_weight dictionary: \[\'unknown\'\]. '
        r'Only expected following keys: \[\'output_1\', \'output_2\'\]'):
      model.fit([x, x], [y, y], epochs=1, class_weight={'unknown': 'something'})

  @keras_parameterized.run_all_keras_modes
  def test_class_weight_invalid_use_case(self):
    num_classes = 5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3
    learning_rate = 0.001

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(num_classes),
              input_shape=(timesteps, input_dim)))
      model.add(keras.layers.Activation('softmax'))
      optimizer = RMSPropOptimizer(learning_rate=learning_rate)
      model.compile(
          optimizer,
          loss='binary_crossentropy',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=test_samples,
          input_shape=(input_dim,),
          num_classes=num_classes)
      # convert class vectors to binary class matrices
      y_train = np_utils.to_categorical(y_train, num_classes)
      class_weight = dict([(i, 1.) for i in range(num_classes)])

      del class_weight[1]
      with self.assertRaises(ValueError):
        model.fit(x_train, y_train,
                  epochs=0, verbose=0, class_weight=class_weight)

      with self.assertRaises(ValueError):
        model.compile(
            optimizer,
            loss='binary_crossentropy',
            sample_weight_mode=[],
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

      # Build multi-output model
      x = keras.Input((3,))
      y1 = keras.layers.Dense(4, name='1')(x)
      y2 = keras.layers.Dense(4, name='2')(x)
      model = keras.models.Model(x, [y1, y2])
      model.compile(
          optimizer,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      x_np = np.random.random((10, 3))
      y_np = np.random.random((10, 4))
      w_np = np.random.random((10,))
      # This will work
      model.fit(x_np, [y_np, y_np], epochs=1,
                sample_weight={'1': w_np})
      # These will not
      with self.assertRaises(ValueError):
        model.fit(x_np, [y_np, y_np], epochs=1,
                  sample_weight=[w_np])
      with self.assertRaises(TypeError):
        model.fit(x_np, [y_np, y_np], epochs=1,
                  sample_weight=w_np)
      with self.assertRaises(ValueError):
        bad_w_np = np.random.random((11,))
        model.fit(x_np, [y_np, y_np], epochs=1,
                  sample_weight={'1': bad_w_np})
      with self.assertRaises(ValueError):
        bad_w_np = np.random.random((10, 2))
        model.fit(x_np, [y_np, y_np], epochs=1,
                  sample_weight={'1': bad_w_np})
      with self.assertRaises(ValueError):
        bad_w_np = np.random.random((10, 2, 2))
        model.fit(x_np, [y_np, y_np], epochs=1,
                  sample_weight={'1': bad_w_np})

  @keras_parameterized.run_all_keras_modes
  def test_default_sample_weight(self):
    """Verifies that fit works without having to set sample_weight."""
    num_classes = 5
    input_dim = 5
    timesteps = 3
    learning_rate = 0.001

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(num_classes),
              input_shape=(timesteps, input_dim)))

      x = np.random.random((10, timesteps, input_dim))
      y = np.random.random((10, timesteps, num_classes))
      optimizer = RMSPropOptimizer(learning_rate=learning_rate)

      # sample_weight_mode is a list and mode value is None
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode=[None],
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a list and mode value is `temporal`
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode=['temporal'],
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is None
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode={'time_distributed': None},
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is `temporal`
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode={'time_distributed': 'temporal'},
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is None
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode=None,
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is `temporal`
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode='temporal',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=1, batch_size=10)

  def test_sample_weight_tensor(self):
    """Tests that sample weight may be defined as a tensor in the graph."""
    with context.graph_mode():
      # Create a simple pass-through model
      input_layer = keras.layers.Input(shape=1, name='input_layer')
      model = keras.Model(inputs=input_layer, outputs=input_layer)
      model.compile(
          loss='mean_absolute_error',
          optimizer='adam')

      # Prepare sample weights iterator tensor
      sample_weights = array_ops.constant(
          [[0, .4, 1, 1], [2, .4, .3, 1]])
      dataset = dataset_ops.Dataset.from_tensor_slices(sample_weights)
      sample_weights = dataset_ops.make_one_shot_iterator(dataset).get_next()
      sample_weights = training_utils.standardize_sample_weights(
          sample_weights, model.output_names)

      # Update model loss with sample weight tensor.
      model._compile_weights_loss_and_weighted_metrics(sample_weights)

      feeds = {'input_layer:0': [[0], [0], [0], [0]],
               'input_layer_target:0': [[1], [1], [1], [1]]}
      with self.cached_session() as sess:
        self.assertAllClose(
            (.4 + 1 + 1) / 4, sess.run(model.total_loss, feed_dict=feeds))
        self.assertAllClose(
            (2+ .4 + .3 + 1) / 4, sess.run(model.total_loss, feed_dict=feeds))

  def test_prepare_sample_weights(self):
    # pylint:disable=anomalous-backslash-in-string
    input_layer = keras.layers.Input(shape=1, name='input_layer')
    model = keras.Model(inputs=input_layer, outputs=[input_layer, input_layer])
    sample_weights = array_ops.constant([0, .4, 1, 1])
    temporal_weights = array_ops.constant([[1, 2], [3, 4], [5, 6]])

    model.compile(
        loss='mean_absolute_error',
        optimizer='adam',
        sample_weight_mode=None)

    with self.assertRaises(AssertionError):
      model._prepare_sample_weights([sample_weights, sample_weights])

    model.compile(loss='mean_absolute_error', optimizer='adam',
                  sample_weight_mode='temporal')
    model._prepare_sample_weights([temporal_weights, temporal_weights])
    with self.assertRaisesRegexp(ValueError, 'Expected shape \[None, None\]'):
      model._prepare_sample_weights([sample_weights, sample_weights])

    with self.assertRaisesRegexp(ValueError,
                                 'sample weights must have same length as the '
                                 'number of outputs'):
      model._prepare_sample_weights([temporal_weights])

    model.compile(loss='mean_absolute_error', optimizer='adam',
                  sample_weight_mode='samplewise')
    model._prepare_sample_weights([sample_weights, sample_weights])
    with self.assertRaisesRegexp(ValueError, 'Expected shape \[None\]'):
      model._prepare_sample_weights([temporal_weights, temporal_weights])
    # pylint:enable=anomalous-backslash-in-string


@keras_parameterized.run_all_keras_modes
class MaskingTest(keras_parameterized.TestCase):

  def _get_model(self, input_shape=None):
    layers = [
        keras.layers.Masking(mask_value=0),
        keras.layers.TimeDistributed(
            keras.layers.Dense(1, kernel_initializer='one'))
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape)
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    return model

  @keras_parameterized.run_with_all_model_types
  def test_masking(self):
    model = self._get_model(input_shape=(2, 1))
    x = np.array([[[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[1], [1]]])
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)

  @keras_parameterized.run_with_all_model_types(exclude_models='functional')
  def test_masking_deferred(self):
    model = self._get_model()
    x = np.array([[[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[1], [1]]])
    loss = model.train_on_batch(x, y)
    self.assertEqual(loss, 0)

  def test_mask_argument_in_layer(self):
    # Test that the mask argument gets correctly passed to a layer in the
    # functional API.

    class CustomMaskedLayer(keras.layers.Layer):

      def __init__(self):
        super(CustomMaskedLayer, self).__init__()
        self.supports_masking = True

      def call(self, inputs, mask=None):
        assert mask is not None
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    x = np.random.random((5, 3))
    inputs = keras.layers.Input((3,))
    masked = keras.layers.Masking(mask_value=0)(inputs)
    outputs = CustomMaskedLayer()(masked)

    model = keras.Model(inputs, outputs)
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    y = np.random.random((5, 3))
    model.train_on_batch(x, y)


@keras_parameterized.run_all_keras_modes
class TestDynamicTrainability(keras_parameterized.TestCase):

  def test_trainable_warning(self):
    x = np.random.random((5, 3))
    y = np.random.random((5, 2))

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim=3))
    model.trainable = False
    model.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.trainable = True
    model.train_on_batch(x, y)
    self.assertRaises(Warning)

  def test_trainable_argument(self):
    with self.cached_session():
      x = np.random.random((5, 3))
      y = np.random.random((5, 2))

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_dim=3, trainable=False))
      model.compile(
          'rmsprop',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

      # test with nesting
      inputs = keras.layers.Input(shape=(3,))
      output = model(inputs)
      model = keras.models.Model(inputs, output)
      model.compile(
          'rmsprop',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

  def test_layer_trainability_switch(self):
    # with constructor argument, in Sequential
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, trainable=False, input_dim=1))
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Sequential
    model = keras.models.Sequential()
    layer = keras.layers.Dense(2, input_dim=1)
    model.add(layer)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # with constructor argument, in Model
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(2, trainable=False)(x)
    model = keras.models.Model(x, y)
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Model
    x = keras.layers.Input(shape=(1,))
    layer = keras.layers.Dense(2)
    y = layer(x)
    model = keras.models.Model(x, y)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_model_trainability_switch(self):
    # a non-trainable model has no trainable weights
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(2)(x)
    model = keras.models.Model(x, y)
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # same for Sequential
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim=1))
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_nested_model_trainability(self):
    # a Sequential inside a Model
    inner_model = keras.models.Sequential()
    inner_model.add(keras.layers.Dense(2, input_dim=1))

    x = keras.layers.Input(shape=(1,))
    y = inner_model(x)
    outer_model = keras.models.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Sequential inside a Sequential
    inner_model = keras.models.Sequential()
    inner_model.add(keras.layers.Dense(2, input_dim=1))
    outer_model = keras.models.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Model
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(2)(x)
    inner_model = keras.models.Model(x, y)
    x = keras.layers.Input(shape=(1,))
    y = inner_model(x)
    outer_model = keras.models.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Sequential
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(2)(x)
    inner_model = keras.models.Model(x, y)
    outer_model = keras.models.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

  def test_gan_workflow(self):
    shared_layer = keras.layers.BatchNormalization()

    inputs1 = keras.Input(10)
    outputs1 = shared_layer(inputs1)
    model1 = keras.Model(inputs1, outputs1)
    shared_layer.trainable = False
    model1.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    inputs2 = keras.Input(10)
    outputs2 = shared_layer(inputs2)
    model2 = keras.Model(inputs2, outputs2)
    shared_layer.trainable = True
    model2.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x, y = np.ones((10, 10)), np.ones((10, 10))

    out1_0 = model1.predict_on_batch(x)
    model1.train_on_batch(x, y)
    out1_1 = model1.predict_on_batch(x)
    self.assertAllClose(out1_0, out1_1)

    out2_0 = model2.predict_on_batch(x)
    model2.train_on_batch(x, y)
    out2_1 = model2.predict_on_batch(x)
    self.assertNotAllClose(out2_0, out2_1)

  def test_toggle_value(self):
    input_0 = keras.layers.Input(shape=(1,))
    dense_0 = keras.layers.Dense(1, kernel_initializer='ones',
                                 bias_initializer='ones')
    dense_1 = keras.layers.Dense(1, kernel_initializer='ones',
                                 bias_initializer='ones')
    result = keras.layers.Add()([dense_0(input_0), dense_1(input_0)])
    model = keras.models.Model(input_0, result)
    dense_0.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((10, 1))
    y = 5 * x + 2
    model.train_on_batch(x, y)
    dense_0.trainable = True
    model.train_on_batch(x, y)
    kernel, bias = dense_0.get_weights()
    self.assertAllEqual([kernel[0, 0], bias[0]], [1., 1.])

    kernel, bias = dense_1.get_weights()
    self.assertAllClose([kernel[0, 0], bias[0]], [1.1176, 1.1176])


class TestTrainingWithDataTensors(keras_parameterized.TestCase):

  @tf_test_util.run_deprecated_v1
  def test_training_and_eval_methods_on_symbolic_tensors_single_io(self):
    x = keras.layers.Input(shape=(3,), name='input')
    y = keras.layers.Dense(4, name='dense')(x)
    model = keras.Model(x, y)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    model.compile(
        optimizer,
        loss,
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    inputs = keras.backend.zeros(shape=(10, 3))
    targets = keras.backend.zeros(shape=(10, 4))

    model.fit(inputs, targets, epochs=1, steps_per_epoch=2, verbose=0)
    model.evaluate(inputs, targets, steps=2, verbose=0)
    model.predict(inputs, steps=2)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    model.fit(inputs, targets,
              epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=(inputs, targets), validation_steps=2)

    # Test with dynamic shape
    inputs = array_ops.placeholder_with_default(
        np.zeros((2, 3)), shape=tensor_shape.TensorShape([None, 3]))
    targets = array_ops.placeholder_with_default(
        np.zeros((2, 4)), shape=tensor_shape.TensorShape([None, 4]))
    self.assertEqual(inputs.shape.dims[0].value, None)
    model.fit(inputs, targets, epochs=1, steps_per_epoch=2, verbose=0)
    model.evaluate(inputs, targets, steps=2, verbose=0)
    model.predict(inputs, steps=2)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    model.fit(inputs, targets,
              epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=(inputs, targets), validation_steps=2)

  @tf_test_util.run_deprecated_v1
  def test_training_and_eval_methods_on_symbolic_tensors_multi_io(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        loss_weights=loss_weights)

    input_a_tf = keras.backend.zeros(shape=(10, 3))
    input_b_tf = keras.backend.zeros(shape=(10, 3))

    output_d_tf = keras.backend.zeros(shape=(10, 4))
    output_e_tf = keras.backend.zeros(shape=(10, 4))

    model.fit(
        [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
        epochs=1,
        steps_per_epoch=2,
        verbose=0)
    with self.assertRaisesRegexp(ValueError,
                                 'should specify the `steps_per_epoch`'):
      model.fit(
          [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
          epochs=1,
          batch_size=5,
          verbose=0)
    model.train_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

    # Test with dictionary inputs
    model.fit(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf},
        epochs=1,
        steps_per_epoch=2,
        verbose=0)
    model.fit(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf},
        validation_data=({'input_a': input_a_tf,
                          'input_b': input_b_tf},
                         {'dense': output_d_tf,
                          'dropout': output_e_tf}),
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        verbose=0)
    model.train_on_batch(
        {'input_a': input_a_tf,
         'input_b': input_b_tf},
        {'dense': output_d_tf,
         'dropout': output_e_tf})

    # Test with validation data
    model.fit(
        [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
        validation_data=([input_a_tf, input_b_tf],
                         [output_d_tf, output_e_tf]),
        epochs=1,
        steps_per_epoch=2,
        validation_steps=2,
        verbose=0)
    # Test with validation split
    with self.assertRaisesRegexp(ValueError,
                                 'you cannot use `validation_split`'):
      model.fit(
          [input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
          epochs=2,
          steps_per_epoch=2,
          verbose=0,
          validation_split=0.2,
          validation_steps=2)

    # Test evaluation / prediction methods
    model.evaluate([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
                   steps=2, verbose=0)
    model.predict([input_a_tf, input_b_tf], steps=2)
    model.test_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

  @tf_test_util.run_deprecated_v1
  def test_model_with_input_feed_tensor(self):
    """We test building a model with a TF variable as input.

    We should be able to call fit, evaluate, predict,
    by only passing them data for the placeholder inputs
    in the model.
    """
    with self.cached_session():
      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 3))

      output_a_np = np.random.random((10, 4))
      output_b_np = np.random.random((10, 3))

      input_v = keras.backend.variables_module.Variable(
          input_a_np, dtype='float32')
      self.evaluate(variables_lib.variables_initializer([input_v]))
      a = keras.Input(tensor=input_v)
      b = keras.Input(shape=(3,), name='input_b')

      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      model = keras.models.Model([a, b], [a_2, b_2])
      model.summary()

      optimizer = 'rmsprop'
      loss = 'mse'
      loss_weights = [1., 0.5]
      model.compile(optimizer, loss, metrics=['mean_squared_error'],
                    loss_weights=loss_weights,
                    sample_weight_mode=None)

      # test train_on_batch
      out = model.train_on_batch(input_b_np,
                                 [output_a_np, output_b_np])
      out = model.train_on_batch({'input_b': input_b_np},
                                 [output_a_np, output_b_np])
      out = model.test_on_batch({'input_b': input_b_np},
                                [output_a_np, output_b_np])
      out = model.predict_on_batch({'input_b': input_b_np})

      # test fit
      out = model.fit({'input_b': input_b_np},
                      [output_a_np, output_b_np], epochs=1, batch_size=10)
      out = model.fit(input_b_np,
                      [output_a_np, output_b_np], epochs=1, batch_size=10)

      # test evaluate
      out = model.evaluate({'input_b': input_b_np},
                           [output_a_np, output_b_np], batch_size=10)
      out = model.evaluate(input_b_np,
                           [output_a_np, output_b_np], batch_size=10)

      # test predict
      out = model.predict({'input_b': input_b_np}, batch_size=10)
      out = model.predict(input_b_np, batch_size=10)
      self.assertEqual(len(out), 2)

      # Now test a model with a single input
      # i.e. we don't pass any data to fit the model.
      self.evaluate(variables_lib.variables_initializer([input_v]))
      a = keras.Input(tensor=input_v)
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      a_2 = keras.layers.Dropout(0.5, name='dropout')(a_2)
      model = keras.models.Model(a, a_2)
      model.summary()

      optimizer = 'rmsprop'
      loss = 'mse'
      model.compile(optimizer, loss, metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None,
                                 output_a_np)
      out = model.train_on_batch(None,
                                 output_a_np)
      out = model.test_on_batch(None,
                                output_a_np)
      out = model.predict_on_batch(None)
      out = model.train_on_batch([],
                                 output_a_np)
      out = model.train_on_batch({},
                                 output_a_np)

      # test fit
      _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=3)
      _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=3)

      # test evaluate
      _ = model.evaluate(None, output_a_np, steps=3)
      _ = model.evaluate(None, output_a_np, steps=3)

      # test predict
      out = model.predict(None, steps=3)
      out = model.predict(None, steps=3)
      self.assertEqual(out.shape, (10 * 3, 4))

      # Same, without learning phase
      # i.e. we don't pass any data to fit the model.
      self.evaluate(variables_lib.variables_initializer([input_v]))
      a = keras.Input(tensor=input_v)
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      model = keras.models.Model(a, a_2)
      model.summary()

      optimizer = 'rmsprop'
      loss = 'mse'
      model.compile(optimizer, loss, metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None,
                                 output_a_np)
      out = model.train_on_batch(None,
                                 output_a_np)
      out = model.test_on_batch(None,
                                output_a_np)
      out = model.predict_on_batch(None)
      out = model.train_on_batch([],
                                 output_a_np)
      out = model.train_on_batch({},
                                 output_a_np)

      # test fit
      _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=10)
      _ = model.fit(None, output_a_np, epochs=1, steps_per_epoch=10)

      # test evaluate
      _ = model.evaluate(None, output_a_np, steps=10)
      _ = model.evaluate(None, output_a_np, steps=10)

      # test predict
      out = model.predict(None, steps=3)
      out = model.predict(None, steps=3)
      self.assertEqual(out.shape, (10 * 3, 4))

  @keras_parameterized.run_all_keras_modes
  def test_model_with_partial_loss(self):
    with self.cached_session():
      a = keras.Input(shape=(3,), name='input_a')
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      a_3 = dp(a_2)
      model = keras.models.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dropout': 'mse'}
      model.compile(optimizer, loss, metrics=['mae'])

      input_a_np = np.random.random((10, 3))
      output_a_np = np.random.random((10, 4))

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, [output_a_np])
      # evaluate
      _ = model.evaluate(input_a_np, [output_a_np])

      # Same without dropout.
      a = keras.Input(shape=(3,), name='input_a')
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      a_3 = keras.layers.Dense(4, name='dense_2')(a_2)
      model = keras.models.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dense_2': 'mse'}
      model.compile(optimizer, loss, metrics={'dense_1': 'mae'})

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, [output_a_np])
      # evaluate
      _ = model.evaluate(input_a_np, [output_a_np])

  @tf_test_util.run_deprecated_v1
  def test_model_with_external_loss(self):
    with self.cached_session():
      # None loss, only regularization loss.
      a = keras.Input(shape=(3,), name='input_a')
      a_2 = keras.layers.Dense(4, name='dense_1',
                               kernel_regularizer='l1',
                               bias_regularizer='l2')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      a_3 = dp(a_2)

      model = keras.models.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = None
      model.compile(optimizer, loss, metrics=['mae'])

      input_a_np = np.random.random((10, 3))

      # test train_on_batch
      out = model.train_on_batch(input_a_np, None)
      out = model.test_on_batch(input_a_np, None)
      # fit
      out = model.fit(input_a_np, None)
      # evaluate
      out = model.evaluate(input_a_np, None)

      # No dropout, external loss.
      a = keras.Input(shape=(3,), name='input_a')
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      a_3 = keras.layers.Dense(4, name='dense_2')(a)

      model = keras.models.Model(a, [a_2, a_3])
      model.add_loss(keras.backend.mean(a_3 + a_2))

      optimizer = 'rmsprop'
      loss = None
      model.compile(optimizer, loss, metrics=['mae'])

      # test train_on_batch
      out = model.train_on_batch(input_a_np, None)
      out = model.test_on_batch(input_a_np, None)
      # fit
      out = model.fit(input_a_np, None)
      # evaluate
      out = model.evaluate(input_a_np, None)

      # Test model with no external data at all.
      input_v = keras.backend.variables_module.Variable(
          input_a_np, dtype='float32')
      self.evaluate(variables_lib.variables_initializer([input_v]))
      a = keras.Input(tensor=input_v)
      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      a_2 = keras.layers.Dropout(0.5, name='dropout')(a_2)
      model = keras.models.Model(a, a_2)
      model.add_loss(keras.backend.mean(a_2))

      model.compile(optimizer='rmsprop',
                    loss=None,
                    metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None, None)
      out = model.test_on_batch(None, None)
      out = model.predict_on_batch(None)

      # test fit
      with self.assertRaises(ValueError):
        out = model.fit(None, None, epochs=1, batch_size=10)
      out = model.fit(None, None, epochs=1, steps_per_epoch=1)

      # test fit with validation data
      with self.assertRaises(ValueError):
        out = model.fit(None, None, epochs=1,
                        steps_per_epoch=None,
                        validation_steps=2)
        out = model.fit(None, None, epochs=1,
                        steps_per_epoch=2,
                        validation_steps=2)

      # test evaluate
      with self.assertRaises(ValueError):
        out = model.evaluate(None, None, batch_size=10)
      out = model.evaluate(None, None, steps=3)

      # test predict
      with self.assertRaises(ValueError):
        out = model.predict(None, batch_size=10)
      out = model.predict(None, steps=3)
      self.assertEqual(out.shape, (10 * 3, 4))

      # Test multi-output model with no external data at all.
      self.evaluate(variables_lib.variables_initializer([input_v]))
      a = keras.Input(tensor=input_v)
      a_1 = keras.layers.Dense(4, name='dense_1')(a)
      a_2 = keras.layers.Dropout(0.5, name='dropout')(a_1)
      model = keras.models.Model(a, [a_1, a_2])
      model.add_loss(keras.backend.mean(a_2))

      model.compile(optimizer='rmsprop',
                    loss=None,
                    metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None, None)
      out = model.test_on_batch(None, None)
      out = model.predict_on_batch(None)

      # test fit
      with self.assertRaises(ValueError):
        out = model.fit(None, None, epochs=1, batch_size=10)
      out = model.fit(None, None, epochs=1, steps_per_epoch=1)

      # test evaluate
      with self.assertRaises(ValueError):
        out = model.evaluate(None, None, batch_size=10)
      out = model.evaluate(None, None, steps=3)

      # test predict
      with self.assertRaises(ValueError):
        out = model.predict(None, batch_size=10, verbose=1)
      out = model.predict(None, steps=3)
      self.assertEqual(len(out), 2)
      self.assertEqual(out[0].shape, (10 * 3, 4))
      self.assertEqual(out[1].shape, (10 * 3, 4))

  @tf_test_util.run_deprecated_v1
  def test_target_tensors(self):
    with self.cached_session():
      # single-output, as list
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,), name='dense'))
      input_val = np.random.random((10, 4))
      target_val = np.random.random((10, 4))
      target = keras.backend.variable(target_val)
      model.compile(optimizer='rmsprop', loss='mse', target_tensors=[target])
      model.train_on_batch(input_val, None)

      # single-output, as single tensor
      model.compile(optimizer='rmsprop', loss='mse', target_tensors=target)
      model.train_on_batch(input_val, None)

      # single-output, as dict
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors={'dense': target})
      model.train_on_batch(input_val, None)

      # test invalid arguments
      with self.assertRaises(TypeError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=set())
      with self.assertRaises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=[target, target])
      with self.assertRaises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors={'dense2': None})
      with self.assertRaises(ValueError):
        model.compile(optimizer='rmsprop', loss='mse',
                      target_tensors=[target])
        model.train_on_batch(input_val, target_val)

      # multi-output, as list
      input_val = np.random.random((10, 4))
      target_val_a = np.random.random((10, 4))
      target_val_b = np.random.random((10, 4))
      target_a = keras.backend.variable(target_val_a)
      target_b = keras.backend.variable(target_val_b)

      inputs = keras.layers.Input(shape=(4,))
      output_a = keras.layers.Dense(4, name='dense_a')(inputs)
      output_b = keras.layers.Dense(4, name='dense_b')(inputs)
      model = keras.models.Model(inputs, [output_a, output_b])
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors=[target_a, target_b])
      model.train_on_batch(input_val, None)

      # multi-output, as dict
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors={'dense_a': target_a,
                                    'dense_b': target_b})
      model.train_on_batch(input_val, None)

      # test with sample weights
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          metrics=['mae', metrics_module.CategoricalAccuracy()],
          target_tensors=[target_a, target_b])
      model.train_on_batch(input_val, None,
                           sample_weight={'dense_a': np.random.random((10,))})

  @tf_test_util.run_deprecated_v1
  def test_model_custom_target_tensors(self):
    with self.cached_session():
      a = keras.Input(shape=(3,), name='input_a')
      b = keras.Input(shape=(3,), name='input_b')

      a_2 = keras.layers.Dense(4, name='dense_1')(a)
      dp = keras.layers.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      y = keras.backend.placeholder([10, 4], name='y')
      y1 = keras.backend.placeholder([10, 3], name='y1')
      y2 = keras.backend.placeholder([7, 5], name='y2')
      model = keras.models.Model([a, b], [a_2, b_2])

      optimizer = 'rmsprop'
      loss = 'mse'
      loss_weights = [1., 0.5]

      # test list of target tensors
      with self.assertRaises(ValueError):
        model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                      sample_weight_mode=None, target_tensors=[y, y1, y2])
      model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                    sample_weight_mode=None, target_tensors=[y, y1])
      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 3))

      output_a_np = np.random.random((10, 4))
      output_b_np = np.random.random((10, 3))

      _ = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np], {
                                   'dense_1': np.random.random((10,)),
                                   'dropout': np.random.random((10,))
                               })
      # test dictionary of target_tensors
      with self.assertRaises(ValueError):
        model.compile(optimizer, loss,
                      metrics=[],
                      loss_weights=loss_weights,
                      sample_weight_mode=None,
                      target_tensors={'does_not_exist': y2})
      # test dictionary of target_tensors
      model.compile(optimizer, loss,
                    metrics=[],
                    loss_weights=loss_weights,
                    sample_weight_mode=None,
                    target_tensors={'dense_1': y, 'dropout': y1})
      _ = model.train_on_batch([input_a_np, input_b_np],
                               [output_a_np, output_b_np], {
                                   'dense_1': np.random.random((10,)),
                                   'dropout': np.random.random((10,))
                               })

      # test with custom TF placeholder as target
      pl_target_a = keras.backend.array_ops.placeholder('float32',
                                                        shape=(None, 4))
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors={'dense_1': pl_target_a})
      model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])


class TestTrainingWithMetrics(keras_parameterized.TestCase):
  """Training tests related to metrics."""

  @keras_parameterized.run_all_keras_modes
  def test_metrics_names(self):
    a = keras.layers.Input(shape=(3,), name='input_a')
    b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = keras.layers.Dropout(0.5, name='dropout')(c)

    model = keras.models.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    metrics = ['mse', metrics_module.BinaryAccuracy()]
    model.compile(
        optimizer,
        loss='mae',
        metrics=metrics,
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    mse_metric = 'mse' if tf2.enabled() else 'mean_squared_error'
    reference_metric_names = [
        'loss', 'dense_loss', 'dropout_loss', 'dense_' + mse_metric,
        'dense_binary_accuracy', 'dropout_' + mse_metric,
        'dropout_binary_accuracy'
    ]
    self.assertEqual(reference_metric_names, model.metrics_names)

    # Verify that model metric names are not altered during training.
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)
    self.assertEqual(reference_metric_names, model.metrics_names)

  @keras_parameterized.run_all_keras_modes
  def test_metric_state_reset_between_fit_and_evaluate(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='relu', input_dim=4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    acc_obj = metrics_module.BinaryAccuracy()
    model.compile(
        loss='mae',
        metrics=[acc_obj],
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x_train = np.random.random((100, 4))
    y_train = np.random.random((100, 1))
    model.fit(x_train, y_train, batch_size=5, epochs=2)
    self.assertEqual(self.evaluate(acc_obj.count), 100)

    x_test = np.random.random((10, 4))
    y_test = np.random.random((10, 1))
    model.evaluate(x_test, y_test, batch_size=5)
    self.assertEqual(self.evaluate(acc_obj.count), 10)

  @keras_parameterized.run_with_all_model_types(exclude_models=['sequential'])
  @keras_parameterized.run_all_keras_modes
  def test_metrics_valid_compile_input_formats(self):
    inp_1 = keras.layers.Input(shape=(1,), name='input_1')
    inp_2 = keras.layers.Input(shape=(1,), name='input_2')
    x = keras.layers.Dense(3, kernel_initializer='ones', trainable=False)
    out_1 = keras.layers.Dense(
        1, kernel_initializer='ones', name='output_1', trainable=False)
    out_2 = keras.layers.Dense(
        1, kernel_initializer='ones', name='output_2', trainable=False)

    branch_a = [inp_1, x, out_1]
    branch_b = [inp_2, x, out_2]
    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    # list of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[keras.metrics.MeanSquaredError()],
        weighted_metrics=[keras.metrics.MeanSquaredError()],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # list of list of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[
            keras.metrics.MeanSquaredError(),
            [keras.metrics.MeanSquaredError(),
             keras.metrics.Accuracy()]
        ],
        weighted_metrics=[
            keras.metrics.MeanSquaredError(),
            [keras.metrics.MeanSquaredError(),
             keras.metrics.Accuracy()]
        ],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # dict of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics={
            'output_1':
                keras.metrics.MeanSquaredError(),
            'output_2': [
                keras.metrics.MeanSquaredError(),
                keras.metrics.Accuracy()
            ],
        },
        weighted_metrics={
            'output_1':
                keras.metrics.MeanSquaredError(),
            'output_2': [
                keras.metrics.MeanSquaredError(),
                keras.metrics.Accuracy()
            ],
        },
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_invalid_metrics(self):
    num_classes = 5
    input_dim = 5

    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=num_classes, input_dim=input_dim)

    with self.assertRaisesRegexp(
        TypeError, 'Type of `metrics` argument not understood. '
        'Expected a list or dictionary, found: '):
      model.compile(
          RMSPropOptimizer(learning_rate=0.001),
          loss='categorical_crossentropy',
          metrics=metrics_module.CategoricalAccuracy(),
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    inp = keras.layers.Input(shape=(1,))
    x = keras.layers.Dense(3, activation='relu')(inp)
    out_1 = keras.layers.Dense(1, activation='sigmoid', name='output_1')(x)
    out_2 = keras.layers.Dense(1, activation='sigmoid', name='output_2')(x)
    model = keras.models.Model(inp, [out_1, out_2])
    with self.assertRaisesRegex(
        ValueError, 'When passing a list of lists as `metrics`, '
        'it should have one entry per model output. '
        'The model has 2 outputs, but you passed metrics='):
      model.compile('rmsprop', loss='mse', metrics=[['mse']])

    with self.assertRaisesRegex(
        ValueError,
        r'Unknown entries in metrics dictionary: \[\'output_3\'\]. Only '
        r'expected following keys: \[\'output_1\', \'output_2\'\]'):
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          metrics={
              'output_1': 'mse',
              'output_3': 'mse',
          },
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    with self.assertRaisesRegex(
        ValueError,
        r'Unknown entries in metrics dictionary: \[\'output_3\'\]. Only '
        r'expected following keys: \[\'output_1\', \'output_2\'\]'):
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          weighted_metrics={
              'output_1': 'mse',
              'output_3': 'mse',
          },
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

  @keras_parameterized.run_all_keras_modes
  def test_metrics_masking(self):
    np.random.seed(1337)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(1, kernel_initializer='ones')))
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='mse',
        weighted_metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # verify that masking is applied.
    x = np.array([[[1], [1]], [[1], [1]], [[0], [0]]])
    y = np.array([[[1], [1]], [[0], [1]], [[1], [1]]])
    scores = model.train_on_batch(x, y)
    self.assertArrayNear(scores, [0.25, 0.75], 0.1)

    # verify that masking is combined with sample weights.
    w = np.array([3, 2, 4])
    scores = model.train_on_batch(x, y, sample_weight=w)
    self.assertArrayNear(scores, [0.3328, 0.8], 0.001)

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_with_tensor_on_model(self):
    x = keras.layers.Input(shape=(1,))
    y = keras.layers.Dense(1, kernel_initializer='ones')(x)
    model = keras.models.Model(x, y)
    model.add_metric(
        math_ops.reduce_sum(y), name='metric_1', aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_2')

    with self.assertRaisesRegex(
        ValueError, 'Using the result of calling a `Metric` object '):
      with keras.backend.get_graph().as_default():
        model.add_metric(metrics_module.Mean(name='metric_2')(y))

    model.compile(
        'sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    inputs = np.ones(shape=(10, 1))
    targets = np.ones(shape=(10, 1))
    history = model.fit(
        inputs,
        targets,
        epochs=2,
        batch_size=5,
        validation_data=(inputs, targets))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertEqual(history.history['val_metric_1'][-1], 5)

    eval_results = model.evaluate(inputs, targets, batch_size=5)
    self.assertEqual(eval_results[-1], 5)

    model.predict(inputs, batch_size=5)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_in_model_call(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')

      def call(self, x):
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_2', aggregation='mean')
        # Provide same name as in the instance created in __init__
        # for eager mode
        self.add_metric(self.mean(x), name='metric_1')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertAlmostEqual(history.history['metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_2'][-1], 5, 0)
    self.assertAlmostEqual(history.history['val_metric_2'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    self.assertAlmostEqual(eval_results[1], 1, 0)
    self.assertAlmostEqual(eval_results[2], 5, 0)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_add_metric_in_layer_call(self):

    class TestLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.a = self.add_variable(
            'a', (1, 1), initializer='ones', trainable=False)
        self.built = True

      def call(self, inputs):
        self.add_metric(
            math_ops.reduce_sum(inputs), name='metric_1', aggregation='mean')
        return inputs + 1

    layers = [
        TestLayer(input_shape=(1,)),
        keras.layers.Dense(2, kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 5, 0)

  @keras_parameterized.run_all_keras_modes
  def test_model_metrics_list(self):

    class LayerWithAddMetric(keras.layers.Layer):

      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = keras.layers.Dense(1, kernel_initializer='ones')

      def __call__(self, inputs):
        outputs = self.dense(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='metric_1', aggregation='mean')
        return outputs

    class LayerWithNestedAddMetricLayer(keras.layers.Layer):

      def __init__(self):
        super(LayerWithNestedAddMetricLayer, self).__init__()
        self.layer = LayerWithAddMetric()

      def call(self, inputs):
        outputs = self.layer(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='metric_2', aggregation='mean')
        return outputs

    x = keras.layers.Input(shape=(1,))
    y = LayerWithNestedAddMetricLayer()(x)

    model = keras.models.Model(x, y)
    model.add_metric(
        math_ops.reduce_sum(y), name='metric_3', aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_4')

    with self.assertRaisesRegex(
        ValueError, 'Using the result of calling a `Metric` object '):
      with keras.backend.get_graph().as_default():
        model.add_metric(metrics_module.Mean(name='metric_4')(y))

    model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('metric_4')],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # Verify that the metrics added using `compile` and `add_metric` API are
    # included
    self.assertEqual([m.name for m in model._compile_metrics], ['metric_4'])
    self.assertEqual([m.name for m in model.metrics],
                     ['metric_4', 'metric_2', 'metric_1', 'metric_3'])

  @keras_parameterized.run_all_keras_modes
  def test_model_metrics_list_in_call(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_1', aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        metrics=[metrics_module.Accuracy('acc')],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    self.assertEqual([m.name for m in model._compile_metrics], ['acc'])
    self.assertEqual([m.name for m in model.metrics], ['acc', 'metric_1'])

  @keras_parameterized.run_all_keras_modes
  def test_multiple_add_metric_calls(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
        self.mean1 = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_2')

      def call(self, x):
        self.add_metric(self.mean2(x), name='metric_2')
        self.add_metric(self.mean1(x), name='metric_1')
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_3', aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertAlmostEqual(history.history['metric_1'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_2'][-1], 1, 0)
    self.assertAlmostEqual(history.history['metric_3'][-1], 5, 0)

    eval_results = model.evaluate(x, y, batch_size=5)
    self.assertArrayNear(eval_results[1:4], [1, 1, 5], 0.1)

    model.predict(x, batch_size=5)
    model.train_on_batch(x, y)
    model.test_on_batch(x, y)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_invalid_metric_tensor(self):

    class TestLayer(keras.layers.Layer):

      def build(self, input_shape):
        self.built = True

      def call(self, inputs):
        self.add_metric(math_ops.reduce_mean(inputs), name='metric_1')
        return inputs + 1

    layers = [TestLayer(input_shape=(1,))]
    layers.append(keras.layers.Dense(2, kernel_initializer='ones'))
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    with self.assertRaisesRegexp(
        ValueError,
        'We do not support adding an aggregated metric result tensor that is '
        'not the output of a `tf.keras.metrics.Metric` metric instance.'):
      model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
      model.compile(
          loss='mse',
          optimizer=RMSPropOptimizer(0.01),
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes
  def test_duplicate_metric_name_in_add_metric(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_1')

      def call(self, x):
        self.add_metric(self.mean(x), name='metric_1')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    with self.assertRaisesRegexp(
        ValueError,
        'Please provide different names for the metrics you have added. '
        'We found 2 metrics with the name: "metric_1"'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_without_name(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(math_ops.reduce_sum(x), aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    with self.assertRaisesRegex(ValueError,
                                'Please provide a name for your metric like'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_correctness(self):
    inputs = keras.Input(shape=(1,))
    targets = keras.Input(shape=(1,))

    class Bias(keras.layers.Layer):

      def build(self, input_shape):
        self.bias = self.add_variable('bias', (1,), initializer='zeros')
        self.mae = metrics_module.MeanAbsoluteError(name='mae_1')

      def call(self, inputs):
        inputs, targets = inputs
        outputs = inputs + self.bias
        self.add_metric(self.mae(targets, outputs), name='mae_1')
        return outputs

    outputs = Bias()([inputs, targets])
    model = keras.Model([inputs, targets], outputs)

    model.add_metric(
        metrics_module.mean_absolute_error(targets, outputs),
        name='mae_2',
        aggregation='mean')

    model.compile(
        loss='mae',
        optimizer=keras.optimizer_v2.gradient_descent.SGD(0.1),
        metrics=[metrics_module.MeanAbsoluteError(name='mae_3')],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    history = model.fit([x, y], y, batch_size=3, epochs=5)

    expected_val = [1., 0.9, 0.8, 0.7, 0.6]
    for key in ['loss', 'mae_1', 'mae_2', 'mae_3']:
      self.assertAllClose(history.history[key], expected_val, 1e-3)

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_order(self):

    class MyLayer(keras.layers.Layer):

      def call(self, inputs, training=None, mask=None):
        self.add_metric(
            array_ops.ones([32]) * 2.0, name='two', aggregation='mean')
        return inputs

    class MyModel(keras.Model):

      def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self._sampler = MyLayer(name='sampler')

      def call(self, inputs, training=None, mask=None):
        z = self._sampler(inputs)
        self.add_metric(
            array_ops.ones([32]) * 1.0, name='one', aggregation='mean')
        self.add_metric(
            array_ops.ones([32]) * 3.0, name='three', aggregation='mean')
        return z

    xdata = np.random.uniform(size=[32, 16]).astype(np.float32)
    dataset_train = dataset_ops.Dataset.from_tensor_slices((xdata, xdata))
    dataset_train = dataset_train.batch(32, drop_remainder=True)

    model = MyModel()
    model.compile(
        optimizer='sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    history = model.fit(dataset_train, epochs=3)
    self.assertDictEqual(
        history.history, {
            'loss': [0.0, 0.0, 0.0],
            'three': [3.0, 3.0, 3.0],
            'two': [2.0, 2.0, 2.0],
            'one': [1.0, 1.0, 1.0]
        })

  @keras_parameterized.run_all_keras_modes
  def test_model_with_nested_compiled_model(self):

    class LayerWithAddMetric(keras.layers.Layer):

      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = keras.layers.Dense(1, kernel_initializer='ones')

      def call(self, inputs):
        outputs = self.dense(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='mean', aggregation='mean')
        return outputs

    x = keras.layers.Input(shape=(1,))
    y = LayerWithAddMetric()(x)

    inner_model = keras.models.Model(x, y)
    inner_model.add_metric(
        math_ops.reduce_sum(y), name='mean1', aggregation='mean')

    inner_model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('acc')],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    self.assertEqual([m.name for m in inner_model.metrics],
                     ['acc', 'mean', 'mean1'])

    x = keras.layers.Input(shape=[1])
    y = inner_model(x)
    outer_model = keras.Model(x, y)
    outer_model.add_metric(
        math_ops.reduce_sum(y), name='mean2', aggregation='mean')

    outer_model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('acc2')],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    self.assertEqual([m.name for m in outer_model.metrics],
                     ['acc2', 'mean', 'mean1', 'mean2'])


class BareUpdateLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.counter = self.add_weight(
        'counter',
        dtype='int32',
        shape=(),
        initializer='zeros',
        trainable=False)

  def call(self, inputs):
    state_ops.assign_add(self.counter, 1)
    return math_ops.cast(self.counter, inputs.dtype) * inputs


class LambdaUpdateLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.counter = self.add_weight(
        'counter',
        dtype='int32',
        shape=(),
        initializer='zeros',
        trainable=False)

  def call(self, inputs):
    # Make sure update isn't run twice.
    self.add_update(lambda: state_ops.assign_add(self.counter, 1))
    return math_ops.cast(self.counter, inputs.dtype) * inputs


class NestedUpdateLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.layer = BareUpdateLayer()
    self.layer.build(input_shape)

  @property
  def counter(self):
    return self.layer.counter

  def call(self, inputs):
    return self.layer(inputs)


class SubgraphUpdateLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.counter = self.add_weight(
        'counter',
        dtype='int32',
        shape=(),
        initializer='zeros',
        trainable=False)

  def call(self, inputs, training=None):
    if training is None:
      training = keras.backend.learning_phase()

    if training:
      self.counter.assign(self.counter + 1)
    return inputs


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestAutoUpdates(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @parameterized.named_parameters(('bare_update', BareUpdateLayer()),
                                  ('lambda_update', LambdaUpdateLayer()),
                                  ('nested_update', NestedUpdateLayer()))
  def test_updates_in_model(self, layer):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, keras.layers.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types
  def test_lambda_updates_trainable_false(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    layer = LambdaUpdateLayer()
    model = testing_utils.get_model_from_layers(
        [layer, keras.layers.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)
    layer.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types
  def test_subgraph_updates_in_model(self):
    layer = SubgraphUpdateLayer()
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, keras.layers.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @parameterized.named_parameters(('bare_update', BareUpdateLayer()),
                                  ('lambda_update', LambdaUpdateLayer()),
                                  ('nested_update', NestedUpdateLayer()))
  def test_updates_standalone_layer(self, layer):
    y = layer(np.ones((10, 10)))
    self.evaluate(layer.counter.initializer)
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)

  def test_trainable_false_standalone_layer(self):
    layer = LambdaUpdateLayer()
    y = layer(np.ones((10, 10)))
    self.evaluate(layer.counter.initializer)
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)
    layer.trainable = False
    y = layer(np.ones((10, 10)))
    self.evaluate(y)
    self.assertEqual(self.evaluate(layer.counter), 1)

  @keras_parameterized.run_with_all_model_types
  def test_batchnorm_trainable_false(self):
    bn = keras.layers.BatchNormalization()
    model = testing_utils.get_model_from_layers([bn, keras.layers.Dense(1)],
                                                input_shape=(10,))
    bn.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertAllEqual(self.evaluate(bn.moving_mean), np.zeros((10,)))
    self.assertAllEqual(self.evaluate(bn.moving_variance), np.ones((10,)))


if __name__ == '__main__':
  test.main()
