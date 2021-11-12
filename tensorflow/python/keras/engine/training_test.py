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

import collections
import io
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers as layers_module
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training as training_module
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.rmsprop import RMSPropOptimizer

try:
  import scipy.sparse as scipy_sparse  # pylint: disable=g-import-not-at-top
except ImportError:
  scipy_sparse = None


class TrainingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types
  def test_model_instrumentation(self):
    layers = [
        layers_module.Dense(10, dtype=np.float64),
        layers_module.Dense(10, dtype=np.float64)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))

    self.assertTrue(model._instrumented_keras_api)
    self.assertTrue(model._instrumented_keras_model_class)
    self.assertFalse(model._instrumented_keras_layer_class)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_fit_training_arg(self):

    class ReturnTraining(layers_module.Layer):

      def call(self, inputs, training):
        if training:
          return inputs + array_ops.constant([100], 'float32')
        else:
          return inputs + array_ops.constant([0], 'float32')

    model = sequential.Sequential([ReturnTraining()])
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    hist = model.fit(x=np.array([0.]), y=np.array([0.]))
    self.assertAllClose(hist.history['loss'][0], 10000)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_fit_on_empty(self):
    model = sequential.Sequential([layers_module.Dense(1)])
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    with self.assertRaisesRegex(ValueError,
                                'Expect x to be a non-empty array or dataset.'):
      model.fit(x=np.array([]), y=np.array([]))

  @keras_parameterized.run_all_keras_modes
  def test_run_eagerly_setting(self):
    model = sequential.Sequential([layers_module.Dense(1)])
    run_eagerly = testing_utils.should_run_eagerly()
    model.compile('sgd', 'mse', run_eagerly=run_eagerly)
    self.assertEqual(model.run_eagerly, run_eagerly)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @parameterized.named_parameters(
      ('train_on_batch', 'train_on_batch'),
      ('test_on_batch', 'test_on_batch'),
      ('predict_on_batch', 'predict_on_batch'),
      ('fit', 'fit'),
      ('evaluate', 'evaluate'),
      ('predict', 'predict'),
  )
  def test_disallow_methods_inside_tf_function(self, method_name):
    model = sequential.Sequential([layers_module.Dense(1)])
    run_eagerly = testing_utils.should_run_eagerly()
    model.compile('sgd', 'mse', run_eagerly=run_eagerly)

    @def_function.function
    def my_fn():
      getattr(model, method_name)(1)

    error_msg = 'inside a `tf.function`'
    with self.assertRaisesRegex(RuntimeError, error_msg):
      my_fn()

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_learning_phase(self):

    class ReturnTraining(layers_module.Layer):

      def call(self, inputs):
        return backend.in_train_phase(lambda: array_ops.ones_like(inputs),
                                      lambda: array_ops.zeros_like(inputs))

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly())

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

    class ReturnTraining(layers_module.Layer):

      def call(self, inputs, training=None):
        return backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs),
            training=training)

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly())

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

    def loss_fn(labels, preds):
      self.assertEqual(labels.dtype, preds.dtype)
      return labels - preds

    layers = [
        layers_module.Dense(10, dtype=np.float64),
        layers_module.Dense(10, dtype=np.float64)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    inputs = np.ones(10, dtype=np.float64)
    targets = np.ones(10, dtype=np.float64)
    model.compile(
        'sgd',
        loss=loss_fn,
        run_eagerly=testing_utils.should_run_eagerly())
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)
    self.assertEqual(model.predict(inputs).dtype, np.float64)

  @keras_parameterized.run_all_keras_modes
  def test_fit_and_validate_nested_training_arg(self):

    class NestedReturnTraining(layers_module.Layer):

      def call(self, inputs, training=None):
        return backend.in_train_phase(
            lambda: array_ops.ones_like(inputs),
            lambda: array_ops.zeros_like(inputs),
            training=training)

    class ReturnTraining(layers_module.Layer):

      def __init__(self, input_shape=None, **kwargs):
        super(ReturnTraining, self).__init__(input_shape=input_shape, **kwargs)
        self._nested_layer = None

      def build(self, input_shape):
        self._nested_layer = NestedReturnTraining()
        self.built = True

      def call(self, inputs):
        return self._nested_layer(inputs)

    model = sequential.Sequential([ReturnTraining(input_shape=(2,))])
    model.compile(
        'sgd',
        loss='mae',
        run_eagerly=testing_utils.should_run_eagerly())

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
    input_a = layers_module.Input(shape=(3,), name='input_a')
    input_b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    dropout = layers_module.Dropout(0.5, name='dropout')
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
        run_eagerly=testing_utils.should_run_eagerly())

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
        run_eagerly=testing_utils.should_run_eagerly())
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
          run_eagerly=testing_utils.should_run_eagerly())
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Build single-input model
    x = layers_module.Input(shape=(3,), name='input_a')
    y = layers_module.Dense(4)(x)
    model = training_module.Model(x, y)
    model.compile(
        optimizer,
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    # This will work
    model.fit([input_a_np], output_d_np, epochs=1)

    # Test model on a list of floats
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 4))

    # Test execution on inputs that are lists of scalars.
    # TF2 and TF1 have slightly different semantics:
    if context.executing_eagerly():
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
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        loss_weights=loss_weights,
        sample_weight_mode=None,
        run_eagerly=testing_utils.should_run_eagerly())

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

  def _make_sequence_input_functions(self, input_type):
    # train and test
    xy_namedtuple = collections.namedtuple('xy_namedtuple', ['x', 'y'])

    # predict
    x_namedtuple = collections.namedtuple('x_namedtuple', ['x'])

    if input_type == 'dataset':
      dataset = dataset_ops.Dataset.range(16).map(
          lambda _: array_ops.ones(shape=(1,)))

      xy_dataset = dataset_ops.Dataset.zip((dataset, dataset)).batch(4)
      x_dataset = dataset.batch(4)
      def xy_function(use_namedtuple):
        return xy_dataset.map(xy_namedtuple) if use_namedtuple else xy_dataset

      def x_function(use_namedtuple):
        return x_dataset.map(x_namedtuple) if use_namedtuple else x_dataset

      return xy_function, x_function

    elif input_type == 'generator':
      def xy_generator(use_namedtuple):
        x, y = np.ones((4, 1)), np.ones((4, 1))
        for _ in range(4):
          if use_namedtuple:
            yield xy_namedtuple(x, y)
          else:
            yield x, y

      def x_generator(use_namedtuple):
        x = np.ones((4, 1))
        for _ in range(4):
          if use_namedtuple:
            yield x_namedtuple(x)
          else:
            yield x

      return xy_generator, x_generator

    elif input_type == 'sequence':
      class XYSequence(data_utils.Sequence):

        def __init__(self, use_namedtuple):
          self._use_namedtuple = use_namedtuple
          super(XYSequence, self).__init__()

        def __getitem__(self, idx):
          x, y = np.ones((4, 1)), np.ones((4, 1))
          if self._use_namedtuple:
            return xy_namedtuple(x, y)
          return x, y

        def __len__(self):
          return 4

      class XSequence(data_utils.Sequence):

        def __init__(self, use_namedtuple):
          self._use_namedtuple = use_namedtuple
          super(XSequence, self).__init__()

        def __getitem__(self, idx):
          x = np.ones((4, 1))
          if self._use_namedtuple:
            return x_namedtuple(x)
          return x

        def __len__(self):
          return 4

      return XYSequence, XSequence

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @keras_parameterized.run_with_all_model_types
  @parameterized.named_parameters(
      ('dataset', 'dataset'),
      ('generator', 'generator'),
      ('sequence', 'sequence'),
  )
  def test_sequence_input_types(self, input_type):
    """Ensure that namedtuples and tuples are plumbed identically."""
    if not context.executing_eagerly():
      self.skipTest('Improved checking is only present in data_adapter.')

    xy_function, x_function = self._make_sequence_input_functions(input_type)
    fit_kwargs, evaluate_kwargs, predict_kwargs = {}, {}, {}
    if input_type == 'generator':
      fit_kwargs['steps_per_epoch'] = 4
      evaluate_kwargs['steps'] = 4
      predict_kwargs['steps'] = 4

    model = testing_utils.get_small_mlp(1, 1, 1)
    model.compile(
        loss='mse',
        optimizer='sgd',
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit(xy_function(use_namedtuple=False), **fit_kwargs)
    model.evaluate(xy_function(use_namedtuple=False), **evaluate_kwargs)
    model.predict(x_function(use_namedtuple=False), **predict_kwargs)

  @keras_parameterized.run_all_keras_modes
  def test_custom_mapping_in_config(self):

    class MyModel(training_module.Model):

      def call(self, inputs):
        return inputs

      def get_config(self):
        self.a = {}
        return {'a': self.a}

    model = MyModel()
    self.assertIn('{"a": {}}', model.to_json())

  def test_training_on_sparse_data_with_dense_placeholders_v1(self):
    with ops.Graph().as_default():
      if scipy_sparse is None:
        return

      test_inputs = [
          scipy_sparse.random(6, 3, density=0.25).tocsr() for _ in range(2)
      ]
      test_outputs = [
          scipy_sparse.random(6, i, density=0.25).tocsr() for i in range(3, 5)
      ]
      in1 = layers_module.Input(shape=(3,))
      in2 = layers_module.Input(shape=(3,))
      out1 = layers_module.Dropout(0.5, name='dropout')(in1)
      out2 = layers_module.Dense(4, name='dense_1')(in2)
      model = training_module.Model([in1, in2], [out1, out2])
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
    inputs = layers_module.Input(shape=(10,), sparse=True)
    weights = variables_module.Variable(
        np.ones((10, 1)).astype(np.float32), name='weights')
    weights_mult = lambda x: sparse_ops.sparse_tensor_dense_matmul(x, weights)
    output_layer = layers_module.Lambda(weights_mult)(inputs)
    model = training_module.Model([inputs], output_layer)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())

  @keras_parameterized.run_all_keras_modes
  def test_weight_deduplication(self):

    class WatchingLayer(layers_module.Layer):

      def __init__(self, dense_to_track):
        # This will cause the kernel and bias to be double counted, effectively
        # doubling the learning rate if weights are not deduped.
        self._kernel = dense_to_track.kernel
        self._bias = dense_to_track.bias
        super(WatchingLayer, self).__init__()

    inp = layers_module.Input(shape=(1,))
    dense_layer = layers_module.Dense(1)
    dense_output = dense_layer(inp)  # This will build the dense kernel

    # Deterministically set weights to make the test repeatable.
    dense_layer.set_weights([np.ones((1, 1)), np.zeros((1,))])
    output = WatchingLayer(dense_layer)(dense_output)

    model = training_module.Model(inp, output)

    # 0.25 is the edge of the radius of convergence for the double apply case.
    # At lr=0.24, the double apply case will very slowly descend while the
    # correct case will drop very quickly.
    model.compile(
        loss='mse',
        optimizer=optimizer_v2.gradient_descent.SGD(0.24),
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones((64 * 2,))
    y = 4.5 * x - 3.

    history = model.fit(x, y, batch_size=64, epochs=2, verbose=2)

    # If the gradient apply is duplicated then the loss after 2 epochs will
    # be ~0.15, compared to the correct answer of O(1e-7).
    self.assertLess(history.history['loss'][-1], 1e-6)

  @keras_parameterized.run_all_keras_modes
  def test_weight_shared_across_layers(self):

    class AddWeightLayer(layers_module.Layer):

      def __init__(self, trainable_var, non_trainable_var):
        self.trainable_var = trainable_var
        self.non_trainable_var = non_trainable_var
        super(AddWeightLayer, self).__init__()

      def call(self, inputs):
        return inputs + self.trainable_var

    class LayerWithWeightSharedLayers(layers_module.Layer):

      def __init__(self):
        super(LayerWithWeightSharedLayers, self).__init__()
        shared_trainable_var = variables_module.Variable(1.)
        shared_non_trainable_var = variables_module.Variable(
            1., trainable=False)
        self.layer1 = AddWeightLayer(shared_trainable_var,
                                     shared_non_trainable_var)
        self.layer2 = AddWeightLayer(shared_trainable_var,
                                     shared_non_trainable_var)

      def call(self, inputs):
        return self.layer2(self.layer1(inputs))

    l = LayerWithWeightSharedLayers()
    layers = list(l._flatten_layers(include_self=False, recursive=False))
    self.assertEqual(layers, [l.layer1, l.layer2])
    self.assertEqual(l.variables,
                     [l.layer1.trainable_var, l.layer1.non_trainable_var])
    self.assertEqual(l.trainable_variables, [l.layer1.trainable_var])
    self.assertEqual(l.non_trainable_variables, [l.layer1.non_trainable_var])
    self.assertLen(l.get_weights(), 2)

  @keras_parameterized.run_all_keras_modes
  def test_weight_tracking_for_template(self):
    def variable_scoped_function(trainable=True):
      return variable_scope.get_variable(
          'dummy', shape=[1], trainable=trainable,
          initializer=init_ops.zeros_initializer())
    def nested_template():
      nested1 = template.make_template('nested', variable_scoped_function)
      nested2 = template.make_template('nested', variable_scoped_function)
      v1 = nested1()
      v2 = nested2()

      # nested1 and nested2 should not share variables
      self.assertIsNot(v1, v2)

      # Variables created by nested1 should be isolated from variables
      # created by nested2.
      self.assertEqual(1, len(nested1.variables))
      self.assertEqual(1, len(nested2.variables))
      self.assertIs(nested1.variables[0], v1)
      self.assertIs(nested2.variables[0], v2)
      self.assertEqual(1, len(nested1.trainable_variables))
      self.assertEqual(1, len(nested2.trainable_variables))
      self.assertIs(nested1.trainable_variables[0], v1)
      self.assertIs(nested2.trainable_variables[0], v2)
      self.assertEqual(len(nested1.non_trainable_variables), 0)
      self.assertEqual(len(nested2.non_trainable_variables), 0)
      return v1, v2

    tmpl1 = template.make_template('s1', nested_template)
    tmpl2 = template.make_template('s1', nested_template)

    v1, v2 = tmpl1()
    v5, v6 = tmpl2()

    model = training_module.Model()
    model.template = tmpl1
    self.assertEqual(2, len(model.variables))
    self.assertIs(model.variables[0], v1)
    self.assertIs(model.variables[1], v2)
    self.assertEqual(2, len(model.variables))
    self.assertIs(model.trainable_variables[0], v1)
    self.assertIs(model.trainable_variables[1], v2)
    self.assertEqual(len(model.non_trainable_variables), 0)
    model.templates = [tmpl2]
    for v, w in zip(model.variables, [v1, v2, v5, v6]):
      self.assertIs(v, w)
    for v, w in zip(model.trainable_variables, [v1, v2, v5, v6]):
      self.assertIs(v, w)
    self.assertEqual(len(model.non_trainable_variables), 0)
    # Make sure losses, layers, and updates aren't broken by having a Template
    # in the mix, which does not expose any updates or losses.
    self.assertEqual([], model.layers)
    self.assertEqual([], model.updates)
    self.assertEqual([], model.losses)
    self.assertEqual([], model.templates.layers)
    self.assertEqual([], model.templates.updates)
    self.assertEqual([], model.templates.losses)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_logs_passed_to_callbacks(self):
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
        optimizer=RMSPropOptimizer(learning_rate=0.01),
        run_eagerly=testing_utils.should_run_eagerly())

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

    self.assertSetEqual(
        set(test_callback.batch_end_logs.keys()), set(['acc', 'loss', 'mae']))
    self.assertSetEqual(
        set(test_callback.epoch_end_logs.keys()),
        set(['acc', 'loss', 'mae', 'val_acc', 'val_loss', 'val_mae']))

  @keras_parameterized.run_all_keras_modes
  def test_mismatched_output_shape_and_target_shape(self):
    model = sequential.Sequential([
        layers_module.Dense(2, input_shape=(3, 4)),
        layers_module.Dense(5),
    ])
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    # Test with Numpy data
    x_train = np.random.random((10, 3, 4)).astype(np.float32)
    y_train = np.random.randint(0, 5, size=(10, 3)).astype(np.float32)
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

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_losses_in_defun(self):
    layer = layers_module.Dense(1, kernel_regularizer='l1')
    layer(array_ops.ones([1, 10]))

    @def_function.function
    def get_losses():
      return layer.losses

    self.assertAllEqual(
        self.evaluate(layer.losses), self.evaluate(get_losses()))

  @keras_parameterized.run_all_keras_modes
  def test_logging(self):
    mock_stdout = io.StringIO()
    model = sequential.Sequential()
    model.add(layers_module.Dense(10, activation='relu'))
    model.add(layers_module.Dense(1, activation='sigmoid'))
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      model.fit(
          np.ones((10, 10), 'float32'), np.ones((10, 1), 'float32'), epochs=10)
    self.assertTrue('Epoch 5/10' in mock_stdout.getvalue())

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_training_with_loss_instance(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])
    loss_weights = [1., 0.5]
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss=losses.MeanSquaredError(),
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights)

    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 3))

    output_d_np = np.random.random((10, 4))
    output_e_np = np.random.random((10, 4))

    model.fit([input_a_np, input_b_np], [output_d_np, output_e_np],
              epochs=1,
              batch_size=5)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_static_batch_in_input_layer(self):
    if context.executing_eagerly():
      self.skipTest('Not inferred in eager.')

    class Counter(Callback):

      def __init__(self):
        self.batches = 0

      def on_batch_end(self, batch, logs=None):
        self.batches += 1

    x, y = np.ones((64, 10), 'float32'), np.ones((64, 1), 'float32')

    for batch_size, expected_batches in [(None, 2), (4, 16)]:
      inputs = input_layer.Input(batch_size=batch_size, shape=(10,))
      outputs = layers_module.Dense(1, activation='sigmoid')(inputs)
      model = training_module.Model(inputs, outputs)

      model.compile(optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
      counter = Counter()
      model.fit(x, y, callbacks=[counter])
      self.assertEqual(counter.batches, expected_batches)

      model = sequential.Sequential(
          [layers_module.Dense(1, batch_input_shape=(batch_size, 10))])
      model.compile(optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
      counter = Counter()
      model.fit(x, y, callbacks=[counter])
      self.assertEqual(counter.batches, expected_batches)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_static_batch_in_input_layer_consistency_checks(self):
    if context.executing_eagerly():
      self.skipTest('Not inferred in eager.')
    x, y = np.ones((64, 10), 'float32'), np.ones((64, 1), 'float32')

    inputs = input_layer.Input(batch_size=2, shape=(10,))
    outputs = layers_module.Dense(1, activation='sigmoid')(inputs)
    model = training_module.Model(inputs, outputs)
    model.compile(optimizer_v2.adam.Adam(0.001), 'binary_crossentropy')
    with self.assertRaisesRegex(ValueError,
                                'incompatible with the specified batch size'):
      model.fit(x, y, batch_size=4)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_compatible_batch_size_functional_model(self):

    class MyLayer(layers_module.Layer):

      def call(self, inputs):
        return array_ops.concat(inputs, axis=0)

    input1 = input_layer.Input(batch_size=2, shape=(10,))
    input2 = input_layer.Input(batch_size=3, shape=(10,))
    outputs = MyLayer()([input1, input2])
    with self.assertRaisesRegex(ValueError,
                                'specified batch sizes of the Input Layers'):
      training_module.Model([input1, input2], outputs)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_calling_subclass_model_on_different_datasets(self):

    class SubclassedModel(training_module.Model):

      def call(self, inputs):
        return inputs * 2

    model = SubclassedModel()
    dataset_one = dataset_ops.Dataset.range(2).batch(2)
    dataset_two = dataset_ops.Dataset.range(3, 10).batch(2)
    self.assertAllEqual([[0], [2]], model.predict(dataset_one, steps=1))
    self.assertAllEqual([[6], [8], [10], [12]],
                        model.predict(dataset_two, steps=2))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_training_on_sparse_categorical_crossentropy_loss_with_softmax(self):
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

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_training_on_categorical_crossentropy_loss_with_softmax(self):
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

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_training_on_binary_crossentropy_loss(self):
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
        run_eagerly=testing_utils.should_run_eagerly())

    class ValCounter(Callback):

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
    if context.executing_eagerly():
      self.skipTest('Check removed in new `fit`')
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_small_mlp(2, 1, 10)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

    with self.assertRaisesRegex(
        ValueError, '`validation_steps` should not be specified if '
        '`validation_data` is None.'):
      model.fit(x, y, epochs=4, validation_data=None, validation_steps=3)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_layer_with_variable_output(self):

    class VariableOutputLayer(layers_module.Layer):

      def build(self, input_shape):
        self.v = self.add_weight('output_var', shape=(2, 5), initializer='ones')

      def call(self, inputs):
        return self.v

    model = testing_utils.get_model_from_layers(
        [VariableOutputLayer(), layers_module.Dense(1)], input_shape=(10,))
    # TODO(omalleyt): Make this work with `run_eagerly=True`.
    model.compile('sgd', 'mse', run_eagerly=False)
    model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2, epochs=5)

    self.assertLen(model.trainable_variables, 3)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_model_dtype(self):

    class AssertTypeLayer(layers_module.Layer):

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
          run_eagerly=testing_utils.should_run_eagerly())

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
        run_eagerly=testing_utils.should_run_eagerly())
    x = np.ones((10, 10)).astype(np.float64)
    y = np.ones((10, 10)).astype(np.float64)
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
    model.fit(dataset)
    self.assertEqual(model._compute_dtype, 'float32')

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_subclassed_model_with_training_arg(self):

    class LayerWithTrainingArg(layers_module.Layer):

      def call(self, inputs, training=None):
        self.training = training
        return inputs

    class ModelWithTrainingArg(training_module.Model):

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
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, x, epochs=1)

    if context.executing_eagerly():
      expected_training_arg = True
    else:
      expected_training_arg = backend.symbolic_learning_phase()

    self.assertIs(model.training, expected_training_arg)
    self.assertIs(model.l1.training, expected_training_arg)

  @keras_parameterized.run_all_keras_modes
  def test_error_when_model_is_not_compiled(self):
    inputs = input_layer.Input(shape=(1,))
    outputs = layers_module.Dense(1)(inputs)
    model = training_module.Model(inputs, outputs)
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.ones((1, 1)), np.ones((1, 1)))

    class MyModel(training_module.Model):

      def call(self, x):
        self.add_loss(math_ops.reduce_sum(x))
        return x

    model = MyModel()
    with self.assertRaisesRegex(RuntimeError, 'must compile your model'):
      model.fit(np.random.random((32, 1)), epochs=2)

  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_losses_of_different_dtypes(self):
    inp = input_layer.Input(shape=(2,))
    out_1 = layers_module.Dense(
        2, dtype='float32', kernel_regularizer='l2')(
            inp)
    out_2 = layers_module.Dense(
        2, dtype='float16', kernel_regularizer='l2')(
            inp)
    model = training_module.Model(inp, [out_1, out_2])
    extra_loss = math_ops.reduce_sum(math_ops.cast(out_2, 'float64'))
    model.add_loss(extra_loss)
    model.compile('sgd', ['mse', 'mse'],
                  run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 2)), np.ones((10, 2))
    model.fit(x, [y, y])

  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_losses_of_different_dtypes_with_subclassed_model(self):

    class MyModel(training_module.Model):

      def build(self, _):
        self.dense = layers_module.Dense(2)

      def call(self, inputs):
        self.add_loss(math_ops.cast(nn_ops.l2_loss(inputs), 'float64'))
        return self.dense(inputs)

    model = MyModel(dtype='float32')
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 2)), np.ones((10, 2))
    model.fit(x, y)

  @keras_parameterized.run_all_keras_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_regularizer_of_different_dtype(self):
    inp = input_layer.Input(shape=(2,))

    def regularizer(weight):
      return math_ops.cast(nn_ops.l2_loss(weight), 'float64')

    out = layers_module.Dense(
        2, dtype='float32', kernel_regularizer=regularizer)(
            inp)
    model = training_module.Model(inp, out)
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 2)), np.ones((10, 2))
    model.fit(x, y)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_outputs_are_floats(self):
    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = sequential.Sequential([layers_module.Dense(1)])
    model.compile('sgd', 'mse', metrics=['accuracy'],
                  run_eagerly=testing_utils.should_run_eagerly())

    history = model.fit(x, y, epochs=2)
    self.assertIsInstance(history.history['loss'][0], float)
    self.assertIsInstance(history.history['accuracy'][0], float)

    loss, accuracy = model.train_on_batch(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

    loss, accuracy = model.evaluate(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

    loss, accuracy = model.test_on_batch(x, y)
    self.assertIsInstance(loss, float)
    self.assertIsInstance(accuracy, float)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_int_output(self):
    x, y = np.ones((10, 1)), np.ones((10, 1))
    model = sequential.Sequential([layers_module.Dense(1)])

    class MyMetric(metrics_module.Metric):

      def update_state(self, y_true, y_pred, sample_weight=None):
        del y_true, y_pred, sample_weight

      def result(self):
        return array_ops.constant(1, dtype='int64')

    model.compile('sgd', 'mse', metrics=[MyMetric()],
                  run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x, y, epochs=2)
    self.assertIsInstance(history.history['my_metric'][0], int)

  @keras_parameterized.run_all_keras_modes
  def test_calling_aggregate_gradient(self):

    class _Optimizer(optimizer_v2.gradient_descent.SGD):
      """Mock optimizer to check if _aggregate_gradient is called."""

      _HAS_AGGREGATE_GRAD = True

      def __init__(self):
        self.aggregate_gradients_called = False
        super(_Optimizer, self).__init__(name='MyOptimizer')

      def _aggregate_gradients(self, grads):
        self.aggregate_gradients_called = True
        return super(_Optimizer, self)._aggregate_gradients(grads)

    mock_optimizer = _Optimizer()

    model = sequential.Sequential()
    model.add(layers_module.Dense(10, activation='relu'))

    model.compile(mock_optimizer, 'mse',
                  run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    model.fit(x, y)
    self.assertEqual(model.optimizer.aggregate_gradients_called, True)

    class _OptimizerOverrideApplyGradients(_Optimizer):
      """Override apply_gradients.

      To test the case where the optimizer does not define the
      experimental_aggregate_gradients parameter.
      """

      _HAS_AGGREGATE_GRAD = False

      def apply_gradients(self, grads_and_vars, name=None):  # pylint: disable=useless-super-delegation
        return super(_OptimizerOverrideApplyGradients,
                     self).apply_gradients(grads_and_vars, name)

    mock_optimizer = _OptimizerOverrideApplyGradients()
    model.compile(mock_optimizer, 'mse',
                  run_eagerly=testing_utils.should_run_eagerly())
    x, y = np.ones((10, 10)), np.ones((10, 10))
    model.fit(x, y)
    self.assertEqual(model.optimizer.aggregate_gradients_called, True)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_gradients_are_none(self):

    class DenseWithExtraWeight(layers_module.Dense):

      def build(self, input_shape):
        # Gradients w.r.t. extra_weights are None
        self.extra_weight_1 = self.add_weight('extra_weight_1', shape=(),
                                              initializer='ones')
        super(DenseWithExtraWeight, self).build(input_shape)
        self.extra_weight_2 = self.add_weight('extra_weight_2', shape=(),
                                              initializer='ones')

    model = sequential.Sequential([DenseWithExtraWeight(4, input_shape=(4,))])
    # Test clipping can handle None gradients
    opt = optimizer_v2.adam.Adam(clipnorm=1.0, clipvalue=1.0)
    model.compile(opt, 'mse', run_eagerly=testing_utils.should_run_eagerly())
    inputs = np.random.normal(size=(64, 4))
    targets = np.random.normal(size=(64, 4))
    old_kernel = model.get_weights()[1]
    model.fit(inputs, targets)
    new_kernel = model.get_weights()[1]
    self.assertNotAllEqual(old_kernel, new_kernel)

  @keras_parameterized.run_all_keras_modes
  def test_layer_ordering(self):

    class MyLayer(layers_module.Layer):
      pass

    class MyModel(training_module.Model):

      def __init__(self, name):
        super(MyModel, self).__init__(name=name)

        self.weight = variables_module.Variable(0, name=name)

        self.direct_sublayer = MyLayer(name='direct')
        self.direct_sublayer.d = {'d': MyLayer(name='direct/dict')}

        self.dict_sublayer = {'d': MyLayer(name='dict')}
        self.dict_sublayer['d'].direct = MyLayer(name='dict/direct')

    model = MyModel('model')
    # All sublayers, including self and recursive sublayers.
    self.assertEqual(['model', 'direct', 'direct/dict', 'dict', 'dict/direct'],
                     [l.name for l in model._flatten_layers()])
    # Only direct sublayers, including those in data structures.
    self.assertEqual(['direct', 'dict'], [l.name for l in model.layers])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_trainable_state_setting(self):

    class UpdateLayer(layers_module.Layer):

      def __init__(self):
        super(UpdateLayer, self).__init__()
        self.v = variables_module.Variable(0., trainable=False)

      def call(self, x):
        self.add_update(lambda: self.v.assign_add(1.))
        return x * self.v

    layer = UpdateLayer()
    model_with_updates = sequential.Sequential([layer])
    model_with_updates.compile(
        'sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    layer.trainable = False
    model_without_updates = sequential.Sequential([layer])
    model_without_updates.compile(
        'sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    x, y = np.ones((10, 1)), np.ones((10, 1))

    self.assertEqual(self.evaluate(layer.v), 0.)
    model_with_updates.fit(x, y, batch_size=10)
    # assign_add called.
    self.assertEqual(self.evaluate(layer.v), 1.)
    model_without_updates.fit(x, y, batch_size=10)
    # assign_add not called.
    self.assertEqual(self.evaluate(layer.v), 1.)

  @keras_parameterized.run_all_keras_modes(
      always_skip_v1=True)
  @parameterized.named_parameters(
      ('numpy_array', 'numpy_array'),
      ('dataset_array', 'dataset_array'),
      ('dataset_dict', 'dataset_dict'))
  def test_single_input_no_tuple_wrapping(self, input_type):
    x = np.ones((10, 1))

    if input_type == 'numpy_array':
      batch_size = 3
      expected_data_type = ops.Tensor
    elif input_type == 'dataset_array':
      x = dataset_ops.Dataset.from_tensor_slices(x).batch(3)
      batch_size = None
      expected_data_type = ops.Tensor
    else:
      x = {'my_input': x}
      x = dataset_ops.Dataset.from_tensor_slices(x).batch(3)
      batch_size = None
      expected_data_type = dict

    test_case = self

    class MyModel(training_module.Model):

      def train_step(self, data):
        # No tuple wrapping for single x input and no targets.
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).train_step(data)

      def test_step(self, data):
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).test_step(data)

      def predict_step(self, data):
        test_case.assertIsInstance(data, expected_data_type)
        return super(MyModel, self).predict_step(data)

    inputs = layers_module.Input(shape=(1,), name='my_input')
    outputs = layers_module.Dense(1)(inputs)
    model = MyModel(inputs, outputs)
    model.add_loss(math_ops.reduce_sum(outputs))
    model.compile('sgd', 'mse')
    model.fit(x, batch_size=batch_size)
    model.evaluate(x, batch_size=batch_size)
    model.predict(x, batch_size=batch_size)

  @keras_parameterized.run_all_keras_modes(
      always_skip_v1=True)
  @parameterized.named_parameters(
      ('custom_metrics', False, True),
      ('compiled_metrics', True, False),
      ('both_compiled_and_custom_metrics', True, True))
  def test_evaluate_with_custom_test_step(
      self, use_compiled_metrics, use_custom_metrics):

    class MyModel(training_module.Model):

      def test_step(self, data):
        x, y = data
        pred = self(x)
        metrics = {}
        if use_compiled_metrics:
          self.compiled_metrics.update_state(y, pred)
          self.compiled_loss(y, pred)
          for metric in self.metrics:
            metrics[metric.name] = metric.result()
        if use_custom_metrics:
          custom_metrics = {
              'mean': math_ops.reduce_mean(pred),
              'sum': math_ops.reduce_sum(pred)
          }
          metrics.update(custom_metrics)
        return metrics

    inputs = layers_module.Input((2,))
    outputs = layers_module.Dense(3)(inputs)
    model = MyModel(inputs, outputs)
    if use_compiled_metrics:
      model.compile('adam', 'mse', metrics=['mae', 'mape'],
                    run_eagerly=testing_utils.should_run_eagerly())
    else:
      model.compile('adam', 'mse',
                    run_eagerly=testing_utils.should_run_eagerly())
    x = np.random.random((4, 2))
    y = np.random.random((4, 3))
    results_list = model.evaluate(x, y)
    results_dict = model.evaluate(x, y, return_dict=True)
    self.assertLen(results_list, len(results_dict))
    if use_compiled_metrics and use_custom_metrics:
      self.assertLen(results_list, 5)
      self.assertEqual(results_list,
                       [results_dict['loss'],
                        results_dict['mae'], results_dict['mape'],
                        results_dict['mean'], results_dict['sum']])
    if use_compiled_metrics and not use_custom_metrics:
      self.assertLen(results_list, 3)
      self.assertEqual(results_list,
                       [results_dict['loss'],
                        results_dict['mae'], results_dict['mape']])
    if not use_compiled_metrics and use_custom_metrics:
      self.assertLen(results_list, 2)
      self.assertEqual(results_list,
                       [results_dict['mean'], results_dict['sum']])


class TestExceptionsAndWarnings(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_compile_warning_for_loss_missing_output(self):
    with self.cached_session():
      inp = layers_module.Input(shape=(16,), name='input_a')
      out_1 = layers_module.Dense(8, name='dense_1')(inp)
      out_2 = layers_module.Dense(
          3, activation='softmax', name='dense_2')(
              out_1)
      model = training_module.Model(inputs=[inp], outputs=[out_1, out_2])
      optimizer = RMSPropOptimizer(learning_rate=0.001)

      model.compile(
          optimizer,
          loss={
              'dense_2': 'categorical_crossentropy',
          },
          metrics={
              'dense_2': 'categorical_accuracy',
              'dense_1': metrics_module.CategoricalAccuracy(),
          },
          run_eagerly=testing_utils.should_run_eagerly())

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_predict_error_with_empty_x(self):
    inputs = layers_module.Input(shape=(2,))
    outputs = layers_module.Dense(4)(inputs)
    model = training_module.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse')

    with self.assertRaisesRegex(ValueError,
                                'Expect x to be a non-empty array or dataset.'):
      model.predict(np.array([]))

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_on_batch_error_inconsistent_batch_size(self):
    input_node1 = layers_module.Input(shape=(5,))
    input_node2 = layers_module.Input(shape=(5,))
    output_node = layers_module.Concatenate()([input_node1, input_node2])
    output_node = layers_module.Dense(4)(output_node)
    model = training_module.Model([input_node1, input_node2], output_node)
    model.compile(loss='mse')

    with self.assertRaisesRegex(ValueError, 'Data cardinality is ambiguous'):
      model.train_on_batch([np.ones((10, 5)), np.ones((10, 5))],
                           np.ones((11, 4)))

    with self.assertRaisesRegex(ValueError, 'Data cardinality is ambiguous'):
      model.test_on_batch([np.ones((10, 5)), np.ones((10, 5))],
                          np.ones((11, 4)))

    with self.assertRaisesRegex(ValueError, 'Data cardinality is ambiguous'):
      model.predict_on_batch([np.ones((10, 5)), np.ones((11, 5))])


class LossWeightingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 10
    weighted_class = 3
    weight = .5
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
        run_eagerly=testing_utils.should_run_eagerly())

    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_test = y_test.copy()
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
    ref_score = model.evaluate(x_test, y_test, verbose=0)  # pylint: disable=unused-variable
    score = model.evaluate(  # pylint: disable=unused-variable
        x_test[test_ids, :], y_test[test_ids, :], verbose=0)
    # TODO(b/152990697): Fix the class weights test here.
    # self.assertLess(score[0], ref_score[0])

  @keras_parameterized.run_all_keras_modes
  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  def test_fit_with_incorrect_weights(self):
    input_a = layers_module.Input(shape=(3,), name='input_a')
    input_b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(2, name='output_1')
    dropout = layers_module.Dropout(0.5, name='output_2')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)
    model.compile(
        optimizer='adam',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    x = np.random.random((10, 3))
    y = np.random.random((10, 2))

    with self.assertRaises(ValueError):
      model.fit([x, x], [y, y], epochs=1, sample_weight={'unknown': x})

    with self.assertRaises(ValueError):
      model.fit([x, x], [y, y], epochs=1, class_weight={'unknown': 1})

  def test_sample_weight_tensor(self):
    """Tests that sample weight may be defined as a tensor in the graph."""
    with ops.get_default_graph().as_default():
      # Create a simple pass-through model
      inputs = layers_module.Input(shape=1, name='input_layer')
      model = training_module.Model(inputs=inputs, outputs=inputs)
      model.compile(
          loss='mean_absolute_error',
          optimizer='adam')

      # Prepare sample weights iterator tensor
      sample_weights = array_ops.constant(
          [[0, .4, 1, 1], [2, .4, .3, 1]])
      dataset = dataset_ops.Dataset.from_tensor_slices(sample_weights)
      sample_weights = dataset_ops.make_one_shot_iterator(dataset).get_next()
      sample_weights = training_utils_v1.standardize_sample_weights(
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


@keras_parameterized.run_all_keras_modes
class MaskingTest(keras_parameterized.TestCase):

  def test_mask_argument_in_layer(self):
    # Test that the mask argument gets correctly passed to a layer in the
    # functional API.

    class CustomMaskedLayer(layers_module.Layer):

      def __init__(self):
        super(CustomMaskedLayer, self).__init__()
        self.supports_masking = True

      def call(self, inputs, mask=None):
        assert mask is not None
        return inputs

      def compute_output_shape(self, input_shape):
        return input_shape

    x = np.random.random((5, 3))
    inputs = layers_module.Input((3,))
    masked = layers_module.Masking(mask_value=0)(inputs)
    outputs = CustomMaskedLayer()(masked)

    model = training_module.Model(inputs, outputs)
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly())
    y = np.random.random((5, 3))
    model.train_on_batch(x, y)


@keras_parameterized.run_all_keras_modes
class TestDynamicTrainability(keras_parameterized.TestCase):

  def test_trainable_warning(self):
    x = np.random.random((5, 3))
    y = np.random.random((5, 2))

    model = sequential.Sequential()
    model.add(layers_module.Dense(2, input_dim=3))
    model.trainable = False
    model.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.trainable = True
    model.train_on_batch(x, y)
    self.assertRaises(Warning)

  def test_trainable_argument(self):
    with self.cached_session():
      x = np.random.random((5, 3))
      y = np.random.random((5, 2))

      model = sequential.Sequential()
      model.add(layers_module.Dense(2, input_dim=3, trainable=False))
      model.compile(
          'rmsprop',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly())
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

      # test with nesting
      inputs = layers_module.Input(shape=(3,))
      output = model(inputs)
      model = training_module.Model(inputs, output)
      model.compile(
          'rmsprop',
          'mse',
          run_eagerly=testing_utils.should_run_eagerly())
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

  def test_layer_trainability_switch(self):
    # with constructor argument, in Sequential
    model = sequential.Sequential()
    model.add(layers_module.Dense(2, trainable=False, input_dim=1))
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Sequential
    model = sequential.Sequential()
    layer = layers_module.Dense(2, input_dim=1)
    model.add(layer)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # with constructor argument, in Model
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2, trainable=False)(x)
    model = training_module.Model(x, y)
    self.assertListEqual(model.trainable_weights, [])

    # by setting the `trainable` argument, in Model
    x = layers_module.Input(shape=(1,))
    layer = layers_module.Dense(2)
    y = layer(x)
    model = training_module.Model(x, y)
    self.assertListEqual(model.trainable_weights, layer.trainable_weights)
    layer.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_model_trainability_switch(self):
    # a non-trainable model has no trainable weights
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    model = training_module.Model(x, y)
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

    # same for Sequential
    model = sequential.Sequential()
    model.add(layers_module.Dense(2, input_dim=1))
    model.trainable = False
    self.assertListEqual(model.trainable_weights, [])

  def test_nested_model_trainability(self):
    # a Sequential inside a Model
    inner_model = sequential.Sequential()
    inner_model.add(layers_module.Dense(2, input_dim=1))

    x = layers_module.Input(shape=(1,))
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Sequential inside a Sequential
    inner_model = sequential.Sequential()
    inner_model.add(layers_module.Dense(2, input_dim=1))
    outer_model = sequential.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Model
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    inner_model = training_module.Model(x, y)
    x = layers_module.Input(shape=(1,))
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

    # a Model inside a Sequential
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(2)(x)
    inner_model = training_module.Model(x, y)
    outer_model = sequential.Sequential()
    outer_model.add(inner_model)
    self.assertListEqual(outer_model.trainable_weights,
                         inner_model.trainable_weights)
    inner_model.trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])
    inner_model.trainable = True
    inner_model.layers[-1].trainable = False
    self.assertListEqual(outer_model.trainable_weights, [])

  def test_toggle_value(self):
    input_0 = layers_module.Input(shape=(1,))
    dense_0 = layers_module.Dense(
        1, kernel_initializer='ones', bias_initializer='ones')
    dense_1 = layers_module.Dense(
        1, kernel_initializer='ones', bias_initializer='ones')
    result = layers_module.Add()([dense_0(input_0), dense_1(input_0)])
    model = training_module.Model(input_0, result)
    dense_0.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

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

  def test_training_and_eval_methods_on_symbolic_tensors_single_io(self):
    with ops.Graph().as_default():
      x = layers_module.Input(shape=(3,), name='input')
      y = layers_module.Dense(4, name='dense')(x)
      model = training_module.Model(x, y)

      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      model.compile(
          optimizer,
          loss,
          metrics=['mae', metrics_module.CategoricalAccuracy()])

      inputs = backend.zeros(shape=(10, 3))
      targets = backend.zeros(shape=(10, 4))

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

  def test_training_and_eval_methods_on_symbolic_tensors_multi_io(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(
        optimizer,
        loss,
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        loss_weights=loss_weights)

    input_a_tf = array_ops.zeros(shape=(10, 3))
    input_b_tf = array_ops.zeros(shape=(10, 3))

    output_d_tf = array_ops.zeros(shape=(10, 4))
    output_e_tf = array_ops.zeros(shape=(10, 4))

    model.fit([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
              epochs=1,
              steps_per_epoch=2,
              verbose=0)
    model.train_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

    # Test with dictionary inputs
    model.fit({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    },
              epochs=1,
              steps_per_epoch=2,
              verbose=0)
    model.fit({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    },
              validation_data=({
                  'input_a': input_a_tf,
                  'input_b': input_b_tf
              }, {
                  'dense': output_d_tf,
                  'dropout': output_e_tf
              }),
              epochs=1,
              steps_per_epoch=2,
              validation_steps=2,
              verbose=0)
    model.train_on_batch({
        'input_a': input_a_tf,
        'input_b': input_b_tf
    }, {
        'dense': output_d_tf,
        'dropout': output_e_tf
    })

    # Test with validation data
    model.fit([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
              validation_data=([input_a_tf,
                                input_b_tf], [output_d_tf, output_e_tf]),
              epochs=1,
              steps_per_epoch=2,
              validation_steps=2,
              verbose=0)
    # Test evaluation / prediction methods
    model.evaluate([input_a_tf, input_b_tf], [output_d_tf, output_e_tf],
                   steps=2,
                   verbose=0)
    model.predict([input_a_tf, input_b_tf], steps=2)
    model.test_on_batch([input_a_tf, input_b_tf], [output_d_tf, output_e_tf])

  @tf_test_util.run_deprecated_v1
  def test_model_with_input_feed_tensor(self):
    """We test building a model with a TF variable as input.

    We should be able to call fit, evaluate, predict,
    by only passing them data for the placeholder inputs
    in the model.
    """
    with ops.Graph().as_default(), self.cached_session():
      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 3))

      output_a_np = np.random.random((10, 4))
      output_b_np = np.random.random((10, 3))

      input_v = variables_module.Variable(input_a_np, dtype='float32')
      self.evaluate(variables_module.variables_initializer([input_v]))
      a = input_layer.Input(tensor=input_v)
      b = input_layer.Input(shape=(3,), name='input_b')

      a_2 = layers_module.Dense(4, name='dense_1')(a)
      dp = layers_module.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      model = training_module.Model([a, b], [a_2, b_2])
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
      self.evaluate(variables_module.variables_initializer([input_v]))
      a = input_layer.Input(tensor=input_v)
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      a_2 = layers_module.Dropout(0.5, name='dropout')(a_2)
      model = training_module.Model(a, a_2)
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
      self.evaluate(variables_module.variables_initializer([input_v]))
      a = input_layer.Input(tensor=input_v)
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      model = training_module.Model(a, a_2)
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
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      dp = layers_module.Dropout(0.5, name='dropout')
      a_3 = dp(a_2)
      model = training_module.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dropout': 'mse'}
      model.compile(optimizer, loss, metrics=['mae'])

      input_a_np = np.random.random((10, 3))
      output_a_np = np.random.random((10, 4))

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, output_a_np)
      # evaluate
      _ = model.evaluate(input_a_np, output_a_np)

      # Same without dropout.
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      a_3 = layers_module.Dense(4, name='dense_2')(a_2)
      model = training_module.Model(a, [a_2, a_3])

      optimizer = 'rmsprop'
      loss = {'dense_2': 'mse'}
      model.compile(optimizer, loss, metrics={'dense_1': 'mae'})

      # test train_on_batch
      _ = model.train_on_batch(input_a_np, output_a_np)
      _ = model.test_on_batch(input_a_np, output_a_np)
      # fit
      _ = model.fit(input_a_np, output_a_np)
      # evaluate
      _ = model.evaluate(input_a_np, output_a_np)

  def test_model_with_external_loss(self):
    with ops.Graph().as_default(), self.cached_session():
      # None loss, only regularization loss.
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(
          4, name='dense_1', kernel_regularizer='l1', bias_regularizer='l2')(
              a)
      dp = layers_module.Dropout(0.5, name='dropout')
      a_3 = dp(a_2)

      model = training_module.Model(a, [a_2, a_3])

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
      a = input_layer.Input(shape=(3,), name='input_a')
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      a_3 = layers_module.Dense(4, name='dense_2')(a)

      model = training_module.Model(a, [a_2, a_3])
      model.add_loss(backend.mean(a_3 + a_2))

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
      input_v = variables_module.Variable(input_a_np, dtype='float32')
      self.evaluate(variables_module.variables_initializer([input_v]))
      a = input_layer.Input(tensor=input_v)
      a_2 = layers_module.Dense(4, name='dense_1')(a)
      a_2 = layers_module.Dropout(0.5, name='dropout')(a_2)
      model = training_module.Model(a, a_2)
      model.add_loss(backend.mean(a_2))

      model.compile(optimizer='rmsprop',
                    loss=None,
                    metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None, None)
      out = model.test_on_batch(None, None)
      out = model.predict_on_batch(None)

      # Test multi-output model with no external data at all.
      self.evaluate(variables_module.variables_initializer([input_v]))
      a = input_layer.Input(tensor=input_v)
      a_1 = layers_module.Dense(4, name='dense_1')(a)
      a_2 = layers_module.Dropout(0.5, name='dropout')(a_1)
      model = training_module.Model(a, [a_1, a_2])
      model.add_loss(backend.mean(a_2))

      model.compile(optimizer='rmsprop',
                    loss=None,
                    metrics=['mean_squared_error'])

      # test train_on_batch
      out = model.train_on_batch(None, None)
      out = model.test_on_batch(None, None)
      out = model.predict_on_batch(None)

      out = model.predict(None, steps=3)
      self.assertEqual(len(out), 2)
      self.assertEqual(out[0].shape, (10 * 3, 4))
      self.assertEqual(out[1].shape, (10 * 3, 4))

  def test_target_tensors(self):
    with ops.Graph().as_default(), self.cached_session():
      # single-output, as list
      model = sequential.Sequential()
      model.add(layers_module.Dense(4, input_shape=(4,), name='dense'))
      input_val = np.random.random((10, 4))
      target_val = np.random.random((10, 4))
      target = backend.variable(target_val)
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
      target_a = backend.variable(target_val_a)
      target_b = backend.variable(target_val_b)

      inputs = layers_module.Input(shape=(4,))
      output_a = layers_module.Dense(4, name='dense_a')(inputs)
      output_b = layers_module.Dense(4, name='dense_b')(inputs)
      model = training_module.Model(inputs, [output_a, output_b])
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

  def test_model_custom_target_tensors(self):
    with ops.Graph().as_default(), self.cached_session():
      a = input_layer.Input(shape=(3,), name='input_a')
      b = input_layer.Input(shape=(3,), name='input_b')

      a_2 = layers_module.Dense(4, name='dense_1')(a)
      dp = layers_module.Dropout(0.5, name='dropout')
      b_2 = dp(b)

      y = backend.placeholder([10, 4], name='y')
      y1 = backend.placeholder([10, 3], name='y1')
      y2 = backend.placeholder([7, 5], name='y2')
      model = training_module.Model([a, b], [a_2, b_2])

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
      pl_target_a = array_ops.placeholder('float32', shape=(None, 4))
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors={'dense_1': pl_target_a})
      model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])


class TestTrainingWithMetrics(keras_parameterized.TestCase):
  """Training tests related to metrics."""

  @keras_parameterized.run_all_keras_modes
  def test_metrics_names(self):
    a = layers_module.Input(shape=(3,), name='input_a')
    b = layers_module.Input(shape=(3,), name='input_b')

    dense = layers_module.Dense(4, name='dense')
    c = dense(a)
    d = dense(b)
    e = layers_module.Dropout(0.5, name='dropout')(c)

    model = training_module.Model([a, b], [d, e])

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    metrics = ['mse', metrics_module.BinaryAccuracy()]
    model.compile(
        optimizer,
        loss='mae',
        metrics=metrics,
        run_eagerly=testing_utils.should_run_eagerly())

    mse_metric = 'mse' if context.executing_eagerly() else 'mean_squared_error'
    reference_metric_names = [
        'loss', 'dense_loss', 'dropout_loss', 'dense_' + mse_metric,
        'dense_binary_accuracy', 'dropout_' + mse_metric,
        'dropout_binary_accuracy'
    ]

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
    model = sequential.Sequential()
    model.add(layers_module.Dense(3, activation='relu', input_dim=4))
    model.add(layers_module.Dense(1, activation='sigmoid'))
    acc_obj = metrics_module.BinaryAccuracy()
    model.compile(
        loss='mae',
        metrics=[acc_obj],
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly())

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
    inp_1 = layers_module.Input(shape=(1,), name='input_1')
    inp_2 = layers_module.Input(shape=(1,), name='input_2')
    x = layers_module.Dense(3, kernel_initializer='ones', trainable=False)
    out_1 = layers_module.Dense(
        1, kernel_initializer='ones', name='output_1', trainable=False)
    out_2 = layers_module.Dense(
        1, kernel_initializer='ones', name='output_2', trainable=False)

    branch_a = [inp_1, x, out_1]
    branch_b = [inp_2, x, out_2]
    model = testing_utils.get_multi_io_model(branch_a, branch_b)

    # list of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics_module.MeanSquaredError()],
        weighted_metrics=[metrics_module.MeanSquaredError()],
        run_eagerly=testing_utils.should_run_eagerly())

    # list of list of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[
            metrics_module.MeanSquaredError(),
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()]
        ],
        weighted_metrics=[
            metrics_module.MeanSquaredError(),
            [metrics_module.MeanSquaredError(),
             metrics_module.Accuracy()]
        ],
        run_eagerly=testing_utils.should_run_eagerly())

    # dict of metrics.
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics={
            'output_1':
                metrics_module.MeanSquaredError(),
            'output_2': [
                metrics_module.MeanSquaredError(),
                metrics_module.Accuracy()
            ],
        },
        weighted_metrics={
            'output_1':
                metrics_module.MeanSquaredError(),
            'output_2': [
                metrics_module.MeanSquaredError(),
                metrics_module.Accuracy()
            ],
        },
        run_eagerly=testing_utils.should_run_eagerly())

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_with_tensor_on_model(self):
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(1, kernel_initializer='ones')(x)
    model = training_module.Model(x, y)
    model.add_metric(
        math_ops.reduce_sum(y), name='metric_1', aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_2')
    else:
      with self.assertRaisesRegex(
          ValueError, 'Using the result of calling a `Metric` object '):
        with backend.get_graph().as_default():
          model.add_metric(metrics_module.Mean(name='metric_2')(y))

    model.compile(
        'sgd',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())

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

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
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
        run_eagerly=testing_utils.should_run_eagerly())

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

    class TestLayer(layers_module.Layer):

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
        layers_module.Dense(2, kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(1,))
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 5, 0)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_model_metrics_list(self):

    class LayerWithAddMetric(layers_module.Layer):

      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = layers_module.Dense(1, kernel_initializer='ones')

      def __call__(self, inputs):
        outputs = self.dense(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='metric_1', aggregation='mean')
        return outputs

    class LayerWithNestedAddMetricLayer(layers_module.Layer):

      def __init__(self):
        super(LayerWithNestedAddMetricLayer, self).__init__()
        self.layer = LayerWithAddMetric()

      def call(self, inputs):
        outputs = self.layer(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='metric_2', aggregation='mean')
        return outputs

    x = layers_module.Input(shape=(1,))
    y = LayerWithNestedAddMetricLayer()(x)

    model = training_module.Model(x, y)
    model.add_metric(
        math_ops.reduce_sum(y), name='metric_3', aggregation='mean')

    if context.executing_eagerly():
      # This is not a use case in v1 graph mode.
      mean_result = metrics_module.Mean()(y)
      with self.assertRaisesRegex(
          ValueError, 'Expected a symbolic Tensor for the metric value'):
        model.add_metric(mean_result, name='metric_4')

    else:
      with self.assertRaisesRegex(
          ValueError, 'Using the result of calling a `Metric` object '):
        with backend.get_graph().as_default():
          model.add_metric(metrics_module.Mean(name='metric_4')(y))

    model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('metric_4')],
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

    # Verify that the metrics added using `compile` and `add_metric` API are
    # included
    self.assertEqual([m.name for m in model.metrics],
                     ['loss', 'metric_4', 'metric_2', 'metric_1', 'metric_3'])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_model_metrics_list_in_call(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_1', aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        metrics=[metrics_module.Accuracy('acc')],
        run_eagerly=testing_utils.should_run_eagerly())
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

    self.assertEqual([m.name for m in model.metrics],
                     ['loss', 'acc', 'metric_1'])

  @keras_parameterized.run_all_keras_modes
  def test_multiple_add_metric_calls(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean1 = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_2')

      def call(self, x):
        self.add_metric(self.mean2(x), name='metric_2')
        self.add_metric(self.mean1(x), name='metric_1')
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_3', aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    self.assertListEqual([m.name for m in model.metrics],
                         ['metric_1', 'metric_2'])
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

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

  @keras_parameterized.run_all_keras_modes
  def test_multiple_add_metric_calls_layer(self):

    class TestLayer(layers_module.Layer):

      def __init__(self):
        super(TestLayer, self).__init__(name='test_layer')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.m1 = metrics_module.Mean(name='m_1')
        self.m2 = [
            metrics_module.Mean(name='m_2'),
            metrics_module.Mean(name='m_3')
        ]
        self.m3 = {
            'mean4': metrics_module.Mean(name='m_4'),
            'mean5': metrics_module.Mean(name='m_5')
        }

      def call(self, x):
        self.add_metric(self.m2[0](x))
        self.add_metric(self.m2[1](x))
        self.add_metric(self.m1(x))
        self.add_metric(self.m3['mean4'](x))
        self.add_metric(self.m3['mean5'](x))
        self.add_metric(math_ops.reduce_sum(x), name='m_6', aggregation='mean')
        return self.dense1(x)

    layer = TestLayer()
    self.assertListEqual([m.name for m in layer.metrics],
                         ['m_1', 'm_2', 'm_3', 'm_4', 'm_5'])

    layer(np.ones((10, 10)))
    self.assertListEqual([m.name for m in layer.metrics],
                         ['m_1', 'm_2', 'm_3', 'm_4', 'm_5', 'm_6'])

  @keras_parameterized.run_all_keras_modes
  def test_duplicate_metric_name_in_add_metric(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')
        self.mean2 = metrics_module.Mean(name='metric_1')

      def call(self, x):
        self.add_metric(self.mean(x), name='metric_1')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    with self.assertRaisesRegex(
        ValueError,
        'Please provide different names for the metrics you have added. '
        'We found 2 metrics with the name: "metric_1"'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_without_name(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(math_ops.reduce_sum(x), aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))

    with self.assertRaisesRegex(ValueError,
                                'Please provide a name for your metric like'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_correctness(self):
    inputs = input_layer.Input(shape=(1,))
    targets = input_layer.Input(shape=(1,))

    class Bias(layers_module.Layer):

      def build(self, input_shape):
        self.bias = self.add_variable('bias', (1,), initializer='zeros')
        self.mae = metrics_module.MeanAbsoluteError(name='mae_1')

      def call(self, inputs):
        inputs, targets = inputs
        outputs = inputs + self.bias
        self.add_metric(self.mae(targets, outputs), name='mae_1')
        return outputs

    outputs = Bias()([inputs, targets])
    model = training_module.Model([inputs, targets], outputs)

    model.add_metric(
        metrics_module.mean_absolute_error(targets, outputs),
        name='mae_2',
        aggregation='mean')

    model.compile(
        loss='mae',
        optimizer=optimizer_v2.gradient_descent.SGD(0.1),
        metrics=[metrics_module.MeanAbsoluteError(name='mae_3')],
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.array([[0.], [1.], [2.]])
    y = np.array([[0.5], [2.], [3.5]])
    history = model.fit([x, y], y, batch_size=3, epochs=5)

    expected_val = [1., 0.9, 0.8, 0.7, 0.6]
    for key in ['loss', 'mae_1', 'mae_2', 'mae_3']:
      self.assertAllClose(history.history[key], expected_val, 1e-3)

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_order(self):

    class MyLayer(layers_module.Layer):

      def call(self, inputs, training=None, mask=None):
        self.add_metric(
            array_ops.ones([32]) * 2.0, name='two', aggregation='mean')
        return inputs

    class MyModel(training_module.Model):

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
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(dataset_train, epochs=3)
    self.assertDictEqual(
        history.history, {
            'loss': [0.0, 0.0, 0.0],
            'three': [3.0, 3.0, 3.0],
            'two': [2.0, 2.0, 2.0],
            'one': [1.0, 1.0, 1.0]
        })

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_aggregation_mean(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(
            math_ops.reduce_sum(x), name='metric_1', aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(
        'rmsprop', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

  @keras_parameterized.run_all_keras_modes
  def test_add_metric_aggregation_none(self):

    class TestModel(training_module.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = layers_module.Dense(2, kernel_initializer='ones')
        self.mean = metrics_module.Mean(name='metric_1')

      def call(self, x):
        self.add_metric(self.mean(x), name='metric_1', aggregation=None)
        return self.dense1(x)

    model = TestModel()
    model.compile(
        'rmsprop', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    model.fit(np.ones(shape=(10, 1)), np.ones(shape=(10, 2)), batch_size=5)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def DISABLED_test_add_metric_invalid_aggregation(self):
    # TODO(psv): Reenable test once it is fixed.
    x = layers_module.Input(shape=(1,))
    y = layers_module.Dense(1, kernel_initializer='ones')(x)
    model = training_module.Model(x, y)
    with self.assertRaisesRegex(ValueError,
                                'only `mean` sample-wise metric aggregation'):
      model.add_metric(
          math_ops.reduce_sum(y), name='metric_1', aggregation='sum')

    with self.assertRaisesRegex(ValueError,
                                'only `mean` sample-wise metric aggregation'):
      model.add_metric(
          math_ops.reduce_sum(y), name='metric_1', aggregation=None)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_calling_evaluate_in_callback_during_fit(self):
    # Check fix for a bug that caused `evaluate` to hit a cached dataset
    # when run from inside a fit callback.
    x = layers_module.Input(shape=(2,))
    y = layers_module.Dense(2, kernel_initializer='ones', use_bias=False)(x)
    model = training_module.Model(x, y)

    ones = np.ones((10, 2), dtype=np.float32)
    zeros = np.zeros((10, 2), dtype=np.float32)
    train_ds = dataset_ops.Dataset.from_tensor_slices(
        (ones, ones)).batch(5)
    val_ds_1 = dataset_ops.Dataset.from_tensor_slices(
        (ones, ones)).batch(5)
    val_ds_2 = dataset_ops.Dataset.from_tensor_slices(
        (zeros, zeros)).batch(5)
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    class MyCallback(Callback):

      def on_epoch_end(self, *args, **kwargs):
        eval_result = self.model.evaluate(val_ds_2)
        if abs(eval_result) > 1e-7:
          raise AssertionError(
              'Expected to hit the zeros dataset but got high loss value of %s'
              % eval_result)

    history = model.fit(
        train_ds, validation_data=val_ds_1, callbacks=[MyCallback()])
    # Evaluate at the end of fit should hit the ones dataset (cached)
    self.assertGreater(abs(history.history['val_loss'][-1]), 0.1)
    # Standalone call to evaluate should not hit the cached dataset
    eval_result = model.evaluate(val_ds_2)
    self.assertLess(abs(eval_result), 1e-7)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_model_with_nested_compiled_model(self):

    class LayerWithAddMetric(layers_module.Layer):

      def __init__(self):
        super(LayerWithAddMetric, self).__init__()
        self.dense = layers_module.Dense(1, kernel_initializer='ones')

      def call(self, inputs):
        outputs = self.dense(inputs)
        self.add_metric(
            math_ops.reduce_sum(outputs), name='mean', aggregation='mean')
        return outputs

    x = layers_module.Input(shape=(1,))
    y = LayerWithAddMetric()(x)

    inner_model = training_module.Model(x, y)
    inner_model.add_metric(
        math_ops.reduce_sum(y), name='mean1', aggregation='mean')

    inner_model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('acc')],
        run_eagerly=testing_utils.should_run_eagerly())
    inner_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)

    self.assertEqual([m.name for m in inner_model.metrics],
                     ['loss', 'acc', 'mean', 'mean1'])

    x = layers_module.Input(shape=[1])
    y = inner_model(x)
    outer_model = training_module.Model(x, y)
    outer_model.add_metric(
        math_ops.reduce_sum(y), name='mean2', aggregation='mean')

    outer_model.compile(
        'sgd',
        loss='mse',
        metrics=[metrics_module.Accuracy('acc2')],
        run_eagerly=testing_utils.should_run_eagerly())
    outer_model.fit(np.ones((10, 1)), np.ones((10, 1)), batch_size=10)
    self.assertEqual([m.name for m in outer_model.metrics],
                     ['loss', 'acc2', 'mean', 'mean1', 'mean2'])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_model_with_metric_class_that_returns_dict(self):
    x = layers_module.Input(shape=(2,))
    y = layers_module.Dense(3)(x)
    model = training_module.Model(x, y)

    class DictMetric(metrics_module.Metric):

      def __init__(self):
        super(DictMetric, self).__init__()
        self.sample_count = variables_module.Variable(0)
        self.l2_sum = variables_module.Variable(0.)

      def update_state(self, y_true, y_pred, sample_weight=None):
        self.l2_sum.assign_add(
            math_ops.reduce_sum(math_ops.square(y_true - y_pred)))
        self.sample_count.assign_add(array_ops.shape(y_true)[0])

      def reset_state(self):
        self.sample_count.assign(0)
        self.l2_sum.assign(0.)

      def result(self):
        mse = self.l2_sum / math_ops.cast(self.sample_count, 'float32')
        rmse = math_ops.sqrt(mse)
        return {'my_mse': mse,
                'my_rmse': rmse}

    model.compile('sgd',
                  'mse',
                  metrics=['mae', DictMetric()],
                  run_eagerly=testing_utils.should_run_eagerly())

    history = model.fit(np.ones((10, 2)), np.ones((10, 3)))
    self.assertEqual(list(history.history.keys()),
                     ['loss', 'mae', 'my_mse', 'my_rmse'])
    list_evaluate_res = model.evaluate(
        np.ones((10, 2)), np.ones((10, 3)))
    self.assertEqual(len(list_evaluate_res), 4)
    dict_evaluate_res = model.evaluate(
        np.ones((10, 2)), np.ones((10, 3)), return_dict=True)
    self.assertEqual(list(dict_evaluate_res.keys()),
                     ['loss', 'mae', 'my_mse', 'my_rmse'])
    list_train_on_batch_res = model.train_on_batch(
        np.ones((10, 2)), np.ones((10, 3)))
    self.assertEqual(len(list_train_on_batch_res), 4)
    dict_train_on_batch_res = model.train_on_batch(
        np.ones((10, 2)), np.ones((10, 3)), return_dict=True)
    self.assertEqual(list(dict_train_on_batch_res.keys()),
                     ['loss', 'mae', 'my_mse', 'my_rmse'])
    list_test_on_batch_res = model.test_on_batch(
        np.ones((10, 2)), np.ones((10, 3)))
    self.assertEqual(len(list_test_on_batch_res), 4)
    dict_test_on_batch_res = model.test_on_batch(
        np.ones((10, 2)), np.ones((10, 3)), return_dict=True)
    self.assertEqual(list(dict_test_on_batch_res.keys()),
                     ['loss', 'mae', 'my_mse', 'my_rmse'])


class BareUpdateLayer(layers_module.Layer):

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


class LambdaUpdateLayer(layers_module.Layer):

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


class NestedUpdateLayer(layers_module.Layer):

  def build(self, input_shape):
    self.layer = BareUpdateLayer()
    self.layer.build(input_shape)

  @property
  def counter(self):
    return self.layer.counter

  def call(self, inputs):
    return self.layer(inputs)


class SubgraphUpdateLayer(layers_module.Layer):

  def build(self, input_shape):
    self.counter = self.add_weight(
        'counter',
        dtype='int32',
        shape=(),
        initializer='zeros',
        trainable=False)

  def call(self, inputs, training=None):
    if training is None:
      training = backend.learning_phase()

    if training:
      self.counter.assign(self.counter + 1)
    return inputs


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestAutoUpdates(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @parameterized.named_parameters(
      ('bare_update', BareUpdateLayer),
      ('lambda_update', LambdaUpdateLayer),
      ('nested_update', NestedUpdateLayer))
  def test_updates_in_model(self, layer_builder):
    layer = layer_builder()
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types
  def test_lambda_updates_trainable_false(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    layer = LambdaUpdateLayer()
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)
    layer.trainable = False
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @keras_parameterized.run_with_all_model_types
  def test_subgraph_updates_in_model(self):
    layer = SubgraphUpdateLayer()
    x, y = np.ones((10, 10)), np.ones((10, 1))
    model = testing_utils.get_model_from_layers(
        [layer, layers_module.Dense(1)], input_shape=(10,))
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2, epochs=1)
    self.assertEqual(self.evaluate(layer.counter), 5)

  @parameterized.named_parameters(
      ('bare_update', BareUpdateLayer),
      ('lambda_update', LambdaUpdateLayer),
      ('nested_update', NestedUpdateLayer))
  def test_updates_standalone_layer(self, layer_builder):
    layer = layer_builder()
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


class TestFunctionTracing(keras_parameterized.TestCase):

  def _seq_model_and_data(self):
    model = sequential.Sequential([layers_module.Dense(4, activation='relu')])
    model.compile(loss='mse', optimizer='rmsprop')
    x = np.random.random((10, 6))
    y = np.random.random((10, 4))
    return model, x, y

  @keras_parameterized.run_all_keras_modes(
      always_skip_v1=True, always_skip_eager=True)
  def test_no_tracing_between_epoch(self):
    model, x, y = self._seq_model_and_data()

    logging.set_verbosity(1)
    with self.assertLogs(level=1) as logs:
      model.fit(x, y, epochs=10, batch_size=5, validation_data=(x, y))

    new_func_graph = 'INFO:absl:Creating new FuncGraph for Python function'
    self.assertEqual(sum(new_func_graph in log for log in logs.output), 9)

  @keras_parameterized.run_all_keras_modes(
      always_skip_v1=True, always_skip_eager=True)
  def test_evaluate_no_cached_data(self):
    model, x, y = self._seq_model_and_data()

    new_func_graph = 'INFO:absl:Creating new FuncGraph for Python function'
    logging.set_verbosity(1)
    with self.assertLogs(level=1) as eval_logs:
      for _ in range(6):
        model.evaluate(x, y, batch_size=5)
    self.assertEqual(sum(new_func_graph in log for log in eval_logs.output), 20)


class TestBuildCustomModel(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_build_list_of_inputs(self):

    class MyModel(training_module.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layers_module.Dense(1)
        self.l2 = layers_module.Dense(2)

      def call(self, x):
        a, b = x
        return self.l1(a) + self.l2(b)

    # List of tuples
    model = MyModel()
    model.build([(None, 1), (None, 2)])
    self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
    self.assertEqual(model.l2.kernel.shape.as_list(), [2, 2])
    # List of lists
    model = MyModel()
    model.build([[None, 1], [None, 2]])
    self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
    self.assertEqual(model.l2.kernel.shape.as_list(), [2, 2])

  @keras_parameterized.run_all_keras_modes
  def test_build_single_inputs(self):

    class MyModel(training_module.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layers_module.Dense(1)

      def call(self, x):
        return self.l1(x)

    model = MyModel()
    model.build((None, 1))
    self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])
    model = MyModel()
    model.build([None, 1])
    self.assertEqual(model.l1.kernel.shape.as_list(), [1, 1])

  @keras_parameterized.run_all_keras_modes
  def test_build_dict_inputs(self):

    class MyModel(training_module.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layers_module.Dense(1)

      def call(self, inputs):
        return self.l1(inputs['x'])

    model = MyModel()
    model.build({'x': [None, 16]})
    self.assertEqual(model.l1.kernel.shape.as_list(), [16, 1])


if __name__ == '__main__':
  test.main()
