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

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.engine.training_utils import weighted_masked_objective
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.rmsprop import RMSPropOptimizer

try:
  import scipy.sparse as scipy_sparse  # pylint: disable=g-import-not-at-top
except ImportError:
  scipy_sparse = None


class TrainingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_fit_on_arrays(self):
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
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'],
        loss_weights=loss_weights)

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

    # Test model with input data as a list of lists
    model.fit(
        [np.ndarray.tolist(input_a_np), np.ndarray.tolist(input_b_np)],
        [output_d_np, output_e_np],
        epochs=2,
        batch_size=5,
        verbose=2)

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
        metrics=[metrics_module.CategoricalAccuracy(), 'mae'])
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Test with dictionaries for loss, metrics, loss weights
    loss = {'dense': 'mse', 'dropout': 'mae'}
    loss_weights = {'dense': 1., 'dropout': 0.5}
    metrics = {
        'dense': 'mse',
        'dropout': metrics_module.CategoricalAccuracy()
    }
    model.compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Invalid use cases
    with self.assertRaises(ValueError):
      model.train_on_batch({'input_a': input_a_np},
                           [output_d_np, output_e_np])
    with self.assertRaises(AttributeError):
      model.fit(
          [input_a_np, input_b_np], [output_d_np, output_e_np],
          epochs=1,
          validation_data=([input_a_np, input_b_np], 0, 0),
          verbose=0)
    with self.assertRaises(ValueError):
      model.train_on_batch([input_a_np], [output_d_np, output_e_np])
    with self.assertRaises(AttributeError):
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
    model.compile(optimizer, loss='mse')
    # This will work
    model.fit([input_a_np], output_d_np, epochs=1)
    with self.assertRaises(ValueError):
      model.fit([input_a_np, input_a_np], output_d_np, epochs=1)

    # Test model on a list of floats
    input_a_np = np.random.random((10, 3))
    input_b_np = np.random.random((10, 4))

    model.fit([np.ndarray.tolist(input_a_np)],
              [np.ndarray.tolist(input_b_np)],
              epochs=2,
              batch_size=5,
              verbose=2)

  @tf_test_util.run_in_graph_and_eager_modes
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
        sample_weight_mode=None)

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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_activity_regularizer_fit(self):
    loss = {}
    for reg in [None, 'l2']:
      inputs = keras.layers.Input(shape=(10,))
      x = keras.layers.Dense(
          10, activation='relu', activity_regularizer=reg,
          kernel_initializer='ones', use_bias=False)(inputs)
      outputs = keras.layers.Dense(1, activation='sigmoid',
                                   kernel_initializer='ones', use_bias=False)(x)
      model = keras.Model(inputs, outputs)

      x = np.ones((10, 10), 'float32')
      y = np.ones((10, 1), 'float32')

      optimizer = RMSPropOptimizer(learning_rate=0.001)
      model.compile(optimizer, 'binary_crossentropy')
      model.fit(x, y, batch_size=2, epochs=5)
      loss[reg] = model.evaluate(x, y)
    self.assertLess(loss[None], loss['l2'])

  @tf_test_util.run_in_graph_and_eager_modes
  def test_activity_regularizer_loss_value(self):
    inputs = keras.layers.Input(shape=(10,))
    outputs = keras.layers.Dense(
        1,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=keras.initializers.ones(),
        activity_regularizer='l2')(
            inputs)
    model = keras.Model(inputs, outputs)
    x = np.ones((10, 10), 'float32')
    y = np.ones((10, 1), 'float32')
    optimizer = RMSPropOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'binary_crossentropy')
    loss = model.test_on_batch(x, y)
    self.assertAlmostEqual(0.01, loss, places=4)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_activity_regularizer_batch_independent(self):
    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(
        10, activation='relu', activity_regularizer='l2')(
            inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    model.compile(optimizer, 'binary_crossentropy')

    x = np.ones((10, 10), 'float32')
    y = np.ones((10, 1), 'float32')
    loss_small_batch = model.test_on_batch(x, y)

    x2 = np.ones((20, 10), 'float32')
    y2 = np.ones((20, 1), 'float32')
    loss_big_batch = model.test_on_batch(x2, y2)

    self.assertAlmostEqual(loss_small_batch, loss_big_batch, places=4)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_activity_regularizer_in_model_call(self):

    class MyModel(keras.Model):

      def call(self, inputs):
        self.add_loss(inputs)
        return inputs

    x = ops.convert_to_tensor(1.)
    model = MyModel()
    _ = model(x)
    self.assertEqual(1, len(model.losses))

  def test_training_on_sparse_data_with_dense_placeholders(self):
    if scipy_sparse is None:
      return

    with self.cached_session():
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
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      model.compile(
          optimizer,
          'mse',
          metrics=['mae', metrics_module.CategoricalAccuracy()])
      model.fit(test_inputs, test_outputs,
                epochs=1, batch_size=2, validation_split=0.5)
      model.evaluate(test_inputs, test_outputs, batch_size=2)

  def test_compile_with_sparse_placeholders(self):
    with self.cached_session():
      input_layer = keras.layers.Input(shape=(10,), sparse=True)
      weights = variables_lib.Variable(
          np.ones((10, 1)).astype(np.float32), name='weights')
      weights_mult = lambda x: sparse_ops.sparse_tensor_dense_matmul(x, weights)
      output_layer = keras.layers.Lambda(weights_mult)(input_layer)
      model = keras.Model([input_layer], output_layer)
      model.compile(
          loss='binary_crossentropy',
          optimizer=keras.optimizers.Adam(lr=0.0001),
          metrics=['accuracy'])

  def test_that_trainable_disables_updates(self):
    val_a = np.random.random((10, 4))
    val_out = np.random.random((10, 4))

    with self.cached_session():
      a = keras.layers.Input(shape=(4,))
      layer = keras.layers.BatchNormalization(input_shape=(4,))
      b = layer(a)
      model = keras.Model(a, b)

      model.trainable = False
      assert not model.updates

      model.compile('sgd', 'mse')
      assert not model.updates

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

      model.trainable = True
      model.compile('sgd', 'mse')
      assert model.updates

      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      assert np.abs(np.sum(x1 - x2)) > 1e-5

      layer.trainable = False
      model.compile('sgd', 'mse')
      assert not model.updates

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

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
      self.assertSetEqual(
          set(test_callback.batch_end_logs.keys()),
          set(['batch', 'size', 'acc', 'loss', 'weighted_mean_absolute_error']))
      self.assertSetEqual(
          set(test_callback.epoch_end_logs.keys()),
          set([
              'acc', 'loss', 'weighted_mean_absolute_error', 'val_acc',
              'val_loss', 'val_weighted_mean_absolute_error'
          ]))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_mismatched_output_shape_and_target_shape(self):
    model = keras.Sequential([
        keras.layers.Dense(2, input_shape=(3, 4)),
        keras.layers.Dense(5),
    ])
    model.compile(RMSPropOptimizer(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy')
    # Test with Numpy data
    x_train = np.random.random((10, 3, 4))
    y_train = np.random.randint(0, 5, size=(10, 3))
    model.fit(x_train, y_train, batch_size=5, epochs=1)

    # Test with iterator
    dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(10)
    iterator = dataset.make_one_shot_iterator()
    model.fit(iterator, epochs=1, steps_per_epoch=2)

    if context.executing_eagerly():
      # Test with eager execution
      model.compile(RMSPropOptimizer(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    run_eagerly=True)
      model.fit(x_train, y_train, batch_size=5, epochs=1)

      # Test with eager execution and iterator
      model.fit(iterator, epochs=1, steps_per_epoch=2)

  def test_losses_in_defun(self):
    with context.eager_mode():
      layer = keras.layers.Dense(1, kernel_regularizer='l1')
      layer(array_ops.ones([1, 10]))

      @function.defun
      def get_losses():
        return layer.losses

      self.assertAllEqual(
          self.evaluate(layer.losses), self.evaluate(get_losses()))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_logging(self):
    mock_stdout = io.BytesIO() if six.PY2 else io.StringIO()
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        RMSPropOptimizer(learning_rate=0.001), loss='binary_crossentropy')
    with test.mock.patch.object(sys, 'stdout', mock_stdout):
      model.fit(
          np.ones((10, 10), 'float32'), np.ones((10, 1), 'float32'), epochs=10)
    self.assertTrue('Epoch 5/10' in mock_stdout.getvalue())


class TestExceptionsAndWarnings(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
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
        model.compile(optimizer, loss=None)

  @tf_test_util.run_in_graph_and_eager_modes
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
            })
        msg = ('Output "dense_1" missing from loss dictionary. We assume this '
               'was done on purpose. The fit and evaluate APIs will not be '
               'expecting any data to be passed to "dense_1".')
        self.assertRegexpMatches(str(mock_log.call_args), msg)


class LossWeightingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
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
        optimizer=RMSPropOptimizer(learning_rate=learning_rate))

    np.random.seed(1337)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_test = y_test.copy()
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    test_ids = np.where(int_y_test == np.array(weighted_class))[0]

    class_weight = dict([(i, 1.) for i in range(num_classes)])
    class_weight[weighted_class] = 2.

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = 2.

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs // 3,
        verbose=0,
        class_weight=class_weight,
        validation_data=(x_train, y_train, sample_weight))
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
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
        loss='categorical_crossentropy')

    np.random.seed(43)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_test = y_test.copy()
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    test_ids = np.where(int_y_test == np.array(weighted_class))[0]

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = 2.

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
    ref_score = model.evaluate(x_test, y_test, verbose=0)
    if not context.executing_eagerly():
      score = model.evaluate(
          x_test[test_ids, :], y_test[test_ids, :], verbose=0)
      self.assertLess(score[0], ref_score[0])

  @tf_test_util.run_in_graph_and_eager_modes
  def test_warning_for_concurrent_sample_and_class_weights(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(3,)))
    model.compile(
        loss='mse',
        optimizer=RMSPropOptimizer(learning_rate=0.01))
    x_train = np.random.random((10, 3))
    y_train = np.random.random((10, 10))
    sample_weight = np.ones((y_train.shape[0]))
    class_weight = {0: 1., 1: 1.}

    with test.mock.patch.object(logging, 'warning') as mock_log:
      model.fit(
          x_train,
          y_train,
          epochs=1,
          verbose=0,
          sample_weight=sample_weight,
          class_weight=class_weight)
      msg = ('The `class_weight` argument will be ignored.')
      self.assertRegexpMatches(str(mock_log.call_args), msg)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_temporal_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
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
      y_train = keras.utils.to_categorical(y_train, num_classes)
      y_test = keras.utils.to_categorical(y_test, num_classes)
      test_ids = np.where(int_y_test == np.array(weighted_class))[0]

      sample_weight = np.ones((y_train.shape[0]))
      sample_weight[int_y_train == weighted_class] = 2.

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
          loss='binary_crossentropy',
          metrics=['acc', metrics_module.CategoricalAccuracy()],
          weighted_metrics=['mae', metrics_module.CategoricalAccuracy()],
          sample_weight_mode='temporal')

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

  @tf_test_util.run_in_graph_and_eager_modes
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
      model.compile(optimizer, loss='binary_crossentropy')

      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=test_samples,
          input_shape=(input_dim,),
          num_classes=num_classes)
      # convert class vectors to binary class matrices
      y_train = keras.utils.to_categorical(y_train, num_classes)
      class_weight = dict([(i, 1.) for i in range(num_classes)])

      del class_weight[1]
      with self.assertRaises(ValueError):
        model.fit(x_train, y_train,
                  epochs=0, verbose=0, class_weight=class_weight)

      with self.assertRaises(ValueError):
        model.compile(
            optimizer, loss='binary_crossentropy', sample_weight_mode=[])

      # Build multi-output model
      x = keras.Input((3,))
      y1 = keras.layers.Dense(4, name='1')(x)
      y2 = keras.layers.Dense(4, name='2')(x)
      model = keras.models.Model(x, [y1, y2])
      model.compile(optimizer, loss='mse')
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

  @tf_test_util.run_in_graph_and_eager_modes
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
      model.compile(optimizer, loss='mse', sample_weight_mode=[None])
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a list and mode value is `temporal`
      model.compile(optimizer, loss='mse', sample_weight_mode=['temporal'])
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is None
      model.compile(
          optimizer, loss='mse', sample_weight_mode={'time_distributed': None})
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a dict and mode value is `temporal`
      model.compile(
          optimizer,
          loss='mse',
          sample_weight_mode={'time_distributed': 'temporal'})
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is None
      model.compile(optimizer, loss='mse', sample_weight_mode=None)
      model.fit(x, y, epochs=1, batch_size=10)

      # sample_weight_mode is a not a list/dict and mode value is `temporal`
      model.compile(optimizer, loss='mse', sample_weight_mode='temporal')
      model.fit(x, y, epochs=1, batch_size=10)


class LossMaskingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_masking_graph_sequential(self):
    with self.cached_session():
      x = np.array([[[1], [1]], [[0], [0]]])
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='one')))
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
      y = np.array([[[1], [1]], [[1], [1]]])
      loss = model.train_on_batch(x, y)
      self.assertEqual(float(loss), 0.)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_masking_deferred_sequential(self):
    with self.cached_session():
      x = np.array([[[1], [1]], [[0], [0]]])
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='one')))
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
      y = np.array([[[1], [1]], [[1], [1]]])
      loss = model.train_on_batch(x, y)
      self.assertEqual(float(loss), 0.)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_masking_functional(self):
    with self.cached_session():
      x = np.array([[[1], [1]], [[0], [0]]])
      inputs = keras.layers.Input((2, 1))
      outputs = keras.layers.Masking(mask_value=0)(inputs)
      outputs = keras.layers.TimeDistributed(
          keras.layers.Dense(1, kernel_initializer='one'))(outputs)
      model = keras.Model(inputs, outputs)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
      y = np.array([[[1], [1]], [[1], [1]]])
      loss = model.train_on_batch(x, y)
      self.assertEqual(float(loss), 0.)

  @tf_test_util.run_in_graph_and_eager_modes
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

    with self.cached_session():
      x = np.random.random((5, 3))
      inputs = keras.layers.Input((3,))
      masked = keras.layers.Masking(mask_value=0)(inputs)
      outputs = CustomMaskedLayer()(masked)

      model = keras.Model(inputs, outputs)
      model.compile(loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
      y = np.random.random((5, 3))
      model.train_on_batch(x, y)

  def test_loss_masking(self):
    with self.cached_session():
      weighted_loss = weighted_masked_objective(keras.losses.get('mae'))
      shape = (3, 4, 2)
      x = np.arange(24).reshape(shape)
      y = 2 * x

      # Normally the trailing 1 is added by standardize_weights
      weights = np.ones((3,))
      mask = np.ones((3, 4))
      mask[1, 0] = 0

      keras.backend.eval(
          weighted_loss(
              keras.backend.variable(x),
              keras.backend.variable(y),
              keras.backend.variable(weights), keras.backend.variable(mask)))


class TestDynamicTrainability(test.TestCase):

  def test_trainable_warning(self):
    with self.cached_session():
      x = np.random.random((5, 3))
      y = np.random.random((5, 2))

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_dim=3))
      model.trainable = False
      model.compile('rmsprop', 'mse')
      model.trainable = True
      model.train_on_batch(x, y)
      self.assertRaises(Warning)

  def test_trainable_argument(self):
    with self.cached_session():
      x = np.random.random((5, 3))
      y = np.random.random((5, 2))

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_dim=3, trainable=False))
      model.compile('rmsprop', 'mse')
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

      # test with nesting
      inputs = keras.layers.Input(shape=(3,))
      output = model(inputs)
      model = keras.models.Model(inputs, output)
      model.compile('rmsprop', 'mse')
      out = model.predict(x)
      model.train_on_batch(x, y)
      out_2 = model.predict(x)
      self.assertAllClose(out, out_2)

  def test_layer_trainability_switch(self):
    with self.cached_session():
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
    with self.cached_session():
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
    with self.cached_session():
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


class TestTrainingWithDataTensors(test.TestCase):

  def test_training_and_eval_methods_on_symbolic_tensors_single_io(self):
    with self.cached_session():
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

  def test_training_and_eval_methods_on_symbolic_tensors_multi_io(self):
    with self.cached_session():
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

      # test fit with validation data
      out = model.fit(None, None, epochs=1,
                      steps_per_epoch=2,
                      validation_steps=2)

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
                               [output_a_np, output_b_np],
                               {y: np.random.random((10, 4)),
                                y1: np.random.random((10, 3))})
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
                               [output_a_np, output_b_np],
                               {y: np.random.random((10, 4)),
                                y1: np.random.random((10, 3))})

      # test with custom TF placeholder as target
      pl_target_a = keras.backend.array_ops.placeholder('float32',
                                                        shape=(None, 4))
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors={'dense_1': pl_target_a})
      model.train_on_batch([input_a_np, input_b_np],
                           [output_a_np, output_b_np])


class TestTrainingWithMetrics(test.TestCase):
  """Training tests related to metrics."""

  @tf_test_util.run_in_graph_and_eager_modes
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
    model.compile(optimizer, loss='mae', metrics=metrics)
    reference_metric_names = [
        'loss', 'dense_loss', 'dropout_loss', 'dense_mean_squared_error',
        'dense_binary_accuracy', 'dropout_mean_squared_error',
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_metrics_correctness(self):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            3, activation='relu', input_dim=4, kernel_initializer='ones'))
    model.add(
        keras.layers.Dense(
            1, activation='sigmoid', kernel_initializer='ones'))
    model.compile(
        loss='mae',
        metrics=['accuracy', metrics_module.BinaryAccuracy()],
        optimizer=RMSPropOptimizer(learning_rate=0.001))

    # verify correctness of stateful and stateless metrics.
    x = np.ones((100, 4))
    y = np.ones((100, 1))
    outs = model.evaluate(x, y)
    self.assertEqual(outs[1], 1.)
    self.assertEqual(outs[2], 1.)

    y = np.zeros((100, 1))
    outs = model.evaluate(x, y)
    self.assertEqual(outs[1], 0.)
    self.assertEqual(outs[2], 0.)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_metrics_correctness_with_weighted_metrics(self):
    np.random.seed(1337)
    x = np.array([[[1.], [1.]], [[0.], [0.]]])
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(1, kernel_initializer='ones'),
            input_shape=(2, 1)))
    model.compile(
        RMSPropOptimizer(learning_rate=0.001),
        loss='mse',
        sample_weight_mode='temporal',
        weighted_metrics=['accuracy', 'mse'])
    y = np.array([[[1.], [1.]], [[1.], [1.]]])

    outs = model.evaluate(x, y)
    self.assertEqual(outs, [0.5, 0.5, 0.5])

    w = np.array([[0., 0.], [0., 0.]])
    outs = model.evaluate(x, y, sample_weight=w)
    self.assertEqual(outs, [0., 0., 0.])

    w = np.array([[3., 4.], [1., 2.]])
    outs = model.evaluate(x, y, sample_weight=w)
    self.assertArrayNear(outs, [0.3, 0.7, 0.3], .001)

    # Verify that metric value is same with arbitrary weights and batch size.
    x = np.random.random((50, 2, 1))
    y = np.random.random((50, 2, 1))
    w = np.random.random((50, 2))
    mse1 = model.evaluate(x, y, sample_weight=w, batch_size=5)[2]
    mse2 = model.evaluate(x, y, sample_weight=w, batch_size=10)[2]
    self.assertNear(mse1, mse2, err=1e-7)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_metric_state_reset_between_fit_and_evaluate(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation='relu', input_dim=4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    acc_obj = metrics_module.BinaryAccuracy()
    model.compile(
        loss='mae',
        metrics=[acc_obj],
        optimizer=RMSPropOptimizer(learning_rate=0.001))

    x_train = np.random.random((100, 4))
    y_train = np.random.random((100, 1))
    model.fit(x_train, y_train, batch_size=5, epochs=2)
    self.assertEqual(self.evaluate(acc_obj.count), 100)

    x_test = np.random.random((10, 4))
    y_test = np.random.random((10, 1))
    model.evaluate(x_test, y_test, batch_size=5)
    self.assertEqual(self.evaluate(acc_obj.count), 10)

  @tf_test_util.run_in_graph_and_eager_modes
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
          metrics=metrics_module.CategoricalAccuracy())

  @tf_test_util.run_in_graph_and_eager_modes
  def test_metrics_masking(self):
    with self.cached_session():
      np.random.seed(1337)
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='ones')))
      model.compile(
          RMSPropOptimizer(learning_rate=0.001),
          loss='mse',
          weighted_metrics=['accuracy'])

      # verify that masking is applied.
      x = np.array([[[1], [1]], [[1], [1]], [[0], [0]]])
      y = np.array([[[1], [1]], [[0], [1]], [[1], [1]]])
      scores = model.train_on_batch(x, y)
      self.assertArrayNear(scores, [0.25, 0.75], 0.1)

      # verify that masking is combined with sample weights.
      w = np.array([3, 2, 4])
      scores = model.train_on_batch(x, y, sample_weight=w)
      self.assertArrayNear(scores, [0.2, 0.8], 0.1)

  def test_add_metric_with_tensor_on_model_in_graph_mode(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(1,))
      y = keras.layers.Dense(1, kernel_initializer='ones')(x)
      model = keras.models.Model(x, y)
      model.add_metric(
          math_ops.reduce_sum(y), name='metric_1', aggregation='mean')

      # test with a metric which does not have the standard signature:
      # (y_true, y_pred, sample_Weight)
      model.add_metric(metrics_module.Mean(name='metric_2')(y))
      model.compile('sgd', loss='mse')

      inputs = np.ones(shape=(10, 1))
      targets = np.ones(shape=(10, 1))
      history = model.fit(
          inputs,
          targets,
          epochs=2,
          batch_size=5,
          validation_data=(inputs, targets))
      self.assertEqual(history.history['metric_1'][-1], 5)
      self.assertEqual(history.history['metric_2'][-1], 1)
      self.assertEqual(history.history['val_metric_1'][-1], 5)
      self.assertEqual(history.history['val_metric_2'][-1], 1)

      eval_results = model.evaluate(inputs, targets, batch_size=5)
      self.assertEqual(eval_results[-1], 1)
      self.assertEqual(eval_results[-2], 5)

      model.predict(inputs, batch_size=5)
      model.train_on_batch(inputs, targets)
      model.test_on_batch(inputs, targets)

  @tf_test_util.run_in_graph_and_eager_modes
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
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

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

  def test_add_metric_in_model_call_run_eagerly(self):
    with context.eager_mode():

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
          loss='mse', optimizer=RMSPropOptimizer(0.01), run_eagerly=True)

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

  @tf_test_util.run_in_graph_and_eager_modes
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

    model = keras.Sequential()
    model.add(TestLayer(input_shape=(1,)))
    model.add(keras.layers.Dense(2, kernel_initializer='ones'))
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertEqual(history.history['metric_1'][-1], 5)
    self.assertAlmostEqual(history.history['val_metric_1'][-1], 5, 0)

  def test_add_metric_in_layer_call_run_eagerly(self):
    with context.eager_mode():

      class TestLayer(keras.layers.Layer):

        def build(self, input_shape):
          self.a = self.add_variable(
              'a', (1, 1), initializer='ones', trainable=False)
          self.built = True

        def call(self, inputs):
          self.add_metric(
              math_ops.reduce_sum(inputs), name='metric_1', aggregation='mean')
          return inputs + 1

      model = keras.Sequential()
      model.add(TestLayer(input_shape=(1,)))
      model.add(keras.layers.Dense(2, kernel_initializer='ones'))
      model.compile(
          loss='mse', optimizer=RMSPropOptimizer(0.01), run_eagerly=True)

      x = np.ones(shape=(10, 1))
      y = np.ones(shape=(10, 2))
      history = model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
      self.assertEqual(history.history['metric_1'][-1], 5)
      self.assertAlmostEqual(history.history['val_metric_1'][-1], 5, 0)

  def test_model_metrics_list(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(1,))
      y = keras.layers.Dense(1, kernel_initializer='ones')(x)
      model = keras.models.Model(x, y)
      model.add_metric(
          math_ops.reduce_sum(y), name='metric_1', aggregation='mean')
      model.add_metric(metrics_module.Mean(name='metric_2')(y))
      model.compile('sgd', loss='mse', metrics=['acc'])

      # Verify that the metrics added using `compile` and `add_metric` API are
      # included
      self.assertEqual(model._compile_metrics, ['acc'])
      names = []
      for m in model.metrics:
        if isinstance(m, metrics_module.Metric):
          names.append(m.name)
        else:
          names.append(m.__name__)
      self.assertEqual(names, ['binary_accuracy', 'metric_1', 'metric_2'])

  def test_model_eager_metrics_list(self):
    with context.eager_mode():

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
          metrics=['acc'],
          run_eagerly=True)
      x = np.ones(shape=(10, 1))
      y = np.ones(shape=(10, 2))
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

      self.assertEqual(model._compile_metrics, ['acc'])
      names = []
      for m in model.metrics:
        if isinstance(m, metrics_module.Metric):
          names.append(m.name)
        else:
          names.append(m.__name__)
      self.assertEqual(names, ['categorical_accuracy', 'metric_1'])

  @tf_test_util.run_in_graph_and_eager_modes
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
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

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

  def test_invalid_metric_tensor_in_call(self):
    with context.eager_mode():

      class TestLayer(keras.layers.Layer):

        def call(self, inputs):
          self.add_metric(metrics_module.Mean(name='metric_1')(inputs))
          return inputs + 1

      model = keras.Sequential()
      model.add(TestLayer(input_shape=(1,)))
      model.add(keras.layers.Dense(2, kernel_initializer='ones'))
      model.compile(
          loss='mse', optimizer=RMSPropOptimizer(0.01), run_eagerly=True)

      x = np.ones(shape=(10, 1))
      y = np.ones(shape=(10, 2))
      with self.assertRaisesRegexp(
          ValueError,
          'We do not support adding an aggregated metric tensor in `call` in '
          'eager execution.'):
        model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @tf_test_util.run_in_graph_and_eager_modes
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
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))

    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    with self.assertRaisesRegexp(
        ValueError,
        'Please provide different names for the metrics you have added. '
        'We found 2 metrics with the name: "metric_1"'):
      model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))

  @tf_test_util.run_in_graph_and_eager_modes
  def test_multiple_no_name_input_to_add_metric(self):

    class TestModel(keras.Model):

      def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense1 = keras.layers.Dense(2, kernel_initializer='ones')

      def call(self, x):
        self.add_metric(math_ops.reduce_sum(x), aggregation='mean')
        self.add_metric(math_ops.reduce_sum(x), aggregation='mean')
        return self.dense1(x)

    model = TestModel()
    model.compile(loss='mse', optimizer=RMSPropOptimizer(0.01))
    x = np.ones(shape=(10, 1))
    y = np.ones(shape=(10, 2))
    model.fit(x, y, epochs=2, batch_size=5, validation_data=(x, y))
    self.assertEqual([m.name for m in model.metrics], ['mean', 'mean_1'])


if __name__ == '__main__':
  test.main()
