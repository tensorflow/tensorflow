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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class TrainingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  @keras_parameterized.run_all_keras_modes
  def test_model_methods_with_eager_tensors_multi_io(self):
    if not context.executing_eagerly():
      # Only test V2 Function and V2 Eager modes, as V1 Graph mode with
      # symbolic tensors has different requirements.
      return

    input_a = keras.layers.Input(shape=(3,), name='input_a')
    input_b = keras.layers.Input(shape=(3,), name='input_b')

    dense = keras.layers.Dense(4, name='dense')
    dropout = keras.layers.Dropout(0.5, name='dropout')

    model = testing_utils.get_multi_io_model(
        [input_a, dense], [input_b, dense, dropout])

    optimizer = rmsprop.RMSprop(learning_rate=0.001)
    loss = 'mse'
    loss_weights = [1., 0.5]
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        loss_weights=loss_weights,
        run_eagerly=testing_utils.should_run_eagerly(),
        sample_weight_mode=None)

    input_a = array_ops.zeros(shape=(10, 3))
    input_b = array_ops.zeros(shape=(10, 3))
    target_a = array_ops.zeros(shape=(10, 4))
    target_b = array_ops.zeros(shape=(10, 4))

    model.fit(
        [input_a, input_b], [target_a, target_b],
        epochs=1,
        batch_size=5,
        verbose=0)
    # Test: no shuffle.
    model.fit(
        [input_a, input_b], [target_a, target_b],
        epochs=1,
        batch_size=5,
        verbose=0,
        shuffle=False)
    # Test: validation data.
    model.fit([input_a, input_b], [target_a, target_b],
              epochs=1, batch_size=2, verbose=0,
              validation_data=([input_a, input_b], [target_a, target_b]))
    model.train_on_batch([input_a, input_b], [target_a, target_b])
    model.predict([input_a, input_b], batch_size=5)
    model.evaluate([input_a, input_b], [target_a, target_b],
                   batch_size=2, verbose=0)
    model.test_on_batch([input_a, input_b], [target_a, target_b])

    # Test: mix np and tensors.
    input_b = np.zeros(shape=(10, 3)).astype('float32')
    target_b = np.zeros(shape=(10, 4)).astype('float32')
    model.fit(
        [input_a, input_b], [target_a, target_b],
        epochs=1,
        batch_size=5,
        verbose=0)
    model.fit([input_a, input_b], [target_a, target_b],
              epochs=1, batch_size=2, verbose=0,
              validation_data=([input_a, input_b], [target_a, target_b]))
    model.fit(
        [input_a, input_b], [target_a, target_b],
        epochs=1,
        batch_size=5,
        verbose=0,
        shuffle=False)
    model.train_on_batch([input_a, input_b], [target_a, target_b])
    model.predict([input_a, input_b], batch_size=5)
    model.evaluate([input_a, input_b], [target_a, target_b],
                   batch_size=2, verbose=0)
    model.test_on_batch([input_a, input_b], [target_a, target_b])

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_model_methods_with_eager_tensors_single_io(self):
    if not context.executing_eagerly():
      # Only test V2 Function and V2 Eager modes, as V1 Graph mode with
      # symbolic tensors has different requirements.
      return

    model = testing_utils.get_small_mlp(10, 4, 3)

    optimizer = rmsprop.RMSprop(learning_rate=0.001)
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        run_eagerly=testing_utils.should_run_eagerly())

    inputs = array_ops.zeros(shape=(10, 3))
    targets = array_ops.zeros(shape=(10, 4))

    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=0)
    model.fit(inputs, targets, epochs=1, batch_size=3, verbose=0, shuffle=False)
    model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0,
              validation_data=(inputs, targets))
    model.evaluate(inputs, targets, batch_size=2, verbose=0)
    model.predict(inputs, batch_size=2)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)

  @keras_parameterized.run_with_all_model_types
  def test_model_fit_and_validation_with_missing_arg_errors(self):
    model = testing_utils.get_small_mlp(10, 4, 3)
    model.compile(optimizer=rmsprop.RMSprop(learning_rate=0.001),
                  loss='mse',
                  run_eagerly=True)

    x = array_ops.zeros(shape=(10, 3))
    y = array_ops.zeros(shape=(10, 4))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat(10).batch(5)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    validation_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).repeat().batch(5)  # Infinite dataset.
    validation_iterator = dataset_ops.make_one_shot_iterator(validation_dataset)

    with self.assertRaisesRegexp(
        ValueError, r'specify .* `steps_per_epoch`'):
      model.fit(iterator, epochs=1, verbose=0)
    if not context.executing_eagerly():
      # In eager execution, `array_ops.zeros` returns value tensors
      # which can be used for validation without a `validation_steps` argument.
      with self.assertRaisesRegexp(
          ValueError, r'provide either `batch_size` or `validation_steps`'):
        model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                  validation_data=(x, y))
    # Step argument is required for infinite datasets.
    with self.assertRaisesRegexp(ValueError,
                                 'specify the `validation_steps` argument.'):
      model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                validation_data=validation_dataset)
    with self.assertRaisesRegexp(ValueError,
                                 'specify the `validation_steps` argument.'):
      model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                validation_data=validation_iterator)

  # TODO(b/120931266): Enable test on subclassed models after bug causing an
  # extra dimension to be added to predict outputs is fixed.
  @keras_parameterized.run_with_all_model_types(exclude_models='subclass')
  def test_generator_methods(self):
    model = testing_utils.get_small_mlp(10, 4, 3)
    optimizer = rmsprop.RMSprop(learning_rate=0.001)
    model.compile(
        optimizer,
        loss='mse',
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=True)

    x = np.random.random((10, 3))
    y = np.random.random((10, 4))

    def numpy_iterator():
      while True:
        yield x, y

    model.fit_generator(numpy_iterator(), steps_per_epoch=3, epochs=1)
    model.evaluate_generator(numpy_iterator(), steps=3)

    def inference_numpy_iterator():
      while True:
        yield x

    out = model.predict_generator(inference_numpy_iterator(), steps=3)
    self.assertEqual(out.shape, (30, 4))


class CorrectnessTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_loss_correctness(self):
    # Test that training loss is the same in eager and graph
    # (by comparing it to a reference value in a deterministic case)
    layers = [
        keras.layers.Dense(3, activation='relu',
                           kernel_initializer='ones'),
        keras.layers.Dense(2, activation='softmax', kernel_initializer='ones')]
    model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop.RMSprop(learning_rate=0.001),
                  run_eagerly=testing_utils.should_run_eagerly())
    x = np.ones((100, 4))
    np.random.seed(123)
    y = np.random.randint(0, 1, size=(100, 1))
    history = model.fit(x, y, epochs=1, batch_size=10)
    self.assertAlmostEqual(history.history['loss'][-1], 0.5836, 4)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_loss_correctness_with_iterator(self):
    # Test that training loss is the same in eager and graph
    # (by comparing it to a reference value in a deterministic case)
    layers = [
        keras.layers.Dense(3, activation='relu',
                           kernel_initializer='ones'),
        keras.layers.Dense(2, activation='softmax', kernel_initializer='ones')]
    model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=rmsprop.RMSprop(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly())
    x = np.ones((100, 4), dtype=np.float32)
    np.random.seed(123)
    y = np.random.randint(0, 1, size=(100, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    history = model.fit(iterator, epochs=1, steps_per_epoch=10)
    self.assertAlmostEqual(history.history['loss'][-1], 0.5836, 4)

  def test_loss_in_call(self):

    class HasLoss(keras.layers.Layer):

      def call(self, x):
        self.add_loss(x)
        return x

    layer = HasLoss()
    layer(1.)  # Plain-value inputs are only valid in eager mode.
    self.assertEqual(1, len(layer.losses))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
