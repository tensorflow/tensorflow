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
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class TrainingTest(test.TestCase):

  def test_model_methods_with_eager_tensors_multi_io(self):
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
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        loss_weights=loss_weights,
        run_eagerly=True,
        sample_weight_mode=None)

    input_a = keras.backend.zeros(shape=(10, 3))
    input_b = keras.backend.zeros(shape=(10, 3))
    target_d = keras.backend.zeros(shape=(10, 4))
    target_e = keras.backend.zeros(shape=(10, 4))

    model.fit(
        [input_a, input_b], [target_d, target_e],
        epochs=1,
        batch_size=5,
        verbose=0)
    # Test: no shuffle.
    model.fit(
        [input_a, input_b], [target_d, target_e],
        epochs=1,
        batch_size=5,
        verbose=0,
        shuffle=False)
    # Test: validation data.
    model.fit([input_a, input_b], [target_d, target_e],
              epochs=1, batch_size=2, verbose=0,
              validation_data=([input_a, input_b], [target_d, target_e]))
    model.train_on_batch([input_a, input_b], [target_d, target_e])
    model.predict([input_a, input_b], batch_size=5)
    model.evaluate([input_a, input_b], [target_d, target_e],
                   batch_size=2, verbose=0)
    model.test_on_batch([input_a, input_b], [target_d, target_e])

    # Test: mix np and tensors.
    input_b = np.zeros(shape=(10, 3)).astype('float32')
    target_e = np.zeros(shape=(10, 4)).astype('float32')
    model.fit(
        [input_a, input_b], [target_d, target_e],
        epochs=1,
        batch_size=5,
        verbose=0)
    model.fit([input_a, input_b], [target_d, target_e],
              epochs=1, batch_size=2, verbose=0,
              validation_data=([input_a, input_b], [target_d, target_e]))
    model.fit(
        [input_a, input_b], [target_d, target_e],
        epochs=1,
        batch_size=5,
        verbose=0,
        shuffle=False)
    model.train_on_batch([input_a, input_b], [target_d, target_e])
    model.predict([input_a, input_b], batch_size=5)
    model.evaluate([input_a, input_b], [target_d, target_e],
                   batch_size=2, verbose=0)
    model.test_on_batch([input_a, input_b], [target_d, target_e])

  def test_model_methods_with_eager_tensors_single_io(self):
    x = keras.layers.Input(shape=(3,), name='input')
    y = keras.layers.Dense(4, name='dense')(x)
    model = keras.Model(x, y)

    optimizer = RMSPropOptimizer(learning_rate=0.001)
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(optimizer, loss, metrics=metrics, run_eagerly=True)

    inputs = keras.backend.zeros(shape=(10, 3))
    targets = keras.backend.zeros(shape=(10, 4))

    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=0)
    model.fit(inputs, targets, epochs=1, batch_size=3, verbose=0, shuffle=False)
    model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0,
              validation_data=(inputs, targets))
    model.evaluate(inputs, targets, batch_size=2, verbose=0)
    model.predict(inputs, batch_size=2)
    model.train_on_batch(inputs, targets)
    model.test_on_batch(inputs, targets)

  def test_model_fit_and_validation_with_missing_arg_errors(self):
    x = keras.layers.Input(shape=(3,), name='input')
    y = keras.layers.Dense(4, name='dense')(x)
    model = keras.Model(x, y)
    model.compile(optimizer=RMSPropOptimizer(learning_rate=0.001),
                  loss='mse',
                  run_eagerly=True)

    x = keras.backend.zeros(shape=(10, 3))
    y = keras.backend.zeros(shape=(10, 4))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat(10).batch(5)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    validation_dataset = dataset_ops.Dataset.from_tensor_slices(
        (x, y)).repeat(10).batch(5)
    validation_iterator = dataset_ops.make_one_shot_iterator(validation_dataset)

    with self.assertRaisesRegexp(
        ValueError, r'specify .* `steps_per_epoch`'):
      model.fit(iterator, epochs=1, verbose=0)
    if not context.executing_eagerly():
      # In eager execution, `keras.backend.zeros` returns value tensors
      # which can be used for validation without a `validation_steps` argument.
      with self.assertRaisesRegexp(
          ValueError, r'provide either `batch_size` or `validation_steps`'):
        model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                  validation_data=(x, y))
    with self.assertRaisesRegexp(ValueError,
                                 'specify the `validation_steps` argument.'):
      model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                validation_data=validation_dataset)
    with self.assertRaisesRegexp(ValueError,
                                 'specify the `validation_steps` argument.'):
      model.fit(iterator, steps_per_epoch=2, epochs=1, verbose=0,
                validation_data=validation_iterator)

  def test_generator_methods(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(3,)))
    optimizer = RMSPropOptimizer(learning_rate=0.001)
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


class CorrectnessTest(test.TestCase):

  def test_loss_correctness(self):
    # Test that training loss is the same in eager and graph
    # (by comparing it to a reference value in a deterministic case)
    model = keras.Sequential()
    model.add(keras.layers.Dense(3,
                                 activation='relu',
                                 input_dim=4,
                                 kernel_initializer='ones'))
    model.add(keras.layers.Dense(2,
                                 activation='softmax',
                                 kernel_initializer='ones'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=RMSPropOptimizer(learning_rate=0.001),
                  run_eagerly=False)
    x = np.ones((100, 4))
    np.random.seed(123)
    y = np.random.randint(0, 1, size=(100, 1))
    history = model.fit(x, y, epochs=1, batch_size=10)
    self.assertAlmostEqual(history.history['loss'][-1], 0.6173, 4)

  def test_loss_correctness_with_iterator(self):
    # Test that training loss is the same in eager and graph
    # (by comparing it to a reference value in a deterministic case)
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            3, activation='relu', input_dim=4, kernel_initializer='ones'))
    model.add(
        keras.layers.Dense(2, activation='softmax', kernel_initializer='ones'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=RMSPropOptimizer(learning_rate=0.001),
        run_eagerly=True)
    x = np.ones((100, 4), dtype=np.float32)
    np.random.seed(123)
    y = np.random.randint(0, 1, size=(100, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    history = model.fit(iterator, epochs=1, steps_per_epoch=10)
    self.assertAlmostEqual(history.history['loss'][-1], 0.6173, 4)

  def test_loss_in_call(self):

    class HasLoss(keras.layers.Layer):

      def call(self, x):
        self.add_loss(x)
        return x

    layer = HasLoss()
    layer(1.)  # Plain-value inputs are only valid in eager mode.
    self.assertEqual(1, len(layer.losses))

  def test_predict_correctness(self):
    i1 = keras.layers.Input(shape=(4, 5))
    i2 = keras.layers.Input(shape=(4, 5))
    i3 = keras.layers.Input(shape=(4, 5))
    o = keras.layers.add([i1, i2, i3])
    model = keras.models.Model([i1, i2, i3], o)
    model.run_eagerly = True

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    x3 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2, x3])

    self.assertAllClose(out, x1 + x2 + x3)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
