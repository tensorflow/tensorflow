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

import os
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class TrainingTest(test.TestCase):

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
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights)

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
    model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])

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
        }, {'dense': output_d_np,
            'dropout': output_e_np},
        epochs=1,
        batch_size=5,
        verbose=0)
    model.fit(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {'dense': output_d_np,
            'dropout': output_e_np},
        epochs=1,
        batch_size=5,
        verbose=1)
    model.fit(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {'dense': output_d_np,
            'dropout': output_e_np},
        validation_data=({'input_a': input_a_np,
                          'input_b': input_b_np
                         },
                         {
                             'dense': output_d_np,
                             'dropout': output_e_np
                         }),
        epochs=1,
        batch_size=5,
        verbose=0)
    model.train_on_batch({
        'input_a': input_a_np,
        'input_b': input_b_np
    }, {'dense': output_d_np,
        'dropout': output_e_np})
    # Test with lists for loss, metrics
    loss = ['mae', 'mse']
    metrics = ['acc', 'mae']
    model.compile(optimizer, loss, metrics=metrics)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Test with dictionaries for loss, metrics, loss weights
    loss = {'dense': 'mse', 'dropout': 'mae'}
    loss_weights = {'dense': 1., 'dropout': 0.5}
    metrics = {'dense': 'mse', 'dropout': 'mae'}
    model.compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights)
    model.fit(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        epochs=1,
        batch_size=5,
        verbose=0)

    # Invalid use cases
    with self.assertRaises(AttributeError):
      model.fit(
          [input_a_np, input_b_np], [output_d_np, output_e_np],
          epochs=1,
          validation_data=([input_a_np, input_b_np], 0, 0),
          verbose=0)
    with self.assertRaises(ValueError):
      model.train_on_batch({'input_a': input_a_np},
                           [output_d_np, output_e_np])
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
    model.compile(optimizer=RMSPropOptimizer(learning_rate=0.001), loss='mse')
    # This will work
    model.fit([input_a_np], output_d_np, epochs=1)
    with self.assertRaises(ValueError):
      model.fit([input_a_np, input_a_np], output_d_np, epochs=1)

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
    metrics = ['mae']
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
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
    self.assertEqual(len(out), 5)
    out = model.evaluate(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        batch_size=5,
        verbose=1)
    self.assertEqual(len(out), 5)
    out = model.evaluate(
        [input_a_np, input_b_np], [output_d_np, output_e_np],
        batch_size=5,
        verbose=2)
    self.assertEqual(len(out), 5)
    out = model.test_on_batch([input_a_np, input_b_np],
                              [output_d_np, output_e_np])
    self.assertEqual(len(out), 5)

    # Test evaluate with dictionary inputs
    model.evaluate(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {'dense': output_d_np,
            'dropout': output_e_np},
        batch_size=5,
        verbose=0)
    model.evaluate(
        {
            'input_a': input_a_np,
            'input_b': input_b_np
        }, {'dense': output_d_np,
            'dropout': output_e_np},
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

  def test_invalid_loss_or_metrics(self):
    num_classes = 5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSPropOptimizer(learning_rate=0.001))
    np.random.seed(1337)

    (x_train, y_train), (_, _) = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)

    with self.assertRaises(ValueError):
      model.fit(x_train, np.concatenate([y_train, y_train], axis=-1))

    with self.assertRaises(TypeError):
      model.compile(loss='categorical_crossentropy',
                    optimizer=RMSPropOptimizer(learning_rate=0.001),
                    metrics=set(0))

    with self.assertRaises(ValueError):
      model.compile(loss=None,
                    optimizer='rms')


if __name__ == '__main__':
  # Bazel sets these environment variables to very long paths.
  # Tempfile uses them to create long paths, and in turn multiprocessing
  # library tries to create sockets named after paths. Delete whatever bazel
  # writes to these to avoid tests failing due to socket addresses being too
  # long.
  for var in ('TMPDIR', 'TMP', 'TEMP'):
    if var in os.environ:
      del os.environ[var]

  ops.enable_eager_execution()
  test.main()
