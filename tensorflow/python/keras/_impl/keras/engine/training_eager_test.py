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

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util as tf_test_util
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
    metrics = ['acc', 'mae']
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
    metrics = ['mae']
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        loss_weights=loss_weights,
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
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics)

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


class LossWeightingTest(test.TestCase):

  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    weighted_class = 3
    train_samples = 300
    test_samples = 300
    input_dim = 5

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSPropOptimizer(learning_rate=0.001))

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
    class_weight[weighted_class] = 4.

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = 4.

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
        verbose=0,
        class_weight=class_weight,
        validation_data=(x_train, y_train, sample_weight))
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
        verbose=0,
        class_weight=class_weight)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
        verbose=0,
        class_weight=class_weight,
        validation_split=0.1)

    model.train_on_batch(
        x_train[:batch_size], y_train[:batch_size], class_weight=class_weight)
    ref_score = model.evaluate(x_test, y_test, verbose=0)
    score = model.evaluate(
        x_test[test_ids, :], y_test[test_ids, :], verbose=0)
    self.assertLess(score, ref_score)

  def test_sample_weights(self):
    num_classes = 5
    batch_size = 5
    weighted_class = 3
    train_samples = 300
    test_samples = 300
    input_dim = 5

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSPropOptimizer(learning_rate=0.001))

    np.random.seed(43)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_train = y_train.copy()
    y_train = keras.utils.to_categorical(y_train, num_classes)

    class_weight = dict([(i, 1.) for i in range(num_classes)])
    class_weight[weighted_class] = 4.

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = 4.

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
        verbose=0,
        sample_weight=sample_weight)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=2,
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

  def test_temporal_sample_weights(self):
    num_classes = 5
    weighted_class = 3
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3

    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(num_classes),
            input_shape=(timesteps, input_dim)))
    model.add(keras.layers.Activation('softmax'))

    np.random.seed(1337)
    (_, y_train), _ = testing_utils.get_test_data(
        train_samples=train_samples,
        test_samples=test_samples,
        input_shape=(input_dim,),
        num_classes=num_classes)
    int_y_train = y_train.copy()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    class_weight = dict([(i, 1.) for i in range(num_classes)])
    class_weight[weighted_class] = 2.

    sample_weight = np.ones((y_train.shape[0]))
    sample_weight[int_y_train == weighted_class] = 2.
    with self.assertRaises(ValueError):
      model.compile(
          loss='binary_crossentropy',
          optimizer=RMSPropOptimizer(learning_rate=0.001),
          sample_weight_mode='temporal')

  def test_class_weight_invalid_use_case(self):
    num_classes = 5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3

    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(num_classes),
            input_shape=(timesteps, input_dim)))
    model.add(keras.layers.Activation('softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSPropOptimizer(learning_rate=0.001))

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
          loss='binary_crossentropy',
          optimizer=RMSPropOptimizer(learning_rate=0.001),
          sample_weight_mode=[])

    # Build multi-output model
    x = keras.Input((3,))
    y1 = keras.layers.Dense(4, name='1')(x)
    y2 = keras.layers.Dense(4, name='2')(x)
    model = keras.models.Model(x, [y1, y2])
    model.compile(optimizer=RMSPropOptimizer(learning_rate=0.001), loss='mse')
    x_np = np.random.random((10, 3))
    y_np = np.random.random((10, 4))
    w_np = np.random.random((10,))
    # This will work
    model.fit(x_np, [y_np, y_np], epochs=1, sample_weight={'1': w_np})
    # These will not
    with self.assertRaises(ValueError):
      model.fit(x_np, [y_np, y_np], epochs=1, sample_weight=[w_np])
    with self.assertRaises(TypeError):
      model.fit(x_np, [y_np, y_np], epochs=1, sample_weight=w_np)
    with self.assertRaises(ValueError):
      bad_w_np = np.random.random((11,))
      model.fit(x_np, [y_np, y_np], epochs=1, sample_weight={'1': bad_w_np})
    with self.assertRaises(ValueError):
      bad_w_np = np.random.random((10, 2))
      model.fit(x_np, [y_np, y_np], epochs=1, sample_weight={'1': bad_w_np})
    with self.assertRaises(ValueError):
      bad_w_np = np.random.random((10, 2, 2))
      model.fit(x_np, [y_np, y_np], epochs=1, sample_weight={'1': bad_w_np})


class CorrectnessTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
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
                  optimizer=RMSPropOptimizer(learning_rate=0.001))
    x = np.ones((100, 4))
    np.random.seed(123)
    y = np.random.randint(0, 1, size=(100, 1))
    history = model.fit(x, y, epochs=1, batch_size=10)
    self.assertEqual(
        np.around(history.history['loss'][-1], decimals=4), 0.6173)

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_metrics_correctness(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(3,
                                 activation='relu',
                                 input_dim=4,
                                 kernel_initializer='ones'))
    model.add(keras.layers.Dense(1,
                                 activation='sigmoid',
                                 kernel_initializer='ones'))
    model.compile(loss='mae',
                  metrics=['acc'],
                  optimizer=RMSPropOptimizer(learning_rate=0.001))
    x = np.ones((100, 4))
    y = np.ones((100, 1))
    outs = model.evaluate(x, y)
    self.assertEqual(outs[1], 1.)
    y = np.zeros((100, 1))
    outs = model.evaluate(x, y)
    self.assertEqual(outs[1], 0.)

if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
