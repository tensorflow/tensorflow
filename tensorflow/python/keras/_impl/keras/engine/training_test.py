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
import unittest

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.keras._impl.keras.engine.training import _weighted_masked_objective
from tensorflow.python.platform import test


class TrainingTest(test.TestCase):

  def test_fit_on_arrays(self):
    with self.test_session():
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
      with self.assertRaises(ValueError):
        model.train_on_batch({'input_a': input_a_np},
                             [output_d_np, output_e_np])
      with self.assertRaises(TypeError):
        model.fit(
            [input_a_np, input_b_np], [output_d_np, output_e_np],
            epochs=1,
            validation_data=([input_a_np, input_b_np], 0, 0),
            verbose=0)
      with self.assertRaises(ValueError):
        model.train_on_batch([input_a_np], [output_d_np, output_e_np])
      with self.assertRaises(TypeError):
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
      model.compile(optimizer='rmsprop', loss='mse')
      # This will work
      model.fit([input_a_np], output_d_np, epochs=1)
      with self.assertRaises(ValueError):
        model.fit([input_a_np, input_a_np], output_d_np, epochs=1)

  def test_evaluate_predict_on_arrays(self):
    with self.test_session():
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

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
      model.add(keras.layers.Activation('relu'))
      model.add(keras.layers.Dense(num_classes))
      model.add(keras.layers.Activation('softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
      np.random.seed(1337)
      (x_train, y_train), (_, _) = testing_utils.get_test_data(
          train_samples=train_samples,
          test_samples=test_samples,
          input_shape=(input_dim,),
          num_classes=num_classes)
      with self.assertRaises(ValueError):
        model.fit(x_train, y_train)

      with self.assertRaises(ValueError):
        model.fit(x_train, np.concatenate([y_train, y_train], axis=-1))

      with self.assertRaises(TypeError):
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=set(0))

      with self.assertRaises(ValueError):
        model.compile(loss=None,
                      optimizer='rmsprop')


class LossWeightingTest(test.TestCase):

  def test_class_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
    train_samples = 1000
    test_samples = 1000
    input_dim = 5

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
      model.add(keras.layers.Activation('relu'))
      model.add(keras.layers.Dense(num_classes))
      model.add(keras.layers.Activation('softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

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
      self.assertLess(score, ref_score)

  def test_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
    train_samples = 1000
    test_samples = 1000
    input_dim = 5

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(10, input_shape=(input_dim,)))
      model.add(keras.layers.Activation('relu'))
      model.add(keras.layers.Dense(num_classes))
      model.add(keras.layers.Activation('softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

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
      score = model.evaluate(
          x_test[test_ids, :], y_test[test_ids, :], verbose=0)
      self.assertLess(score, ref_score)

  def test_temporal_sample_weights(self):
    num_classes = 5
    batch_size = 5
    epochs = 5
    weighted_class = 3
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3

    with self.test_session():
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

      class_weight = dict([(i, 1.) for i in range(num_classes)])
      class_weight[weighted_class] = 2.

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
          loss='binary_crossentropy',
          optimizer='rmsprop',
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
      score = model.evaluate(
          temporal_x_test[test_ids], temporal_y_test[test_ids], verbose=0)
      self.assertLess(score, ref_score)

  def test_class_weight_invalid_use_case(self):
    num_classes = 5
    train_samples = 1000
    test_samples = 1000
    input_dim = 5
    timesteps = 3

    with self.test_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(num_classes),
              input_shape=(timesteps, input_dim)))
      model.add(keras.layers.Activation('softmax'))
      model.compile(
          loss='binary_crossentropy',
          optimizer='rmsprop')

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
            optimizer='rmsprop',
            sample_weight_mode=[])

      # Build multi-output model
      x = keras.Input((3,))
      y1 = keras.layers.Dense(4, name='1')(x)
      y2 = keras.layers.Dense(4, name='2')(x)
      model = keras.models.Model(x, [y1, y2])
      model.compile(optimizer='rmsprop', loss='mse')
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


class LossMaskingTest(test.TestCase):

  def test_masking(self):
    with self.test_session():
      np.random.seed(1337)
      x = np.array([[[1], [1]], [[0], [0]]])
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='one')))
      model.compile(loss='mse', optimizer='sgd')
      y = np.array([[[1], [1]], [[1], [1]]])
      loss = model.train_on_batch(x, y)
      self.assertEqual(loss, 0)

  def test_loss_masking(self):
    with self.test_session():
      weighted_loss = _weighted_masked_objective(keras.losses.get('mae'))
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
    with self.test_session():
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
    with self.test_session():
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
    with self.test_session():
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
    with self.test_session():
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
    with self.test_session():
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


class TestGeneratorMethods(test.TestCase):

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  def test_generator_methods(self):
    arr_data = np.random.random((50, 2))
    arr_labels = np.random.random((50,))

    def custom_generator():
      batch_size = 10
      n_samples = 50
      while True:
        batch_index = np.random.randint(0, n_samples - batch_size)
        start = batch_index
        end = start + batch_size
        x = arr_data[start: end]
        y = arr_labels[start: end]
        yield x, y

    with self.test_session():
      x = keras.Input((2,))
      y = keras.layers.Dense(1)(x)
      fn_model = keras.models.Model(x, y)
      fn_model.compile(loss='mse', optimizer='sgd')

      seq_model = keras.models.Sequential()
      seq_model.add(keras.layers.Dense(1, input_shape=(2,)))
      seq_model.compile(loss='mse', optimizer='sgd')

      for model in [fn_model, seq_model]:
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            workers=4,
                            use_multiprocessing=True)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False,
                            validation_data=custom_generator(),
                            validation_steps=10)
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            validation_data=custom_generator(),
                            validation_steps=1,
                            workers=0)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                workers=2,
                                use_multiprocessing=True)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                use_multiprocessing=False)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                workers=0)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 workers=2,
                                 use_multiprocessing=True)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False,
                                 workers=0)

        # Test legacy API
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_q_size=10,
                            workers=4,
                            pickle_safe=True)
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_q_size=10,
                                workers=2,
                                pickle_safe=True)
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_q_size=10,
                                 workers=2,
                                 pickle_safe=True)

  def test_generator_methods_with_sample_weights(self):
    arr_data = np.random.random((50, 2))
    arr_labels = np.random.random((50,))
    arr_sample_weights = np.random.random((50,))

    def custom_generator():
      batch_size = 10
      n_samples = 50
      while True:
        batch_index = np.random.randint(0, n_samples - batch_size)
        start = batch_index
        end = start + batch_size
        x = arr_data[start: end]
        y = arr_labels[start: end]
        w = arr_sample_weights[start: end]
        yield x, y, w

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
      model.compile(loss='mse', optimizer='sgd')

      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False)
      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False,
                          validation_data=custom_generator(),
                          validation_steps=10)
      model.predict_generator(custom_generator(),
                              steps=5,
                              max_queue_size=10,
                              use_multiprocessing=False)
      model.evaluate_generator(custom_generator(),
                               steps=5,
                               max_queue_size=10,
                               use_multiprocessing=False)

  def test_generator_methods_invalid_use_case(self):

    def custom_generator():
      while 1:
        yield 0

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
      model.compile(loss='mse', optimizer='sgd')

      with self.assertRaises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False)
      with self.assertRaises(ValueError):
        model.fit_generator(custom_generator(),
                            steps_per_epoch=5,
                            epochs=1,
                            verbose=1,
                            max_queue_size=10,
                            use_multiprocessing=False,
                            validation_data=custom_generator(),
                            validation_steps=10)
      with self.assertRaises(TypeError):
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                use_multiprocessing=False)
      with self.assertRaises(ValueError):
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False)


class TestTrainingUtils(test.TestCase):

  def test_check_array_lengths(self):
    keras.engine.training._check_array_lengths(None, None, None)
    a_np = np.random.random((4, 3, 3))
    keras.engine.training._check_array_lengths(a_np, a_np, a_np)
    keras.engine.training._check_array_lengths(
        [a_np, a_np], [a_np, a_np], [a_np, a_np])
    keras.engine.training._check_array_lengths([None], [None], [None])

    b_np = np.random.random((3, 4))
    with self.assertRaises(ValueError):
      keras.engine.training._check_array_lengths(a_np, None, None)
    with self.assertRaises(ValueError):
      keras.engine.training._check_array_lengths(a_np, a_np, None)
    with self.assertRaises(ValueError):
      keras.engine.training._check_array_lengths([a_np], [None], None)
    with self.assertRaises(ValueError):
      keras.engine.training._check_array_lengths([a_np], [b_np], None)
    with self.assertRaises(ValueError):
      keras.engine.training._check_array_lengths([a_np], None, [b_np])

  def test_slice_arrays(self):
    input_a = np.random.random((10, 3))
    keras.engine.training._slice_arrays(None)
    keras.engine.training._slice_arrays(input_a, 0)
    keras.engine.training._slice_arrays(input_a, 0, 1)
    keras.engine.training._slice_arrays(input_a, stop=2)
    input_a = [None, [1, 1], None, [1, 1]]
    keras.engine.training._slice_arrays(input_a, 0)
    keras.engine.training._slice_arrays(input_a, 0, 1)
    keras.engine.training._slice_arrays(input_a, stop=2)
    input_a = [None]
    keras.engine.training._slice_arrays(input_a, 0)
    keras.engine.training._slice_arrays(input_a, 0, 1)
    keras.engine.training._slice_arrays(input_a, stop=2)
    input_a = None
    keras.engine.training._slice_arrays(input_a, 0)
    keras.engine.training._slice_arrays(input_a, 0, 1)
    keras.engine.training._slice_arrays(input_a, stop=2)


class TestTrainingWithDataTensors(test.TestCase):

  def test_model_with_input_feed_tensor(self):
    """We test building a model with a TF variable as input.

    We should be able to call fit, evaluate, predict,
    by only passing them data for the placeholder inputs
    in the model.
    """
    with self.test_session():
      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 3))

      output_a_np = np.random.random((10, 4))
      output_b_np = np.random.random((10, 3))

      a = keras.Input(
          tensor=keras.backend.variables_module.Variable(input_a_np,
                                                         dtype='float32'))
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
      a = keras.Input(
          tensor=keras.backend.variables_module.Variable(input_a_np,
                                                         dtype='float32'))
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
      out = model.fit(None,
                      output_a_np, epochs=1, batch_size=10)
      out = model.fit(None,
                      output_a_np, epochs=1, batch_size=10)

      # test evaluate
      out = model.evaluate(None,
                           output_a_np, batch_size=10)
      out = model.evaluate(None,
                           output_a_np, batch_size=10)

      # test predict
      out = model.predict(None, steps=3)
      out = model.predict(None, steps=3)
      self.assertEqual(out.shape, (10 * 3, 4))

      # Same, without learning phase
      # i.e. we don't pass any data to fit the model.
      a = keras.Input(
          tensor=keras.backend.variables_module.Variable(input_a_np,
                                                         dtype='float32'))
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
      out = model.fit(None,
                      output_a_np, epochs=1, batch_size=10)
      out = model.fit(None,
                      output_a_np, epochs=1, batch_size=10)

      # test evaluate
      out = model.evaluate(None,
                           output_a_np, batch_size=10)
      out = model.evaluate(None,
                           output_a_np, batch_size=10)

      # test predict
      out = model.predict(None, steps=3)
      out = model.predict(None, steps=3)
      self.assertEqual(out.shape, (10 * 3, 4))

  def test_model_with_partial_loss(self):
    with self.test_session():
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
    with self.test_session():
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
      a = keras.Input(
          tensor=keras.backend.variables_module.Variable(input_a_np,
                                                         dtype='float32'))
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
      a = keras.Input(
          tensor=keras.backend.variables_module.Variable(input_a_np,
                                                         dtype='float32'))
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
    with self.test_session():
      # single-output, as list
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,), name='dense'))
      input_val = np.random.random((10, 4))
      target_val = np.random.random((10, 4))
      target = keras.backend.variable(target_val)
      model.compile(optimizer='rmsprop', loss='mse', target_tensors=[target])
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
      model.compile(optimizer='rmsprop', loss='mse',
                    target_tensors=[target_a, target_b])
      model.train_on_batch(input_val, None,
                           sample_weight={'dense_a': np.random.random((10,))})

  def test_model_custom_target_tensors(self):
    with self.test_session():
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


if __name__ == '__main__':
  # Bazel sets these environment variables to very long paths.
  # Tempfile uses them to create long paths, and in turn multiprocessing
  # library tries to create sockets named after paths. Delete whatever bazel
  # writes to these to avoid tests failing due to socket addresses being too
  # long.
  for var in ('TMPDIR', 'TMP', 'TEMP'):
    if var in os.environ:
      del os.environ[var]

  test.main()
