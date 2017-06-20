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

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import testing_utils
from tensorflow.contrib.keras.python.keras.engine.training import _weighted_masked_objective
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

  def test_generator_methods(self):
    arr_data = np.random.randint(0, 256, (50, 2))
    arr_labels = np.random.randint(0, 2, 50)

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

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(2,)))
    model.compile(loss='mse', optimizer='sgd')

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        workers=4,
                        pickle_safe=True)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_q_size=10,
                        pickle_safe=False,
                        validation_data=custom_generator(),
                        validation_steps=10)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_q_size=10,
                            workers=2,
                            pickle_safe=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_q_size=10,
                            pickle_safe=False)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_q_size=10,
                             workers=2,
                             pickle_safe=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_q_size=10,
                             pickle_safe=False)


if __name__ == '__main__':
  test.main()
