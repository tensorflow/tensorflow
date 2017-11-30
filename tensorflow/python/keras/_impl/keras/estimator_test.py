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

from math import log10
import os
import tempfile

import numpy as np

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(
      keras.layers.Conv2D(
          32, kernel_size=(3, 3), activation='relu', input_shape=(14, 14, 3)))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(16, activation='relu'))
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Dense(3, activation='softmax'))
  return model


def simple_functional_model():
  a = keras.layers.Input(shape=(14, 14, 3))
  b = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(a)
  b = keras.layers.MaxPooling2D(pool_size=(2, 2))(b)
  b = keras.layers.Dropout(0.25)(b)
  b = keras.layers.Flatten()(b)
  b = keras.layers.Dense(16, activation='relu')(b)
  b = keras.layers.Dropout(0.25)(b)
  b = keras.layers.Dense(3, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def get_resource_for_simple_model(is_sequential, is_evaluate):
  model = simple_sequential_model(
  ) if is_sequential else simple_functional_model()
  if is_sequential:
    model.build()
  input_name = model.input_names[0]

  np.random.seed(1337)
  (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
      train_samples=200,
      test_samples=100,
      input_shape=(14, 14, 3),
      num_classes=3)
  y_train = keras.utils.to_categorical(y_train)
  y_test = keras.utils.to_categorical(y_test)

  train_input_fn = numpy_io.numpy_input_fn(
      x={input_name: np.array(x_train, dtype=np.float32)},
      y=np.array(y_train, dtype=np.float32),
      shuffle=False,
      num_epochs=None,
      batch_size=16)

  evaluate_input_fn = numpy_io.numpy_input_fn(
      x={input_name: np.array(x_test, dtype=np.float32)},
      y=np.array(y_test, dtype=np.float32),
      num_epochs=1,
      shuffle=False)

  predict_input_fn = numpy_io.numpy_input_fn(
      x={input_name: np.array(x_test, dtype=np.float32)},
      num_epochs=1,
      shuffle=False)

  inference_input_fn = evaluate_input_fn if is_evaluate else predict_input_fn

  return model, (x_train, y_train), (x_test,
                                     y_test), train_input_fn, inference_input_fn


def multi_inputs_multi_outputs_model():
  # test multi-input layer
  a = keras.layers.Input(shape=(32,), name='input_a')
  b = keras.layers.Input(shape=(32,), name='input_b')
  dense = keras.layers.Dense(16, name='dense_1')
  a_2 = dense(a)
  b_2 = dense(b)
  merged = keras.layers.concatenate([a_2, b_2], name='merge')
  c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(inputs=[a, b], outputs=[c, d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics={'dense_2': 'accuracy',
               'dense_3': 'accuracy'})
  return model


class TestKerasEstimator(test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'keras_estimator_test')
    gfile.MakeDirs(self._base_dir)

  def tearDown(self):
    gfile.DeleteRecursively(self._base_dir)

  def test_train(self):
    for is_sequential in [True, False]:
      keras_model, (_, _), (
          _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
              is_sequential=is_sequential, is_evaluate=True)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy', 'mse', keras.metrics.categorical_accuracy])

      with self.test_session():
        est_keras = keras.estimator.model_to_estimator(
            keras_model=keras_model,
            model_dir=tempfile.mkdtemp(dir=self._base_dir))
        est_keras.train(input_fn=train_input_fn, steps=200 * 10 / 16)
        eval_results = est_keras.evaluate(input_fn=eval_input_fn)
        self.assertGreater(eval_results['accuracy'], 0.9)
        self.assertGreater(eval_results['categorical_accuracy'], 0.9)
        self.assertLess(eval_results['mse'], 0.1)

  def test_evaluate(self):
    keras_model, (x_train, y_train), (
        x_test, y_test), _, eval_input_fn = get_resource_for_simple_model(
            is_sequential=False, is_evaluate=True)

    with self.test_session():
      metrics = [
          'binary_accuracy', 'binary_crossentropy', 'categorical_accuracy',
          'categorical_crossentropy', 'cosine_proximity', 'hinge',
          'kullback_leibler_divergence', 'mean_absolute_error',
          'mean_absolute_percentage_error', 'mean_squared_error',
          'mean_squared_logarithmic_error', 'poisson', 'squared_hinge',
          'top_k_categorical_accuracy'
      ]
      keras_model.compile(
          loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
      keras_model.fit(x_train, y_train, epochs=1)
      keras_eval = keras_model.evaluate(x_test, y_test, batch_size=32)

    with self.test_session():
      keras_est = keras.estimator.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est_eval = keras_est.evaluate(input_fn=eval_input_fn)

    metrics = ['loss'] + metrics

    # Check loss and all metrics match between keras and estimator.
    def shift(val):
      return val / 10**int(log10(abs(val)))

    for i, metric_name in enumerate(metrics):
      self.assertAlmostEqual(
          shift(est_eval[metric_name]),
          shift(keras_eval[i]),
          places=4,
          msg='%s mismatch, keras model: %s, estimator: %s' %
          (metric_name, est_eval[metric_name], keras_eval[i]))

  def test_predict(self):
    # Check that predict on a pretrained model yield the same result.
    keras_model, (x_train, y_train), (
        x_test, _), _, pred_input_fn = get_resource_for_simple_model(
            is_sequential=True, is_evaluate=False)

    with self.test_session():
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
      keras_model.fit(x_train, y_train, epochs=1)
      keras_pred = [np.argmax(y) for y in keras_model.predict(x_test)]

    with self.test_session():
      keras_est = keras.estimator.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est_pred = [
          np.argmax(y[keras_model.output_names[0]])
          for y in keras_est.predict(input_fn=pred_input_fn)
      ]
    self.assertAllEqual(est_pred, keras_pred)

  def test_multi_inputs_multi_outputs(self):
    np.random.seed(1337)
    (a_train, c_train), (a_test, c_test) = testing_utils.get_test_data(
        train_samples=200, test_samples=100, input_shape=(32,), num_classes=3)
    (b_train, d_train), (b_test, d_test) = testing_utils.get_test_data(
        train_samples=200, test_samples=100, input_shape=(32,), num_classes=2)
    c_train = keras.utils.to_categorical(c_train)
    c_test = keras.utils.to_categorical(c_test)
    d_train = keras.utils.to_categorical(d_train)
    d_test = keras.utils.to_categorical(d_test)

    def train_input_fn():
      input_dict = {
          'input_a':
              ops.convert_to_tensor(
                  np.array(a_train, dtype=np.float32), dtype=dtypes.float32),
          'input_b':
              ops.convert_to_tensor(
                  np.array(b_train, dtype=np.float32), dtype=dtypes.float32)
      }
      output_dict = {
          'dense_2':
              ops.convert_to_tensor(
                  np.array(c_train, dtype=np.float32), dtype=dtypes.float32),
          'dense_3':
              ops.convert_to_tensor(
                  np.array(d_train, dtype=np.float32), dtype=dtypes.float32)
      }
      return input_dict, output_dict

    def evaluate_input_fn():
      input_dict = {
          'input_a':
              ops.convert_to_tensor(
                  np.array(a_test, dtype=np.float32), dtype=dtypes.float32),
          'input_b':
              ops.convert_to_tensor(
                  np.array(b_test, dtype=np.float32), dtype=dtypes.float32)
      }
      output_dict = {
          'dense_2':
              ops.convert_to_tensor(
                  np.array(c_test, dtype=np.float32), dtype=dtypes.float32),
          'dense_3':
              ops.convert_to_tensor(
                  np.array(d_test, dtype=np.float32), dtype=dtypes.float32)
      }
      return input_dict, output_dict

    with self.test_session():
      model = multi_inputs_multi_outputs_model()
      est_keras = keras.estimator.model_to_estimator(
          keras_model=model, model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est_keras.train(input_fn=train_input_fn, steps=200 * 10 / 16)
      eval_results = est_keras.evaluate(input_fn=evaluate_input_fn, steps=1)
      self.assertGreater(eval_results['accuracy_dense_2'], 0.5)
      self.assertGreater(eval_results['accuracy_dense_3'], 0.5)

  def test_init_from_file(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    keras_model, (x_train, y_train), (
        x_test, _), _, pred_input_fn = get_resource_for_simple_model(
            is_sequential=False, is_evaluate=False)

    with self.test_session():
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])
      keras_model.fit(x_train, y_train, epochs=1)
      keras_pred = [np.argmax(y) for y in keras_model.predict(x_test)]
      fname = os.path.join(self._base_dir, 'keras_model.h5')
      keras.models.save_model(keras_model, fname)

    with self.test_session():
      keras_est = keras.estimator.model_to_estimator(
          keras_model_path=fname,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est_pred = [
          np.argmax(y[keras_model.output_names[0]])
          for y in keras_est.predict(input_fn=pred_input_fn)
      ]
    self.assertAllEqual(est_pred, keras_pred)

  def test_keras_model_init_error(self):
    with self.assertRaisesRegexp(ValueError, 'Either'):
      keras.estimator.model_to_estimator()

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'not both'):
        keras.estimator.model_to_estimator(
            keras_model=keras_model,
            keras_model_path=tempfile.mkdtemp(dir=self._base_dir))

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'compiled'):
        keras.estimator.model_to_estimator(keras_model=keras_model)

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'not a local path'):
        keras.estimator.model_to_estimator(
            keras_model_path='gs://bucket/object')

  def test_invalid_ionames_error(self):
    np.random.seed(1337)
    (x_train, y_train), (_, _) = testing_utils.get_test_data(
        train_samples=200, test_samples=100, input_shape=(10,), num_classes=2)
    y_train = keras.utils.to_categorical(y_train)

    def invald_input_name_input_fn():
      input_dict = {
          'invalid_input_name':
              ops.convert_to_tensor(
                  np.array(x_train, dtype=np.float32), dtype=dtypes.float32),
      }
      output = ops.convert_to_tensor(
          np.array(y_train, dtype=np.float32), dtype=dtypes.float32)
      return input_dict, output

    def invald_output_name_input_fn():
      input_dict = {
          'input_1':
              ops.convert_to_tensor(
                  np.array(x_train, dtype=np.float32), dtype=dtypes.float32),
      }
      output_dict = {
          'invalid_output_name':
              ops.convert_to_tensor(
                  np.array(y_train, dtype=np.float32), dtype=dtypes.float32),
      }
      return input_dict, output_dict

    model = simple_functional_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    est_keras = keras.estimator.model_to_estimator(
        keras_model=model, model_dir=tempfile.mkdtemp(dir=self._base_dir))

    with self.test_session():
      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_input_name_input_fn, steps=100)

      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_output_name_input_fn, steps=100)

  def test_custom_objects(self):
    keras_model, (_, _), (
        _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
            is_sequential=True, is_evaluate=True)

    class CustomOp(keras.optimizers.RMSprop):
      pass

    def custom_loss(y_true, y_pred):
      return keras.losses.categorical_crossentropy(y_true, y_pred)

    keras_model.compile(
        loss=custom_loss, optimizer=CustomOp(), metrics=['accuracy'])

    with self.test_session():
      est_keras = keras.estimator.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est_keras.train(input_fn=train_input_fn, steps=200 * 10 / 16)
      eval_results = est_keras.evaluate(input_fn=eval_input_fn)
      self.assertGreater(eval_results['accuracy'], 0.9)


if __name__ == '__main__':
  test.main()
