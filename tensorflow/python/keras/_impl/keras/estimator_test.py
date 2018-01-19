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

from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.keras._impl.keras.applications import mobilenet
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache


try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2


def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model():
  a = keras.layers.Input(shape=_INPUT_SIZE)
  b = keras.layers.Dense(16, activation='relu')(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def get_resource_for_simple_model(is_sequential, is_evaluate):
  model = simple_sequential_model(
  ) if is_sequential else simple_functional_model()
  if is_sequential:
    model.build()
  input_name = model.input_names[0]
  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = keras.utils.to_categorical(y_train)
  y_test = keras.utils.to_categorical(y_test)

  train_input_fn = numpy_io.numpy_input_fn(
      x={input_name: x_train},
      y=y_train,
      shuffle=False,
      num_epochs=None,
      batch_size=16)

  evaluate_input_fn = numpy_io.numpy_input_fn(
      x={input_name: x_test}, y=y_test, num_epochs=1, shuffle=False)

  predict_input_fn = numpy_io.numpy_input_fn(
      x={input_name: x_test}, num_epochs=1, shuffle=False)

  inference_input_fn = evaluate_input_fn if is_evaluate else predict_input_fn

  return model, (x_train, y_train), (x_test,
                                     y_test), train_input_fn, inference_input_fn


def multi_inputs_multi_outputs_model():
  # test multi-input layer
  a = keras.layers.Input(shape=(16,), name='input_a')
  b = keras.layers.Input(shape=(16,), name='input_b')
  dense = keras.layers.Dense(8, name='dense_1')
  a_2 = dense(a)
  b_2 = dense(b)
  merged = keras.layers.concatenate([a_2, b_2], name='merge')
  c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(inputs=[a, b], outputs=[c, d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics={
          'dense_2': 'categorical_accuracy',
          'dense_3': 'categorical_accuracy'
      })
  return model


class TestKerasEstimator(test_util.TensorFlowTestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'keras_estimator_test')
    gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)

  def tearDown(self):
    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      gfile.DeleteRecursively(self._base_dir)

  def test_train(self):
    for is_sequential in [True, False]:
      keras_model, (_, _), (
          _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
              is_sequential=is_sequential, is_evaluate=True)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['mse', keras.metrics.categorical_accuracy])

      with self.test_session():
        est_keras = keras.estimator.model_to_estimator(
            keras_model=keras_model, config=self._config)
        before_eval_results = est_keras.evaluate(
            input_fn=eval_input_fn, steps=1)
        est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
        after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
        self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

      writer_cache.FileWriterCache.clear()
      gfile.DeleteRecursively(self._config.model_dir)

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
          keras_model=keras_model, config=self._config)
      est_eval = keras_est.evaluate(input_fn=eval_input_fn)

    metrics = ['loss'] + metrics

    # Check loss and all metrics match between keras and estimator.
    def shift(val):
      if val == 0:
        return 0
      else:
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
          keras_model=keras_model, config=self._config)
      est_pred = [
          np.argmax(y[keras_model.output_names[0]])
          for y in keras_est.predict(input_fn=pred_input_fn)
      ]
    self.assertAllEqual(est_pred, keras_pred)

  def test_multi_inputs_multi_outputs(self):
    np.random.seed(_RANDOM_SEED)
    (a_train, c_train), (a_test, c_test) = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=50,
        input_shape=(16,),
        num_classes=3)
    np.random.seed(_RANDOM_SEED)
    (b_train, d_train), (b_test, d_test) = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=50,
        input_shape=(16,),
        num_classes=2)
    c_train = keras.utils.to_categorical(c_train)
    c_test = keras.utils.to_categorical(c_test)
    d_train = keras.utils.to_categorical(d_train)
    d_test = keras.utils.to_categorical(d_test)

    def train_input_fn():
      input_dict = {'input_a': a_train, 'input_b': b_train}
      output_dict = {'dense_2': c_train, 'dense_3': d_train}
      return input_dict, output_dict

    def eval_input_fn():
      input_dict = {'input_a': a_test, 'input_b': b_test}
      output_dict = {'dense_2': c_test, 'dense_3': d_test}
      return input_dict, output_dict

    with self.test_session():
      model = multi_inputs_multi_outputs_model()
      est_keras = keras.estimator.model_to_estimator(
          keras_model=model, config=self._config)
      before_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

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
          metrics=['categorical_accuracy'])
      keras_model.fit(x_train, y_train, epochs=1)
      keras_pred = [np.argmax(y) for y in keras_model.predict(x_test)]
      fname = os.path.join(self._base_dir, 'keras_model.h5')
      keras.models.save_model(keras_model, fname)

    with self.test_session():
      keras_est = keras.estimator.model_to_estimator(
          keras_model_path=fname, config=self._config)
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
    (x_train, y_train), (_, _) = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=100,
        input_shape=(10,),
        num_classes=2)
    y_train = keras.utils.to_categorical(y_train)

    def invald_input_name_input_fn():
      input_dict = {'invalid_input_name': x_train}
      return input_dict, y_train

    def invald_output_name_input_fn():
      input_dict = {'input_1': x_train}
      output_dict = {'invalid_output_name': y_train}
      return input_dict, output_dict

    model = simple_functional_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    est_keras = keras.estimator.model_to_estimator(
        keras_model=model, config=self._config)

    with self.test_session():
      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_input_name_input_fn, steps=100)

      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_output_name_input_fn, steps=100)

  def test_custom_objects(self):
    keras_mobile = mobilenet.MobileNet(weights=None)
    keras_mobile.compile(loss='categorical_crossentropy', optimizer='adam')
    custom_objects = {
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D
    }
    with self.assertRaisesRegexp(ValueError, 'relu6'):
      with self.test_session():
        keras.estimator.model_to_estimator(
            keras_model=keras_mobile,
            model_dir=tempfile.mkdtemp(dir=self._base_dir))

    with self.test_session():
      keras.estimator.model_to_estimator(
          keras_model=keras_mobile,
          model_dir=tempfile.mkdtemp(dir=self._base_dir),
          custom_objects=custom_objects)


if __name__ == '__main__':
  test.main()
