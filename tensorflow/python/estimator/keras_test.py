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

import json
from math import log10
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.estimator import keras as keras_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.parsing_ops import gen_parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import rmsprop
from tensorflow.python.training import session_run_hook


try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2

_TMP_DIR = '/tmp'


def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model(activation='relu'):
  a = keras.layers.Input(shape=_INPUT_SIZE)
  b = keras.layers.Dense(16, activation=activation)(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def simple_subclassed_model():

  class SimpleModel(keras.Model):

    def __init__(self):
      super(SimpleModel, self).__init__()
      self.dense1 = keras.layers.Dense(16, activation='relu')
      self.dp = keras.layers.Dropout(0.1)
      self.dense2 = keras.layers.Dense(_NUM_CLASS, activation='softmax')

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.dp(x)
      return self.dense2(x)

  return SimpleModel()


def get_resource_for_simple_model(model_type='sequential',
                                  is_evaluate=False,):
  if model_type == 'sequential':
    model = simple_sequential_model()
    model.build()
  elif model_type == 'subclass':
    model = simple_subclassed_model()
  else:
    assert model_type == 'functional'
    model = simple_functional_model()

  if model_type == 'subclass':
    input_name = 'input_1'
    output_name = 'output_1'
  else:
    input_name = model.input_names[0]
    output_name = model.output_names[0]

  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = keras.utils.to_categorical(y_train)
  y_test = keras.utils.to_categorical(y_test)

  train_input_fn = numpy_io.numpy_input_fn(
      x=randomize_io_type(x_train, input_name),
      y=randomize_io_type(y_train, output_name),
      shuffle=False,
      num_epochs=None,
      batch_size=16)

  evaluate_input_fn = numpy_io.numpy_input_fn(
      x=randomize_io_type(x_test, input_name),
      y=randomize_io_type(y_test, output_name),
      num_epochs=1, shuffle=False)

  predict_input_fn = numpy_io.numpy_input_fn(
      x=randomize_io_type(x_test, input_name), num_epochs=1, shuffle=False)

  inference_input_fn = evaluate_input_fn if is_evaluate else predict_input_fn

  return model, (x_train, y_train), (x_test,
                                     y_test), train_input_fn, inference_input_fn


def randomize_io_type(array, name):
  switch = np.random.random()
  if switch > 0.5:
    return array
  else:
    return {name: array}


def multi_inputs_multi_outputs_model():
  a = keras.layers.Input(shape=(16,), name='input_a')
  b = keras.layers.Input(shape=(16,), name='input_b')
  m = keras.layers.Input(shape=(8,), dtype='string', name='input_m')
  dense = keras.layers.Dense(8, name='dense_1')

  a_2 = dense(a)
  # Read m
  m_2 = keras.layers.Lambda(gen_parsing_ops.string_to_number)(m)
  s_2 = keras.layers.Lambda(lambda k: k[0] * k[1])([m_2, a_2])
  b_2 = dense(b)
  merged = keras.layers.concatenate([s_2, b_2], name='merge')
  c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(inputs=[a, b, m], outputs=[c, d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics={
          'dense_2': 'categorical_accuracy',
          'dense_3': 'categorical_accuracy'
      })
  return model


class MyHook(session_run_hook.SessionRunHook):

  def begin(self):
    _ = variable_scope.get_variable('temp', [1])


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
    for model_type in ['sequential', 'functional']:
      keras_model, (_, _), (
          _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
              model_type=model_type, is_evaluate=True)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['mse', keras.metrics.categorical_accuracy])

      with self.test_session():
        est_keras = keras_lib.model_to_estimator(
            keras_model=keras_model, config=self._config)
        before_eval_results = est_keras.evaluate(
            input_fn=eval_input_fn, steps=1)
        est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
        after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
        self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

      writer_cache.FileWriterCache.clear()
      gfile.DeleteRecursively(self._config.model_dir)

  # see b/109935364
  @test_util.run_in_graph_and_eager_modes
  def test_train_with_hooks(self):
    for model_type in ['sequential', 'functional']:
      keras_model, (_, _), (
          _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
              model_type=model_type, is_evaluate=True)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer=rmsprop.RMSPropOptimizer(1e-3),
          metrics=['mse', keras.metrics.categorical_accuracy])

      my_hook = MyHook()
      with self.test_session():
        est_keras = keras_lib.model_to_estimator(
            keras_model=keras_model, config=self._config)
        before_eval_results = est_keras.evaluate(
            input_fn=eval_input_fn, steps=1)
        est_keras.train(input_fn=train_input_fn, hooks=[my_hook],
                        steps=_TRAIN_SIZE / 16)
        after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
        self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

      writer_cache.FileWriterCache.clear()
      gfile.DeleteRecursively(self._config.model_dir)

  @test_util.run_in_graph_and_eager_modes
  def test_train_with_model_fit_and_hooks(self):
    keras_model, (x_train, y_train), _, \
      train_input_fn, eval_input_fn = get_resource_for_simple_model(
          model_type='sequential', is_evaluate=True)

    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(1e-3),
        metrics=['mse', keras.metrics.categorical_accuracy])
    my_hook = MyHook()
    with self.test_session():
      keras_model.fit(x_train, y_train, epochs=1)

      keras_est = keras_lib.model_to_estimator(
          keras_model=keras_model, config=self._config)
      before_eval_results = keras_est.evaluate(input_fn=eval_input_fn)
      keras_est.train(input_fn=train_input_fn, hooks=[my_hook],
                      steps=_TRAIN_SIZE / 16)
      after_eval_results = keras_est.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

  @test_util.run_in_graph_and_eager_modes
  def test_train_with_tf_optimizer(self):
    for model_type in ['sequential', 'functional']:
      keras_model, (_, _), (
          _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
              model_type=model_type, is_evaluate=True)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer=rmsprop.RMSPropOptimizer(1e-3),
          metrics=['mse', keras.metrics.categorical_accuracy])

      with self.test_session():
        est_keras = keras_lib.model_to_estimator(
            keras_model=keras_model,
            config=self._config)
        before_eval_results = est_keras.evaluate(
            input_fn=eval_input_fn, steps=1)
        est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
        after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
        self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

      writer_cache.FileWriterCache.clear()
      gfile.DeleteRecursively(self._config.model_dir)

  @test_util.run_in_graph_and_eager_modes
  def test_train_with_subclassed_model(self):
    keras_model, (_, _), (
        _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
            model_type='subclass', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(1e-3),
        metrics=['mse', keras.metrics.categorical_accuracy])

    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=self._config)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      before_eval_results = est_keras.evaluate(
          input_fn=eval_input_fn, steps=1)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

  def test_train_with_subclassed_model_with_existing_state(self):
    keras_model, (_, _), (
        _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
            model_type='subclass', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(1e-3),
        metrics=['mse', keras.metrics.categorical_accuracy])

    with self.test_session():
      # Create state
      keras_model.train_on_batch(np.random.random((10,) + _INPUT_SIZE),
                                 np.random.random((10, _NUM_CLASS)))
      original_preds = keras_model.predict(np.ones((10,) + _INPUT_SIZE))

      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=self._config)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      before_eval_results = est_keras.evaluate(
          input_fn=eval_input_fn, steps=1)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

      # Check that original model state was not altered
      preds = keras_model.predict(np.ones((10,) + _INPUT_SIZE))
      self.assertAllClose(original_preds, preds, atol=1e-5)
      # Check that the original model compilation did not break
      keras_model.train_on_batch(np.random.random((10,) + _INPUT_SIZE),
                                 np.random.random((10, _NUM_CLASS)))

  def test_evaluate(self):
    keras_model, (x_train, y_train), (
        x_test, y_test), _, eval_input_fn = get_resource_for_simple_model(
            model_type='functional', is_evaluate=True)

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
      keras_est = keras_lib.model_to_estimator(
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
            model_type='sequential', is_evaluate=False)

    with self.test_session():
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
      keras_model.fit(x_train, y_train, epochs=1)
      keras_pred = [np.argmax(y) for y in keras_model.predict(x_test)]

    with self.test_session():
      keras_est = keras_lib.model_to_estimator(
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
    np.random.seed(_RANDOM_SEED)
    (input_m_train, _), (input_m_test, _) = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=50,
        input_shape=(8,),
        num_classes=2)

    c_train = keras.utils.to_categorical(c_train)
    c_test = keras.utils.to_categorical(c_test)
    d_train = keras.utils.to_categorical(d_train)
    d_test = keras.utils.to_categorical(d_test)

    def train_input_fn():
      input_dict = {'input_a': a_train, 'input_b': b_train,
                    'input_m': input_m_train.astype(np.str)}
      output_dict = {'dense_2': c_train, 'dense_3': d_train}
      return input_dict, output_dict

    def eval_input_fn():
      input_dict = {'input_a': a_test, 'input_b': b_test,
                    'input_m': input_m_test.astype(np.str)}
      output_dict = {'dense_2': c_test, 'dense_3': d_test}
      return input_dict, output_dict

    with self.test_session():
      model = multi_inputs_multi_outputs_model()
      est_keras = keras_lib.model_to_estimator(
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
            model_type='functional', is_evaluate=False)

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
      keras_est = keras_lib.model_to_estimator(
          keras_model_path=fname, config=self._config)
      est_pred = [
          np.argmax(y[keras_model.output_names[0]])
          for y in keras_est.predict(input_fn=pred_input_fn)
      ]
    self.assertAllEqual(est_pred, keras_pred)

  def test_keras_model_init_error(self):
    with self.assertRaisesRegexp(ValueError, 'Either'):
      keras_lib.model_to_estimator()

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'not both'):
        keras_lib.model_to_estimator(
            keras_model=keras_model,
            keras_model_path=tempfile.mkdtemp(dir=self._base_dir))

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'compiled'):
        keras_lib.model_to_estimator(keras_model=keras_model)

    with self.test_session():
      keras_model = simple_sequential_model()
      with self.assertRaisesRegexp(ValueError, 'not a local path'):
        keras_lib.model_to_estimator(
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
    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=model, config=self._config)

    with self.test_session():
      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_input_name_input_fn, steps=100)

      with self.assertRaises(ValueError):
        est_keras.train(input_fn=invald_output_name_input_fn, steps=100)

  def test_custom_objects(self):

    def relu6(x):
      return keras.backend.relu(x, max_value=6)

    keras_model = simple_functional_model(activation=relu6)
    keras_model.compile(loss='categorical_crossentropy', optimizer='adam')
    custom_objects = {
        'relu6': relu6
    }

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=50,
        input_shape=(10,),
        num_classes=2)
    y_train = keras.utils.to_categorical(y_train, 2)
    input_name = keras_model.input_names[0]
    output_name = keras_model.output_names[0]
    train_input_fn = numpy_io.numpy_input_fn(
        x=randomize_io_type(x_train, input_name),
        y=randomize_io_type(y_train, output_name),
        shuffle=False,
        num_epochs=None,
        batch_size=16)
    with self.assertRaisesRegexp(ValueError, 'relu6'):
      with self.test_session():
        est = keras_lib.model_to_estimator(
            keras_model=keras_model,
            model_dir=tempfile.mkdtemp(dir=self._base_dir))
        est.train(input_fn=train_input_fn, steps=1)

    with self.test_session():
      est = keras_lib.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir),
          custom_objects=custom_objects)
      est.train(input_fn=train_input_fn, steps=1)

  def test_tf_config(self):
    keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.categorical_accuracy])

    tf_config = json.dumps({
        'cluster': {
            run_config_lib.TaskType.PS: ['localhost:1234'],
            run_config_lib.TaskType.WORKER: ['localhost:1236'],
            run_config_lib.TaskType.MASTER: ['localhost:1238']
        },
        'task': {
            'type': run_config_lib.TaskType.MASTER,
            'index': 0
        }
    })
    with test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      with self.test_session():
        keras_lib.model_to_estimator(
            keras_model=keras_model,
            model_dir=tempfile.mkdtemp(dir=self._base_dir))

  def test_gpu_config(self):
    with ops.Graph().as_default():
      keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['mse', keras.metrics.categorical_accuracy])

      gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.3)
      sess_config = config_pb2.ConfigProto(gpu_options=gpu_options)
      self._config._session_config = sess_config
      with self.test_session():
        keras_lib.model_to_estimator(
            keras_model=keras_model, config=self._config)
        self.assertEqual(
            keras.backend.get_session()
            ._config.gpu_options.per_process_gpu_memory_fraction,
            gpu_options.per_process_gpu_memory_fraction)

  def test_with_empty_config(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.categorical_accuracy])

    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, model_dir=self._base_dir,
          config=run_config_lib.RunConfig())
      self.assertEqual(run_config_lib.get_default_session_config(),
                       est_keras._session_config)
      self.assertEqual(est_keras._session_config,
                       est_keras._config.session_config)
      self.assertEqual(self._base_dir, est_keras._config.model_dir)
      self.assertEqual(self._base_dir, est_keras._model_dir)

    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, model_dir=self._base_dir,
          config=None)
      self.assertEqual(run_config_lib.get_default_session_config(),
                       est_keras._session_config)
      self.assertEqual(est_keras._session_config,
                       est_keras._config.session_config)
      self.assertEqual(self._base_dir, est_keras._config.model_dir)
      self.assertEqual(self._base_dir, est_keras._model_dir)

  def test_with_empty_config_and_empty_model_dir(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.categorical_accuracy])

    with self.test_session():
      with test.mock.patch.object(tempfile, 'mkdtemp', return_value=_TMP_DIR):
        est_keras = keras_lib.model_to_estimator(
            keras_model=keras_model,
            config=run_config_lib.RunConfig())
        self.assertEqual(est_keras._model_dir, _TMP_DIR)

  def test_with_conflicting_model_dir_and_config(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.categorical_accuracy])

    with self.test_session():
      with self.assertRaisesRegexp(ValueError, '`model_dir` are set both in '
                                   'constructor and `RunConfig`'):
        keras_lib.model_to_estimator(
            keras_model=keras_model, model_dir=self._base_dir,
            config=run_config_lib.RunConfig(model_dir=_TMP_DIR))

  def test_pretrained_weights(self):
    keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(1e-3),
        metrics=['mse', keras.metrics.categorical_accuracy])
    with self.test_session():
      keras_model.train_on_batch(
          np.random.random((10,) + _INPUT_SIZE),
          np.random.random((10, _NUM_CLASS)))
      weights = keras_model.get_weights()
      keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
      keras_model.set_weights(weights)
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer=SGD(lr=0.0001, momentum=0.9),
          metrics=['mse', keras.metrics.categorical_accuracy])
      keras_lib.model_to_estimator(
          keras_model=keras_model, config=self._config)


if __name__ == '__main__':
  test.main()
