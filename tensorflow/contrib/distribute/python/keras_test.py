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
"""Tests for Keras Sequential and Functional models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import keras as keras_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import rmsprop

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


def get_ds_train_input_fn():
  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), _ = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = keras.utils.to_categorical(y_train)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.batch(32)
  return dataset


def get_ds_test_input_fn():
  np.random.seed(_RANDOM_SEED)
  _, (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_test = keras.utils.to_categorical(y_test)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  dataset = dataset.batch(32)
  return dataset


class TestKerasDistributionStrategy(test_util.TensorFlowTestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(),
                                  'keras_mirrored_strategy_test')
    gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)

  def tearDown(self):
    writer_cache.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      gfile.DeleteRecursively(self._base_dir)

  def test_train_functional_with_distribution_strategy(self):
    dist = mirrored_strategy.MirroredStrategy(
        devices=['/device:GPU:0', '/device:GPU:1'])
    keras_model = simple_functional_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=dist)
    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

  def test_train_sequential_with_distribution_strategy(self):
    dist = mirrored_strategy.MirroredStrategy(
        devices=['/device:GPU:0', '/device:GPU:1'])
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=dist)
    with self.test_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

if __name__ == '__main__':
  test.main()
