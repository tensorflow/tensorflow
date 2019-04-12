# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for accuracy and mathematical correctness of tf.keras multi-worker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib.distribute.python import collective_all_reduce_strategy as collective_strategy
from tensorflow.contrib.distribute.python import keras_multi_worker_test_base
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


np.random.seed(99)
EMBED_INPUTS = np.random.randint(0, 10, (6400, 1)).astype(np.int32)
EMBED_TARGETS = np.random.normal(0, 0.1, (6400, 1)).astype(np.float32)
IMAGE_INPUTS = np.random.normal(0, 0.1, (6400, 28, 28, 3)).astype(np.float32)
IMAGE_TARGETS = np.random.randint(0, 10, (6400, 1))
LSTM_INPUTS = np.random.normal(0, 0.1, (6400, 10, 20)).astype(np.float32)
LSTM_TARGETS = np.random.normal(0, 0.1, (6400, 1)).astype(np.float32)


def get_num_workers():
  cluster_resolver = TFConfigClusterResolver()
  cluster_spec = cluster_resolver.cluster_spec().as_dict()
  if cluster_spec:
    task_type = cluster_resolver.task_type
    return int(multi_worker_util.worker_count(cluster_spec, task_type))
  return 1


class Bias(keras.layers.Layer):

  def build(self, input_shape):
    self.bias = self.add_weight(shape=(), initializer='zeros', name='bias')

  def call(self, inputs):
    return inputs + self.bias


class SimpleBiasTest(
    keras_multi_worker_test_base.KerasIndependentWorkerTestBase,
    parameterized.TestCase):

  @keras_multi_worker_test_base.run_sync_strategies
  def test_multi_worker_simple_bias_fit(self, strategy_cls):

    def _worker_fn(results_without_ds=None):
      # Make sure Session is cleared at the start of each run.
      keras.backend._SESSION.session = None

      x = ops.convert_to_tensor([[0.], [1.], [2.], [0.], [1.], [2.]])
      y = ops.convert_to_tensor([[0.5], [2.], [3.5], [0.5], [2.], [3.5]])
      ds = dataset_ops.Dataset.from_tensor_slices((x, y))
      ds = ds.batch(6)
      model = keras.Sequential([Bias(input_shape=(1,))])
      model.compile(
          keras.optimizer_v2.gradient_descent.SGD(0.1), 'mae', metrics=['mae'])
      history = model.fit(ds, epochs=5)
      self.assertAllClose(history.history['loss'], [1., 0.9, 0.8, 0.7, 0.6])
      self.assertAllClose(history.history['mean_absolute_error'],
                          [1., 0.9, 0.8, 0.7, 0.6])

      results = {'training': history.history}
      if results_without_ds:
        for key in results:
          self.assertAllClose(
              results[key],
              results_without_ds[key],
              msg='Fail to assert {}'.format(key))

      return results

    results_without_ds = _worker_fn()
    self.run_independent_workers(
        _worker_fn,
        strategy_cls,
        num_workers=2,
        results_without_ds=results_without_ds)


def make_image_model(initial_weights=None):
  image = keras.layers.Input(shape=(28, 28, 3), name='image')
  c1 = keras.layers.Conv2D(
      name='conv1',
      filters=16,
      kernel_size=(3, 3),
      strides=(4, 4),
      kernel_regularizer=keras.regularizers.l2(1e-4))(
          image)
  c1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)
  c1 = keras.layers.Flatten()(c1)
  logits = keras.layers.Dense(10, activation='softmax', name='pred')(c1)
  model = keras.Model(inputs=[image], outputs=[logits])

  if initial_weights:
    model.set_weights(initial_weights)

  model.compile(
      'sgd',
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])

  return model, IMAGE_INPUTS, IMAGE_TARGETS


# TODO(b/130243026): Re-enable this test.
def make_lstm_model(initial_weights=None):
  inputs = keras.layers.Input(shape=(10, 20))
  rnn1_out = keras.layers.LSTM(20, return_sequences=True)(inputs)
  rnn2_out = keras.layers.LSTM(10)(rnn1_out)
  outputs = keras.layers.Dense(1)(rnn2_out)
  model = keras.Model(inputs, outputs)

  if initial_weights:
    model.set_weights(initial_weights)

  model.compile('adam', 'binary_crossentropy', metrics=['mse'])

  return model, LSTM_INPUTS, LSTM_TARGETS


def make_embedding_model(initial_weights=None):
  # TODO(b/130231718): Remove batch_size here.
  inputs = keras.layers.Input(
      batch_size=64 // get_num_workers(), shape=(1,), dtype='int32')
  embeddings = keras.layers.Embedding(100, 5)(inputs)
  outputs = keras.layers.Dense(1, activation='softmax')(embeddings)
  model = keras.Model(inputs, outputs)

  if initial_weights:
    model.set_weights(initial_weights)

  model.compile('rmsprop', 'mae', metrics=['binary_crossentropy'])

  return model, EMBED_INPUTS, EMBED_TARGETS


class ModelCorrectnessTest(
    keras_multi_worker_test_base.KerasIndependentWorkerTestBase,
    parameterized.TestCase):

  def make_dataset(self, inputs, targets, batch_size=64):
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size)
    return dataset

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[
              collective_strategy.CollectiveAllReduceStrategy,
          ],
          make_model=[make_image_model, make_embedding_model],
          required_gpus=[0])) # TODO(b/130299192): Enable for 1 gpu case.
  def test_correctness(self, strategy_cls, make_model):

    def _worker_fn(initial_weights=None, results_without_ds=None):
      # Make sure Session is cleared at each run
      # so that it can be configured properly for the DistributionStrategy.
      keras.backend._SESSION.session = None

      results = {}
      model, inputs, targets = make_model(initial_weights)

      data = self.make_dataset(inputs, targets)

      # TODO(b/129363441): Remove `steps_per_epoch`.
      results['training'] = model.fit(
          data, steps_per_epoch=50, epochs=2).history
      results['trained_weights'] = model.get_weights()

      eval_data = self.make_dataset(inputs, targets)
      results['evaluation'] = model.evaluate(eval_data, steps=50)

      if results_without_ds:
        for key in results:
          self.assertAllClose(
              results[key],
              results_without_ds[key],
              msg='Fail to assert {}'.format(key))

      return results

    model, _, _ = make_model()
    initial_weights = model.get_weights()
    results_without_ds = _worker_fn(initial_weights=initial_weights)
    self.run_independent_workers(
        _worker_fn,
        strategy_cls,
        num_workers=2,
        initial_weights=initial_weights,
        results_without_ds=results_without_ds)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
