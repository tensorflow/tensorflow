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
from tensorflow.contrib.distribute.python import keras_multi_worker_test_base
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def batch_and_maybe_shard_dataset(dataset, global_batch_size):
  """Shard the dataset if running in multi-node environment."""

  cluster_resolver = TFConfigClusterResolver()
  cluster_spec = cluster_resolver.cluster_spec().as_dict()
  if cluster_spec:
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    num_workers = int(multi_worker_util.worker_count(cluster_spec, task_type))
    id_in_cluster = int(
        multi_worker_util.id_in_cluster(cluster_spec, task_type, task_id))
    dataset = dataset.shard(num_workers, id_in_cluster)
  return dataset.batch(global_batch_size)


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
      ds = batch_and_maybe_shard_dataset(ds, global_batch_size=6)
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


class ImageModelTest(
    keras_multi_worker_test_base.KerasIndependentWorkerTestBase,
    parameterized.TestCase):

  inputs = np.random.normal(0, 0.1, (6400, 28, 28, 3)).astype(np.float32)
  targets = np.random.randint(0, 10, (6400, 1))

  def _get_model(self, initial_weights=None):
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
    return model

  def _get_inputs(self):
    inputs = ImageModelTest.inputs
    targets = ImageModelTest.targets
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = batch_and_maybe_shard_dataset(dataset, global_batch_size=64)
    return dataset

  @keras_multi_worker_test_base.run_sync_strategies
  def test_cnn_correctness(self, strategy_cls):

    model = self._get_model()
    initial_weights = model.get_weights()

    def _worker_fn(initial_weights=None, results_without_ds=None):
      # Make sure Session is cleared at each run.
      keras.backend._SESSION.session = None
      results = {}
      model = self._get_model(initial_weights)

      data = self._get_inputs()

      # TODO(b/129363441): Remove `steps_per_epoch`.
      results['training'] = model.fit(
          data, steps_per_epoch=50, epochs=2).history
      results['trained_weights'] = model.get_weights()

      eval_data = self._get_inputs()
      results['evaluation'] = model.evaluate(eval_data, steps=50)

      if results_without_ds:
        for key in results:
          self.assertAllClose(
              results[key],
              results_without_ds[key],
              atol=1e-4,
              rtol=1e-4,
              msg='Fail to assert {}'.format(key))

      return results

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
