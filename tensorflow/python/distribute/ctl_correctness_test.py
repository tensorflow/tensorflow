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
"""Custom Training Loop correctness test.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

_NUM_SAMPLES = 66
_BATCH_SIZE = 32
_RANDOM_SEED = 1337
_NUM_EPOCHS = 2
_STEPS_PER_EPOCH = 2


class MaybeStrategyScope(object):
  """Provides a context allowing no distribution strategy."""

  def __init__(self, strategy):
    self._strategy = strategy
    self._scope = None

  def __enter__(self):
    if self._strategy:
      self._scope = self._strategy.scope()
      self._scope.__enter__()

  def __exit__(self, exc_type, value, traceback):
    if self._strategy:
      self._scope.__exit__(exc_type, value, traceback)
      self._scope = None


def get_model(sync_batchnorm=False):
  model = keras.Sequential()
  model.add(keras.layers.Dense(10, activation='relu', input_shape=(1,)))
  model.add(keras.layers.Dense(
      10, activation='relu',
      kernel_regularizer=keras.regularizers.l2(1e-4)))
  if sync_batchnorm:
    model.add(keras.layers.SyncBatchNormalization())
  else:
    model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Dense(10, activation='relu'))
  model.add(keras.layers.Dense(1))
  return model


def get_data():
  x_train = np.random.rand(_NUM_SAMPLES, 1)
  y_train = 3 * x_train
  x_train = x_train.astype('float32')
  y_train = y_train.astype('float32')
  train_dataset = dataset_ops.DatasetV2.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.batch(_BATCH_SIZE)
  return train_dataset


def compute_loss(labels, logits, reg_losses):
  pred_loss = keras.losses.mean_squared_error(labels, logits)
  scaled_loss = nn.compute_average_loss(
      pred_loss, global_batch_size=_BATCH_SIZE)
  l2_loss = nn.scale_regularization_loss(reg_losses)
  return scaled_loss + l2_loss


def iteration_inside_func(initial_weights, dataset, optimizer_fn,
                          iteration_type, strategy=None, sync_batchnorm=None):
  """Helper function to test iterating over data inside a tf.function."""
  with MaybeStrategyScope(strategy):
    if strategy and sync_batchnorm:
      model = get_model(sync_batchnorm)
    else:
      model = get_model()
    model.set_weights(initial_weights)
    optimizer = optimizer_fn()

    training_accuracy = keras.metrics.CategoricalAccuracy(
        'training_accuracy', dtype=dtypes.float32)

    @def_function.function
    def train_epoch(dist_input):
      """Training StepFn."""
      def step_fn(inputs):
        samples, labels = inputs
        with backprop.GradientTape() as tape:
          logits = model(samples)
          loss = compute_loss(labels, logits, model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        training_accuracy.update_state(labels, logits)
        return loss

      total_loss = 0.0
      num_batches = 0
      if iteration_type == 'dataset':
        for x in dist_input:
          if strategy:
            per_replica_losses = strategy.run(step_fn, args=(x,))
            total_loss += strategy.reduce(reduce_util.ReduceOp.SUM,
                                          per_replica_losses,
                                          axis=None)
          else:
            total_loss += step_fn(x)
          num_batches += 1
      else:
        iterator = iter(dist_input)
        for _ in range(_STEPS_PER_EPOCH):
          if strategy:
            per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
            total_loss += strategy.reduce(reduce_util.ReduceOp.SUM,
                                          per_replica_losses,
                                          axis=None)
          else:
            total_loss += step_fn(next(iterator))
          num_batches += 1

      return total_loss / math_ops.cast(num_batches, dtype=dtypes.float32)

    if strategy:
      dataset = strategy.experimental_distribute_dataset(dataset)

    for _ in range(_NUM_EPOCHS):
      loss = train_epoch(dataset)

    return (model.get_weights(),
            loss,
            training_accuracy.result())


def iteration_outside_func(initial_weights, dataset, optimizer_fn,
                           iteration_type, strategy=None, sync_batchnorm=None):
  """Helper function to test iterating over data outside a tf.function."""
  with MaybeStrategyScope(strategy):
    model = get_model(sync_batchnorm=sync_batchnorm)
    model.set_weights(initial_weights)
    optimizer = optimizer_fn()

    training_accuracy = keras.metrics.CategoricalAccuracy(
        'training_accuracy', dtype=dtypes.float32)

    @def_function.function
    def train_step(dist_inputs):
      """Training StepFn."""
      def step_fn(inputs):
        samples, labels = inputs
        with backprop.GradientTape() as tape:
          logits = model(samples)
          loss = compute_loss(labels, logits, model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        training_accuracy.update_state(labels, logits)
        return loss

      if strategy:
        per_replica_losses = strategy.run(step_fn, args=(dist_inputs,))
        return strategy.reduce(reduce_util.ReduceOp.SUM,
                               per_replica_losses,
                               axis=None)
      else:
        return step_fn(dist_inputs)

    if strategy:
      dataset = strategy.experimental_distribute_dataset(dataset)

    total_loss = 0.0
    num_batches = 0
    if iteration_type == 'dataset':
      for _ in range(_NUM_EPOCHS):
        for x in dataset:
          total_loss += train_step(x)
          num_batches += 1
    else:
      for _ in range(_NUM_EPOCHS):
        iterator = iter(dataset)
        for _ in range(_STEPS_PER_EPOCH):
          total_loss += train_step(next(iterator))
          num_batches += 1

    return (model.get_weights(),
            total_loss / math_ops.cast(num_batches, dtype=dtypes.float32),
            training_accuracy.result())


class TestDistributionStrategyDnnCorrectness(test.TestCase,
                                             parameterized.TestCase):
  """Test custom training loop correctness with a simple DNN model."""

  def setUp(self):
    super(TestDistributionStrategyDnnCorrectness, self).setUp()
    v2_compat.enable_v2_behavior()
    np.random.seed(_RANDOM_SEED)
    random_seed.set_random_seed(_RANDOM_SEED)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          optimizer_fn=strategy_combinations.optimizers_v1_and_v2,
          mode=['eager'],
          iteration_type=['iterator', 'dataset'],
          inside_func=[False, True],
          sync_batchnorm=[True, False]
      ))
  def test_dnn_correctness_minus_tpus(self, distribution, optimizer_fn,
                                      iteration_type, inside_func,
                                      sync_batchnorm):
    # TODO(anjs): Identify why this particular V1 optimizer needs a higher tol.
    if 'FtrlV1' in optimizer_fn._name and 'TPU' in type(distribution).__name__:
      self.skipTest('Reduced tolerance of the order of 1e-1 required.')
    self.dnn_correctness(distribution, optimizer_fn, iteration_type,
                         inside_func, sync_batchnorm)

  def dnn_correctness(self, distribution, optimizer_fn, iteration_type,
                      inside_func, sync_batchnorm=None):
    model = get_model(sync_batchnorm)
    initial_weights = model.get_weights()
    dataset = get_data()
    if inside_func:
      iteration_func = iteration_inside_func
    else:
      iteration_func = iteration_outside_func
    wts_with_ds, loss_with_ds, acc_with_ds = iteration_func(
        initial_weights, dataset, optimizer_fn, iteration_type,
        strategy=distribution, sync_batchnorm=sync_batchnorm)
    wts, loss, acc = iteration_func(initial_weights, dataset, optimizer_fn,
                                    iteration_type,
                                    sync_batchnorm=sync_batchnorm)

    self.assertAllClose(wts, wts_with_ds, atol=1e-3, rtol=1e-3)
    self.assertAllClose(loss, loss_with_ds, atol=1e-3, rtol=1e-3)
    self.assertAllClose(acc, acc_with_ds, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
  test.main()
