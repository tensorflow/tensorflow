# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU Embeddings mid level API on TPU."""
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v1
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


_SLOT_NAME_MAPPING = {
    # Slot names in Keras optimizer v2 are different compared to the slot names
    # in our API.
    adagrad.Adagrad: {'accumulators': 'accumulator'},
    adam.Adam: {'momenta': 'm', 'velocities': 'v'},
    ftrl.Ftrl: {'accumulators': 'accumulator', 'linears': 'linear'},
}


class TPUEmbeddingV0CorrectnessTest(tpu_embedding_base_test.TPUEmbeddingBaseTest
                                   ):

  def _get_strategy(self):
    if hasattr(self, 'strategy'):
      return self.strategy
    return super(TPUEmbeddingV0CorrectnessTest, self)._get_strategy()

  def _create_mid_level(self, optimizer=None):
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    return tpu_embedding_v1.TPUEmbeddingV0(
        feature_config=self.feature_config, optimizer=optimizer)

  def _get_slot_variable_creation_fn(self, optimizer):
    # This is needed so that the mid level API can create slots using a user
    # passed optimizer rather than the built-in methods. This allows a user to
    # train the same model on CPU and TPU.
    def slot_variable_creation_fn(table, slot_names, slot_initializers):
      slots = {}
      for slot, initializer in zip(slot_names, slot_initializers):
        slots[slot] = optimizer.add_slot(
            table, _SLOT_NAME_MAPPING[type(optimizer)][slot], initializer)
      return slots

    return slot_variable_creation_fn

  def _create_strategy_and_mid_level(self, optimizer_name):
    strategy = self._get_strategy()

    # Keras optimizers has to be translated to embedding optimizer with slot
    # variable creation fn properly populated.
    with strategy.scope():
      if optimizer_name == 'sgd':
        optimizer = gradient_descent.SGD(learning_rate=0.1)
        embedding_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      elif optimizer_name == 'adagrad':
        optimizer = adagrad.Adagrad(learning_rate=0.1)
        embedding_optimizer = tpu_embedding_v2_utils.Adagrad(
            learning_rate=0.1,
            slot_variable_creation_fn=self._get_slot_variable_creation_fn(
                optimizer))
      elif optimizer_name == 'adam':
        optimizer = adam.Adam(learning_rate=0.1)
        embedding_optimizer = tpu_embedding_v2_utils.Adam(
            learning_rate=0.1,
            slot_variable_creation_fn=self._get_slot_variable_creation_fn(
                optimizer))
      elif optimizer_name == 'ftrl':
        optimizer = ftrl.Ftrl(learning_rate=0.1)
        embedding_optimizer = tpu_embedding_v2_utils.FTRL(
            learning_rate=0.1,
            slot_variable_creation_fn=self._get_slot_variable_creation_fn(
                optimizer))
      else:
        raise ValueError('optimizer is not recognized: ', optimizer_name)

      mid_level_api = self._create_mid_level(optimizer=embedding_optimizer)

    return strategy, mid_level_api, optimizer

  @parameterized.parameters(
      *itertools.product(['sgd', 'adagrad', 'adam', 'ftrl'], [True, False],
                         [True, False], [True, False]))
  def test_embedding(self, optimizer_name, training, sparse,
                     is_high_dimensional):
    strategy, mid_level_api, optimizer = (
        self._create_strategy_and_mid_level(optimizer_name))

    if sparse:
      if is_high_dimensional:
        dataset = self._create_high_dimensional_sparse_dataset(strategy)
      else:
        dataset = self._create_sparse_dataset(strategy)
    else:
      if is_high_dimensional:
        dataset = self._create_high_dimensional_sparse_dataset(strategy)
      else:
        dataset = self._create_ragged_dataset(strategy)

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      """Create and run computation that returns the embedding activations."""

      def step(data):
        if not training:
          activations = mid_level_api(data)
          total_loss = self._get_total_loss_tensor(activations)
          ret_val = [total_loss] + list(activations)
          return ret_val
        else:
          with backprop.GradientTape() as tape:
            tape.watch(mid_level_api.embedding_tables.values())
            activations = mid_level_api(data)
            total_loss = self._get_total_loss_tensor(activations)
            loss_per_replica = total_loss / strategy.num_replicas_in_sync
          gradients = tape.gradient(loss_per_replica,
                                    mid_level_api.embedding_tables.values())
          optimizer.apply_gradients(
              list(zip(gradients, mid_level_api.embedding_tables.values())))
        ret_val = [total_loss] + list(activations)
        return ret_val

      return strategy.run(step, args=(next(dist_iter),))

    # Run model.
    shard_out_val = test_fn()

    # Compute sparse tensors for global batch.
    if is_high_dimensional:
      input_data = next(
          iter(self._create_high_dimensional_sparse_dataset(strategy)))
    else:
      input_data = next(iter(self._create_sparse_dataset(strategy)))

    # Check results.
    self._check_results(strategy, shard_out_val, training, input_data,
                        mid_level_api._variables, optimizer,
                        is_high_dimensional)

  def _check_embedding_and_slot_variables(self, embedding_table_user_before,
                                          gradients_wrt_user,
                                          embedding_table_video_before,
                                          gradients_wrt_video, optimizer,
                                          table_to_variable):
    if isinstance(optimizer, gradient_descent.SGD):
      check_fn = self._check_embedding_and_slot_variables_for_sgd
    elif isinstance(optimizer, adagrad.Adagrad):
      check_fn = self._check_embedding_and_slot_variables_for_adagrad
    elif isinstance(optimizer, adam.Adam):
      check_fn = self._check_embedding_and_slot_variables_for_adam
    elif isinstance(optimizer, ftrl.Ftrl):
      check_fn = self._check_embedding_and_slot_variables_for_ftrl
    else:
      raise ValueError('optimizer is not recognized: ', type(optimizer))
    check_fn(embedding_table_user_before, gradients_wrt_user, optimizer,
             table_to_variable[self.table_user.name])
    check_fn(embedding_table_video_before, gradients_wrt_video, optimizer,
             table_to_variable[self.table_video.name])

  def _check_embedding_and_slot_variables_for_sgd(self, embedding_table_before,
                                                  gradients, optimizer,
                                                  variables):
    embedding_table = np.copy(embedding_table_before)
    config = optimizer.get_config()
    embedding_table -= config['learning_rate'] * np.sum(gradients, axis=0)
    self.assertAllClose(
        self._get_variable(variables['parameters']).numpy(), embedding_table)

  def _check_embedding_and_slot_variables_for_adagrad(self,
                                                      embedding_table_before,
                                                      gradients, optimizer,
                                                      variables):
    embedding_table = np.copy(embedding_table_before)
    config = optimizer.get_config()
    accumulator = (
        config['initial_accumulator_value'] + np.sum(gradients, axis=0)**2)
    embedding_table -= (
        config['learning_rate'] * np.sum(gradients, axis=0) /
        np.sqrt(accumulator))
    self.assertAllClose(
        self._get_variable(variables['parameters']).numpy(), embedding_table)
    self.assertAllClose(
        self._get_variable(variables['accumulators']).numpy(), accumulator)

  def _check_embedding_and_slot_variables_for_adam(self, embedding_table_before,
                                                   gradients, optimizer,
                                                   variables):
    embedding_table = np.copy(embedding_table_before)
    config = optimizer.get_config()
    g = np.sum(gradients, axis=0)
    v = g**2 * (1 - config['beta_2'])
    m = g * (1 - config['beta_1'])
    epsilon = config['epsilon']
    lr_modifier = np.sqrt(1 - config['beta_2']) / (1 - config['beta_1'])
    embedding_table -= (
        m * config['learning_rate'] * lr_modifier / (np.sqrt(v) + epsilon))
    self.assertAllClose(
        self._get_variable(variables['parameters']).numpy(),
        embedding_table,
        rtol=1e-3)
    self.assertAllClose(
        self._get_variable(variables['momenta']).numpy(), m, rtol=1e-4)
    self.assertAllClose(
        self._get_variable(variables['velocities']).numpy(), v, rtol=1e-4)

  def _check_embedding_and_slot_variables_for_ftrl(self, embedding_table_before,
                                                   gradients, optimizer,
                                                   variables):
    embedding_table = np.copy(embedding_table_before)
    config = optimizer.get_config()
    neg_lr_p = -config['learning_rate_power']
    accumulator = (
        config['initial_accumulator_value'] + np.sum(gradients, axis=0)**2)
    sigma = (accumulator**neg_lr_p - config['initial_accumulator_value']**
             neg_lr_p) / config['learning_rate']
    linear = np.sum(gradients, axis=0) - sigma * embedding_table
    quadratic = accumulator**neg_lr_p / config['learning_rate']
    embedding_table = -linear / quadratic
    actual_parameters = self._get_variable(variables['parameters']).numpy()
    # For entries where `linear` == 0, it is not worth comparing since the
    # initial values have not been touched yet and they will not agree with what
    # the actual values should be.
    actual_parameters *= (linear != 0.0)
    # FTRL has a bit more precision diff on parameters.
    self.assertAllClose(actual_parameters, embedding_table, rtol=5e-5)
    self.assertAllClose(
        self._get_variable(variables['linears']).numpy(), linear, rtol=5e-4)
    self.assertAllClose(
        self._get_variable(variables['accumulators']).numpy(), accumulator)

  @parameterized.parameters(True, False)
  def test_enqueue_with_weights(self, ragged):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    weight = 0.5
    if ragged:
      dataset = self._create_ragged_dataset(
          strategy, include_weights=True, weight=weight)
    else:
      dataset = self._create_sparse_dataset(
          strategy, include_weights=True, weight=weight)

    dataset_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def embedding_lookup(features, weights):

      def step(features, weights):
        return mid_level_api(features, weights)

      return strategy.run(step, args=(features, weights))

    features, weights = next(dataset_iter)
    # Replace the weight for the second feature by None to test.
    weights = (weights[0], None, weights[2])

    no_weights_activations = embedding_lookup(features, weights=None)
    weights_activations = embedding_lookup(features, weights=weights)

    no_weights0 = (self._unpack(strategy, no_weights_activations[0]),
                   self._unpack(strategy, no_weights_activations[1]),
                   self._unpack(strategy, no_weights_activations[2]))
    weights0 = (self._unpack(strategy, weights_activations[0]),
                self._unpack(strategy, weights_activations[1]),
                self._unpack(strategy, weights_activations[2]))
    # videos table has sum combiner and users table has mean combiner.
    # i.e. users table lookups isn't affected by the weights as all the weights
    # are the same.
    # Tuple entry 0 and 1 are the watched and favorited features from the videos
    # table and entry 2 is the friends feature from the users table.
    # Note that None was passed as a weight for entry 1 so weight should have no
    # effect.
    weight = (0.5, 1.0, 1.0)
    golden = tuple([no_weight * w for no_weight, w in zip(no_weights0, weight)])

    self.assertAllClose(golden, weights0)

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
