# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.util import nest

FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')
flags.DEFINE_string('model_dir', os.environ.get('TEST_TMPDIR'),
                    'A temporary directory.')


class TPUEmbeddingCorrectness(parameterized.TestCase, test.TestCase):

  def setUp(self):
    super(TPUEmbeddingCorrectness, self).setUp()
    self.embedding_values = np.array(list(range(32)), dtype=np.float64)
    self.initializer = init_ops_v2.Constant(self.embedding_values)
    # Embedding for video initialized to
    # 0 1 2 3
    # 4 5 6 7
    # ...
    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=8,
        dim=4,
        initializer=self.initializer,
        combiner='sum',
        name='video')
    # Embedding for user initialized to
    # 0 1
    # 2 3
    # 4 5
    # 6 7
    # ...
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=16,
        dim=2,
        initializer=self.initializer,
        combiner='mean',
        name='user')
    self.feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched'),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='favorited'),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends'))

    self.batch_size = 2
    self.data_batch_size = 4

    # One (global) batch of inputs
    # sparse tensor for watched:
    # row 0: 0
    # row 1: 0, 1
    # row 2: 0, 1
    # row 3: 1
    self.feature_watched_indices = [[0, 0], [1, 0], [1, 1],
                                    [2, 0], [2, 1], [3, 0]]
    self.feature_watched_values = [0, 0, 1, 0, 1, 1]
    self.feature_watched_row_lengths = [1, 2, 2, 1]
    # sparse tensor for favorited:
    # row 0: 0, 1
    # row 1: 1
    # row 2: 0
    # row 3: 0, 1
    self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0],
                                      [2, 0], [3, 0], [3, 1]]
    self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
    self.feature_favorited_row_lengths = [2, 1, 1, 2]
    # sparse tensor for friends:
    # row 0: 3
    # row 1: 0, 1, 2
    # row 2: 3
    # row 3: 0, 1, 2
    self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2],
                                    [2, 0], [3, 0], [3, 1], [3, 2]]
    self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
    self.feature_friends_row_lengths = [1, 3, 1, 3]
    self.resolver = None

  def _get_strategy(self):
    self.resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    remote.connect_to_cluster(self.resolver)
    tpu_strategy_util.initialize_tpu_system(self.resolver)
    return tpu_strategy.TPUStrategy(self.resolver)

  def _create_strategy_and_mid_level(self, optimizer_name):
    strategy = self._get_strategy()

    with strategy.scope():
      if optimizer_name == 'sgd':
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      elif optimizer_name == 'adagrad':
        optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
      elif optimizer_name == 'adam':
        optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
      elif optimizer_name == 'ftrl':
        optimizer = tpu_embedding_v2_utils.FTRL(learning_rate=0.1)
      else:
        raise ValueError('optimizer is not recognized: ', optimizer_name)
      mid_level_api = self._create_mid_level(optimizer=optimizer)

    return strategy, mid_level_api, optimizer

  @parameterized.parameters(*itertools.product(
      ['sgd', 'adagrad', 'adam', 'ftrl'], [True, False], [True, False]))
  def test_embedding(self, optimizer_name, training, sparse):
    strategy, mid_level_api, optimizer = (
        self._create_strategy_and_mid_level(optimizer_name))

    if sparse:
      dataset = self._create_sparse_dataset(strategy)
    else:
      dataset = self._create_ragged_dataset(strategy)

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():

      def step():
        """Create and run computation that returns the embedding activations."""
        if not training:
          activations = mid_level_api.dequeue()
          total_loss = _get_total_loss_tensor(activations)
          ret_val = [total_loss] + list(activations)
          return ret_val
        else:
          with backprop.GradientTape() as tape:
            activations = mid_level_api.dequeue()
            tape.watch(activations)
            total_loss = _get_total_loss_tensor(activations)
            loss_per_replica = total_loss / strategy.num_replicas_in_sync
          gradients = tape.gradient(loss_per_replica, activations)
          mid_level_api.apply_gradients(gradients)
        ret_val = [total_loss] + list(activations)
        return ret_val

      mid_level_api.enqueue(next(dist_iter), training=training)
      result = strategy.run(step)
      return result

    # Run model.
    shard_out_val = test_fn()

    # Retrieve TPU weights to CPU.
    mid_level_api._retrieve_variables()

    # Compute sparse tensors for global batch.
    input_data = next(iter(self._create_sparse_dataset(strategy)))

    # Check results.
    self._check_results(strategy, shard_out_val, training, input_data,
                        mid_level_api._variables,
                        optimizer)

  def _create_mid_level(self, optimizer=None):
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    return tpu_embedding_v2.TPUEmbedding(
        feature_config=self.feature_config,
        optimizer=optimizer)

  def _create_sparse_data(self, include_weights, weight=0.5):
    sparse_features = (
        sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices,
            values=self.feature_watched_values,
            dense_shape=[self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_favorited_indices,
            values=self.feature_favorited_values,
            dense_shape=[self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_friends_indices,
            values=self.feature_friends_values,
            dense_shape=[self.data_batch_size, 3]))
    if include_weights:
      weights = []
      for sparse in sparse_features:
        values = (
            array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight)
        weights.append(sparse_tensor.SparseTensor(
            indices=sparse.indices,
            values=values,
            dense_shape=sparse.dense_shape))
      sparse_features = (sparse_features, tuple(weights))
    return sparse_features

  def _create_sparse_dataset(self, strategy, include_weights=False, weight=0.5):
    # Create dataset for enqueue operation
    sparse_features = self._create_sparse_data(include_weights, weight)

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
    # Create dataset for enqueue operation
    sparse_features = self._create_sparse_data(include_weights, weight)
    ragged_features = nest.map_structure(ragged_tensor.RaggedTensor.from_sparse,
                                         sparse_features)

    dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_dense_input_fn(self, strategy, include_weights=False, weight=0.5):

    def input_fn(ctx):
      del ctx
      features = (
          constant_op.constant(self.feature_watched_values[-2:],
                               dtype=dtypes.int32),
          constant_op.constant(self.feature_favorited_values[-2:],
                               dtype=dtypes.int32),
          constant_op.constant(self.feature_friends_values[-2:],
                               dtype=dtypes.int32))
      if include_weights:
        weights = [array_ops.ones_like(t, dtype=dtypes.float32) * weight
                   for t in features]
        features = (features, tuple(weights))
      return dataset_ops.DatasetV2.from_tensors(features).repeat()

    return input_fn

  def _check_results(self, strategy, shard_out_val, training, input_data,
                     table_to_variable, optimizer):
    num_replicas = strategy.num_replicas_in_sync

    # Unpack the values `strategy.run()` returns.
    loss = _unpack(strategy, shard_out_val[0])
    activation_watched = _unpack(strategy, shard_out_val[1])
    activation_favorited = _unpack(strategy, shard_out_val[2])
    activation_friends = _unpack(strategy, shard_out_val[3])

    # Core 0:
    # Calculate the values of embedding activations.
    activation_watched_gold0 = np.array([[0, 1, 2, 3], [4, 6, 8, 10]])
    activation_favorited_gold0 = np.array([[4, 6, 8, 10], [4, 5, 6, 7]])
    # Second row of `activation_friends_gold0` is the mean of the following.
    # row 0: 0 1
    # row 1: 2 3
    # row 2: 4 5
    activation_friends_gold0 = np.array([[6, 7], [2, 3]])

    loss_gold0 = _compute_loss(activation_watched_gold0,
                               activation_favorited_gold0,
                               activation_friends_gold0)

    # Add on values from other cores:
    # Activations for watched are an alternating sequence of
    # activation_watched_gold0 and activation_favorited_gold0.
    # For favorited it is the same but in the opposite order.
    activation_watched_gold = np.concatenate(
        (np.concatenate((np.expand_dims(activation_watched_gold0, axis=0),) *
                        (num_replicas // 2)),
         np.concatenate((np.expand_dims(activation_favorited_gold0, axis=0),) *
                        (num_replicas // 2))),
        axis=1).reshape([self.batch_size * num_replicas, 4])
    activation_favorited_gold = np.concatenate(
        (activation_watched_gold[self.batch_size:,],
         activation_watched_gold[0:self.batch_size,]))
    activation_friends_gold = np.concatenate(
        (activation_friends_gold0,) * num_replicas)

    loss_gold = [loss_gold0] * num_replicas

    # Test values.
    self.assertAllClose(activation_watched_gold, activation_watched)
    self.assertAllClose(activation_favorited_gold, activation_favorited)
    self.assertAllClose(activation_friends_gold, activation_friends)

    self.assertAllClose(loss_gold, loss)

    embedding_table_video_before = np.copy(
        np.reshape(self.embedding_values, [8, 4]))
    embedding_table_user_before = np.copy(
        np.reshape(self.embedding_values, [16, 2]))

    global_batch_size = self.batch_size * num_replicas
    if training:
      gradient_wrt_watched_gold = (2 * activation_watched_gold /
                                   global_batch_size)
      gradient_wrt_favorited_gold = (2 * activation_favorited_gold /
                                     global_batch_size)
      gradient_wrt_friends_gold = (2 * activation_friends_gold /
                                   global_batch_size)

      # Calculate gradients wrt embedding tables.
      gradients_wrt_user = (
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_friends_gold,
              embedding_table_user_before, input_data[2].indices.numpy(),
              input_data[2].values.numpy(), self.table_user.combiner))
      gradients_wrt_video = (
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_favorited_gold,
              embedding_table_video_before, input_data[1].indices.numpy(),
              input_data[1].values.numpy(), self.table_video.combiner) +
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_watched_gold,
              embedding_table_video_before, input_data[0].indices.numpy(),
              input_data[0].values.numpy(), self.table_video.combiner))

      self._check_embedding_and_slot_variables(embedding_table_user_before,
                                               gradients_wrt_user,
                                               embedding_table_video_before,
                                               gradients_wrt_video,
                                               optimizer,
                                               table_to_variable)

  def _check_embedding_and_slot_variables(self, embedding_table_user_before,
                                          gradients_wrt_user,
                                          embedding_table_video_before,
                                          gradients_wrt_video,
                                          optimizer,
                                          table_to_variable):
    if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
      check_fn = self._check_embedding_and_slot_variables_for_sgd
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
      check_fn = self._check_embedding_and_slot_variables_for_adagrad
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
      check_fn = self._check_embedding_and_slot_variables_for_adam
    elif isinstance(optimizer, tpu_embedding_v2_utils.FTRL):
      check_fn = self._check_embedding_and_slot_variables_for_ftrl
    else:
      raise ValueError('optimizer is not recognized: ', type(optimizer))
    check_fn(embedding_table_user_before, gradients_wrt_user,
             optimizer, table_to_variable[self.table_user.name])
    check_fn(embedding_table_video_before, gradients_wrt_video,
             optimizer, table_to_variable[self.table_video.name])

  def _check_embedding_and_slot_variables_for_sgd(self, embedding_table_before,
                                                  gradients,
                                                  optimizer,
                                                  variables):
    embedding_table = np.copy(embedding_table_before)
    embedding_table -= optimizer.learning_rate * np.sum(gradients, axis=0)
    self.assertAllClose(_get_variable(variables['parameters']).numpy(),
                        embedding_table)

  def _check_embedding_and_slot_variables_for_adagrad(self,
                                                      embedding_table_before,
                                                      gradients,
                                                      optimizer,
                                                      variable):
    embedding_table = np.copy(embedding_table_before)
    accumulator = (
        optimizer.initial_accumulator_value + np.sum(gradients, axis=0)**2)
    embedding_table -= (
        optimizer.learning_rate * np.sum(gradients, axis=0) /
        np.sqrt(accumulator))
    self.assertAllClose(_get_variable(variable['parameters']).numpy(),
                        embedding_table)
    self.assertAllClose(_get_variable(variable['accumulators']).numpy(),
                        accumulator)

  def _check_embedding_and_slot_variables_for_adam(self, embedding_table_before,
                                                   gradients,
                                                   optimizer,
                                                   variable):
    embedding_table = np.copy(embedding_table_before)
    g = np.sum(gradients, axis=0)
    v = g**2 * (1 - optimizer.beta_2)
    m = g * (1 - optimizer.beta_1)
    epsilon = optimizer.epsilon
    # TPU Embeddings don't have the LR decay factor for Adam.
    lr_modifier = 1
    embedding_table -= (
        m * optimizer.learning_rate * lr_modifier / (np.sqrt(v) + epsilon))
    self.assertAllClose(_get_variable(variable['parameters']).numpy(),
                        embedding_table, rtol=1e-4)
    self.assertAllClose(_get_variable(variable['momenta']).numpy(),
                        m, rtol=1e-4)
    self.assertAllClose(_get_variable(variable['velocities']).numpy(),
                        v, rtol=1e-4)

  def _check_embedding_and_slot_variables_for_ftrl(self, embedding_table_before,
                                                   gradients, optimizer,
                                                   variable):
    embedding_table = np.copy(embedding_table_before)
    neg_lr_p = -optimizer.learning_rate_power
    accumulator = (
        optimizer.initial_accumulator_value + np.sum(gradients, axis=0)**2)
    sigma = (accumulator**neg_lr_p - optimizer.initial_accumulator_value**
             neg_lr_p) / optimizer.learning_rate
    linear = np.sum(gradients, axis=0) - sigma * embedding_table
    quadratic = accumulator**neg_lr_p / optimizer.learning_rate
    embedding_table = -linear / quadratic
    actual_parameters = _get_variable(variable['parameters']).numpy()
    # For entries where `linear` == 0, it is not worth comparing since the
    # initial values have not been touched yet and they will not agree with what
    # the actual values should be.
    actual_parameters *= (linear != 0.0)
    # FTRL has a bit more precision diff on parameters.
    self.assertAllClose(actual_parameters, embedding_table, rtol=5e-5)
    self.assertAllClose(
        _get_variable(variable['linears']).numpy(), linear, rtol=5e-4)
    self.assertAllClose(
        _get_variable(variable['accumulators']).numpy(), accumulator)

  def _get_replica_numpy(self, structured, strategy, replica_id):
    def select_replica(x):
      x = strategy.experimental_local_results(x)
      if len(x) == 1:
        return x.numpy()
      return x[replica_id].numpy()
    return nest.map_structure(select_replica, structured)

  def test_dense_lookup(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    input_fn = self._create_dense_input_fn(strategy)
    dist = strategy.distribute_datasets_from_function(
        input_fn,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(dist_iter), training=False)
      return strategy.run(step)

    # Run model.
    shard0 = self._get_replica_numpy(test_fn(), strategy, 0)

    # embedding_values is a linear list, so we reshape to match the correct
    # shape of the corresponding table before performing the lookup.
    numpy_videos = np.reshape(self.embedding_values, (8, 4))
    numpy_users = np.reshape(self.embedding_values, (16, 2))
    golden = ((numpy_videos[self.feature_watched_values[-2:]],
               numpy_videos[self.feature_favorited_values[-2:]],
               numpy_users[self.feature_friends_values[-2:]]))
    self.assertAllClose(shard0, golden)

  @parameterized.parameters([True, False])
  def test_sequence_embeddings(self, sparse):
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched',
            max_sequence_length=2),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='favorited',
            max_sequence_length=2),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends',
            max_sequence_length=3))
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    strategy = self._get_strategy()
    num_replicas = strategy.num_replicas_in_sync
    with strategy.scope():
      mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config,
          optimizer=optimizer)
    # Call build here. We call 'next' outside of the tf.function and this
    # results in data where the shape of the sparse tensor is a tensor which we
    # can't tell the shape of at tracing time.
    mid_level.build(self.batch_size)
    if sparse:
      dataset = self._create_sparse_dataset(strategy)
    else:
      dataset = self._create_ragged_dataset(strategy)
    data = next(
        iter(
            strategy.experimental_distribute_dataset(
                dataset,
                options=distribute_lib.InputOptions(
                    experimental_fetch_to_device=False))))

    @def_function.function
    def embedding_and_set_gradients(data):
      def tpu_fn():
        activations = mid_level.dequeue()
        mid_level.apply_gradients(nest.map_structure(array_ops.ones_like,
                                                     activations))
        return activations
      mid_level.enqueue(data)
      return strategy.run(tpu_fn)

    @def_function.function
    def embedding_only(data):
      def tpu_fn():
        return mid_level.dequeue()
      mid_level.enqueue(data)
      return strategy.run(tpu_fn)

    # Only check core 0.
    before_update = self._get_replica_numpy(
        embedding_and_set_gradients(data), strategy, 0)
    after_update = self._get_replica_numpy(embedding_only(data), strategy, 0)

    # For videos table, row 0 and row 1 are looked up 3*num_replicas times as
    # they occur 3 times per replica (considering the features 0 and 1 which are
    # both looked up in the videos table).
    # Feature 0 has ids [0, 0, 1], [0, 1, 1], ... repeated over num_replicas
    # Feature 1 has ids [0, 1, 1], [0, 0, 1], ... repeated over num_replicas
    # This means that both rows 0 and 1 get a -0.1*3*num_replicas update
    # For users table, each row is looked up twice:
    # Feature 2 has ids [3, 0, 1, 2], .. repeated over num_replicas
    # This means that we get a -0.1*num_replicas update to the third feature.

    # In general this means that after the update, if we lookup feature 0 and 1
    # the values will be 0.3*num_replicas lower per entry and for feature 2 they
    # will be 0.1*num_replicas lower.
    # The one issue is that these lookups contain padding values.
    # For core 0, we get the first 2 elements of the 4 element batch.
    # For feature 0, the indices are [[0, 0], [1, 0], [1, 1]] with max sequence
    # length of 2, which means that [0, 1] will be 0s.
    # For feature 1, the indices are [[0, 0], [0, 1], [1, 0]] with max sequence
    # length of 2, which means that [1, 1] will be 0s.
    # For feature 2, the indices are [[0, 0], [1, 0], [1, 1], [1, 2]] with max
    # sequence length of 3, which means that [0, 1], [0, 2] will be 0s.
    # The following masks represent that so that we only apply the above updates
    # to the non-padding rows:
    masks = (
        np.array([[[1], [0]], [[1], [1]]]),
        np.array([[[1], [1]], [[1], [0]]]),
        np.array([[[1], [0], [0]], [[1], [1], [1]]]))

    per_row_update = (0.3 * num_replicas,
                      0.3 * num_replicas,
                      0.1 * num_replicas)
    golden = tuple([before - update * mask for before, update, mask in
                    zip(before_update, per_row_update, masks)])
    self.assertAllClose(golden, after_update)


def _compute_gradients_wrt_embedding_table(batch_size,
                                           gradient_wrt_activation,
                                           embedding_table,
                                           feature_indices,
                                           feature_values,
                                           combiner,
                                           max_sequence_length=0):
  """Compute gradients wrt embedding_table.

  Args:
    batch_size: `int`, batch size.
    gradient_wrt_activation: `np.array` with shape `batch_size` by
      embedding `dimension`.
    embedding_table: `np.array` with shape `vocabulary_size` by embedding
      `dimension`.
    feature_indices: `indices` as used to construct `SparseTensor`.
    feature_values: `values` as used to construct `SparseTensor`.
    combiner: `String`, 'mean' or 'sum'.
    max_sequence_length: If non-zero, a sequence feature with the given length.

  Returns:
    Gradients wrt `embedding_table`, an `np.array`s with shape
      `batch_size` by `vocabulary_size` by
      embedding `dimension`.

  Raises:
    ValueError: if `combiner` is not one of 'mean' or 'sum'.
  """
  if combiner not in ('mean', 'sum'):
    raise ValueError('`combiner` must be mean or sum; got {}.'.format(combiner))
  grads = []
  for i in range(batch_size):
    grad = np.zeros_like(embedding_table)
    count = 0
    for (batch_i, seq_index), vocabulary_id in zip(feature_indices,
                                                   feature_values):
      if batch_i == i:
        count += 1
        if max_sequence_length > 0:
          if seq_index < max_sequence_length:
            grad[vocabulary_id, :] += gradient_wrt_activation[i, seq_index, :]
        else:
          grad[vocabulary_id, :] += gradient_wrt_activation[i, :]
    if combiner == 'mean' and not max_sequence_length:
      grad = grad / count
    grads.append(grad)
  return np.stack(grads)


def _unpack(strategy, per_replica_output):
  per_replica_output = strategy.experimental_local_results(per_replica_output)
  per_replica_output = array_ops.concat(per_replica_output, axis=0).numpy()
  return per_replica_output


def _get_total_loss_tensor(activations):
  losses = []
  for activation in activations:
    losses.append(
        math_ops.reduce_mean(
            math_ops.reduce_sum(
                gen_math_ops.squared_difference(activation, 0), 1)))
  total_loss = array_ops.expand_dims_v2(sum(losses), 0)
  return total_loss


def _compute_loss(activation_watched, activation_favorited, activation_friends):
  watched_loss = np.mean(np.sum(activation_watched**2, axis=1))
  if len(activation_favorited.shape) == 2:
    favorited_loss = np.mean(np.sum(activation_favorited**2, axis=1))
  else:
    favorited_loss = np.mean(np.sum(activation_favorited**2, axis=(1, 2)))
  if len(activation_friends.shape) == 2:
    friends_loss = np.mean(np.sum(activation_friends**2, axis=1))
  else:
    friends_loss = np.mean(np.sum(activation_friends**2, axis=(1, 2)))
  loss = watched_loss + favorited_loss + friends_loss
  return loss


def _get_variable(variable):
  if isinstance(variable, tpu_embedding_v2.TPUShardedVariable):
    return variable.variables[0]
  return variable


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
