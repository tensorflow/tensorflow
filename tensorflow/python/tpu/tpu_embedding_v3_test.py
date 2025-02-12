# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import itertools
import logging

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest


def create_input_data_based_on_hw_requirement(
    num_chip,
    max_unique_ids_per_partition,
    per_sc_vocab_size,
    per_sc_sample_count,
    num_minibatches_per_physical_sparse_core,
):
  """Create the coo tensor based on hardware requirements.

  Args:
    num_chip: number of chips in the tpu system.
    max_unique_ids_per_partition: max unique ids per physical replica
    per_sc_vocab_size: per sc shard of table size.
    per_sc_sample_count: per sc sample count.
    num_minibatches_per_physical_sparse_core: per sc minibatch number.

  Returns:
    row_ids, col_ids, gains and splits
  """
  num_sc_per_chip = 4
  num_physical_replica = num_chip * num_sc_per_chip

  col_ids = []
  row_ids = []
  gains = []

  smallest_num_division = np.power(
      2, np.ceil(np.log2(num_minibatches_per_physical_sparse_core))
  )
  division_size = (
      per_sc_vocab_size + smallest_num_division - 1
  ) // smallest_num_division

  assert division_size >= max_unique_ids_per_partition, (
      'The max_unique_ids_per_partition is set to'
      f' {max_unique_ids_per_partition} and the number of minibatches per'
      f' sparse core is set to {num_minibatches_per_physical_sparse_core}.'
      f' But the vocab size per sparse core is {per_sc_vocab_size} which is'
      ' not going to fit that many minibatches, consider setting the number of'
      ' minibatches smaller.'
  )

  # Generating id nums for each sc on a chip. Since each chip will have the
  # same number of ids, we can shuffle this array to get random id numbers for
  # each chip.
  # Make sure that at least 1 replica contains

  per_sc_per_minibatch_id_nums_for_each_replica = np.random.randint(
      max_unique_ids_per_partition
      * (num_minibatches_per_physical_sparse_core - 1)
      + 1,
      max_unique_ids_per_partition * num_minibatches_per_physical_sparse_core
      + 1,
      size=num_physical_replica,
  )

  per_chip_sample_count = per_sc_sample_count * num_sc_per_chip

  for chip_id in range(num_chip):
    for sc_id in range(num_sc_per_chip):
      np.random.shuffle(per_sc_per_minibatch_id_nums_for_each_replica)
      for physical_replica_id in range(num_physical_replica):
        physical_replica_id_nums = (
            per_sc_per_minibatch_id_nums_for_each_replica[physical_replica_id]
        )
        # Generate local col ids based on the minibatch constrains.
        # Make sure that the generated col ids are all unique.
        local_col_ids = np.array([])
        for i in range(num_minibatches_per_physical_sparse_core):
          local_col_ids_minibatch_size = max_unique_ids_per_partition
          if i == num_minibatches_per_physical_sparse_core - 1:
            local_col_ids_minibatch_size = (
                physical_replica_id_nums - i * max_unique_ids_per_partition
            )

          local_col_ids = np.append(
              local_col_ids,
              np.random.choice(
                  np.arange(division_size),
                  size=local_col_ids_minibatch_size,
                  replace=False,
              )
              + i * division_size,
          )
        local_row_ids = np.random.randint(
            low=0,
            high=per_sc_sample_count,
            size=physical_replica_id_nums,
        )

        row_ids += list(
            local_row_ids
            + chip_id * per_chip_sample_count
            + sc_id * per_sc_sample_count
        )
        col_ids += list(
            local_col_ids * num_physical_replica + physical_replica_id
        )

        gains += list(np.random.random(size=physical_replica_id_nums))

  return np.array(row_ids), np.array(col_ids), np.array(gains)


class TPUEmbeddingV3Test(parameterized.TestCase, test.TestCase):

  def setUp(self):
    super().setUp()
    self.vocabulary_size = 16384
    self.embedding_dim = 127
    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=init_ops_v2.Constant(1.0),
        combiner='sum',
        name='video',
    )
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=init_ops_v2.Constant(2.0),
        combiner='sum',
        name='user',
    )

  def test_single_feature_single_table_lookup_with_static_buffer_size(self):
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn():
      def step(data):
        return mid_level_api(data)

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu(data)

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result)
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_two_features_single_table_lookup_with_csr_input(self):
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn():
      def step(data):
        return mid_level_api([data, data])

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu([data, data])

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result[0])
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_two_features_two_tables_stacked_lookup_with_csr_input(self):
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        allow_id_dropping=False,
    )

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          sparse_core_embedding_config=sparse_core_embedding_config,
      )
    self.assertLen(mid_level_api.embedding_tables, 1)

    @def_function.function
    def test_fn():
      def step(data):
        return mid_level_api([data, data])

      return strategy.run(step, args=(data,))

    result = test_fn()

    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu([data, data])

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result[0])
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_embedding_initialization(self):

    def element_id_initializer(shape, dtype):
      values = math_ops.range(0, shape[0] * shape[1], dtype=dtype)
      return array_ops.reshape(values, shape)

    table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=128,
        initializer=init_ops_v2.Constant(1.0),
        combiner='sum',
        name='video',
    )

    table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=128,
        initializer=element_id_initializer,
        combiner='sum',
        name='user',
    )

    feature_config = {
        'watched': tpu_embedding_v2_utils.FeatureConfig(
            table=table_video, output_shape=[16]
        ),
        'user': tpu_embedding_v2_utils.FeatureConfig(
            table=table_user, output_shape=[16]
        ),
    }

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)
    sparse_features = {
        'watched': sparse_ops.sparse_reorder(
            sparse_tensor.SparseTensor(
                indices=[[i, 0] for i in range(16)],
                values=np.arange(16),
                dense_shape=[16, 1],
            )
        ),
        'user': sparse_ops.sparse_reorder(
            sparse_tensor.SparseTensor(
                indices=[[i, 0] for i in range(16)],
                values=np.arange(16) + 16,
                dense_shape=[16, 1],
            )
        ),
    }

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )

    @def_function.function
    def test_fn(data):
      def step(partitioned_tensor):
        activations, _ = mid_level_api.dequeue(partitioned_tensor)
        return activations

      partitioned_tensor = mid_level_api.enqueue(
          data, device=strategy.extended.worker_devices[0]
      )
      return strategy.run(step, args=(partitioned_tensor,))

    result = test_fn(data)
    watched = result['watched']
    user = result['user']
    self.assertEqual(watched.shape, (16, 128))
    self.assertEqual(user.shape, (16, 128))
    self.assertAllClose(
        watched, np.ones(16 * 128).reshape((16, 128)), atol=1e-5, rtol=1e-5
    )
    self.assertAllClose(
        user,
        2048 + np.arange(16 * 128).reshape((16, 128)),
        atol=1e-5,
        rtol=1e-5,
    )

  def _recover_same_sized_tables(self, table, strategy, num_tables=1):
    # This table has num_sparse_cores mod shards, so we need to slice,
    # reconcat and reshape.
    def _unshuffle_from_sc_to_cpu(num_sc_devices, t):
      old_shape = t.shape
      # The width of the table must be a multiple of number of SC devices. The
      # tpu strategy does this round off at training time so we expect the
      # checkpoints value to meet this requirement.
      assert t.shape[0] % num_sc_devices == 0
      intermediate_tensor = array_ops.reshape(
          t, (num_sc_devices, t.shape[0] // num_sc_devices, t.shape[1])
      )
      intermediate_tensor = array_ops.transpose(intermediate_tensor, (1, 0, 2))
      return array_ops.reshape(intermediate_tensor, old_shape)

    table_partitions = [
        shard.numpy()[:, : self.embedding_dim] for shard in table.values
    ]
    full_table = np.concatenate(table_partitions, axis=0)
    full_table = _unshuffle_from_sc_to_cpu(
        strategy.num_replicas_in_sync * 4, full_table
    )

    # If we have multiple tables stacked, assume each has the same vocab sizez
    # and so was rounded the same (before stacking)
    slice_size = full_table.shape[0] // num_tables
    tables = []
    for i in range(num_tables):
      table = full_table[
          i * slice_size : i * slice_size + self.vocabulary_size, :
      ]
      # Since we apply the table stacking shift to the stacked table, we
      # are shifting it back here.
      table = np.reshape(
          table, (4 * strategy.num_replicas_in_sync, -1, table.shape[-1])
      )
      table = np.roll(table, -4 * i, axis=0)
      table = np.reshape(table, (-1, table.shape[-1]))
      tables.append(table)

    if num_tables == 1:
      return tables[0]

    return tables

  def test_single_feature_single_table_backwards_pass_with_csr_input(self):
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      # Feed in learning rate as a variable.
      weight = tf_variables.Variable(initial_value=1.0)
      # Note that we return the function read_value and not its result here
      # that is important.
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=weight.read_value)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config, optimizer=optimizer
      )
      mid_level_api.build()
    random = np.random.uniform(size=(16, self.embedding_dim)).astype(np.float32)

    @def_function.function
    def test_fn(random_input):
      def step(data, random_input):
        partitioned_tensors = mid_level_api.enqueue(data)
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn(random)

    full_table = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['video'], strategy
    )

    golden = np.ones(
        [self.vocabulary_size, self.embedding_dim], dtype=np.float32
    )
    for i in range(1024):
      golden[i, :] = golden[i, :] - random[i % 16]

    self.assertAllClose(full_table, golden)

  def test_two_feature_single_table_backwards_pass_with_csr_input(self):
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched2', output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    with strategy.scope():
      # Feed in learning rate as a variable.
      weight = tf_variables.Variable(initial_value=1.0)
      # Note that we return the function read_value and not its result here
      # that is important.
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=weight.read_value)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config, optimizer=optimizer
      )
      mid_level_api.build()
    random1 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )
    random2 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )

    @def_function.function
    def test_fn(random_input):
      def step(data, random_input):
        partitioned_tensors = mid_level_api.enqueue([data, data])
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn([random1, random2])

    full_table = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['video'], strategy
    )

    golden = np.ones(
        [self.vocabulary_size, self.embedding_dim], dtype=np.float32
    )
    for i in range(512):
      golden[i, :] = golden[i, :] - random1[i % 16] - random2[i % 16]

    self.assertAllClose(full_table, golden)

  def test_two_feature_two_tables_stacked_backwards_pass_with_csr_input(self):
    feature_config = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='watched2', output_shape=[16]
        ),
    ]

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(512)],
            values=np.arange(512),
            dense_shape=[16, 512],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        allow_id_dropping=False,
    )

    with strategy.scope():
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=1.0)
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=optimizer,
          sparse_core_embedding_config=sparse_core_embedding_config,
      )
      mid_level_api.build()
    random1 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )
    random2 = np.random.uniform(size=(16, self.embedding_dim)).astype(
        np.float32
    )

    @def_function.function
    def test_fn(random_input):
      def step(data, random_input):
        partitioned_tensors = mid_level_api.enqueue([data, data])
        preprocessed_results = tpu_replication.outside_compilation(
            mid_level_api._copy_tensors_to_device,
            partitioned_tensors=partitioned_tensors,
        )
        mid_level_api.apply_gradients(random_input, preprocessed_results)

      strategy.run(step, args=(data, random_input))

    test_fn([random1, random2])

    full_tables = self._recover_same_sized_tables(
        mid_level_api.embedding_tables['user_video'], strategy, num_tables=2
    )

    goldens = [
        np.full(
            [self.vocabulary_size, self.embedding_dim], 2, dtype=np.float32
        ),
        np.ones([self.vocabulary_size, self.embedding_dim], dtype=np.float32),
    ]

    for i in range(512):
      goldens[0][i, :] = goldens[0][i, :] - random2[i % 16]
      goldens[1][i, :] = goldens[1][i, :] - random1[i % 16]

    self.assertAllClose(full_tables, goldens)

  def test_compute_sparse_core_stats_and_pass_it_to_api(self):
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    dist_iter = iter(dist)
    data = next(dist_iter)

    num_tpu_chips = strategy.num_replicas_in_sync

    # profile the dataset to get the max ids per table and max unique ids per
    # table.
    table_to_max_ids, table_to_max_unique_ids = (
        tpu_embedding_v3.TPUEmbeddingV2.compute_sparse_core_stats(
            features=data,
            feature_config=feature_config,
            num_tpu_chips=num_tpu_chips,
            optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
        )
    )
    self.assertEqual(
        feature_config.table.dim, 127, 'Unexpected update to FeatureConfig'
    )

    sparse_core_embedding_config = tpu_embedding_v3.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=64,
        max_ids_per_table=table_to_max_ids,
        max_unique_ids_per_table=table_to_max_unique_ids,
        allow_id_dropping=False,
    )

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          sparse_core_embedding_config=sparse_core_embedding_config,
      )

    @def_function.function
    def test_fn():
      def step(data):
        return mid_level_api(data)

      return strategy.run(step, args=(data,))

    result = test_fn()
    self.assertEqual(
        feature_config.table.dim, 127, 'Unexpected update to FeatureConfig'
    )
    mid_level_api_cpu = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config=feature_config,
        optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
    )

    cpu_result = mid_level_api_cpu(data)

    for per_feature_result, per_feature_result_cpu in zip(
        nest.flatten(result[0]), nest.flatten(cpu_result)
    ):
      self.assertAllEqual(per_feature_result, per_feature_result_cpu)

  def test_raise_error_when_weight_decay_is_set(self):
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    with self.assertRaises(NotImplementedError):
      with strategy.scope():
        tpu_embedding_v3.TPUEmbeddingV2(
            feature_config=feature_config,
            optimizer=tpu_embedding_v2_utils.SGD(
                learning_rate=1.0,
                weight_decay_factor=0.1,
                multiply_weight_decay_factor_by_learning_rate=True,
            ),
        )

  @parameterized.parameters(
      *list(itertools.product([True, False], [True, False]))
  )
  def test_pipelining_and_summary_recording(
      self,
      record_summaries,
      is_enqueue_separated,
  ):
    """Test that pipelining is disabled when summaries are recorded and vice versa."""
    num_steps = 10
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video, name='watched', output_shape=[16]
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_strategy.TPUStrategy(resolver)

    sparse_features = sparse_ops.sparse_reorder(
        sparse_tensor.SparseTensor(
            indices=[[i % 16, i] for i in range(1024)],
            values=np.arange(1024),
            dense_shape=[16, 1024],
        )
    )

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
    dataset = (
        dataset.unbatch()
        .repeat()
        .batch(16 * strategy.num_replicas_in_sync, drop_remainder=True)
    )

    dist = strategy.experimental_distribute_dataset(
        dataset,
        options=distribute_lib.InputOptions(experimental_fetch_to_device=False),
    )
    tpu_iter = iter(dist)

    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_config,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          pipeline_execution_with_tensor_core=True,
      )
      weight = tf_variables.Variable(initial_value=10)

    if is_enqueue_separated:

      @def_function.function
      def tpu_test_fn(tpu_iter):
        def step(data):
          activations, preserved_results = mid_level_api.dequeue(data)
          activations = activations * weight
          mid_level_api.apply_gradients(activations, preserved_results)
          return activations

        @def_function.function
        def enqueue(data):
          return mid_level_api.enqueue(
              data, device=strategy.extended.worker_devices[0]
          )

        inner_result = array_ops.zeros(
            [16, self.embedding_dim], dtype=dtypes.float32
        )
        for _ in math_ops.range(num_steps):
          host_partition_result = enqueue(next(tpu_iter))
          inner_result = strategy.run(step, args=(host_partition_result,))
        return inner_result

    else:

      @def_function.function
      def tpu_test_fn(tpu_iter):
        def step(data):
          activations, preserved_results = mid_level_api(data)
          activations = activations * weight
          mid_level_api.apply_gradients(activations, preserved_results)
          return activations

        inner_result = array_ops.zeros(
            [16, self.embedding_dim], dtype=dtypes.float32
        )
        for _ in math_ops.range(num_steps):
          inner_result = strategy.run(step, args=(next(tpu_iter),))
        return inner_result

    with summary_ops_v2.record_if(record_summaries):
      is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
      # If we are recording summaries the function should not be pipelined and
      # vice versa.
      self.assertNotEqual(is_pipelined, record_summaries)

  def is_function_pipelined(self, tf_function, *args):
    """Returns whether the tf_function is flagged for embedding pipelining.

    Args:
      tf_function: a tf.function.
      *args: the arguments to the tf_function.

    Returns:
      Whether the tf_function is (will be) pipelined.

    This helper looks for a while loop in the provided function. It then looks
    for any op that has the pipelining attribute (e.g.,
    XlaSparseDenseMatmulWithCsrInput). The presence of the attribute indicates
    that the function is to be pipelined during compilation.

    Example usge:

      with summary_ops_v2.record_if(False):
        is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
        self.assertTrue(is_pipelined)
      with summary_ops_v2.record_if(True):
        is_pipelined = self.is_function_pipelined(tpu_test_fn, tpu_iter)
        self.assertFalse(is_pipelined)
    """
    # The attribute to look for to see if function is pipelined:
    attr_name = tpu_embedding_v3._PIPELINE_ATTRIBUTE

    ############################################################################
    # First, find the while loop op.
    ############################################################################
    func_graph = tf_function.get_concrete_function(*args).graph
    # type(func_graph) is tensorflow.python.framework.func_graph.FuncGraph
    while_op = None
    for op in func_graph.get_operations():
      # type(op) is tensorflow.python.framework.ops.Operation
      if op.name == 'while':
        while_op = op
        break
    self.assertIsNotNone(while_op, 'while op not found')

    ############################################################################
    # Second, find the body of the while loop.
    ############################################################################
    body_name = while_op.get_attr('body').name
    while_body_func = None
    try:
      while_body_func = func_graph.get_concrete_function(body_name)
    except AttributeError as exc:
      for func in while_op.graph._functions.values():
        if func.name.decode() == body_name:
          while_body_func = func
          break
      if while_body_func is None:
        raise ValueError('body not found') from exc
    while_body_graph = while_body_func.graph

    ############################################################################
    # Finally, find any forward pass op by looking for the pipelining attribute.
    ############################################################################
    pipelined_op = None
    for op in while_body_graph.get_operations():
      try:
        attr = op.get_attr(attr_name)
        pipelined_op = op
        logging.info(
            'Op "%s" has pipelining attr: %s : %s', op.name, attr_name, attr
        )
        break
      except ValueError:
        pass
    has_pipelining_attr = pipelined_op is not None
    return has_pipelining_attr


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  config.enable_mlir_bridge()
  test.main()
