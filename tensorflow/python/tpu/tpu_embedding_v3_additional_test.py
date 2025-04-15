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
"""Additional multi/single worker tests for tpu_embedding_v3."""

import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as tf_checkpoint
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import save as tf_save
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3


_TPU = flags.DEFINE_string('tpu', None, 'The TPU to use for TPUStrategy.')
# pylint: disable=g-long-lambda


RowIdInitializer = tpu_embedding_v2_utils.RowIdInitializer


def get_replica_values(per_replica_or_tensor):
  if isinstance(per_replica_or_tensor, values_lib.PerReplica):
    return per_replica_or_tensor.values
  else:
    return [per_replica_or_tensor]


def pad_to_shape_initializer(init_mat):
  """An initializer that pads init_mat out to the given shape."""
  return lambda shape, dtype: array_ops.pad(
      init_mat,
      [
          [0, shape[0] - init_mat.shape[0]],
          [0, shape[1] - init_mat.shape[1]],
      ],
      'CONSTANT',
  )


class TPUEmbeddingLayerV2Test(parameterized.TestCase, test.TestCase):

  def setUp(self):
    super().setUp()
    self.vocabulary_size = 128
    self.embedding_dim = 8

    self.table_video = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=RowIdInitializer(0),
        combiner='sum',
        name='video',
    )
    self.table_user = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.vocabulary_size,
        dim=self.embedding_dim,
        initializer=RowIdInitializer(1000),
        combiner='sum',
        name='user',
    )

    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=_TPU.value)
    if _TPU.value is None:
      remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)

    # FIXME(b/303466959): Remove this device assignment after TPUStrategy
    # can follow the actual device ordering under SC.
    topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
    tpu_metadata = resolver.get_tpu_system_metadata()

    device_assignment = device_assignment_lib.DeviceAssignment.build(
        topology, num_replicas=tpu_metadata.num_cores
    )
    self._strategy = tpu_strategy.TPUStrategyV2(
        resolver, experimental_device_assignment=device_assignment
    )

    self.addCleanup(tpu_cluster_resolver.shutdown_tpu_system, resolver)

    self.feature_video = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_video,
        name='video',
        output_shape=[self.vocabulary_size],
    )
    self.feature_user = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='user', output_shape=[self.vocabulary_size]
    )

    self.assertEqual(
        self._strategy.extended._tpu_devices.shape, (tpu_metadata.num_cores, 1)
    )

  def testSingleTableInitializeAndLookup(self):
    # This test sets up devices to lookup the entire table.

    feature_config = [self.feature_video]

    strategy = self._strategy

    with strategy.scope():
      embedding_layer = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config,
          tpu_embedding_v2_utils.Adagrad(),
          pipeline_execution_with_tensor_core=True,
      )

      @def_function.function
      def train_step(features):
        def train_step_fn(features):
          return embedding_layer(features)[0]

        return strategy.run(train_step_fn, args=(features,))

    def value_fn(ctx):
      del ctx  # unused
      return [
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
      ]

    features = strategy.experimental_distribute_values_from_function(value_fn)

    [embeddings] = train_step(features)

    expected = RowIdInitializer(0)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    for replica in get_replica_values(embeddings):
      self.assertAllClose(replica, expected)

  def testStackedTableInitializeAndLookup(self):
    # This test sets up devices to lookup the entire table.
    feature_config = [self.feature_video, self.feature_user]

    strategy = self._strategy

    def value_fn(ctx):
      del ctx  # unused
      return [
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
          sparse_tensor.SparseTensor(
              indices=[[i, 0] for i in range(0, self.vocabulary_size)],
              values=np.arange(0, self.vocabulary_size),
              dense_shape=[self.vocabulary_size, 1],
          ),
      ]

    features = strategy.experimental_distribute_values_from_function(value_fn)

    with strategy.scope():
      embedding_layer = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config,
          tpu_embedding_v2_utils.Adagrad(),
          pipeline_execution_with_tensor_core=True,
      )

      @def_function.function
      def train_step(features):
        def train_step_fn(features):
          return embedding_layer(features)[0]

        return strategy.run(train_step_fn, args=(features,))

    [embeddings_video, embeddings_user] = train_step(features)

    expected_video = RowIdInitializer(0)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    expected_user = RowIdInitializer(1000)(
        shape=(self.vocabulary_size, self.embedding_dim), dtype=dtypes.float32
    )
    for replica_video, replica_user in zip(
        get_replica_values(embeddings_video),
        get_replica_values(embeddings_user),
    ):
      self.assertAllClose(replica_video, expected_video)
      self.assertAllClose(replica_user, expected_user)

  def testTwoTablesStackedHaveCorrectInitialValues(self):
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=10,
        dim=2,
        initializer=pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=240, dtype=np.float32
    ).reshape([20, 2])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=2,
        initializer=pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
          pipeline_execution_with_tensor_core=True,
      )
      # The two tables should be stacked into the same variable.
      self.assertLen(mid_level_api.embedding_tables, 1)

  def testCpuRestoreForNoStackedTables(self):
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=12,
        dim=2,
        initializer=pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=380, dtype=np.float32
    ).reshape([20, 9])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=9,  # to ensure stacking does not occur
        initializer=pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      # The two tables should be *not* be stacked.
      self.assertLen(mid_level_api.embedding_tables, 2)
      # Save v3 embedding
      checkpoint = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in serving embedding
    with distribute_lib.get_strategy().scope():
      serving_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(serving_embedding)
      checkpoint_for_restore.restore(checkpoint_path)
      serving_embedding.build()
    # Check that 2 tables exist in serving.
    self.assertLen(serving_embedding.embedding_tables, 2)
    look_for_row_idx_1 = [
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
    ]
    row_lookup = serving_embedding(look_for_row_idx_1)
    self.assertAllEqual(
        row_lookup[0],
        tf_constant(
            [
                102.0,
                103.0,
            ],
            shape=(1, 2),
        ),
    )
    self.assertAllEqual(
        row_lookup[1],
        tf_constant(
            [209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0],
            shape=(1, 9),
        ),
    )

  def testCpuRestoreForStackedTables(self):
    table1_initial_value = np.arange(
        start=100, stop=120, dtype=np.float32
    ).reshape([10, 2])
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=12,
        dim=2,
        initializer=pad_to_shape_initializer(table1_initial_value),
        combiner='sum',
        name='table1',
    )
    table2_initial_value = np.arange(
        start=200, stop=240, dtype=np.float32
    ).reshape([20, 2])
    table2 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=20,
        dim=2,
        initializer=pad_to_shape_initializer(table2_initial_value),
        combiner='sum',
        name='table2',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
        tpu_embedding_v2_utils.FeatureConfig(
            table=table2, name='feature2', output_shape=[16]
        ),
    ]

    with self._strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      # The two tables should be stacked into the same variable.
      self.assertLen(mid_level_api.embedding_tables, 1)
      # Save v3 embedding
      checkpoint = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in serving embedding
    with distribute_lib.get_strategy().scope():
      serving_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.SGD(learning_rate=1.0),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(serving_embedding)
      checkpoint_for_restore.restore(checkpoint_path)
      serving_embedding.build()
    # Check that unstacking happens on restore
    self.assertLen(serving_embedding.embedding_tables, 2)
    look_for_row_idx_1 = [
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]
        ),
    ]
    row_lookup = serving_embedding(look_for_row_idx_1)
    self.assertAllEqual(
        row_lookup[0],
        tf_constant([102.0, 103.0], shape=(1, 2)),
    )
    self.assertAllEqual(
        row_lookup[1],
        tf_constant([202.0, 203.0], shape=(1, 2)),
    )
    saved_model_path = os.path.join(
        self.create_tempdir().full_path, 'saved_model'
    )
    tf_save.save(
        serving_embedding, saved_model_path
    )
    loaded_embedding = tf_load.load(saved_model_path)
    self.assertLen(loaded_embedding._variables, 2)

  def testUnshardedToTpuRestore(self):
    table1 = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=25,
        dim=6,
        initializer=RowIdInitializer(0),
        combiner='sum',
        name='table1',
    )
    feature_configs = [
        tpu_embedding_v2_utils.FeatureConfig(
            table=table1, name='feature1', output_shape=[16]
        ),
    ]

    with distribute_lib.get_strategy().scope():
      cpu_embedding = tpu_embedding_for_serving.TPUEmbeddingForServing(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.Adagrad(0.1),
      )

      self.assertLen(cpu_embedding.embedding_tables, 1)
      # Save unsharded embedding
      checkpoint = tf_checkpoint.Checkpoint(cpu_embedding)
      checkpoint_prefix = os.path.join(self.create_tempdir().full_path, 'ckpt')
      checkpoint_path = checkpoint.save(checkpoint_prefix)

    # Restore in TPU embedding
    strategy = self._strategy
    with strategy.scope():
      mid_level_api = tpu_embedding_v3.TPUEmbeddingV2(
          feature_config=feature_configs,
          optimizer=tpu_embedding_v2_utils.Adagrad(0.1),
      )
      checkpoint_for_restore = tf_checkpoint.Checkpoint(mid_level_api)
      checkpoint_for_restore.restore(checkpoint_path)
      mid_level_api.build()

    replicas, cores_per_replica = strategy.extended._tpu_devices.shape
    total_sc_shards = (
        replicas * cores_per_replica * mid_level_api._num_sc_per_chip
    )
    padded_vocab = 8 * total_sc_shards
    unsharded_full_value = cpu_embedding._variables['table1']['parameters']
    shard_shape = [padded_vocab // total_sc_shards, 8]
    offset = 0
    ordered_devices = []
    for devices in strategy.extended._tpu_devices:  # pylint: disable=protected-access
      ordered_devices.extend(devices)
    for device in ordered_devices:
      partition = []
      for _ in range(mid_level_api._num_sc_per_chip):
        sh = unsharded_full_value[offset::total_sc_shards, :]
        padded_sh = pad_to_shape_initializer(sh)(shard_shape, dtypes.float32)
        partition.append(padded_sh)
        offset += 1
      # Check value at each partition
      self.assertAllEqual(
          mid_level_api._variables['table1']['parameters'].read_from_device(
              device
          ),
          array_ops.concat(partition, axis=0),
      )


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  config.enable_mlir_bridge()
  test.main()
