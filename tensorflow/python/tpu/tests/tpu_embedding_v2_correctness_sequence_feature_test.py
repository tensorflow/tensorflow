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
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_v2_correctness_base_test
from tensorflow.python.util import nest


class TPUEmbeddingCorrectnessTest(
    tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

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


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
