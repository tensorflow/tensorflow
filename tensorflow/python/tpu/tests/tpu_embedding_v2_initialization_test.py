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
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def test_enqueue_dequeue_apply_gradients_on_cpu(self):
    # Dequeue on CPU.
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.dequeue()
    # Enqueue on CPU.
    features = {
        'watched': sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices,
            values=self.feature_watched_values,
            dense_shape=[2, 2])}
    with self.assertRaises(RuntimeError):
      mid_level_api.enqueue(features)
    # Apply gradient on CPU.
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.apply_gradients(None)

  def test_multiple_creation(self):
    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=self.table_user, name='friends', max_sequence_length=2)
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    strategy = self._get_strategy()
    with strategy.scope():
      embedding_one = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config, optimizer=optimizer)
      embedding_two = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config, optimizer=optimizer)

    # The first TPU embedding should be able to be built.
    # The second one should fail with a runtime error indicating another TPU
    # embedding has already been initialized on TPU.
    embedding_one.build(64)
    with self.assertRaisesRegex(RuntimeError,
                                'TPU is already initialized for embeddings.'):
      embedding_two.build(64)

  def test_same_config_different_instantiations(self):
    num_tables = 30
    table_dim = np.random.randint(1, 128, size=[num_tables])
    table_vocab_size = np.random.randint(100, 1000, size=[num_tables])
    table_names = ['table{}'.format(i) for i in range(num_tables)]
    table_data = list(zip(table_dim, table_vocab_size, table_names))
    strategy = self._get_strategy()

    def tpu_embedding_config():
      feature_configs = []
      for dim, vocab, name in table_data:
        feature_configs.append(
            tpu_embedding_v2_utils.FeatureConfig(
                table=tpu_embedding_v2_utils.TableConfig(
                    vocabulary_size=int(vocab),
                    dim=int(dim),
                    initializer=init_ops_v2.Zeros(),
                    name=name)))
      optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
      with strategy.scope():
        mid_level_api = tpu_embedding_v2.TPUEmbedding(
            feature_config=feature_configs, optimizer=optimizer)
      mid_level_api._output_shapes = [TensorShape(128)] * len(feature_configs)
      return mid_level_api._create_config_proto()

    self.assertProtoEquals(tpu_embedding_config(), tpu_embedding_config())


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
