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
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def test_tables_with_same_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Multiple tables with name table found.'):
      with self._get_strategy().scope():
        tpu_embedding_v2.TPUEmbedding(
            (tpu_embedding_v2_utils.FeatureConfig(
                table=tpu_embedding_v2_utils.TableConfig(
                    name='table',
                    vocabulary_size=4,
                    dim=2,
                    initializer=self.initializer,),
                name='watched'),
             tpu_embedding_v2_utils.FeatureConfig(
                 table=tpu_embedding_v2_utils.TableConfig(
                     name='table',
                     vocabulary_size=4,
                     dim=2,
                     initializer=self.initializer),
                 name='favorited')),
            tpu_embedding_v2_utils.SGD(learning_rate=0.1))

  def test_pass_non_tensor_to_apply_gradients(self):
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    # We aren't going to actually run anything, so the batch_size here does not
    # matter.
    mid_level_api.build(64)

    # Test pass non tensor to apply_gradients.
    @def_function.function
    def test_apply_1():
      mid_level_api.apply_gradients((1, 2, 3))

    with self.assertRaisesRegex(ValueError, 'found non-tensor type'):
      strategy.run(test_apply_1)

    # Test pass different structure to apply_gradients.
    @def_function.function
    def test_apply_2():
      # This should be a tuple as feature_config is a tuple of 3 configs.
      mid_level_api.apply_gradients([1, 2, 3])

    with self.assertRaisesRegex(
        TypeError, 'The two structures don\'t have the same nested structure.'):
      strategy.run(test_apply_2)

  def test_enqueue_weight_for_dense_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy, include_weights=True)
    dense_iter = iter(
        strategy.experimental_distribute_dataset(
            dataset,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      features, weights = next(dense_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(ValueError, 'Weight specified for dense input'):
      test_fn()

  def test_enqueue_wrong_weight_type_for_sparse_and_ragged_tensor(self):
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    ragged = self._create_ragged_dataset(strategy, include_weights=True)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))
    ragged_iter = iter(
        strategy.experimental_distribute_dataset(
            ragged,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_sparse_fn():
      def step():
        return mid_level_api.dequeue()

      features, _ = next(sparse_iter)
      _, weights = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is SparseTensor.'):
      test_sparse_fn()

    @def_function.function
    def test_ragged_fn():
      def step():
        return mid_level_api.dequeue()

      _, weights = next(sparse_iter)
      features, _ = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is RaggedTensor.'):
      test_ragged_fn()

  def test_enqueue_incorrect_structure_for_features_and_weights(self):
    self.skip_if_oss()
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    @def_function.function
    def test_features_fn():
      def step():
        return mid_level_api.dequeue()

      features = next(sparse_iter)
      features = (features[0],)
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_features_fn()

    @def_function.function
    def test_weights_fn():
      def step():
        return mid_level_api.dequeue()

      features, weights = next(sparse_iter)
      weights = (weights[0],)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_weights_fn()

  def test_enqueue_cpu_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy)
    dense_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def test_fn():
      def get_activations():
        return mid_level_api.dequeue()

      features = next(dense_iter)
      mid_level_api.enqueue(features, training=False)
      activations = strategy.run(get_activations)
      return activations

    with self.assertRaisesRegex(ValueError, 'which is on a TPU input device'):
      test_fn()

  @parameterized.parameters([True, False])
  def test_enqueue_cpu_tensor_with_outside_compilation(self, use_mlir):

    if use_mlir:
      config.enable_mlir_bridge()

    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    dataset = self._create_dense_dataset(strategy)
    dense_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def test_fn():
      def get_activations(features):
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()

      activations = strategy.run(get_activations, args=(next(dense_iter),))
      return activations

    with self.assertRaisesRegex(ValueError, 'which is on a TPU input device'):
      test_fn()

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
