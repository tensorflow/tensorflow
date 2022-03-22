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

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def test_build_incorrect_output_shapes(self):
    _, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    # Output shapes is set in the mid_level_api, but build with incorrect output
    # shapes.
    mid_level_api._output_shapes = [TensorShape((2, 4)) for _ in range(3)]

    with self.assertRaisesRegex(ValueError,
                                'Inconsistent shape founded for input feature'):
      mid_level_api.build([TensorShape([1, 1, 1]) for _ in range(3)])

  def test_enqueue_incorrect_shape_feature(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_high_dimensional_sparse_dataset(strategy)
    sparse_iter = iter(
        strategy.experimental_distribute_dataset(
            sparse,
            options=distribute_lib.InputOptions(
                experimental_fetch_to_device=False)))

    mid_level_api._output_shapes = [TensorShape((1, 1)) for _ in range(3)]
    # The output shape passed to build method is consistent.
    mid_level_api.build([TensorShape([1, 1, 1]) for _ in range(3)])

    @def_function.function
    def test_fn():

      def step():
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(sparse_iter), training=False)
      return strategy.run(step)

    # Enqueued tensor has shape inconsistent with the output shape setting.
    with self.assertRaisesRegex(ValueError,
                                'Inconsistent shape founded for input feature'):
      test_fn()

  def test_not_fully_defined_output_shapes_in_feature_config(self):
    _, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    # Feature config sets undefined output shapes
    mid_level_api._output_shapes = [TensorShape(None) for _ in range(3)]
    with self.assertRaisesRegex(ValueError, 'Input Feature'):
      mid_level_api.build()

  def test_not_fully_defined_output_shapes_for_build(self):
    _, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    # Build with undefined output shape
    with self.assertRaisesRegex(ValueError, 'Input Feature'):
      mid_level_api.build([TensorShape([1, None, None]) for _ in range(3)])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
