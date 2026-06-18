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
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_base_test


class TPUEmbeddingCorrectnessBaseTest(
    tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def _test_embedding(self, optimizer_name, training, sparse,
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

    if is_high_dimensional:
      if sparse:
        mid_level_api.build([
            TensorShape([self.batch_size, self.data_batch_size, 2]),
            TensorShape([self.batch_size, self.data_batch_size, 2]),
            TensorShape([self.batch_size, self.data_batch_size, 3]),
        ])
      else:
        mid_level_api.build([
            TensorShape([self.batch_size, self.data_batch_size, None]),
            TensorShape([self.batch_size, self.data_batch_size, None]),
            TensorShape([self.batch_size, self.data_batch_size, None]),
        ])

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
          total_loss = self._get_total_loss_tensor(activations)
          ret_val = [total_loss] + list(activations)
          return ret_val
        else:
          with backprop.GradientTape() as tape:
            activations = mid_level_api.dequeue()
            tape.watch(activations)
            total_loss = self._get_total_loss_tensor(activations)
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
    if is_high_dimensional:
      input_data = next(
          iter(self._create_high_dimensional_sparse_dataset(strategy)))
    else:
      input_data = next(iter(self._create_sparse_dataset(strategy)))

    # Check results.
    self._check_results(strategy, shard_out_val, training, input_data,
                        mid_level_api._variables, optimizer,
                        is_high_dimensional)

if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
