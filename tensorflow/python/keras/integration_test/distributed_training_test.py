# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test to demonstrate basic Keras training with a variety of strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
ds_combinations = tf.__internal__.distribute.combinations

# Note: Strategy combinations are not (yet) public APIs, so they are subject
# to API changes and backward-compatibility is not guaranteed.
# TODO(b/188763034): Proceed to export the strategy combinations as public APIs.
STRATEGIES = [
    ds_combinations.default_strategy,
    ds_combinations.mirrored_strategy_with_cpu_1_and_2,
    ds_combinations.mirrored_strategy_with_two_gpus,
    ds_combinations.tpu_strategy,
    ds_combinations.cloud_tpu_strategy,
    ds_combinations.parameter_server_strategy_3worker_2ps_cpu,
    ds_combinations.parameter_server_strategy_3worker_2ps_1gpu,
    ds_combinations.multi_worker_mirrored_2x1_cpu,
    ds_combinations.multi_worker_mirrored_2x2_gpu,
    ds_combinations.central_storage_strategy_with_two_gpus,
]


@ds_combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=STRATEGIES, mode="eager"))
class DistributedTrainingTest(tf.test.TestCase):
  """Test to demonstrate basic Keras training with a variety of strategies."""

  def testKerasTrainingAPI(self, strategy):

    # A `dataset_fn` is required for `Model.fit` to work across all strategies.
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(
          global_batch_size=64)
      x = tf.random.uniform((10, 10))
      y = tf.random.uniform((10,))
      dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
      dataset = dataset.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id)
      return dataset.batch(batch_size).prefetch(2)

    with strategy.scope():
      model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
      optimizer = tf.keras.optimizers.SGD()
      model.compile(optimizer, loss="mse", steps_per_execution=10)

    x = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

    model.fit(x, epochs=2, steps_per_epoch=10)


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
