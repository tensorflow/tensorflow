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
"""Demonstrate Keras preprocessing layers applied in tf.data.Dataset.map."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.integration_test import preprocessing_test_utils as utils

ds_combinations = tf.__internal__.distribute.combinations
multi_process_runner = tf.__internal__.distribute.multi_process_runner
test_combinations = tf.__internal__.test.combinations

# Note: Strategy combinations are not (yet) public APIs, so they are subject
# to API changes and backward-compatibility is not guaranteed.
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
    test_combinations.combine(strategy=STRATEGIES, mode="eager"))
class PreprocessingAppliedInDatasetCreatorTest(tf.test.TestCase):
  """Demonstrate Keras preprocessing layers applied in tf.data.Dataset.map."""

  def testDistributedModelFit(self, strategy):
    with strategy.scope():
      preprocessing_model = utils.make_preprocessing_model(self.get_temp_dir())
      training_model = utils.make_training_model()
      training_model.compile(optimizer="sgd", loss="binary_crossentropy")

    def dataset_fn(input_context):
      dataset = utils.make_dataset()
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      batch_size = input_context.get_per_replica_batch_size(
          global_batch_size=utils.BATCH_SIZE)
      dataset = dataset.batch(batch_size).repeat().prefetch(2)
      return dataset.map(lambda x, y: (preprocessing_model(x), y))

    dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
    training_model.fit(dataset_creator, epochs=2, steps_per_epoch=utils.STEPS)


if __name__ == "__main__":
  multi_process_runner.test_main()
