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
"""Demonstrate Keras preprocessing layers applied inside a Model."""
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
    # TODO(b/183044870) TPU strategies with soft placement do not yet work.
    # ds_combinations.tpu_strategy,
    # ds_combinations.cloud_tpu_strategy,
    ds_combinations.parameter_server_strategy_3worker_2ps_cpu,
    ds_combinations.parameter_server_strategy_3worker_2ps_1gpu,
    ds_combinations.multi_worker_mirrored_2x1_cpu,
    ds_combinations.multi_worker_mirrored_2x2_gpu,
    ds_combinations.central_storage_strategy_with_two_gpus,
]


@ds_combinations.generate(
    test_combinations.combine(strategy=STRATEGIES, mode="eager"))
class PreprocessingAppliedInModelTest(tf.test.TestCase):
  """Demonstrate Keras preprocessing layers applied inside a Model."""

  def testDistributedModelFit(self, strategy):
    with strategy.scope():
      preprocessing_model = utils.make_preprocessing_model(self.get_temp_dir())
      training_model = utils.make_training_model()
      # Merge the two separate models into a single model for training.
      inputs = preprocessing_model.inputs
      outputs = training_model(preprocessing_model(inputs))
      merged_model = tf.keras.Model(inputs, outputs)
      merged_model.compile(optimizer="sgd", loss="binary_crossentropy")

    def dataset_fn(input_context):
      dataset = utils.make_dataset()
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      batch_size = input_context.get_per_replica_batch_size(
          global_batch_size=utils.BATCH_SIZE)
      return dataset.batch(batch_size).repeat().prefetch(2)

    dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
    merged_model.fit(dataset_creator, epochs=2, steps_per_epoch=utils.STEPS)


if __name__ == "__main__":
  multi_process_runner.test_main()
