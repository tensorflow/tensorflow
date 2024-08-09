# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os

import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test


class TrainingCheckpointTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(
        combinations.combine(
            distributions=[
                strategy_combinations.mirrored_strategy_with_one_cpu,
                strategy_combinations.mirrored_strategry_with_gpu_and_cpu,
                strategy_combinations.tpu_strategy,
                strategy_combinations.tpu_strategy_packed_var,
                strategy_combinations.central_storage_strategy_with_two_gpus,
            ],
            mode=["eager"]
        )
    )

    def test_initialize_from_checkpoint(
        self, distribution: tf.distribute.Strategy
    ):
        """
        Tests that a variable can be restored from a checkpoint.
        
        Args:
        distribution: A `tf.distribute.Strategy` to use for the test.
        """

        variable_shape = tf.TensorShape([5])
        save_checkpoint = trackable_utils.Checkpoint(
            v=tf.Variable(tf.constant(1, shape=variable_shape), name="v")
        )
        save_path = save_checkpoint.save(
            os.path.join(self.get_temp_dir(), "checkpoint")
        )

        with distribution.scope():
            restore_checkpoint = trackable_utils.Checkpoint()
            restore_checkpoint.restore(save_path)
            initial_value = restore_checkpoint._preload_simple_restoration("v")
            v = tf.Variable(initial_value)

            # Check that the variable is now tagged as restored. `Checkpoint` then
            # knows it doesn't have to restore `v`'s value when it's assigned to an 
            # object
            self.assertGreater(v.update_uid, 0)


if __name__ == "__main__":
    test.main()