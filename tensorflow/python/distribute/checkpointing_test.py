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

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training.tracking import util as trackable_utils


class TrainingCheckpointTests(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testInitializeFromCheckpoint(self, distribution):
    variable_shape = [5]
    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(
        array_ops.ones(variable_shape)))
    save_path = save_checkpoint.save(
        os.path.join(self.get_temp_dir(), "checkpoint"))
    with distribution.scope():
      restore_checkpoint = trackable_utils.Checkpoint()
      restore_checkpoint.restore(save_path)
      initial_value = restore_checkpoint._preload_simple_restoration(
          "v")
      v = variables_lib.Variable(initial_value)
      # Check that the variable is now tagged as restored. `Checkpoint` then
      # knows it doesn't have to restore `v`'s value when it's assigned to an
      # object.
      self.assertGreater(v._update_uid, 0)
      self.assertAllClose(array_ops.ones(variable_shape), v)
      v.assign(array_ops.zeros(variable_shape))
      # Assignment to an object should not trigger restoration, since we already
      # restored the object through an initializer. This wouldn't be a
      # correctness issue, but it would mean that models would use twice as much
      # memory when loading (the buffer already assigned to the variable, and
      # the new restoration).
      restore_checkpoint.v = v
      self.assertAllClose(array_ops.zeros(variable_shape), v)


if __name__ == "__main__":
  test.main()
