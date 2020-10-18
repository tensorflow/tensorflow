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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.tracking import util as trackable_utils


class TrainingCheckpointTests(test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testCheckpointRestoreOptimizerSlots(self, distribution):
    def state():
      with distribution.scope():
        v = variables_lib.Variable(random_ops.random_normal([]))
      opt = adam.Adam(0.001)

      @def_function.function
      def step():
        def f():
          with backprop.GradientTape() as tape:
            loss = v + v
          gradients = tape.gradient(loss, [v])
          opt.apply_gradients(zip(gradients, [v]))

        distribution.run(f)

      return v, opt, step

    def checkpoint():
      v, opt, step = state()
      step()

      # Save random weights into checkpoint.
      checkpoint = trackable_utils.Checkpoint(v=v, opt=opt)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      with self.test_session():
        save_path = checkpoint.save(prefix)
      return save_path

    save_path = checkpoint()

    v, opt, step = state()
    checkpoint = trackable_utils.Checkpoint(v=v, opt=opt)
    # Restore from the checkpoint inside a distribution.scope().
    with self.test_session():
      with distribution.scope():
        checkpoint.restore(save_path)
    step()
    slot = opt.get_slot(v, "m")
    self.assertEqual(v._distribute_strategy, slot._distribute_strategy)

    v, opt, step = state()
    checkpoint = trackable_utils.Checkpoint(v=v, opt=opt)
    # Restore from the checkpoint outside a distribution.scope().
    with self.test_session():
      with self.assertRaisesRegex(
          ValueError, "optimizer slot variable under the scope"):
        checkpoint.restore(save_path)

  @ds_combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          mode=["eager"]))
  def testCheckpointSaveRestoreIoDevice(self, distribution):

    def state():
      with distribution.scope():
        v = variables_lib.Variable(random_ops.random_normal([]))
        return v

    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_io_device="/job:localhost")

    def checkpoint():
      v = state()
      # Save random weights into checkpoint.
      checkpoint = trackable_utils.Checkpoint(v=v)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      with self.test_session():
        save_path = checkpoint.save(prefix, options=ckpt_options)
      return save_path

    save_path = checkpoint()

    v = state()
    checkpoint = trackable_utils.Checkpoint(v=v)
    # Restore from the checkpoint inside a distribution.scope().
    # Check that restore works without error.
    with self.test_session():
      with distribution.scope():
        checkpoint.restore(save_path, options=ckpt_options)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
