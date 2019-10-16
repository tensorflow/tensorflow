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

import functools
import os

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training import adam as adam_v1
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils


class NonLayerTrackable(tracking.AutoTrackable):

  def __init__(self):
    super(NonLayerTrackable, self).__init__()
    self.a_variable = trackable_utils.add_variable(
        self, name="a_variable", shape=[])


class Subclassed(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(Subclassed, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Trackables which aren't Layers.
    self._non_layer = NonLayerTrackable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class TrainingCheckpointTests(test.TestCase, parameterized.TestCase):

  def testEagerTPUDistributionStrategy(self):
    self.skipTest("b/121387144")
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    def _train_fn(optimizer, model):
      input_value = constant_op.constant([[3.]])
      optimizer.minimize(
          functools.partial(model, input_value),
          global_step=root.optimizer_step)

    for training_continuation in range(3):
      strategy = tpu_strategy.TPUStrategy()
      with strategy.scope():
        model = Subclassed()
        optimizer = adam_v1.AdamOptimizer(0.001)
        root = trackable_utils.Checkpoint(
            optimizer=optimizer, model=model,
            optimizer_step=training_util.get_or_create_global_step())
        root.restore(checkpoint_management.latest_checkpoint(
            checkpoint_directory))

        for _ in range(num_training_steps):
          strategy.extended.call_for_each_replica(
              functools.partial(_train_fn, optimizer, model))
        root.save(file_prefix=checkpoint_prefix)
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         root.optimizer_step.numpy())

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
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
        distribution.experimental_run_v2(f)

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


if __name__ == "__main__":
  test.main()
