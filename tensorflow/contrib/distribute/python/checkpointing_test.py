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

from tensorflow.compiler.tests import xla_test
from tensorflow.contrib.distribute.python import tpu_strategy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.platform import test
from tensorflow.python.training import adam as adam_v1
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.checkpointable import util as checkpointable_utils


class NonLayerCheckpointable(tracking.Checkpointable):

  def __init__(self):
    super(NonLayerCheckpointable, self).__init__()
    self.a_variable = checkpointable_utils.add_variable(
        self, name="a_variable", shape=[])


class Subclassed(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(Subclassed, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Checkpointables which aren't Layers.
    self._non_layer = NonLayerCheckpointable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class TrainingCheckpointTests(xla_test.XLATestCase):

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
        root = checkpointable_utils.Checkpoint(
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


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
