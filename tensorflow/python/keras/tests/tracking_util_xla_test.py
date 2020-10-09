# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.module import module
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_utils


class NonLayerTrackable(module.Module):

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


class CheckpointingTests(xla_test.XLATestCase):

  def testDeferredRestorationUsageEager(self):
    """An idiomatic eager execution example."""
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    for training_continuation in range(3):
      with self.test_scope():
        model = Subclassed()
        optimizer = adam.Adam(0.001)
        root = trackable_utils.Checkpoint(
            optimizer=optimizer, model=model)
        manager = checkpoint_management.CheckpointManager(
            root, checkpoint_directory, max_to_keep=2)
        root.restore(manager.latest_checkpoint)
        for _ in range(num_training_steps):
          input_value = constant_op.constant([[3.]])
          with backprop.GradientTape() as tape:
            loss = model(input_value)
          variables = model.trainable_variables
          gradients = tape.gradient(loss, variables)
          optimizer.apply_gradients(zip(gradients, variables))
        manager.save()
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         root.optimizer.iterations.numpy())


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
