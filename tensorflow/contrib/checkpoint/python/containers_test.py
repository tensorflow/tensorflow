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

import os

import six

from tensorflow.contrib.checkpoint.python import containers
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util


class UniqueNameTrackerTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testNames(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    x1 = resource_variable_ops.ResourceVariable(2.)
    x2 = resource_variable_ops.ResourceVariable(3.)
    x3 = resource_variable_ops.ResourceVariable(4.)
    y = resource_variable_ops.ResourceVariable(5.)
    slots = containers.UniqueNameTracker()
    slots.track(x1, "x")
    slots.track(x2, "x")
    slots.track(x3, "x_1")
    slots.track(y, "y")
    self.evaluate((x1.initializer, x2.initializer, x3.initializer,
                   y.initializer))
    save_root = util.Checkpoint(slots=slots)
    save_path = save_root.save(checkpoint_prefix)

    restore_slots = tracking.AutoTrackable()
    restore_root = util.Checkpoint(
        slots=restore_slots)
    status = restore_root.restore(save_path)
    restore_slots.x = resource_variable_ops.ResourceVariable(0.)
    restore_slots.x_1 = resource_variable_ops.ResourceVariable(0.)
    restore_slots.x_1_1 = resource_variable_ops.ResourceVariable(0.)
    restore_slots.y = resource_variable_ops.ResourceVariable(0.)
    status.assert_consumed().run_restore_ops()
    self.assertEqual(2., self.evaluate(restore_slots.x))
    self.assertEqual(3., self.evaluate(restore_slots.x_1))
    self.assertEqual(4., self.evaluate(restore_slots.x_1_1))
    self.assertEqual(5., self.evaluate(restore_slots.y))

  @test_util.run_in_graph_and_eager_modes
  def testExample(self):
    class SlotManager(tracking.AutoTrackable):

      def __init__(self):
        self.slotdeps = containers.UniqueNameTracker()
        slotdeps = self.slotdeps
        slots = []
        slots.append(slotdeps.track(
            resource_variable_ops.ResourceVariable(3.), "x"))
        slots.append(slotdeps.track(
            resource_variable_ops.ResourceVariable(4.), "y"))
        slots.append(slotdeps.track(
            resource_variable_ops.ResourceVariable(5.), "x"))
        self.slots = data_structures.NoDependency(slots)

    manager = SlotManager()
    self.evaluate([v.initializer for v in manager.slots])
    checkpoint = util.Checkpoint(slot_manager=manager)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)
    metadata = util.object_metadata(save_path)
    dependency_names = []
    for node in metadata.nodes:
      for child in node.children:
        dependency_names.append(child.local_name)
    six.assertCountEqual(
        self,
        dependency_names,
        ["x", "x_1", "y", "slot_manager", "slotdeps", "save_counter"])

  @test_util.run_in_graph_and_eager_modes
  def testLayers(self):
    tracker = containers.UniqueNameTracker()
    tracker.track(layers.Dense(3), "dense")
    tracker.layers[0](array_ops.zeros([1, 1]))
    self.assertEqual(2, len(tracker.trainable_weights))

if __name__ == "__main__":
  test.main()
