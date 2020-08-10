# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import util as trackable_utils


class NonLayerTrackable(module.Module):

  def __init__(self):
    super(NonLayerTrackable, self).__init__()
    self.a_variable = trackable_utils.add_variable(
        self, name="a_variable", shape=[])


class MyModel(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(MyModel, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Trackables which aren't Layers.
    self._non_layer = NonLayerTrackable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class TrackableCompatibilityTests(test.TestCase):

  def _initialized_model(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    optimizer = adam.AdamOptimizer(0.001)
    optimizer_step = training_util.get_or_create_global_step()
    root_trackable = trackable_utils.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=optimizer_step)
    train_op = optimizer.minimize(
        functools.partial(model, input_value),
        global_step=optimizer_step)
    self.evaluate(trackable_utils.gather_initializers(
        root_trackable))
    self.evaluate(train_op)
    # A regular variable, a slot variable, and a non-slot Optimizer variable
    # with known values to check when loading.
    self.evaluate(model._named_dense.bias.assign([1.]))
    self.evaluate(optimizer.get_slot(
        var=model._named_dense.bias, name="m").assign([2.]))
    beta1_power, _ = optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(3.))
    return root_trackable

  def _set_sentinels(self, root_trackable):
    self.evaluate(root_trackable.model._named_dense.bias.assign([101.]))
    self.evaluate(
        root_trackable.optimizer.get_slot(
            var=root_trackable.model._named_dense.bias, name="m")
        .assign([102.]))
    beta1_power, _ = root_trackable.optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(103.))

  def _check_sentinels(self, root_trackable):
    self.assertAllEqual(
        [1.], self.evaluate(root_trackable.model._named_dense.bias))
    self.assertAllEqual([2.], self.evaluate(
        root_trackable.optimizer.get_slot(
            var=root_trackable.model._named_dense.bias, name="m")))
    beta1_power, _ = root_trackable.optimizer._get_beta_accumulators()
    self.assertAllEqual(3., self.evaluate(beta1_power))

  def testLoadFromObjectBasedGraph(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    save_graph = ops_lib.Graph()
    with save_graph.as_default(), self.session(graph=save_graph) as sess:
      root = self._initialized_model()
      object_saver = trackable_utils.Checkpoint(root=root)
      save_path = object_saver.save(file_prefix=checkpoint_prefix)

      # An incompatible object-based checkpoint to check error messages
      var = variables.Variable(1., name="a")
      self.evaluate(var.initializer)
      second_saver = trackable_utils.Checkpoint(v=var)
      second_path = second_saver.save(file_prefix=os.path.join(
          checkpoint_directory, "second"))

    restore_graph = ops_lib.Graph()
    with restore_graph.as_default(), self.session(
        graph=restore_graph) as sess:
      root = self._initialized_model()
      self._set_sentinels(root)
      saver = saver_module.Saver()
      saver.restore(sess=sess, save_path=save_path)
      self._check_sentinels(root)
      before_second_restore_ops = restore_graph.get_operations()
      # Test that multiple restores do not pollute the graph
      saver.restore(sess=sess, save_path=save_path)
      self.assertEqual(before_second_restore_ops,
                       restore_graph.get_operations())
      with self.assertRaisesRegex(errors.NotFoundError,
                                  "Could not find some variables"):
        saver.restore(sess=sess, save_path=second_path)

  def testLoadFromObjectBasedEager(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    save_graph = ops_lib.Graph()
    with save_graph.as_default(), self.session(graph=save_graph):
      root = self._initialized_model()
      object_saver = trackable_utils.Checkpoint(root=root)
      save_path = object_saver.save(file_prefix=checkpoint_prefix)

    with context.eager_mode():
      root = self._initialized_model()
      self._set_sentinels(root)
      saver = saver_module.Saver(
          root.model.variables + root.optimizer.variables())
      saver.restore(sess=None, save_path=save_path)
      self._check_sentinels(root)


if __name__ == "__main__":
  test.main()
