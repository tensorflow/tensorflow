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
"""Tests for object-based saving which use tf.train.* optimizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import adam
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils


class CheckpointingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDeferredSlotRestoration(self):
    checkpoint_directory = self.get_temp_dir()

    root = trackable_utils.Checkpoint()
    root.var = trackable_utils.add_variable(
        root, name="var", initializer=0.)
    optimizer = adam.AdamOptimizer(0.1)
    if context.executing_eagerly():
      optimizer.minimize(root.var.read_value)
    else:
      train_op = optimizer.minimize(root.var)
      # Note that `optimizer` has not been added as a dependency of
      # `root`. Create a one-off grouping so that slot variables for `root.var`
      # get initialized too.
      self.evaluate(trackable_utils.gather_initializers(
          trackable_utils.Checkpoint(root=root, optimizer=optimizer)))
      self.evaluate(train_op)
    self.evaluate(state_ops.assign(root.var, 12.))
    no_slots_path = root.save(os.path.join(checkpoint_directory, "no_slots"))
    root.optimizer = optimizer
    self.evaluate(state_ops.assign(root.var, 13.))
    self.evaluate(state_ops.assign(optimizer.get_slot(name="m", var=root.var),
                                   14.))
    slots_path = root.save(os.path.join(checkpoint_directory, "with_slots"))
    new_root = trackable_utils.Checkpoint()
    # Load the slot-containing checkpoint (deferred), then immediately overwrite
    # the non-slot variable (also deferred).
    slot_status = new_root.restore(slots_path)
    no_slot_status = new_root.restore(no_slots_path)
    with self.assertRaises(AssertionError):
      no_slot_status.assert_consumed()
    new_root.var = trackable_utils.add_variable(
        new_root, name="var", shape=[])
    no_slot_status.assert_consumed()
    no_slot_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(new_root.var))
    new_root.optimizer = adam.AdamOptimizer(0.1)
    slot_status.assert_existing_objects_matched()
    with self.assertRaisesRegex(AssertionError, "beta1_power"):
      slot_status.assert_consumed()
    self.assertEqual(12., self.evaluate(new_root.var))
    if context.executing_eagerly():
      # Slot variables are only created with restoring initializers when
      # executing eagerly.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
    else:
      self.assertIs(new_root.optimizer.get_slot(name="m", var=new_root.var),
                    None)
    if context.executing_eagerly():
      new_root.optimizer.minimize(new_root.var.read_value)
    else:
      train_op = new_root.optimizer.minimize(new_root.var)
      # The slot variable now exists; restore() didn't create it, but we should
      # now have a restore op for it.
      slot_status.run_restore_ops()
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
      self.evaluate(train_op)
    slot_status.assert_consumed()

  def testManySavesGraph(self):
    """Saves after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(trackable_utils.gather_initializers(obj))
        obj.save(checkpoint_prefix)
        before_ops = graph.get_operations()
        obj.save(checkpoint_prefix)
        self.assertEqual(before_ops, graph.get_operations())

  def testManyRestoresGraph(self):
    """Restores after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(trackable_utils.gather_initializers(obj))
        save_path = obj.save(checkpoint_prefix)
        obj.restore(save_path)
        before_ops = graph.get_operations()
        obj.restore(save_path)
        self.assertEqual(before_ops, graph.get_operations())

  def testMultipleGraphsNonSlotVariables(self):
    with context.graph_mode():
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      optimizer = adam.AdamOptimizer(0.001)
      # Construct a model in one graph
      first_graph = ops.Graph()
      first_session = session_lib.Session(graph=first_graph)
      with first_graph.as_default(), first_session.as_default():
        first_variable = resource_variable_ops.ResourceVariable([1.])
        first_root_trackable = trackable_utils.Checkpoint(
            optimizer=optimizer, variable=first_variable)
        train_op = optimizer.minimize(first_variable.read_value)
        self.evaluate(trackable_utils.gather_initializers(
            first_root_trackable))
        self.evaluate(train_op)
        self.evaluate(first_variable.assign([1.]))
        self.evaluate(optimizer.get_slot(
            var=first_variable, name="m").assign([2.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(3.))

      # Save and load in a second graph
      second_graph = ops.Graph()
      with second_graph.as_default(), session_lib.Session(graph=second_graph):
        second_variable = resource_variable_ops.ResourceVariable([1.])
        second_root_trackable = trackable_utils.Checkpoint(
            optimizer=optimizer, variable=second_variable)
        train_op = optimizer.minimize(second_variable.read_value)
        second_root_trackable.restore(None).initialize_or_restore()
        self.evaluate(train_op)
        self.evaluate(second_variable.assign([4.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([5.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(6.))
        save_path = second_root_trackable.save(checkpoint_prefix)
        self.evaluate(second_variable.assign([7.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([8.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))
        status = second_root_trackable.restore(save_path)
        status.assert_consumed().run_restore_ops()
        self.assertAllEqual([4.], self.evaluate(second_variable))
        self.assertAllEqual([5.], self.evaluate(optimizer.get_slot(
            var=second_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))

      # Check that the first graph is unmolested
      with first_graph.as_default(), first_session.as_default():
        self.assertAllEqual([1.], self.evaluate(first_variable))
        self.assertAllEqual([2.], self.evaluate(optimizer.get_slot(
            var=first_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(3., self.evaluate(beta1_power))


class _ManualScope(tracking.AutoTrackable):

  def __call__(self):
    with variable_scope.variable_scope("ManualScope") as vs:
      self.variable_scope = vs
      with trackable_utils.capture_dependencies(template=self):
        return self._build()

  def _build(self):
    return variable_scope.get_variable(name="in_manual_scope", shape=[])


class TemplateTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_trackable_save_restore(self):

    def _templated():
      v = variable_scope.get_variable(
          "v", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      v2 = variable_scope.get_variable(
          "v2", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      manual = _ManualScope()
      return v, v + 1., v2, manual, manual()

    save_template = template.make_template("s1", _templated)
    v1_save, _, v2_save, manual_scope, manual_scope_v = save_template()
    six.assertCountEqual(self, [
        id(obj) for obj in
        [v1_save, v2_save, manual_scope, manual_scope_v, save_template]
    ], [id(obj) for obj in trackable_utils.list_objects(save_template)])
    manual_dep, = manual_scope._checkpoint_dependencies
    self.assertEqual("in_manual_scope", manual_dep.name)
    self.assertIs(manual_scope_v, manual_dep.ref)
    optimizer = adam.AdamOptimizer(0.0)
    save_root = trackable_utils.Checkpoint(
        my_template=save_template, optimizer=optimizer)
    optimizer.minimize(v1_save.read_value)
    self.evaluate([v.initializer for v in save_template.variables])
    self.evaluate([v.initializer for v in optimizer.variables()])
    self.evaluate(v1_save.assign([12.]))
    self.evaluate(v2_save.assign([14.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_root.save(checkpoint_prefix)

    load_template = template.make_template("s2", _templated)
    load_optimizer = adam.AdamOptimizer(0.0)
    load_root = trackable_utils.Checkpoint(
        my_template=load_template, optimizer=load_optimizer)
    status = load_root.restore(save_path)
    var, var_plus_one, var2, _, _ = load_template()
    load_optimizer.minimize(var.read_value)
    self.assertEqual(3, len(load_template._checkpoint_dependencies))
    self.assertEqual("v", load_template._checkpoint_dependencies[0].name)
    self.assertEqual("v2", load_template._checkpoint_dependencies[1].name)
    self.assertEqual("ManualScope",
                     load_template._checkpoint_dependencies[2].name)
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual([12.], self.evaluate(var))
    self.assertAllEqual([13.], self.evaluate(var_plus_one))
    self.assertAllEqual([14.], self.evaluate(var2))


if __name__ == "__main__":
  test.main()
