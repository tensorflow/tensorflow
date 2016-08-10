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
# ==============================================================================

"""Tests for variable store."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope


class VariableScopeTest(tf.test.TestCase):

  def testGetVar(self):
    vs = variable_scope._get_default_variable_store()
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    assert v == v1

  def testNameExists(self):
    vs = variable_scope._get_default_variable_store()
    # No check by default, so we can both create and get existing names.
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    assert v == v1
    # When reuse is False, we fail when variables are already there.
    vs.get_variable("w", [1], reuse=False)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("v", [1], reuse=False)  # That fails.
    # When reuse is True, we fail when variables are new.
    vs.get_variable("v", [1], reuse=True)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("u", [1], reuse=True)  # That fails.

  def testNamelessStore(self):
    vs = variable_scope._get_default_variable_store()
    vs.get_variable("v1", [2])
    vs.get_variable("v2", [2])
    expected_names = ["%s:0" % name for name in ["v1", "v2"]]
    self.assertEqual(set(expected_names),
                     set([v.name for v in vs._vars.values()]))

  def testVarScopeInitializer(self):
    with self.test_session() as sess:
      init = tf.constant_initializer(0.3)
      with tf.variable_scope("tower") as tower:
        with tf.variable_scope("foo", initializer=init):
          v = tf.get_variable("v", [])
          sess.run(tf.initialize_variables([v]))
          self.assertAllClose(v.eval(), 0.3)
        with tf.variable_scope(tower, initializer=init):
          w = tf.get_variable("w", [])
          sess.run(tf.initialize_variables([w]))
          self.assertAllClose(w.eval(), 0.3)

  def testVarScopeDType(self):
    with self.test_session():
      with tf.variable_scope("tower") as tower:
        with tf.variable_scope("foo", dtype=tf.float16):
          v = tf.get_variable("v", [])
          self.assertEqual(v.dtype, tf.float16_ref)
        with tf.variable_scope(tower, dtype=tf.float16):
          w = tf.get_variable("w", [])
          self.assertEqual(w.dtype, tf.float16_ref)

  def testInitFromNonTensorValue(self):
    with self.test_session() as sess:
      v = tf.get_variable("v", initializer=4, dtype=tf.int32)
      sess.run(tf.initialize_variables([v]))
      self.assertAllClose(v.eval(), 4)

      w = tf.get_variable("w",
                          initializer=numpy.array([1, 2, 3]),
                          dtype=tf.int32)
      sess.run(tf.initialize_variables([w]))
      self.assertAllClose(w.eval(), [1, 2, 3])

      with self.assertRaises(TypeError):
        tf.get_variable("x", initializer={})

  def testVarScopeCachingDevice(self):
    with self.test_session():
      caching_device = "/job:moo"
      with tf.variable_scope("tower"):
        with tf.variable_scope("caching", caching_device=caching_device):
          v = tf.get_variable("v", [])
          self.assertTrue(v.value().device.startswith(caching_device))

          with tf.variable_scope("child"):
            v2 = tf.get_variable("v", [])
            self.assertTrue(v2.value().device.startswith(caching_device))

          with tf.variable_scope("not_cached", caching_device=""):
            v2_not_cached = tf.get_variable("v", [])
            self.assertFalse(
                v2_not_cached.value().device.startswith(caching_device))

          with tf.variable_scope(
              "not_cached_identity_device",
              caching_device=lambda op: op.device):
            v2_identity_device = tf.get_variable("v", [])
            self.assertFalse(
                v2_identity_device.value().device.startswith(caching_device))

          with tf.variable_scope("we_will_do_it_live") as vs_live:
            vs_live.set_caching_device("/job:live")
            v_live = tf.get_variable("v", [])
            self.assertTrue(v_live.value().device.startswith("/job:live"))

        v_tower = tf.get_variable("v", [])
        self.assertFalse(v_tower.value().device.startswith(caching_device))

  def testVarScopeRegularizer(self):
    with self.test_session() as sess:
      init = tf.constant_initializer(0.3)
      def regularizer1(v):
        return tf.reduce_mean(v) + 0.1
      def regularizer2(v):
        return tf.reduce_mean(v) + 0.2
      with tf.variable_scope("tower", regularizer=regularizer1) as tower:
        with tf.variable_scope("foo", initializer=init):
          v = tf.get_variable("v", [])
          sess.run(tf.initialize_variables([v]))
          losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(1, len(losses))
          self.assertAllClose(losses[0].eval(), 0.4)
        with tf.variable_scope(tower, initializer=init) as vs:
          u = tf.get_variable("u", [])
          vs.set_regularizer(regularizer2)
          w = tf.get_variable("w", [])
          # Next 3 variable not regularized to test disabling regularization.
          x = tf.get_variable("x", [], regularizer=tf.no_regularizer)
          with tf.variable_scope("baz", regularizer=tf.no_regularizer):
            y = tf.get_variable("y", [])
          vs.set_regularizer(tf.no_regularizer)
          z = tf.get_variable("z", [])
          # Check results.
          losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(3, len(losses))
          sess.run(tf.initialize_variables([u, w, x, y, z]))
          self.assertAllClose(losses[0].eval(), 0.4)
          self.assertAllClose(losses[1].eval(), 0.4)
          self.assertAllClose(losses[2].eval(), 0.5)
        with tf.variable_scope("foo", reuse=True):
          v = tf.get_variable("v", [])  # "v" is alredy there, reused
          losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(3, len(losses))  # No new loss added.

  def testIntializeFromValue(self):
    with self.test_session() as sess:
      init = tf.constant(0.1)
      w = tf.get_variable("v", initializer=init)
      sess.run(tf.initialize_variables([w]))
      self.assertAllClose(w.eval(), 0.1)

      with self.assertRaisesRegexp(ValueError, "shape"):
        # We disallow explicit shape specification when initializer is constant.
        tf.get_variable("u", [1], initializer=init)

      with tf.variable_scope("foo", initializer=init):
        # Constant initializer can be passed through scopes if needed.
        v = tf.get_variable("v")
        sess.run(tf.initialize_variables([v]))
        self.assertAllClose(v.eval(), 0.1)

  def testControlDeps(self):
    with self.test_session() as sess:
      v0 = tf.get_variable("v0", [1], initializer=tf.constant_initializer(0))
      with tf.control_dependencies([v0.value()]):
        v1 = tf.get_variable("v1", [1], initializer=tf.constant_initializer(1))
        add = v1 + v0
      # v0 should be uninitialized.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(v0)
      # We should be able to initialize and run v1 without initializing
      # v0, even if the variable was created with a control dep on v0.
      sess.run(v1.initializer)
      self.assertEqual(1, sess.run(v1))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(v0)
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      sess.run(v0.initializer)
      sess.run(add)

  def testControlFlow(self):
    with self.test_session() as sess:
      v0 = tf.get_variable("v0", [], initializer=tf.constant_initializer(0))
      var_dict = {}
      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        v1 = tf.get_variable("v1", [1], initializer=tf.constant_initializer(1))
        var_dict["v1"] = v1
        return v1 + v0
      def var_in_else_clause():
        v2 = tf.get_variable("v2", [1], initializer=tf.constant_initializer(2))
        var_dict["v2"] = v2
        return v2 + v0
      add = control_flow_ops.cond(tf.less(v0, 10),
                                  var_in_then_clause,
                                  var_in_else_clause)
      v1 = var_dict["v1"]
      v2 = var_dict["v2"]
      # We should be able to initialize and run v1 and v2 without initializing
      # v0, even if the variable was created with a control dep on v0.
      sess.run(v1.initializer)
      self.assertEqual([1], sess.run(v1))
      sess.run(v2.initializer)
      self.assertEqual([2], sess.run(v2))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegexp(tf.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      sess.run(v0.initializer)
      sess.run(add)

  def testGetVariableScope(self):
    # Test the get_variable_scope() function and setting properties of result.
    with self.test_session() as sess:
      init = tf.constant_initializer(0.3)
      with tf.variable_scope("foo"):
        new_init1 = tf.get_variable_scope().initializer
        self.assertEqual(new_init1, None)
        # Check that we can set initializer like this.
        tf.get_variable_scope().set_initializer(init)
        v = tf.get_variable("v", [])
        sess.run(tf.initialize_variables([v]))
        self.assertAllClose(v.eval(), 0.3)
        # Check that we can set reuse.
        tf.get_variable_scope().reuse_variables()
        with self.assertRaises(ValueError):  # Fail, w does not exist yet.
          tf.get_variable("w", [1])
      # Check that the set initializer goes away.
      new_init = tf.get_variable_scope().initializer
      self.assertEqual(new_init, None)

  def testVarScope(self):
    with self.test_session():
      with tf.variable_scope("tower") as tower:
        self.assertEqual(tower.name, "tower")
        with tf.name_scope("scope") as sc:
          self.assertEqual(sc, "tower/scope/")

      with tf.variable_scope("foo"):
        with tf.variable_scope("bar") as bar:
          self.assertEqual(bar.name, "foo/bar")
          with tf.name_scope("scope") as sc:
            self.assertEqual(sc, "foo/bar/scope/")

      with tf.variable_scope("foo"):
        with tf.variable_scope(tower, reuse=True) as tower_shared:
          self.assertEqual(tower_shared.name, "tower")
          with tf.name_scope("scope") as sc:
            self.assertEqual(sc, "foo_1/tower/scope/")

  def testVarScopeNameScope(self):
    with self.test_session():
      with tf.name_scope("scope1"):
        with tf.variable_scope("tower") as tower:
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with tf.variable_scope(tower):  # Re-entering acts like another "tower".
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")
        with tf.variable_scope("tower"):  # Re-entering by string acts the same.
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_2/scope2/")

      with tf.name_scope("scope3"):
        with tf.variable_scope("tower"):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower/scope2/")
        with tf.variable_scope(tower):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower_1/scope2/")

      root_var_scope = tf.get_variable_scope()
      with tf.name_scope("scope4"):
        with tf.variable_scope(root_var_scope):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope4/scope2/")

  def testVarScopeOriginalNameScope(self):
    with self.test_session():
      with tf.name_scope("scope1"):
        with tf.variable_scope("tower") as tower:
          self.assertEqual(tower.original_name_scope, "scope1/tower/")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
      with tf.name_scope("scope2"):
        with tf.variable_scope(tower) as tower1:
          # Re-entering preserves original name scope.
          self.assertEqual(tower1.original_name_scope, "scope1/tower/")
          with tf.name_scope("foo") as sc2:
            self.assertEqual(sc2, "scope2/tower/foo/")
        # Test re-entering original name scope.
        with tf.name_scope(tower.original_name_scope):
          with tf.name_scope("bar") as sc3:
            self.assertEqual(sc3, "scope1/tower/bar/")
      with tf.name_scope("scope2"):
        with tf.variable_scope(tower):
          with tf.name_scope(tower.original_name_scope):
            with tf.name_scope("bar") as sc3:
              self.assertEqual(sc3, "scope1/tower/bar_1/")

  def testVarScopeObjectReuse(self):
    with self.test_session():
      vs = None
      with tf.variable_scope("jump", reuse=True) as scope:
        vs = scope

      with tf.variable_scope(vs) as jump:
        self.assertTrue(jump.reuse)

      with tf.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with tf.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertFalse(jump_no_reuse.reuse)

      with tf.variable_scope("jump", reuse=False) as scope:
        vs = scope

      with tf.variable_scope(vs) as jump:
        self.assertFalse(jump.reuse)

      with tf.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with tf.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertFalse(jump_no_reuse.reuse)

  def testVarOpScope(self):
    with self.test_session():
      with tf.name_scope("scope1"):
        with tf.variable_op_scope([], "tower", "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with tf.variable_op_scope([], "tower", "default"):
          with self.assertRaises(ValueError):
            tf.get_variable("w", [])
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")

      with tf.name_scope("scope2"):
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "default_1/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default_1/scope2/")

  def testVarOpScopeReuse(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        with tf.variable_op_scope([], "tower", "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with tf.variable_scope(outer, reuse=True) as outer:
        with tf.variable_op_scope([], "tower", "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testVarScopeGetVar(self):
    with self.test_session():
      with tf.variable_scope("root"):
        with tf.variable_scope("towerA") as tower_a:
          va = tf.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with tf.variable_scope(tower_a, reuse=True):
          va2 = tf.get_variable("v", [1])
          self.assertEqual(va2, va)

        with tf.variable_scope("towerB"):
          vb = tf.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with self.assertRaises(ValueError):
          with tf.variable_scope("towerA"):
            va2 = tf.get_variable("v", [1])

        with tf.variable_scope("towerA", reuse=True):
          va2 = tf.get_variable("v", [1])
          self.assertEqual(va2, va)

        with tf.variable_scope("foo"):
          with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with tf.variable_scope(tower_a, reuse=True):
              va3 = tf.get_variable("v", [1])
              self.assertEqual(va, va3)

        with self.assertRaises(ValueError):
          with tf.variable_scope(tower_a, reuse=True):
            with tf.variable_scope("baz"):
              tf.get_variable("v", [1])

        with self.assertRaises(ValueError) as exc:
          with tf.variable_scope(tower_a, reuse=True):
            tf.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with tf.variable_scope(tower_a, reuse=True):
            tf.get_variable("v", [1], dtype=tf.int32)
        self.assertEqual("dtype" in str(exc.exception), True)

  def testVarScopeOuterScope(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        pass
      with tf.variable_scope(outer):
        self.assertEqual(tf.get_variable("w", []).name,
                         "outer/w:0")
        with tf.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with tf.variable_scope("default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with tf.variable_scope(outer, reuse=True):
        self.assertEqual(tf.get_variable("w", []).name,
                         "outer/w:0")
        with tf.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        with tf.variable_scope("default", reuse=True):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  def testVarScopeNestedOuterScope(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        with tf.variable_scope(outer):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with tf.variable_scope("default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

        with tf.variable_scope(outer, reuse=True):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer_1/scope2/")
        with tf.variable_scope("default", reuse=True):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default_1/scope2/")

  def testVarOpScopeReuseParam(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        with tf.variable_op_scope([], "tower", "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with tf.variable_scope(outer) as outer:
        with tf.variable_op_scope([], "tower", "default", reuse=True):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        outer.reuse_variables()
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testVarOpScopeReuseError(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        with tf.variable_op_scope([], None, "default", reuse=True):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/tower/w:0")

  def testVarOpScopeOuterScope(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        pass
      with tf.variable_op_scope([], outer, "default"):
        self.assertEqual(tf.get_variable("w", []).name,
                         "outer/w:0")
        with tf.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with tf.variable_op_scope([], outer, "default", reuse=True):
        self.assertEqual(tf.get_variable("w", []).name,
                         "outer/w:0")
        with tf.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        outer.reuse_variables()
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  def testVarOpScopeNestedOuterScope(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        with tf.variable_op_scope([], outer, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with tf.variable_op_scope([], outer, "default", reuse=True):
        self.assertEqual(tf.get_variable("w", []).name,
                         "outer/w:0")
        with tf.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with tf.variable_op_scope([], None, "default"):
          self.assertEqual(tf.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")


def axis0_into1_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  return part


def axis0_into2_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  part[0] = 2
  return part


def axis0_into3_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  part[0] = 3
  return part


class VariableScopeWithPartitioningTest(tf.test.TestCase):

  def testResultNameMatchesRequested(self):
    with tf.variable_scope("scope0", partitioner=axis0_into2_partitioner):
      v = tf.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v.name, "scope0/name0")
      v_concat = v.as_tensor()
      self.assertEqual(v_concat.name, "scope0/name0:0")
      variables = tf.get_collection(tf.GraphKeys.VARIABLES)
      self.assertTrue("scope0/name0/part_0:0" in [x.name for x in variables])
      self.assertTrue("scope0/name0/part_1:0" in [x.name for x in variables])
      self.assertFalse("scope0/name0/part_2:0" in [x.name for x in variables])

  def testBreaksIfPartitioningChanges(self):
    with tf.variable_scope("scope0", partitioner=axis0_into2_partitioner):
      tf.get_variable("name0", shape=(3, 1, 1))

    with tf.variable_scope("scope0",
                           partitioner=axis0_into3_partitioner,
                           reuse=True):
      with self.assertRaisesRegexp(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions .* "
          "and found partitions .*"):
        tf.get_variable("name0", shape=(3, 1, 1))

    with tf.variable_scope("scope0",
                           partitioner=axis0_into1_partitioner,
                           reuse=True):
      with self.assertRaisesRegexp(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions .* "
          "and found partitions .*"):
        tf.get_variable("name0", shape=(3, 1, 1))

  def testReturnsExistingConcatenatedValueIfReuse(self):
    with tf.variable_scope("scope0", partitioner=axis0_into2_partitioner):
      v_concat = tf.get_variable("name0", shape=(3, 1, 1))
      tf.get_variable_scope().reuse_variables()
      v_concat_2 = tf.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v_concat, v_concat_2)

  def testAllowsReuseWithoutPartitioner(self):
    with tf.variable_scope("scope0", partitioner=axis0_into2_partitioner):
      v = tf.get_variable("name0", shape=(3, 1, 1))
    with tf.variable_scope("scope0", reuse=True):
      v_reused = tf.get_variable("name0")
    self.assertEqual(v, v_reused)

  def testPropagatePartitionerOnReopening(self):
    with tf.variable_scope("scope0", partitioner=axis0_into2_partitioner) as vs:
      self.assertEqual(axis0_into2_partitioner, vs.partitioner)
      with tf.variable_scope(vs) as vs1:
        self.assertEqual(axis0_into2_partitioner, vs1.partitioner)

  def testPartitionConcatenatesAlongCorrectAxis(self):
    def _part_axis_0(**unused_kwargs):
      return (2, 1, 1)

    def _part_axis_1(**unused_kwargs):
      return (1, 2, 1)

    with tf.variable_scope("root"):
      v0 = tf.get_variable("n0", shape=(2, 2, 2), partitioner=_part_axis_0)
      v1 = tf.get_variable("n1", shape=(2, 2, 2), partitioner=_part_axis_1)

    self.assertEqual(v0.get_shape(), (2, 2, 2))
    self.assertEqual(v1.get_shape(), (2, 2, 2))

    n0_0 = tf.get_default_graph().get_tensor_by_name("root/n0/part_0:0")
    n0_1 = tf.get_default_graph().get_tensor_by_name("root/n0/part_1:0")
    self.assertEqual(n0_0.get_shape(), (1, 2, 2))
    self.assertEqual(n0_1.get_shape(), (1, 2, 2))

    n1_0 = tf.get_default_graph().get_tensor_by_name("root/n1/part_0:0")
    n1_1 = tf.get_default_graph().get_tensor_by_name("root/n1/part_1:0")
    self.assertEqual(n1_0.get_shape(), (2, 1, 2))
    self.assertEqual(n1_1.get_shape(), (2, 1, 2))


class VariableScopeWithCustomGetterTest(tf.test.TestCase):

  def testNonCallableGetterFails(self):
    with self.assertRaisesRegexp(ValueError, r"custom_getter .* not callable:"):
      with tf.variable_scope("scope0", custom_getter=3):
        tf.get_variable("name0")
    with self.assertRaisesRegexp(ValueError, r"custom_getter .* not callable:"):
      tf.get_variable("name0", custom_getter=3)

  def testNoSideEffectsWithIdentityCustomGetter(self):
    called = [0]
    def custom_getter(getter, *args, **kwargs):
      called[0] += 1
      return getter(*args, **kwargs)
    with tf.variable_scope("scope", custom_getter=custom_getter) as scope:
      v = tf.get_variable("v", [1])
    with tf.variable_scope(scope, reuse=True):
      v2 = tf.get_variable("v", [1])
    with tf.variable_scope("new_scope") as new_scope:
      v3 = tf.get_variable("v3", [1])
    with tf.variable_scope(new_scope, reuse=True, custom_getter=custom_getter):
      v4 = tf.get_variable("v3", [1])

    self.assertEqual(v, v2)
    self.assertEqual(v3, v4)
    self.assertEqual(3, called[0])  # skipped one in the first new_scope

  def testGetterThatCreatesTwoVariablesAndSumsThem(self):
    def custom_getter(getter, name, *args, **kwargs):
      g_0 = getter("%s/0" % name, *args, **kwargs)
      g_1 = getter("%s/1" % name, *args, **kwargs)
      with tf.name_scope("custom_getter"):
        return g_0 + g_1

    with tf.variable_scope("scope", custom_getter=custom_getter):
      v = tf.get_variable("v", [1, 2, 3])

    self.assertEqual([1, 2, 3], v.get_shape())
    true_vars = tf.trainable_variables()
    self.assertEqual(2, len(true_vars))
    self.assertEqual("scope/v/0:0", true_vars[0].name)
    self.assertEqual("scope/v/1:0", true_vars[1].name)
    self.assertEqual("custom_getter/add:0", v.name)
    with self.test_session() as sess:
      tf.initialize_all_variables().run()
      np_vars, np_v = sess.run([true_vars, v])
      self.assertAllClose(np_v, sum(np_vars))

if __name__ == "__main__":
  tf.test.main()
