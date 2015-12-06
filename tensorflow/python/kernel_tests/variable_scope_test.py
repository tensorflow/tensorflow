# Copyright 2015 Google Inc. All Rights Reserved.
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

import tensorflow.python.platform

import tensorflow as tf

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops


class VariableStoreTest(tf.test.TestCase):

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

  def testVarScopeIntializer(self):
    with self.test_session() as sess:
      init = tf.constant_initializer(0.3)
      with variable_scope.variable_scope("tower") as tower:
        with variable_scope.variable_scope("foo", initializer=init):
          v = variable_scope.get_variable("v", [])
          sess.run(tf.initialize_variables([v]))
          self.assertAllClose(v.eval(), 0.3)
        with variable_scope.variable_scope(tower, initializer=init):
          w = variable_scope.get_variable("w", [])
          sess.run(tf.initialize_variables([w]))
          self.assertAllClose(w.eval(), 0.3)

  def testControlDeps(self):
    with self.test_session() as sess:
      v0 = variable_scope.get_variable("v0", [1],
                                       initializer=tf.constant_initializer(0))
      with tf.control_dependencies([v0.value()]):
        v1 = variable_scope.get_variable("v1", [1],
                                         initializer=tf.constant_initializer(1))
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
      v0 = variable_scope.get_variable("v0", [],
                                       initializer=tf.constant_initializer(0))
      var_dict = {}
      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        v1 = variable_scope.get_variable("v1", [1],
                                         initializer=tf.constant_initializer(1))
        var_dict["v1"] = v1
        return v1 + v0
      def var_in_else_clause():
        v2 = variable_scope.get_variable("v2", [1],
                                         initializer=tf.constant_initializer(2))
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
      with variable_scope.variable_scope("foo"):
        new_init1 = variable_scope.get_variable_scope().initializer
        self.assertEqual(new_init1, None)
        # Check that we can set initializer like this.
        variable_scope.get_variable_scope().set_initializer(init)
        v = variable_scope.get_variable("v", [])
        sess.run(tf.initialize_variables([v]))
        self.assertAllClose(v.eval(), 0.3)
        # Check that we can set reuse.
        variable_scope.get_variable_scope().reuse_variables()
        with self.assertRaises(ValueError):  # Fail, w does not exist yet.
          variable_scope.get_variable("w", [1])
      # Check that the set initializer goes away.
      new_init = variable_scope.get_variable_scope().initializer
      self.assertEqual(new_init, None)

  def testVarScope(self):
    with self.test_session():
      with variable_scope.variable_scope("tower") as tower:
        self.assertEqual(tower.name, "tower")
        with tf.name_scope("scope") as sc:
          self.assertEqual(sc, "tower/scope/")

      with variable_scope.variable_scope("foo"):
        with variable_scope.variable_scope("bar") as bar:
          self.assertEqual(bar.name, "foo/bar")
          with tf.name_scope("scope") as sc:
            self.assertEqual(sc, "foo/bar/scope/")

      with variable_scope.variable_scope("foo"):
        with variable_scope.variable_scope(tower, reuse=True) as tower_shared:
          self.assertEqual(tower_shared.name, "tower")
          with tf.name_scope("scope") as sc:
            self.assertEqual(sc, "foo_1/tower/scope/")

  def testVarScopeNameScope(self):
    with self.test_session():
      with tf.name_scope("scope1"):
        with variable_scope.variable_scope("tower") as tower:
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with variable_scope.variable_scope("tower"):  # Re-enter adds suffix.
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")

      with tf.name_scope("scope3"):
        with variable_scope.variable_scope("tower"):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower/scope2/")
        with variable_scope.variable_scope(tower):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower_1/scope2/")

      root_var_scope = variable_scope.get_variable_scope()
      with tf.name_scope("scope4"):
        with variable_scope.variable_scope(root_var_scope):
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope4/scope2/")

  def testVarOpScope(self):
    with self.test_session():
      with tf.name_scope("scope1"):
        with variable_scope.variable_op_scope([], "tower", "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with variable_scope.variable_op_scope([], "tower", "default"):
          with self.assertRaises(ValueError):
            variable_scope.get_variable("w", [])
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")

      with tf.name_scope("scope2"):
        with variable_scope.variable_op_scope([], None, "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default/scope2/")
        with variable_scope.variable_op_scope([], None, "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "default_1/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default_1/scope2/")

  def testVarOpScopeReuse(self):
    with self.test_session():
      with tf.variable_scope("outer") as outer:
        with variable_scope.variable_op_scope([], "tower", "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with variable_scope.variable_op_scope([], None, "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with tf.variable_scope(outer, reuse=True) as outer:
        with variable_scope.variable_op_scope([], "tower", "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "outer/tower/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        with variable_scope.variable_op_scope([], None, "default"):
          self.assertEqual(variable_scope.get_variable("w", []).name,
                           "outer/default/w:0")
          with tf.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testVarScopeGetVar(self):
    with self.test_session():
      with variable_scope.variable_scope("root"):
        with variable_scope.variable_scope("towerA") as tower_a:
          va = variable_scope.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with variable_scope.variable_scope(tower_a, reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertEqual(va2, va)

        with variable_scope.variable_scope("towerB"):
          vb = variable_scope.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope("towerA"):
            va2 = variable_scope.get_variable("v", [1])
        self.assertEqual(str(exc.exception)[:12], "Over-sharing")

        with variable_scope.variable_scope("towerA", reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertEqual(va2, va)

        with variable_scope.variable_scope("foo"):
          with variable_scope.variable_scope("bar"):
            v = variable_scope.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with variable_scope.variable_scope(tower_a, reuse=True):
              va3 = variable_scope.get_variable("v", [1])
              self.assertEqual(va, va3)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            with variable_scope.variable_scope("baz"):
              variable_scope.get_variable("v", [1])
        self.assertEqual(str(exc.exception)[:13], "Under-sharing")

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [1], dtype=tf.int32)
        self.assertEqual("dtype" in str(exc.exception), True)


if __name__ == "__main__":
  tf.test.main()
