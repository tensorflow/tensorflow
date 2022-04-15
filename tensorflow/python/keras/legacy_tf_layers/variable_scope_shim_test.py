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

import gc
import threading

from absl.testing import parameterized
import numpy

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.legacy_tf_layers import core as core_layers
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def run_inside_wrap_function_in_eager_mode(graph_function):
  """Decorator to execute the same graph code in eager and graph modes.

  In graph mode, we just execute the graph_function passed as argument. In eager
  mode, we wrap the function using wrap_function and then execute the wrapped
  result.

  Args:
    graph_function: python function containing graph code to be wrapped

  Returns:
    decorated function
  """
  def wrap_and_execute(self):
    tracker = variable_scope_shim.VariableAndLossTracker()
    with tracker.scope():
      # use the original function
      graph_function(self)
  return wrap_and_execute


class VariableScopeTest(test.TestCase):

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEqual(0, len(gc.garbage))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVar(self):
    vs = variable_scope._get_default_variable_store()
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNameExists(self):
    vs = variable_scope._get_default_variable_store()
    # No check by default, so we can both create and get existing names.
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    self.assertIs(v, v1)

    self.assertIsNot(v, vs.get_variable("u", [1], reuse=False))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNamelessStore(self):
    vs = variable_scope._get_default_variable_store()
    vs.get_variable("v1", [2])
    vs.get_variable("v2", [2])
    expected_names = ["%s:0" % name for name in ["v1", "v2"]]
    self.assertEqual(
        set(expected_names), set(v.name for v in vs._vars.values()))

  # TODO(mihaimaruseac): Not converted to use wrap_function because of
  # TypeError: Expected tf.group() expected Tensor arguments not 'None' with
  # type '<type 'NoneType'>'
  @test_util.run_in_graph_and_eager_modes
  def testVarScopeInitializer(self):
    init = init_ops.constant_initializer(0.3)
    with variable_scope.variable_scope("tower0") as tower:
      with variable_scope.variable_scope("foo", initializer=init):
        v = variable_scope.get_variable("v", [])
        self.evaluate(variables_lib.variables_initializer([v]))
        self.assertAllClose(self.evaluate(v.value()), 0.3)
      with variable_scope.variable_scope(tower, initializer=init):
        w = variable_scope.get_variable("w", [])
        self.evaluate(variables_lib.variables_initializer([w]))
        self.assertAllClose(self.evaluate(w.value()), 0.3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeConstraint(self):
    constraint = lambda x: 0. * x
    with variable_scope.variable_scope("tower1") as tower:
      with variable_scope.variable_scope("foo", constraint=constraint):
        v = variable_scope.get_variable("v", [])
        self.assertIsNotNone(v.constraint)
      with variable_scope.variable_scope(tower, constraint=constraint):
        w = variable_scope.get_variable("w", [])
        self.assertIsNotNone(w.constraint)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeDType(self):
    with variable_scope.variable_scope("tower2") as tower:
      with variable_scope.variable_scope("foo", dtype=dtypes.float16):
        v = variable_scope.get_variable("v", [])
        self.assertEqual(v.dtype.base_dtype, dtypes.float16)
      with variable_scope.variable_scope(tower, dtype=dtypes.float16):
        w = variable_scope.get_variable("w", [])
        self.assertEqual(w.dtype.base_dtype, dtypes.float16)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitFromNonTensorValue(self):
    v = variable_scope.get_variable("v4", initializer=4, dtype=dtypes.int32)
    self.evaluate(variables_lib.variables_initializer([v]))
    self.assertAllClose(self.evaluate(v.value()), 4)

    w = variable_scope.get_variable(
        "w4", initializer=numpy.array([1, 2, 3]), dtype=dtypes.int64)
    self.evaluate(variables_lib.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), [1, 2, 3])

    # A quirk to be revisited?
    error = ValueError if context.executing_eagerly() else TypeError
    with self.assertRaises(error):
      variable_scope.get_variable("x4", initializer={})

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitFromNonInitializer(self):
    # Test various dtypes with zeros initializer as following:
    types = [
        dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32,
        dtypes.int64, dtypes.bool
    ]

    # Use different variable_name to distinguish various dtypes
    for (i, dtype) in enumerate(types):
      x = variable_scope.get_variable(
          name="xx%d" % i, shape=(3, 4), dtype=dtype)
      y = variable_scope.get_variable(
          name="yy%d" % i,
          shape=(3, 4),
          dtype=dtype,
          initializer=init_ops.zeros_initializer(dtype=dtype))

      self.evaluate(variables_lib.global_variables_initializer())
      self.assertAllEqual(self.evaluate(x.value()), self.evaluate(y.value()))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeRegularizer(self):
    init = init_ops.constant_initializer(0.3)

    def regularizer1(v):
      return math_ops.reduce_mean(v) + 0.1

    def regularizer2(v):
      return math_ops.reduce_mean(v) + 0.2

    with variable_scope.variable_scope(
        "tower3", regularizer=regularizer1) as tower:
      with variable_scope.variable_scope("foo", initializer=init):
        v = variable_scope.get_variable("v", [])
        self.evaluate(variables_lib.variables_initializer([v]))
      with variable_scope.variable_scope(tower, initializer=init) as vs:
        variable_scope.get_variable("u", [])
        vs.set_regularizer(regularizer2)
        variable_scope.get_variable("w", [])
        # Next 3 variable not regularized to test disabling regularization.
        variable_scope.get_variable(
            "x", [], regularizer=variable_scope.no_regularizer)
        with variable_scope.variable_scope(
            "baz", regularizer=variable_scope.no_regularizer):
          variable_scope.get_variable("y", [])
        vs.set_regularizer(variable_scope.no_regularizer)
        variable_scope.get_variable("z", [])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testInitializeFromValue(self):
    init = constant_op.constant(0.1)
    w = variable_scope.get_variable("v", initializer=init)
    self.evaluate(variables_lib.variables_initializer([w]))
    self.assertAllClose(self.evaluate(w.value()), 0.1)

    with self.assertRaisesRegex(ValueError, "shape"):
      # We disallow explicit shape specification when initializer is constant.
      variable_scope.get_variable("u", [1], initializer=init)

    with variable_scope.variable_scope("foo", initializer=init):
      # Constant initializer can be passed through scopes if needed.
      v = variable_scope.get_variable("v")
      self.evaluate(variables_lib.variables_initializer([v]))
      self.assertAllClose(self.evaluate(v.value()), 0.1)

    # Check that non-float32 initializer creates a non-float32 variable.
    init = constant_op.constant(1, dtype=dtypes.int32)
    t = variable_scope.get_variable("t", initializer=init)
    self.assertEqual(t.dtype.base_dtype, dtypes.int32)

    # Raise error if `initializer` dtype and `dtype` are not identical.
    with self.assertRaisesRegex(ValueError, "don't match"):
      variable_scope.get_variable("s", initializer=init, dtype=dtypes.float64)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetOrCreateReuse(self):
    with self.cached_session():

      def test_value(value):
        x = constant_op.constant(value)
        with variable_scope.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=variable_scope.AUTO_REUSE):
          _ = state_ops.assign(variable_scope.get_variable("var", []), x)
        with variable_scope.variable_scope(
            "testVarScopeGetOrCreateReuse_bar",
            reuse=variable_scope.AUTO_REUSE):
          _ = variable_scope.get_variable("var", [])
        self.assertEqual(value, self.evaluate(x))

      test_value(42.)  # Variable is created.
      test_value(13.)  # Variable is reused hereafter.
      test_value(17.)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScope(self):
    with self.cached_session():
      with ops.name_scope_v2("testVarOpScope1"):
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "tower/w:0")

      with ops.name_scope_v2("testVarOpScope2"):
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default_1/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesInterleavedSubstringScopes(self):
    with self.cached_session():
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_2/layer/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeUniqueNamesWithJump(self):
    with self.cached_session():
      with variable_scope.variable_scope("default") as default:
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default/layer/w:0")
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_1/w:0")
        with variable_scope.variable_scope(default):
          pass
        # No matter the jump in the middle, unique numbering continues.
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_2/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuse(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

      with variable_scope.variable_scope(outer, reuse=True) as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeGetVar(self):
    with self.cached_session():
      with variable_scope.variable_scope("root"):
        with variable_scope.variable_scope("towerA") as tower_a:
          va = variable_scope.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with variable_scope.variable_scope(tower_a, reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertIs(va2, va)

        with variable_scope.variable_scope("towerB"):
          vb = variable_scope.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with variable_scope.variable_scope("towerA", reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertIs(va2, va)

        with variable_scope.variable_scope("foo"):
          with variable_scope.variable_scope("bar"):
            v = variable_scope.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with variable_scope.variable_scope(tower_a, reuse=True):
              va3 = variable_scope.get_variable("v", [1])
              self.assertIs(va, va3)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [1], dtype=dtypes.int32)
        self.assertEqual("dtype" in str(exc.exception), True)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeOuterScope(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

      with variable_scope.variable_scope(outer, reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarScopeNestedOuterScope(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

        with variable_scope.variable_scope(outer, reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseParam(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

      with variable_scope.variable_scope(outer) as outer:
        with variable_scope.variable_scope("tower", "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeReuseError(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        with variable_scope.variable_scope(None, "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeOuterScope(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer, "default", []):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVarOpScopeNestedOuterScope(self):
    with self.cached_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "outer/w:0")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testBasicWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      with variable_scope.variable_scope(
          "scope", auxiliary_name_scope=False) as scope:
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "scope/w:0")
      with variable_scope.variable_scope(scope, auxiliary_name_scope=False):
        self.assertEqual(
            variable_scope.get_variable("w1", []).name, "scope/w1:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            "inner", auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/inner/w:0")
        with variable_scope.variable_scope(
            inner, auxiliary_name_scope=False) as inner1:
          self.assertEqual(inner1.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w1", []).name, "outer/inner/w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testCreatedByDefaultNameWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      with variable_scope.variable_scope(
          None, default_name="default", auxiliary_name_scope=False):
        self.assertEqual(
            variable_scope.get_variable("w", []).name, "default/w:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            None, default_name="default",
            auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "outer/")
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterRootScopeWhenAuxiliaryNameScopeIsFalse(self):
    with self.cached_session():
      root_scope = variable_scope.get_variable_scope()
      with variable_scope.variable_scope(
          root_scope, auxiliary_name_scope=False):
        self.assertEqual(variable_scope.get_variable("w", []).name, "w:0")

      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope(
            root_scope, auxiliary_name_scope=False) as inner:
          self.assertEqual(inner.original_name_scope, "")
          self.assertEqual(variable_scope.get_variable("w1", []).name, "w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testAuxiliaryNameScopeIsInvalid(self):
    with self.cached_session():
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            None, default_name="scope", auxiliary_name_scope="invalid"):
          pass

      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            "scope", auxiliary_name_scope="invalid"):
          pass

      with variable_scope.variable_scope("scope") as scope:
        pass
      with self.assertRaisesRegex(TypeError, "auxiliary_name_scope"):
        with variable_scope.variable_scope(
            scope, auxiliary_name_scope="invalid"):
          pass

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReuseScopeWithoutNameScopeCollision(self):
    # Github issue: #13429
    with self.cached_session():
      with variable_scope.variable_scope("outer"):
        with variable_scope.variable_scope("inner") as inner:
          pass

      with variable_scope.variable_scope(
          inner, auxiliary_name_scope=False) as scope:
        with ops.name_scope_v2(scope.original_name_scope):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/inner/w:0")

      with variable_scope.variable_scope("another"):
        with variable_scope.variable_scope(
            inner, auxiliary_name_scope=False) as scope1:
          with ops.name_scope_v2(scope1.original_name_scope):
            self.assertEqual(
                variable_scope.get_variable("w1", []).name,
                "outer/inner/w1:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVarWithDevice(self):
    g = ops.Graph()
    varname_type = []

    def device_func(op):
      if op.type in ["Variable", "VariableV2", "VarHandleOp"]:
        varname_type.append((op.name, op.get_attr("dtype")))
      return "/device:GPU:0"

    with g.as_default():
      with ops.device(device_func):
        _ = variable_scope.get_variable("x", (100, 200))
        _ = variable_scope.get_variable(
            "y", dtype=dtypes.int64, initializer=numpy.arange(73))
    self.assertEqual(varname_type[0], ("x", dtypes.float32))
    self.assertEqual(varname_type[1], ("y", dtypes.int64))

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithRefDtype(self):
    v = variable_scope.get_variable("v", shape=[3, 4], dtype=dtypes.float32)
    # Ensure it is possible to do get_variable with a _ref dtype passed in.
    _ = variable_scope.get_variable("w", shape=[5, 6], dtype=v.dtype)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesNoArgs(self):
    v = variable_scope.get_variable("foo", initializer=lambda: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testGetVariableWithInitializerWhichTakesOptionalArgs(self):
    v = variable_scope.get_variable("foo", initializer=lambda x=True: [2])
    self.assertEqual(v.name, "foo:0")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoGraphs(self):

    def f():
      g1 = ops.Graph()
      g2 = ops.Graph()
      with g1.as_default():
        with g2.as_default():
          with variable_scope.variable_scope("_"):
            pass

    self.assertRaisesRegex(ValueError,
                           "'_' is not a valid (?:root )?scope name", f)


class VariableScopeWithCustomGetterTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNonCallableGetterFails(self):
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      with variable_scope.variable_scope("scope0", custom_getter=3):
        variable_scope.get_variable("name0")
    with self.assertRaisesRegex(ValueError, r"custom_getter .* not callable:"):
      variable_scope.get_variable("name0", custom_getter=3)

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testNoSideEffectsWithIdentityCustomGetter(self):
    called = [0]

    def custom_getter(getter, *args, **kwargs):
      called[0] += 1
      return getter(*args, **kwargs)

    with variable_scope.variable_scope(
        "scope", custom_getter=custom_getter) as scope:
      v = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope(scope, reuse=True):
      v2 = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope("new_scope") as new_scope:
      v3 = variable_scope.get_variable("v3", [1])
    with variable_scope.variable_scope(
        new_scope, reuse=True, custom_getter=custom_getter):
      v4 = variable_scope.get_variable("v3", [1])

    self.assertIs(v, v2)
    self.assertIs(v3, v4)
    self.assertEqual(3, called[0])  # skipped one in the first new_scope

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testSynchronizationAndAggregationWithCustomGetter(self):
    called = [0]
    synchronization = variable_scope.VariableSynchronization.AUTO
    aggregation = variable_scope.VariableAggregation.NONE

    def custom_getter(getter, *args, **kwargs):
      called[0] += 1

      # Verify synchronization and aggregation kwargs are as expected.
      self.assertEqual(kwargs["synchronization"], synchronization)
      self.assertEqual(kwargs["aggregation"], aggregation)
      return getter(*args, **kwargs)

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      variable_scope.get_variable("v", [1])
    self.assertEqual(1, called[0])

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      synchronization = variable_scope.VariableSynchronization.ON_READ
      aggregation = variable_scope.VariableAggregation.MEAN
      variable_scope.get_variable(
          "v1", [1], synchronization=synchronization, aggregation=aggregation)

    self.assertEqual(2, called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreator(self):
    variable_names = []

    def creator_a(next_creator, **kwargs):
      variable_names.append(kwargs.get("name", ""))
      return next_creator(**kwargs)

    def creator_b(next_creator, **kwargs):
      kwargs["name"] = "forced_name"
      return next_creator(**kwargs)

    with variable_scope.variable_creator_scope(creator_a):
      with variable_scope.variable_creator_scope(creator_b):
        variable_scope.variable(1.0, name="one_name")

    self.assertEqual(variable_names[0], "forced_name")

    called = [False]

    def creater_c(next_creator, **kwargs):
      called[0] = True
      self.assertEqual(kwargs["synchronization"],
                       variable_scope.VariableSynchronization.ON_WRITE)
      self.assertEqual(kwargs["aggregation"],
                       variable_scope.VariableAggregation.MEAN)
      return next_creator(**kwargs)

    with variable_scope.variable_creator_scope(creater_c):
      variable_scope.get_variable(
          "v", [],
          synchronization=variable_scope.VariableSynchronization.ON_WRITE,
          aggregation=variable_scope.VariableAggregation.MEAN)
    self.assertTrue(called[0])

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testVariableCreatorNestingError(self):

    def creator(next_creator, **kwargs):
      return next_creator(**kwargs)

    # Save the state so we can clean up at the end.
    graph = ops.get_default_graph()
    old_creator_stack = graph._variable_creator_stack

    try:
      scope = variable_scope.variable_creator_scope(creator)
      scope.__enter__()
      with variable_scope.variable_creator_scope(creator):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
    finally:
      graph._variable_creator_stack = old_creator_stack


class VariableScopeMultithreadedTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoThreadsDisjointScopeEntry(self):

    def thread_fn(i, graph):
      with graph.as_default():
        with variable_scope.variable_scope("foo"):
          if i == 0:
            v = variable_scope.get_variable("v", [])
            self.assertEqual("foo/v:0", v.name)
          else:
            # Any thread after the first one should fail to create variable
            # with the same name.
            with self.assertRaises(ValueError):
              variable_scope.get_variable("v", [])

    graph = ops.get_default_graph()
    threads = [
        threading.Thread(target=thread_fn, args=(
            i,
            graph,
        )) for i in range(2)
    ]

    threads[0].start()
    # Allow thread 0 to finish before starting thread 1.
    threads[0].join()
    threads[1].start()
    threads[1].join()

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testTwoThreadsNestedScopeEntry(self):

    def thread_fn(i, graph, run_event, pause_event):
      with graph.as_default():
        with variable_scope.variable_scope("foo"):
          if i == 0:
            v = variable_scope.get_variable("v", [])
            self.assertEqual("foo/v:0", v.name)
          else:
            # Any thread after the first one should fail to create variable
            # with the same name.
            with self.assertRaises(ValueError):
              variable_scope.get_variable("v", [])
          pause_event.set()
          run_event.wait()

    graph = ops.get_default_graph()
    run_events = [threading.Event() for _ in range(2)]
    pause_events = [threading.Event() for _ in range(2)]
    threads = [
        threading.Thread(
            target=thread_fn, args=(i, graph, run_events[i], pause_events[i]))
        for i in range(2)
    ]

    # Start first thread.
    threads[0].start()
    pause_events[0].wait()
    # Start next thread once the first thread has paused.
    threads[1].start()
    pause_events[1].wait()
    # Resume both threads.
    run_events[0].set()
    run_events[1].set()
    threads[0].join()
    threads[1].join()

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testReenterMainScope(self):

    def thread_fn(graph, main_thread_scope):
      with graph.as_default():
        # Variable created with main scope will have prefix "main".
        with variable_scope.variable_scope(main_thread_scope):
          with variable_scope.variable_scope("foo"):
            v = variable_scope.get_variable("v", [])
            self.assertEqual("main/foo/v:0", v.name)

        # Variable created outside main scope will not have prefix "main".
        with variable_scope.variable_scope("bar"):
          v = variable_scope.get_variable("v", [])
          self.assertEqual("bar/v:0", v.name)

    graph = ops.get_default_graph()
    with variable_scope.variable_scope("main") as main_thread_scope:
      thread = threading.Thread(
          target=thread_fn, args=(graph, main_thread_scope))
      thread.start()
      thread.join()


@combinations.generate(combinations.combine(mode=["eager"]))
class TF1VariableScopeWrapperLayerTest(test.TestCase, parameterized.TestCase):

  def test_get_variable(self):
    # Test the shim when using `get_variable` (and regularizers) directly

    class WrappedDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = inputs
        with variable_scope.variable_scope("dense_one"):
          # The weights are created with a `regularizer`,
          # so the layer should track their regularization losses
          kernel = variable_scope.get_variable(
              shape=[out.shape[-1], self.units],
              regularizer=regularizers.L2(),
              initializer=init_ops.ones_initializer(),
              name="kernel")
          bias = variable_scope.get_variable(
              shape=[self.units,],
              initializer=init_ops.zeros_initializer(),
              name="bias")
          out = math_ops.matmul(out, kernel)
          out = nn_ops.bias_add(out, bias)
        with variable_scope.variable_scope("nested_scope"):
          with variable_scope.variable_scope("dense_two"):
            kernel = variable_scope.get_variable(
                shape=[out.shape[-1], self.units],
                regularizer=regularizers.L2(),
                initializer=init_ops.ones_initializer(),
                name="kernel")
            bias = variable_scope.get_variable(
                shape=[self.units,],
                initializer=init_ops.zeros_initializer(),
                name="bias")
            out = math_ops.matmul(out, kernel)
            out = nn_ops.bias_add(out, bias)
        return out

    layer = WrappedDenseLayer(10)
    out = layer(array_ops.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, regularization losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, array_ops.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(math_ops.add_n(layer.losses), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(array_ops.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        array_ops.ones(shape=(10, 10)) * 2)
    out = layer(array_ops.ones(shape=(5, 5)))
    self.assertAllEqual(out, array_ops.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(math_ops.add_n(layer.losses), 6)

  def test_compat_v1_layer(self):
    # Test the shim when using `compat.v1` layers

    class WrappedDenseLayer(variable_scope_shim.VariableScopeWrapperLayer):

      def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

      def forward_pass(self, inputs, training=None):
        out = core_layers.dense(inputs, self.units, name="dense_one",
                                kernel_initializer=init_ops.ones_initializer(),
                                kernel_regularizer="l2")
        with variable_scope.variable_scope("nested_scope"):
          out = core_layers.dense(
              out, self.units, name="dense_two",
              kernel_initializer=init_ops.ones_initializer(),
              kernel_regularizer="l2")
        return out

    layer = WrappedDenseLayer(10)
    out = layer(array_ops.ones(shape=(5, 5)))
    weights = {x.name: x for x in layer.variables}

    # Verify the correct output, losses, + variables were made
    self.assertEqual(weights.keys(), {"dense_one/bias:0",
                                      "dense_one/kernel:0",
                                      "nested_scope/dense_two/bias:0",
                                      "nested_scope/dense_two/kernel:0"})
    self.assertAllEqual(out, array_ops.ones(shape=(5, 10)) * 50)
    self.assertAllEqual(math_ops.add_n(layer.losses), 1.5)

    # Verify reuse by updating the variables then re-running
    weights["dense_one/kernel:0"].assign(array_ops.ones(shape=(5, 10)) * 2)
    weights["nested_scope/dense_two/kernel:0"].assign(
        array_ops.ones(shape=(10, 10)) * 2)
    out = layer(array_ops.ones(shape=(5, 5)))
    self.assertAllEqual(out, array_ops.ones(shape=(5, 10)) * 200)
    self.assertAllEqual(math_ops.add_n(layer.losses), 6)


if __name__ == "__main__":
  test.main()
