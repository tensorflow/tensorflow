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
"""Tests for functions."""

import re
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import function_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _OptimizerOptions():
  for cse in [False, True]:
    for inline in [False, True]:
      for cfold in [False, True]:
        cfg = config_pb2.ConfigProto(
            graph_options=config_pb2.GraphOptions(
                optimizer_options=config_pb2.OptimizerOptions(
                    opt_level=config_pb2.OptimizerOptions.L0,
                    do_common_subexpression_elimination=cse,
                    do_function_inlining=inline,
                    do_constant_folding=cfold)))
        if cse:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.arithmetic_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if inline:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.function_optimization = (
              rewriter_config_pb2.RewriterConfig.OFF)
        if cfold:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.ON)
        else:
          cfg.graph_options.rewrite_options.constant_folding = (
              rewriter_config_pb2.RewriterConfig.OFF)
        yield cfg


class FunctionTest(test.TestCase):
  """Test methods for verifying Function support.

  These test methods are used as mix-ins in two test cases: with
  and without C API support.
  """

  def testIdentity(self):

    @function.Defun(dtypes.float32, func_name="MyIdentity")
    def MyIdentityFunc(a):
      return a

    with ops.Graph().as_default():
      call = MyIdentityFunc([18.0])
      self.assertEqual("MyIdentity", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([18.0], self.evaluate(call))

  @test_util.run_v1_only("b/120545219")
  def testIdentityImplicitDeref(self):

    @function.Defun(dtypes.float32, func_name="MyIdentity")
    def MyIdentityFunc(a):
      return a

    with ops.Graph().as_default():
      var = variables.VariableV1([18.0])
      call = MyIdentityFunc(var._ref())  # pylint: disable=protected-access
      self.assertEqual("MyIdentity", call.op.name)
      for cfg in _OptimizerOptions():
        with session.Session(config=cfg) as sess:
          self.evaluate(var.initializer)
          self.assertAllEqual([18.0], self.evaluate(call))

  def testIdentityOutputName(self):

    @function.Defun(
        dtypes.float32, func_name="MyIdentity", out_names=["my_result_name"])
    def MyIdentityFunc(a):
      return a

    with ops.Graph().as_default():
      call = MyIdentityFunc([18.0])
      self.assertEqual("MyIdentity", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([18.0], self.evaluate(call))

  def testTooManyOutputNames(self):

    @function.Defun(
        dtypes.float32,
        func_name="MyIdentity",
        out_names=["my_result1", "my_result2"])
    def MyIdentityFunc(a):
      return a

    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          (r"output names must be either empty or equal in size to outputs. "
           "output names size = 2 outputs size = 1")):
        MyIdentityFunc([18.0])

  def testDefineFunction2Args(self):

    @function.Defun(dtypes.float32, dtypes.float32, func_name="APlus2B")
    def APlus2B(a, b):
      return a + b * 2

    with ops.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([5.0], self.evaluate(call))

  def testFunctionWithNoOutput(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def APlus2B(a, b):
      c = a + b * 2  # Create some ops to have nodes in the body
      print(c)  # Using 'print' to make lint happy

    with ops.Graph().as_default():
      # Call function. There should be no exceptions.
      APlus2B([1.0], [2.0])

  def testDefineFunction2ArgsOutputName(self):

    @function.Defun(
        dtypes.float32,
        dtypes.float32,
        func_name="APlus2B",
        out_names=["my_result_name"])
    def APlus2B(a, b):
      return a + b * 2

    # APlus2B is stateless.
    self.assertEqual([], APlus2B.stateful_ops)
    with ops.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([5.0], self.evaluate(call))

  def testDefineFunctionDuplicateOutputs(self):

    @function.Defun(dtypes.float32, func_name="Duplicate")
    def Duplicate(a):
      b = a + 1.0
      return b, b

    g = ops.Graph()
    with g.as_default():
      Duplicate([3.0])
      func_sig = g.as_graph_def().library.function[0].signature
      # The names given to both outputs should be different
      # even though the same tensor is emitted to both.
      out_names = [a.name for a in func_sig.output_arg]
      self.assertEqual(2, len(out_names))
      self.assertNotEqual(out_names[0], out_names[1])

  def testGradientFunc(self):

    @function.Defun(dtypes.float32, func_name="XSquarePlusOneFn")
    def XSquarePlusOne(x):
      return x * x + 1.0

    @function.Defun(dtypes.float32, dtypes.float32)
    def XSquarePlusOneGrad(x, dy):
      dx = functional_ops.symbolic_gradient(
          input=[x, dy], Tout=[dtypes.float32], f="XSquarePlusOneFn", name="dx")
      return dx

    g = ops.Graph()
    with g.as_default():
      call_f = XSquarePlusOne([2.0])
      call_g = XSquarePlusOneGrad([2.0], [0.1])

      with session.Session() as sess:
        self.assertAllClose([5.0], self.evaluate(call_f))
        self.assertAllClose([0.4], self.evaluate(call_g))

  def testTanhSymGrad(self):

    @function.Defun(dtypes.float32)
    def Forward(x):
      return math_ops.reduce_sum(math_ops.tanh(x))

    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32)
      y = Forward(x)
      dx = gradients_impl.gradients([y], [x])

    inp = np.array([-1, 1, 2, -2], dtype=np.float32)
    feed = {x: inp}
    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L1,
                do_function_inlining=True)))
    with session.Session(graph=g, config=cfg) as sess:
      out, = sess.run(dx, feed)
    self.assertAllClose(1 - np.square(np.tanh(inp)), out)

  def testCustomGradient(self):
    dtype = dtypes.float32

    @function.Defun(dtype, dtype, dtype)
    def XentLossGrad(logits, labels, dloss):
      dlogits = array_ops.reshape(dloss, [-1, 1]) * (
          nn_ops.softmax(logits) - labels)
      dlabels = array_ops.zeros_like(labels)
      # Takes exp(dlogits) to differentiate it from the "correct" gradient.
      return math_ops.exp(dlogits), dlabels

    @function.Defun(dtype, dtype, grad_func=XentLossGrad)
    def XentLoss(logits, labels):
      return math_ops.reduce_sum(labels * math_ops.log(nn_ops.softmax(logits)),
                                 1)

    g = ops.Graph()
    with g.as_default():
      logits = array_ops.placeholder(dtype)
      labels = array_ops.placeholder(dtype)
      loss = XentLoss(logits, labels)
      dlogits = gradients_impl.gradients([loss], [logits])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    prob = np.exp(x) / np.sum(np.exp(x), 1, keepdims=1)
    y = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      with session.Session(graph=g, config=cfg) as sess:
        out, = sess.run(dlogits, {logits: x, labels: y})
      self.assertAllClose(out, np.exp(prob - y), rtol=1e-5)

  @test_util.disable_xla("b/124286351")  # No error is raised
  def testCustomGradientError(self):
    dtype = dtypes.float32

    @function.Defun(dtype, dtype, dtype)
    def Grad(x, dy, dz):
      # Should have returned 1 result.
      return x, dy + dz

    @function.Defun(dtype, grad_func=Grad)
    def Forward(x):
      return x, x

    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(dtype)
      out = math_ops.add_n(Forward(inp))
      dinp = gradients_impl.gradients(out, [inp])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    with session.Session(graph=g) as sess:
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "SymGrad expects to return 1.*but get 2.*instead"):
        _ = sess.run(dinp, {inp: x})

  def testSymGradShape(self):
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtypes.float32, [25, 4])
      y = array_ops.placeholder(dtypes.float32, [200, 100])
      dz = array_ops.placeholder(dtypes.float32, [1])
      # We assume Foo is a function of (x, y) -> (z) Then, Foo's
      # gradient function is (x, y, dz) -> (dx, dy).  dx's shape
      # should be the same as x's; and dy's shape should be the same
      # as y's.
      dx, dy = functional_ops.symbolic_gradient(
          input=[x, y, dz], Tout=[dtypes.float32] * 2, f="Foo")
      self.assertEqual(x.get_shape(), dx.get_shape())
      self.assertEqual(y.get_shape(), dy.get_shape())

  @test_util.run_deprecated_v1
  def testSymGradAttr(self):

    @function.Defun(noinline=True)
    def Foo(x):
      return x * 2

    self.assertTrue(
        Foo.instantiate([dtypes.float32]).definition.attr["_noinline"].b)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(3.0)
      y = Foo(x)
      dx, = gradients_impl.gradients(y, [x])

    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True)))

    with self.session(graph=g, config=cfg):
      self.assertAllClose(y, 6.)
      self.assertAllClose(dx, 2.)

  def _testZNoDepOnY(self, use_const_grad_ys):

    @function.Defun(dtypes.float32, dtypes.float32)
    def Foo(x, y):  # pylint: disable=unused-argument
      return x * 2

    with ops.Graph().as_default():
      # z = Foo(x, y). z doe
      x = constant_op.constant(1.0)
      y = constant_op.constant(2.0)
      z = Foo(x, y)
      if use_const_grad_ys:
        dx, dy = gradients_impl.gradients([z], [x, y], grad_ys=[1.0])
      else:
        dx, dy = gradients_impl.gradients([z], [x, y])
      with session.Session() as sess:
        dx_val, dy_val = self.evaluate([dx, dy])
        self.assertEqual([2.0], dx_val)
        self.assertEqual([0.0], dy_val)

  def testZNoDepOnY(self):
    self._testZNoDepOnY(False)

  def testZNoDepOnYConstGradYs(self):
    # Tests for constant folding of grad_ys
    self._testZNoDepOnY(True)

  def testDefineFunctionNoArgs(self):

    @function.Defun(func_name="AConstant")
    def AConstant():
      return constant_op.constant([42])

    with ops.Graph().as_default():

      call = AConstant()
      self.assertEqual("AConstant", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([42], self.evaluate(call))

  def testDefineFunctionNames(self):

    @function.Defun(dtypes.float32, func_name="Foo")
    def Foo(a):
      return a + 1

    with ops.Graph().as_default():
      call1 = Foo([1.0])
      self.assertEqual("Foo", call1.op.name)
      call2 = Foo([1.0])
      self.assertEqual("Foo_1", call2.op.name)
      # pylint: disable=unexpected-keyword-arg
      call3 = Foo([1.0], name="mine")
      self.assertEqual("mine", call3.op.name)
      with ops.name_scope("my"):
        call4 = Foo([1.0], name="precious")
        self.assertEqual("my/precious", call4.op.name)

  def testNoOp(self):

    @function.Defun(dtypes.float32)
    def Foo(x):
      y = logging_ops.Print(x, [], "Hello")
      with ops.control_dependencies([y]):
        z = gen_control_flow_ops.no_op()
      with ops.control_dependencies([z]):
        return x * 2

    # @function.Defun creates a non-partitioned function.  If we place this on
    # the GPU then the inner `Print` op cannot be run.
    with ops.Graph().as_default(), self.cached_session(use_gpu=False):
      z = Foo(constant_op.constant(3.0))
      self.assertAllEqual(z, 6.0)

  def testAssertOp(self):

    @function.Defun(dtypes.float32)
    def Foo(x):
      check = gen_logging_ops._assert(math_ops.greater(x, 0), [x])
      with ops.control_dependencies([check]):
        return x * 2

    # Foo contains a stateful op (Assert).
    self.assertEqual([("Assert", "Assert")], Foo.stateful_ops)
    g = ops.Graph()
    with g.as_default(), self.cached_session():
      self.assertAllEqual(Foo(constant_op.constant(3.0)), 6.0)
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion failed.*-3"):
        self.assertAllEqual(Foo(constant_op.constant(-3.0)), 6.0)

  @test_util.run_deprecated_v1
  def testAssertWrapper(self):

    @function.Defun(dtypes.float32)
    def MyFn(x):
      with ops.control_dependencies(
          [control_flow_assert.Assert(math_ops.less_equal(x, 10.0), [x])]):
        return array_ops.identity(x)

    with self.cached_session():
      self.assertEqual(1.0, MyFn(1.0).eval())
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        _ = MyFn(100.0).eval()

  @test_util.run_deprecated_v1
  def testWhileLoopCallsFunc(self):
    with self.session():

      @function.Defun(dtypes.float32)
      def Times2(x):
        constant_two = constant_op.constant(2, dtypes.int32)
        two_on_gpu = math_ops.cast(constant_two, dtypes.float32)
        return x * two_on_gpu

      def Body(x):
        x2 = Times2(x)
        x2.set_shape([])
        return x2

      loop = while_loop.while_loop(lambda x: x < 1e5, Body, [1.0])

      ans = self.evaluate(loop)
      self.assertAllClose(ans, 131072.)

  @test_util.run_deprecated_v1
  def testControlFlowStrictness(self):
    """Inlined functions must not execute in a untaken control flow branch."""

    @function.Defun(dtypes.int32)
    def AssertFail(x):
      # Assertion that always fails and does not have a data dependency on `x`.
      assert_false = control_flow_assert.Assert(False, [42])
      with ops.control_dependencies([assert_false]):
        return array_ops.identity(x)

    with ops.device("CPU"):
      pred = array_ops.placeholder(dtypes.bool)
      x = array_ops.placeholder(dtypes.int32)
      cond = tf_cond.cond(pred, lambda: x + 1, lambda: AssertFail(x))
      # pylint: disable=unnecessary-lambda
      loop = while_loop.while_loop(lambda y: pred, lambda y: AssertFail(y), [x])
      # pylint: enable=unnecessary-lambda

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    # Enables inlining.
    config = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True),
            rewrite_options=rewriter_config))

    with session.Session(config=config) as sess:
      # Since the 'False' branch is not taken, the assertion should not fire.
      self.assertEqual(4, sess.run(cond, {pred: True, x: 3}))

      # The assertion should still fire if the False branch is taken.
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        sess.run(cond, {pred: False, x: 3})

      # Similarly for loops.
      self.assertEqual(3, sess.run(loop, {pred: False, x: 3}))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "assertion"):
        sess.run(loop, {pred: True, x: 3})

  @test_util.run_deprecated_v1
  def testVar(self):

    @function.Defun(dtypes.float32)
    def Foo(x):
      return x * x + 1

    g = ops.Graph()
    with g.as_default():
      v = variables.Variable(constant_op.constant(10.0))
      z = Foo(v)

    with self.session(graph=g):
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(z, 101.)

  @test_util.run_deprecated_v1
  def testResourceVarAsImplicitInput(self):
    g = ops.Graph()
    with g.as_default(), ops.device("cpu:0"):
      expected_type = dtypes.float32
      expected_shape = tensor_shape.TensorShape((4, 4))
      v = variable_scope.get_variable(
          "var", expected_shape, expected_type, use_resource=True)

      @function.Defun()
      def Foo():
        captured = array_ops.identity(v)
        self.assertEqual(expected_type, captured.dtype)
        self.assertEqual(expected_shape, captured.shape)
        return captured, array_ops.shape(captured)

      expected_val = v.value()
      actual_val, actual_shape = Foo()

    with self.session(graph=g):
      v.initializer.run()
      self.assertAllEqual(expected_val, self.evaluate(actual_val))
      self.assertAllEqual(expected_shape, self.evaluate(actual_shape))

  def testDefineErrors(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "can not return None"):

        @function.Defun()
        def TwoNone():
          return None, None

        _ = TwoNone.definition

      with self.assertRaisesRegex(ValueError, "are not supported"):

        @function.Defun()
        def DefaultArg(unused_a=12):
          return constant_op.constant([1])

        _ = DefaultArg.definition
      with self.assertRaisesRegex(ValueError, "are not supported"):

        @function.Defun()
        def KwArgs(**unused_kwargs):
          return constant_op.constant([1])

        _ = KwArgs.definition
      with self.assertRaisesRegex(ValueError, "tf.function input types"):

        @function.Defun(dtypes.float32)
        def PlusMinusV2(a, b):
          return a + b, b - a

        _ = PlusMinusV2.definition
      with self.assertRaisesRegex(ValueError, "tf.function input types"):

        @function.Defun(dtypes.float32, dtypes.float32, dtypes.float32)
        def PlusMinusV3(a, b):
          return a + b, b - a

        _ = PlusMinusV3.definition

  def testCallErrors(self):

    @function.Defun()
    def Const():
      return constant_op.constant(1)

    @function.Defun(dtypes.int32)
    def PlusOne(a):
      return a + 1

    @function.Defun(dtypes.int32, dtypes.int32)
    def PlusMinus(a, b):
      return a + b, b - a

    with ops.Graph().as_default():

      _ = Const()
      # pylint: disable=too-many-function-args
      # pylint: disable=unexpected-keyword-arg
      # pylint: disable=no-value-for-parameter
      with self.assertRaisesRegex(ValueError, "Expected 0"):
        _ = Const(1)
      with self.assertRaisesRegex(ValueError, "Expected 0"):
        _ = Const(1, 2)

      with self.assertRaisesRegex(ValueError, "Expected 1"):
        _ = PlusOne()
      _ = PlusOne(1)
      with self.assertRaisesRegex(ValueError, "Expected 1"):
        _ = PlusOne(1, 2)

      with self.assertRaisesRegex(ValueError, "Expected 2"):
        _ = PlusMinus()
      with self.assertRaisesRegex(ValueError, "Expected 2"):
        _ = PlusMinus(1)
      _ = PlusMinus(1, 2)

      _ = PlusOne(1, name="p1")
      with self.assertRaisesRegex(ValueError, "Unknown keyword arguments"):
        _ = PlusOne(1, device="/device:GPU:0")

  def testFunctionDecorator(self):

    @function.Defun(dtypes.float32, func_name="Minus1")
    def Minus1(b):
      return b - 1.0

    with ops.Graph().as_default():
      call1 = Minus1([2.])
      self.assertTrue(isinstance(Minus1, function._DefinedFunction))
      self.assertEqual(Minus1.name, "Minus1")
      # pylint: disable=unexpected-keyword-arg
      call2 = Minus1(call1, name="next")
      # pylint: enable=unexpected-keyword-arg
      self.assertEqual("next", call2.op.name)
      with session.Session() as sess:
        self.assertAllEqual([1], self.evaluate(call1))
        self.assertAllEqual([0], self.evaluate(call2))

  def testNestedFunction(self):

    @function.Defun(dtypes.float32)
    def Cube(x):
      return x * x * x

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):
      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.cached_session():
        self.assertAllEqual(z, 25.0)

  def testNestedDefinedFunction(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):

      @function.Defun(dtypes.float32)
      def Cube(x):
        return x * x * x

      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.cached_session():
        self.assertAllEqual(z, 25.0)

  def testUnusedFunction(self):
    invoked = False
    # pylint: disable=unused-variable
    @function.Defun()
    def Unused():
      invoked = True
      return constant_op.constant(42.)

    self.assertFalse(invoked)
    g = ops.Graph()
    with g.as_default():

      @function.Defun()
      def Unused2():
        invoked = True
        return constant_op.constant(7.)

      constant_op.constant(3.)
    # pylint: enable=unused-variable
    self.assertFalse(invoked)
    gdef = g.as_graph_def()
    self.assertEqual(0, len(gdef.library.function))

  @test_util.run_deprecated_v1
  def testReduction(self):
    g = ops.Graph()

    # BN0 is computing batch normed matrix along rows.
    def BN0(x):
      mean = math_ops.reduce_mean(x, [0])
      var = math_ops.reduce_mean(math_ops.square(x - mean))  # biased var
      rstd = math_ops.rsqrt(var + 1e-8)
      return (x - mean) * rstd

    # Wraps BatchNorm in a tf function.
    @function.Defun(dtypes.float32)
    def BN1(x):
      return BN0(x)

    with g.as_default():
      x = array_ops.placeholder(dtypes.float32)
      y0 = BN0(x)  # A plain graph
      y1 = BN1(x)  # A tf function
      dx0, = gradients_impl.gradients([y0], [x])
      dx1, = gradients_impl.gradients([y1], [x])

    # Both should produce the same result and gradient.
    with self.session(graph=g) as sess:
      vals = sess.run([y0, y1, dx0, dx1], {x: np.random.uniform(size=(3, 7))})
      self.assertAllClose(vals[0], vals[1])
      self.assertAllClose(vals[2], vals[3])

  @test_util.run_deprecated_v1
  def testCapture(self):
    g = ops.Graph()
    with g.as_default():
      w = variables.Variable(constant_op.constant([[1.0]]))
      b = variables.Variable(constant_op.constant([2.0]))

      # Foo() captures w and b.
      @function.Defun(dtypes.float32)
      def Foo(x):

        # Plus() captures b.
        @function.Defun(dtypes.float32)
        def Plus(y):
          return y + b

        return Plus(math_ops.matmul(w, x))

      y = Foo(constant_op.constant([[10.]]))

      @function.Defun()
      def Bar():
        return w

      z = Bar()

    with self.session(graph=g):
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(y, [[12.0]])
      self.assertAllEqual(z, [[1.0]])

  def testCaptureControls(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant([10.0])
      x = logging_ops.Print(x, [x], "outer")

      @function.Defun(dtypes.float32)
      def Foo(y):
        with ops.control_dependencies([x]):
          y = logging_ops.Print(y, [y], "inner")
        return y

      with self.assertRaisesRegex(ValueError, "not an element of this graph."):
        # NOTE: We still do not support capturing control deps.
        _ = Foo(x)

  @test_util.run_deprecated_v1
  def testCaptureInWhileLoop(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)

      @function.Defun()
      def Foo():
        return while_loop.while_loop(lambda i: i < 10, lambda i: i + x, [0])

      y = Foo()

    with self.session(graph=g) as sess:
      self.assertEqual(self.evaluate(y), 10)

  @test_util.run_deprecated_v1
  def testCaptureInCond(self):
    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(1)

      @function.Defun(dtypes.bool)
      def Foo(pred):
        return tf_cond.cond(pred, lambda: x, lambda: x + 1)

      y = Foo(True)
      z = Foo(False)

    with self.session(graph=g) as sess:
      self.assertEqual(self.evaluate(y), 1)
      self.assertEqual(self.evaluate(z), 2)

  @test_util.run_deprecated_v1
  def testSignatureHash(self):
    # Foo.Inner and Bar.Inner have identical function body but have
    # different signatures. They should be treated as two different functions.

    @function.Defun()
    def Foo(x):

      @function.Defun()
      def Inner(x):
        return x + 10.

      return Inner(x)

    @function.Defun()
    def Bar(x):

      @function.Defun()
      def Inner(x, unused_y, unused_z):
        return x + 10.

      return Inner(x, 2., 3.)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(10.0)
      y = Foo(x)
      z = Bar(x)

    with self.session(graph=g) as sess:
      v0, v1 = self.evaluate([y, z])
      self.assertAllEqual(v0, 20.)
      self.assertAllEqual(v1, 20.)

  def testShapeFunction(self):

    @function.Defun(
        dtypes.float32, shape_func=lambda op: [op.inputs[0].get_shape()])
    def Foo(x):
      return x + 1.0

    @function.Defun(
        shape_func=lambda op: [[1] + op.inputs[0].get_shape().as_list()])
    def Bar(x):
      return array_ops_stack.stack([x])

    g = ops.Graph()
    with g.as_default():
      x = Foo([1.0, 2.0])
      self.assertEqual(x.get_shape().as_list(), [2])
      y = Bar(array_ops.zeros([1, 2, 3]))
      self.assertAllEqual(y.get_shape().as_list(), [1, 1, 2, 3])

  @test_util.run_deprecated_v1
  def testVariableReuse(self):

    def LinearWithReuse(input_tensor, reuse=None):
      size = input_tensor.shape.dims[1]
      with variable_scope.variable_scope("linear", reuse=reuse):
        w = variable_scope.get_variable(
            "w", shape=[size, size], dtype=input_tensor.dtype)
      return math_ops.matmul(input_tensor, w)

    @function.Defun(dtypes.float32)
    def Foo(inputs):
      inputs = array_ops.reshape(inputs, [32, 100])
      hidden = LinearWithReuse(inputs)
      return LinearWithReuse(hidden, reuse=True)

    input_op = array_ops.placeholder(shape=[32, 100], dtype=dtypes.float32)
    output_op = Foo(input_op)

    global_vars = variables.global_variables()
    self.assertEqual(len(global_vars), 1)
    self.assertEqual(global_vars[0].name, "linear/w:0")

    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      output_val = sess.run(
          output_op, feed_dict={input_op: np.random.rand(32, 100)})
      self.assertEqual(output_val.shape, (32, 100))

  @test_util.run_deprecated_v1
  def testFunctionCallInDifferentVariableScopes(self):

    @function.Defun(dtypes.float32)
    def Foo(inputs):
      var = variable_scope.get_variable(
          "var",
          shape=[10],
          dtype=dtypes.float32,
          initializer=init_ops.ones_initializer())
      return inputs + var

    input_op = array_ops.placeholder(shape=[10], dtype=dtypes.float32)
    with variable_scope.variable_scope("vs1"):
      out1_op = Foo(input_op)

    with variable_scope.variable_scope("vs2"):
      out2_op = Foo(input_op)

    global_vars = variables.global_variables()
    self.assertEqual(len(global_vars), 1)
    self.assertEqual(global_vars[0].name, "vs1/var:0")

    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      out1, out2 = sess.run(
          [out1_op, out2_op], feed_dict={input_op: np.linspace(1, 10, 10)})
      self.assertAllEqual(out1, np.linspace(2, 11, 10))
      self.assertAllEqual(out2, np.linspace(2, 11, 10))

  def testTwoInputsSameOp(self):
    g = ops.Graph()
    with g.as_default():
      m = array_ops.placeholder(dtypes.float32)
      s, u, v = linalg_ops.svd(m)
      ss = math_ops.reduce_sum(s)
      uu = math_ops.reduce_sum(u)
      vv = math_ops.reduce_sum(v)
      result = ss + uu + vv
    f = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations()[1:],  # skip the placeholder
        [s, u, v],
        [result])
    self.assertEqual(len(f.signature.input_arg), 3)

  def testGradientWithIntegerFunctionArgument(self):

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(t, x):
      return x[t]

    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(dtypes.float32)
      t = constant_op.constant(0, dtypes.int32)
      out = Foo(t, inp)
      dinp, = gradients_impl.gradients(out, [inp])

    x = np.zeros((2,)).astype(np.float32)
    with session.Session(graph=g) as sess:
      self.assertAllClose(
          np.array([1.0, 0.0]).astype(np.float32), sess.run(dinp, {inp: x}))

  @test_util.run_deprecated_v1
  def testFunctionMarkedStateful(self):

    @function.Defun(dtypes.int32, dtypes.float32)
    def Foo(t, x):
      return x[t]

    @function.Defun(dtypes.int64)
    def Bar(x):
      return x

    # NOTE(mrry): All functions are currently considered stateless by the
    # runtime, so we simulate a "stateful" function.
    # TODO(b/70565970): Remove this hack when we are able to build stateful
    # functions using the API.
    # pylint: disable=protected-access
    Foo._signature.is_stateful = True
    Bar._signature.is_stateful = True
    # pylint: enable=protected-access

    result_1 = Foo(3, [1.0, 2.0, 3.0, 4.0])
    result_2 = Bar(constant_op.constant(100, dtype=dtypes.int64))

    with session.Session() as sess:
      self.assertEqual(4.0, self.evaluate(result_1))
      self.assertEqual(100, self.evaluate(result_2))
      self.assertEqual((4.0, 100), sess.run((result_1, result_2)))

  @test_util.run_deprecated_v1
  def testStatefulFunction(self):

    @function.Defun()
    def FunctionWithStatelessOp():
      return constant_op.constant(42.0)

    @function.Defun()
    def FunctionWithStatefulOp():
      return random_ops.random_uniform([100], maxval=10, dtype=dtypes.int32)

    @function.Defun()
    def FunctionWithStatelessFunctionCall():
      return FunctionWithStatelessOp()

    @function.Defun()
    def FunctionWithStatefulFunctionCall():
      return FunctionWithStatefulOp()

    # Test that the `is_stateful` bit is propagated.
    self.assertFalse(FunctionWithStatelessOp.definition.signature.is_stateful)
    self.assertTrue(FunctionWithStatefulOp.definition.signature.is_stateful)
    self.assertFalse(
        FunctionWithStatelessFunctionCall.definition.signature.is_stateful)
    self.assertTrue(
        FunctionWithStatefulFunctionCall.definition.signature.is_stateful)

    # Ensure that two invocations of the same random-number-generating
    # function produce different results.
    result1 = FunctionWithStatefulFunctionCall()
    result2 = FunctionWithStatefulFunctionCall()

    # Statefulness affects how the function is treated by the various
    # optimization passes, so run the test in each optimizer
    # configuration.
    for config in _OptimizerOptions():
      with session.Session(config=config) as sess:
        val1, val2 = sess.run((result1, result2))
        self.assertFalse(all(val1 == val2))
        val3, val4 = sess.run((result1, result2))
        self.assertFalse(all(val3 == val1))
        self.assertFalse(all(val4 == val2))

  @test_util.run_v1_only("currently failing on v2")
  def testStatefulFunctionWithAllowlisting(self):
    t = random_ops.random_uniform([100], maxval=10, dtype=dtypes.int32)

    @function.Defun(capture_by_value=True)
    def StatefulFn():
      return t + constant_op.constant(3, dtype=dtypes.int32)

    # First time we try to capture a stateful RandomUniform op.
    with self.assertRaisesRegex(ValueError, "Cannot capture a stateful node"):
      res = StatefulFn()

    # This time we allowlist this op, so that its recreated.
    @function.Defun(capture_by_value=True, allowlisted_stateful_ops=set([t.op]))
    def StatefulFn2():
      return t + constant_op.constant(3, dtype=dtypes.int32)

    res = StatefulFn2()
    with session.Session() as sess:
      r = sess.run(res)
      for i in r:
        self.assertGreaterEqual(i, 3)

  @test_util.run_deprecated_v1
  def testSameFunctionOnTwoDevices(self):

    @function.Defun(dtypes.float32)
    def AddOne(x):
      return x + 1.0

    with ops.device("/cpu:0"):
      f_0 = AddOne(41.0)

    with ops.device("/cpu:1"):
      f_1 = AddOne(43.0)

    for config in _OptimizerOptions():
      config.device_count["CPU"] = 2
      with session.Session(config=config) as sess:
        self.assertEqual(42.0, self.evaluate(f_0))
        self.assertEqual(44.0, self.evaluate(f_1))
        self.assertEqual((42.0, 44.0), sess.run((f_0, f_1)))

  @test_util.run_deprecated_v1
  def testGuaranteedConstsAreCaptured(self):
    var = variables.Variable(1.0)
    const = array_ops.guarantee_const(var)
    also_const = array_ops.identity(const)
    still_const = array_ops.identity(also_const)
    not_const = still_const + var
    also_not_const = array_ops.placeholder(dtypes.float32)

    @function.Defun()
    def CapturesGuaranteedConst():
      output = const + also_const + still_const + not_const + also_not_const
      first, second, third, fourth, fifth = function.get_extra_args()
      self.assertEqual("GuaranteeConst", first.consumers()[0].node_def.op)
      self.assertEqual("GuaranteeConst", second.consumers()[0].node_def.op)
      self.assertEqual("GuaranteeConst", third.consumers()[0].node_def.op)
      self.assertNotEqual("GuaranteeConst", fourth.consumers()[0].node_def.op)
      self.assertNotEqual("GuaranteeConst", fifth.consumers()[0].node_def.op)
      return output

    with self.session(use_gpu=False) as sess:
      self.evaluate(var.initializer)
      _ = sess.run(CapturesGuaranteedConst(), {also_not_const: 1.0})

  @test_util.run_deprecated_v1
  def testSameFunctionDifferentGrads(self):

    def PartOne(x):

      # Default grad is dx = dy * 2
      @function.Defun(dtypes.float32)
      def Foo(x):
        return x * 2

      return Foo(x)

    def PartTwo(x):

      @function.Defun(dtypes.float32, dtypes.float32)
      def Bar(x, dy):
        return x + dy  # crazy backprop

      @function.Defun(dtypes.float32, grad_func=Bar)
      def Foo(x):
        return x * 2

      return Foo(x)

    def PartThree(x):

      def Bar(op, dy):
        return op.inputs[0] * dy / 2  # crazy backprop

      @function.Defun(dtypes.float32, python_grad_func=Bar)
      def Foo(x):
        return x * 2

      return Foo(x)

    g = ops.Graph()
    with g.as_default():
      x = constant_op.constant(100.)
      x0 = x
      y0 = PartOne(x0)
      dx0, = gradients_impl.gradients(ys=[y0], xs=[x0])
      x1 = x
      y1 = PartTwo(x1)
      dx1, = gradients_impl.gradients(ys=[y1], xs=[x1])
      x2 = x
      y2 = PartThree(x2)
      dx2, = gradients_impl.gradients(ys=[y2], xs=[x2])

    with self.session(graph=g) as sess:
      v0, v1, v2 = self.evaluate([dx0, dx1, dx2])

    self.assertAllEqual(v0, 2.)
    self.assertAllEqual(v1, 101.)
    self.assertAllEqual(v2, 50.)


class FunctionsFromProtos(test.TestCase):

  def stripInternalFunctionDefAnnotations(self, f_def):
    result = function_pb2.FunctionDef()
    result.CopyFrom(f_def)
    result.attr.pop("_construction_context", None)
    return result

  def expectFunctionsEqual(self, func, grad_func=None, new_func=None):
    if new_func is None:
      # Make a copy of func.definition to avoid any bugs masked by using the
      # same object
      serialized_fdef = func.definition.SerializeToString()
      # Serialize and then deserialize `func` to create `new_func`
      fdef = function_pb2.FunctionDef.FromString(serialized_fdef)
      new_func = function._from_definition(fdef, grad_func=grad_func)
    self.assertEqual(func.name, new_func.name)
    self.assertEqual(
        self.stripInternalFunctionDefAnnotations(func.definition),
        self.stripInternalFunctionDefAnnotations(new_func.definition))
    self.assertEqual(func.grad_func_name, new_func.grad_func_name)
    self.assertEqual(func.declared_input_types, new_func.declared_input_types)
    self.assertEqual(func.captured_inputs, new_func.captured_inputs)

  @test_util.run_deprecated_v1
  def testBasic(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def Foo(x, y):
      return x + y

    self.expectFunctionsEqual(Foo)

  def testGradFunc(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def G(x, dy):
      return x * dy

    @function.Defun(dtypes.float32, grad_func=G)
    def F(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    self.expectFunctionsEqual(F, grad_func=G)

  def testCapturedInputs(self):
    c = constant_op.constant(10, dtypes.int64)

    @function.Defun(dtypes.int64)
    def Foo(x):
      return x + c

    new_func = function._from_definition(Foo.definition)

    self.assertEqual(Foo.name, new_func.name)
    self.assertEqual(
        self.stripInternalFunctionDefAnnotations(Foo.definition),
        self.stripInternalFunctionDefAnnotations(new_func.definition))
    self.assertEqual(Foo.grad_func_name, new_func.grad_func_name)

    # Captured inputs are added as regular inputs to the function definition
    self.assertEqual(new_func.declared_input_types,
                     Foo.declared_input_types + (dtypes.int64,))
    self.assertEqual(len(new_func.captured_inputs), 0)

  def testNestedFunctions(self):

    @function.Defun(dtypes.float32)
    def Outer(x):

      @function.Defun(dtypes.float32)
      def Inner(y):
        return y + 1

      return Inner(Inner(x))

    self.expectFunctionsEqual(Outer)

  def testFromLibrary(self):
    # Define some functions with different gradient functions. Note that many of
    # the below functions are identical since function bodies don't matter for
    # this test.

    @function.Defun(dtypes.float32, dtypes.float32)
    def G1(x, dy):
      return x * dy

    @function.Defun(dtypes.float32, dtypes.float32)
    def G2(x, dy):
      return x * dy

    # F1 and F2 have the same gradient function
    @function.Defun(dtypes.float32, grad_func=G1)
    def F1(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    @function.Defun(dtypes.float32, grad_func=G1)
    def F2(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    # F3 has a different gradient function
    @function.Defun(dtypes.float32, grad_func=G2)
    def F3(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    # F4 has no gradient function
    @function.Defun(dtypes.float32)
    def F4(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    # Instantiate all functions
    g = ops.Graph()
    with g.as_default():
      c = constant_op.constant(1.0, dtypes.float32)
      f1 = F1(c)
      f2 = F2(c)
      f3 = F3(c)
      f4 = F4(c)
      gradients_impl.gradients([f1, f2, f3, f4], c)

    library = g.as_graph_def().library
    new_funcs = function.from_library(library)

    def CheckNewFunc(func):
      new_func = [f for f in new_funcs if f.name == func.name]
      self.assertEqual(len(new_func), 1)
      self.expectFunctionsEqual(func, new_func=new_func[0])

    CheckNewFunc(G1)
    CheckNewFunc(G2)
    CheckNewFunc(F1)
    CheckNewFunc(F2)
    CheckNewFunc(F3)
    CheckNewFunc(F4)

  def testFromLibraryEmptyLib(self):
    library = function_pb2.FunctionDefLibrary()
    self.assertEqual(len(function.from_library(library)), 0)

  def testFromLibraryMissingFuncDef(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def G1(x, dy):
      return x * dy

    @function.Defun(dtypes.float32)
    def F1(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    gradient = function_pb2.GradientDef()
    gradient.function_name = F1.name
    gradient.gradient_func = G1.name

    # Create invalid function def that is missing G1 function def
    library = function_pb2.FunctionDefLibrary()
    library.gradient.extend([gradient])
    library.function.extend([F1.definition])

    with self.assertRaisesRegex(
        ValueError,
        "FunctionDefLibrary missing 'G1_[0-9a-zA-Z]{8,11}' FunctionDef"):
      function.from_library(library)

    # Create invalid function def that is missing F1 function def
    library = function_pb2.FunctionDefLibrary()
    library.gradient.extend([gradient])
    library.function.extend([G1.definition])

    with self.assertRaisesRegex(
        ValueError,
        "FunctionDefLibrary missing 'F1_[0-9a-zA-Z]{8,11}' FunctionDef"):
      function.from_library(library)

  def testFromLibraryCyclicGradFuncs(self):

    @function.Defun(dtypes.float32)
    def F1(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    @function.Defun(dtypes.float32)
    def F2(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    # Create invalid function def library where F1 has gradient function F2 and
    # F2 has gradient function F1
    library = function_pb2.FunctionDefLibrary()
    library.function.extend([F1.definition, F2.definition])

    gradient1 = function_pb2.GradientDef()
    gradient1.function_name = F1.name
    gradient1.gradient_func = F2.name

    gradient2 = function_pb2.GradientDef()
    gradient2.function_name = F2.name
    gradient2.gradient_func = F1.name

    library.gradient.extend([gradient1, gradient2])

    with self.assertRaisesRegex(
        ValueError, "FunctionDefLibrary contains cyclic gradient functions!"):
      function.from_library(library)

  def testExperimentalAttrs(self):

    @function.Defun(dtypes.int32, experimental_tag="tag_value")
    def FunctionWithStrAttr(i):
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=123)
    def FunctionWithIntAttr(i):
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=123.0)
    def FunctionWithFloatAttr(i):
      return array_ops.identity(i)

    @function.Defun(dtypes.int32, experimental_tag=True)
    def FunctionWithBoolAttr(i):
      return array_ops.identity(i)

    self.assertTrue("experimental_tag" in FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["experimental_tag"].s,
                     b"tag_value")
    self.assertTrue("experimental_tag" in FunctionWithIntAttr.definition.attr)
    self.assertEqual(FunctionWithIntAttr.definition.attr["experimental_tag"].i,
                     123)
    self.assertTrue("experimental_tag" in FunctionWithFloatAttr.definition.attr)
    self.assertEqual(
        FunctionWithFloatAttr.definition.attr["experimental_tag"].f, 123.0)
    self.assertTrue("experimental_tag" in FunctionWithBoolAttr.definition.attr)
    self.assertEqual(FunctionWithBoolAttr.definition.attr["experimental_tag"].b,
                     True)

  def testImplementsReferenceAttrs(self):

    @function.Defun(
        dtypes.int32, _implements="org.google.lstm", _reference="arxiv.org")
    def FunctionWithStrAttr(i):
      return array_ops.identity(i)

    self.assertIn("_implements", FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["_implements"].s,
                     b"org.google.lstm")
    self.assertIn("_reference", FunctionWithStrAttr.definition.attr)
    self.assertEqual(FunctionWithStrAttr.definition.attr["_reference"].s,
                     b"arxiv.org")


class FunctionOverloadTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):

    @function.Defun()
    def Sinh(x):
      return 1 / 2. * (math_ops.exp(x) - math_ops.exp(-x))

    g = ops.Graph()
    with g.as_default():
      x = Sinh(constant_op.constant(0.25, dtypes.float32))
      y = Sinh(constant_op.constant(0.25, dtypes.float64))

    with self.session(graph=g):
      self.assertAllClose(x, np.sinh(0.25))
      self.assertAllClose(y, np.sinh(0.25))

  def testGradient(self):

    @function.Defun(func_name="Spec")
    def G(x, dy):
      return x * dy

    @function.Defun(grad_func=G)
    def F(x):
      return math_ops.exp(x) - math_ops.exp(-x)

    for dtype in [dtypes.float32, dtypes.float64]:
      g = ops.Graph()
      with g.as_default():
        x = constant_op.constant(0.25, dtype)
        y = F(x)
        dx, = gradients_impl.gradients(y, x)

        with self.session(graph=g):
          self.assertAllClose(dx, 0.25)

  def testDocString(self):

    @function.Defun()
    def Foo(x):
      """Successor of x."""
      return x + 1

    g = ops.Graph()
    with g.as_default():
      _ = Foo(1)

    self.assertEqual(g.as_graph_def().library.function[0].signature.description,
                     "Successor of x.")


class FunctionCaptureByValueTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCaptureByValue(self):
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant([[1.0]])
      b = constant_op.constant([2.0])

      # Foo() captures w and b.
      @function.Defun(dtypes.float32, capture_by_value=True)
      def Foo(x):

        # Plus() captures b.
        @function.Defun(dtypes.float32, capture_by_value=True)
        def Plus(y):
          return y + b

        self.assertEqual(0, len(Plus.captured_inputs))

        return Plus(math_ops.matmul(w, x))

      y = Foo(constant_op.constant([[10.]]))

    self.assertEqual(0, len(Foo.captured_inputs))

    with self.session(graph=g):
      self.assertAllEqual(y, [[12.0]])


@test_util.run_all_without_tensor_float_32(
    "Calls matmul in custom LSTM function")
class UnrollLSTMTest(test.TestCase):
  BATCH_SIZE = 16
  LSTM_DIMS = 32
  NUM_UNROLL = 20

  def _Weights(self):
    dims = self.LSTM_DIMS
    return random_ops.random_uniform([2 * dims, 4 * dims], -1, 1, seed=123456)

  def _Input(self):
    return random_ops.random_uniform(
        [self.NUM_UNROLL, self.BATCH_SIZE, self.LSTM_DIMS], seed=654321)

  # Helper to construct a LSTM cell graph.
  @classmethod
  def LSTMCell(cls, x, mprev, cprev, weights):
    xm = array_ops.concat([x, mprev], 1)
    i_i, i_g, f_g, o_g = array_ops.split(
        value=math_ops.matmul(xm, weights), num_or_size_splits=4, axis=1)
    new_c = math_ops.sigmoid(f_g) * cprev + math_ops.sigmoid(
        i_g) * math_ops.tanh(i_i)
    new_c = math_ops.maximum(math_ops.minimum(new_c, 50.0), -50.0)
    new_m = math_ops.sigmoid(o_g) * math_ops.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):

    def Loop(cell, w, i):
      x = array_ops_stack.unstack(i, self.NUM_UNROLL)
      m = array_ops.zeros_like(x[0])
      c = array_ops.zeros_like(x[0])
      for i in range(self.NUM_UNROLL):
        m, c = cell(x[i], m, c, w)
      return m

    cell = UnrollLSTMTest.LSTMCell
    if mode == "complete":
      # Constructs the complete graph in python.
      return Loop(cell, weights, inp)

    cell = function.Defun(dtypes.float32, dtypes.float32, dtypes.float32,
                          dtypes.float32)(
                              cell)
    if mode == "cell":
      # Just represent the LSTM as a function.
      return Loop(cell, weights, inp)

    if mode == "loop":
      # Wraps the whole loop as a function.
      @function.Defun(dtypes.float32, dtypes.float32)
      def LSTMLoop(w, i):
        return Loop(cell, w, i)

      return LSTMLoop(weights, inp)

    if mode == "loop10":
      # Wraps 10 lstm steps into one function, and the whole loop
      # into another calling the formers.

      # Groups 10 steps at a time.
      @function.Defun(dtypes.float32, dtypes.float32, dtypes.float32,
                      *([dtypes.float32] * 10))
      def Loop10(w, m, c, *args):
        for x in args:
          m, c = cell(x, m, c, w)
        return m, c

      @function.Defun(dtypes.float32, dtypes.float32)
      def LSTMLoop10(weights, inp):
        x = array_ops_stack.unstack(inp, self.NUM_UNROLL)
        m = array_ops.zeros_like(x[0])
        c = array_ops.zeros_like(x[0])
        assert self.NUM_UNROLL % 10 == 0
        for i in range(0, self.NUM_UNROLL, 10):
          m, c = Loop10(weights, m, c, *x[i:i + 10])
        return m

      return LSTMLoop10(weights, inp)

  def testUnrollLSTM(self):
    # Run one step of the unrolled lstm graph.
    def RunForward(mode, cfg=None):
      tf_logging.info("mode = %s", mode)
      g = ops.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
      gdef = g.as_graph_def()
      finish = time.time()
      tf_logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), session.Session(config=cfg) as sess:
        return self.evaluate(m)

    mv0 = RunForward("complete")
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      mv1 = RunForward("cell", cfg)
      mv2 = RunForward("loop", cfg)
      mv3 = RunForward("loop10", cfg)
      self.assertAllClose(mv0, mv1, rtol=1e-4)
      self.assertAllClose(mv0, mv2, rtol=1e-4)
      self.assertAllClose(mv0, mv3, rtol=1e-4)

  def testUnrollLSTMGrad(self):
    # Run one step of the unrolled lstm graph.
    def RunForwardBackward(mode, cfg=None):
      tf_logging.info("mode = %s", mode)
      g = ops.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
        loss = math_ops.reduce_sum(math_ops.square(m))
        dw = gradients_impl.gradients([loss], [weights])
      gdef = g.as_graph_def()
      finish = time.time()
      tf_logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), session.Session(config=cfg) as sess:
        return self.evaluate(dw)

    d0 = RunForwardBackward("complete")
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      d1 = RunForwardBackward("cell", cfg)
      d2 = RunForwardBackward("loop", cfg)
      d3 = RunForwardBackward("loop10", cfg)
      self.assertAllClose(d0, d1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d3, rtol=1e-4, atol=1e-4)


class FunctionInlineControlTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters((True), (False))
  @test_util.disable_xla("XLA changes the names, breaking graph analysis")
  def testFoo(self, noinline):
    dtype = dtypes.float32
    cfg = config_pb2.ConfigProto(
        graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=True,
                do_function_inlining=True,
                do_constant_folding=True)))
    cell_func_call_pattern = re.compile(r"Cell[^/]*\(")
    @function.Defun(dtype, noinline=noinline)
    def Cell(v):
      # If v is a vector [n, 1], x is a big square matrix.
      x = math_ops.tanh(v + array_ops.transpose(v, [1, 0]))
      return math_ops.reduce_sum(x, 1, keepdims=True)

    @function.Defun(dtype)
    def Forward(x):
      for _ in range(10):
        # pylint: disable=cell-var-from-loop
        x = Cell(x)
      return math_ops.reduce_sum(x, [0, 1])

    # Disabling this check on the ROCm platform, because it fails
    # The failure might not be ROCm specific(see commit message for details)
    if not test.is_built_with_rocm():
      self.assertEqual(noinline, Cell.definition.attr["_noinline"].b)

    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype)
      y = Forward(x)
      dx, = gradients_impl.gradients([y], [x])

    np.random.seed(321)
    inp = np.random.uniform(-1, 1, [16, 1]).astype(np.float32)
    run_metadata = config_pb2.RunMetadata()
    with session.Session(graph=g, config=cfg) as sess:
      ans = sess.run(
          [y, dx], {x: inp},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      self.assertAllClose(ans[0], 255.971, rtol=1e-3)
      self.assertAllClose(np.sum(ans[1]), 13.0408, rtol=1e-3)

    def MetadataHasCell(run_metadata):
      for dev_stats in run_metadata.step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
          if cell_func_call_pattern.search(node_stats.timeline_label):
            return True
      return False

    self.assertEqual(MetadataHasCell(run_metadata), noinline)


class ModuleFunctionTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasic(self):

    @function.Defun(*[dtypes.float32] * 3)
    def LinearWithCApi(w, b, x):
      return nn_ops.relu(math_ops.matmul(x, w) + b)

    @function.Defun(*[dtypes.float32] * 5)
    def Linear2WithCApi(w1, b1, w2, b2, x):
      return LinearWithCApi(w2, b2, LinearWithCApi(w1, b1, x))

    with ops.Graph().as_default():
      a, b, c, d, e = [
          constant_op.constant([[_]], dtype=dtypes.float32) for _ in range(5)
      ]
      y = LinearWithCApi(a, b, c)
      z = Linear2WithCApi(a, b, c, d, e)
      with session.Session() as sess:
        self.assertAllEqual([[1]], self.evaluate(y))
        self.assertAllEqual([[5]], self.evaluate(z))


class VariableHoistingTest(test.TestCase):

  def _testSimpleModel(self, use_forward_func, use_resource=False):

    def _Model(x):
      w = variable_scope.get_variable(
          "w", (64, 64),
          initializer=init_ops.random_uniform_initializer(seed=312),
          use_resource=use_resource)
      b = variable_scope.get_variable(
          "b", (64),
          initializer=init_ops.zeros_initializer(),
          use_resource=use_resource),
      return math_ops.sigmoid(math_ops.matmul(x, w) + b)

    @function.Defun()
    def Model(x):
      return _Model(x)

    cvars = []

    @function.Defun()
    def Grad(x, y0):
      if use_forward_func:
        y = Model(x)
      else:
        y = _Model(x)
      loss = math_ops.reduce_mean(
          math_ops.reduce_sum(y0 * math_ops.log(y), 1), 0)
      arg_w, arg_b = function.get_extra_args()
      self.assertEqual(arg_w.get_shape(), tensor_shape.TensorShape([64, 64]))
      self.assertEqual(arg_b.get_shape(), tensor_shape.TensorShape([64]))
      dw, db = gradients_impl.gradients(loss, [arg_w, arg_b])
      cvars.extend(function.get_extra_vars())
      return loss, dw, db

    g = ops.Graph()
    with g.as_default():
      x = random_ops.random_normal([64, 64], seed=100)
      y0 = random_ops.random_normal([64, 64], seed=200)
      with variable_scope.variable_scope("Foo"):
        loss, dw, db = Grad(x, y0)

    self.assertEqual(2, len(cvars))
    w, b = cvars[:2]
    self.assertEqual("Foo/w", w.op.name)
    self.assertEqual("Foo/b", b.op.name)

    with self.session(graph=g) as sess:
      self.evaluate(variables.global_variables_initializer())
      w, b, x, y0, loss, dw, db = self.evaluate([w, b, x, y0, loss, dw, db])

    self.assertAllEqual(w.shape, (64, 64))
    self.assertAllClose(np.sum(w), 2050.44)
    self.assertAllEqual(b.shape, (64,))
    self.assertAllClose(np.sum(b), 0.0)
    self.assertAllClose(loss, -2.27, rtol=1e-2)
    self.assertAllEqual(dw.shape, (64, 64))
    self.assertAllClose(np.sum(dw), -1.04, rtol=1e-2)
    self.assertAllEqual(db.shape, (64,))
    self.assertAllClose(np.sum(db), 0.509, rtol=1e-2)

  @test_util.run_deprecated_v1
  def testBasic(self):
    self._testSimpleModel(False)
    self._testSimpleModel(True)

  @test_util.run_deprecated_v1
  def testBasicResource(self):
    self._testSimpleModel(False, use_resource=True)
    self._testSimpleModel(True, use_resource=True)


class TemplateTest(test.TestCase):

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasic(self):
    self.assertTemplateVariableSharing(use_resource=True, defun_first=False)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicRef(self):
    self.assertTemplateVariableSharing(use_resource=False, defun_first=False)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicDefunFirst(self):
    self.assertTemplateVariableSharing(use_resource=True, defun_first=True)

  @test_util.run_v1_only("make_template not supported in TF2")
  def testBasicRefDefunFirst(self):
    self.assertTemplateVariableSharing(use_resource=False, defun_first=True)

  def assertTemplateVariableSharing(self, use_resource, defun_first):
    parameters = []

    def MakeModel(x):
      w = variable_scope.get_variable(
          "w", (64, 64),
          initializer=init_ops.random_uniform_initializer(seed=312),
          use_resource=use_resource)
      b = variable_scope.get_variable(
          "b", (64),
          initializer=init_ops.zeros_initializer(),
          use_resource=use_resource)
      parameters.extend((w, b))
      return math_ops.sigmoid(math_ops.matmul(x, w) + b)

    model = template.make_template("f", MakeModel, create_scope_now_=True)

    @function.Defun()
    def ModelDefun(x):
      return model(x)

    x = array_ops.placeholder(dtypes.float32)
    if defun_first:
      ModelDefun(x)
      model(x)
    else:
      model(x)
      ModelDefun(x)
    w1, b1, w2, b2 = parameters  # pylint: disable=unbalanced-tuple-unpacking
    self.assertIs(w1, w2)
    self.assertIs(b1, b2)


class DevicePlacementTest(test.TestCase):

  def testNoDeviceGraph(self):
    with ops.Graph().as_default():

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      Matmul(1., 2.)

      gdef = ops.get_default_graph().as_graph_def()
      self.assertAllEqual(len(gdef.library.function), 1)
      fdef = gdef.library.function[0]

      for node in fdef.node_def:
        self.assertAllEqual(node.device, "")

  def testNestedDevices(self):
    with ops.Graph().as_default(), ops.device("CPU:0"):

      @function.Defun(*[dtypes.float32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      with ops.device("CPU:1"):

        @function.Defun(*[dtypes.float32] * 2)
        def Divide(a, b):
          return math_ops.divide(a, b)

        Divide(Matmul(1., 2.), 3.)

      gdef = ops.get_default_graph().as_graph_def()
      matmul_fdef = [
          f for f in gdef.library.function if "Matmul" in f.signature.name
      ]
      divide_fdef = [
          f for f in gdef.library.function if "Divide" in f.signature.name
      ]
      self.assertAllEqual(len(matmul_fdef), 1)
      self.assertAllEqual(len(divide_fdef), 1)
      for node in matmul_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:0")
      for node in divide_fdef[0].node_def:
        self.assertAllEqual(node.device, "/device:CPU:1")

  def _testNestedDeviceWithSameFunction(self, func_name):

    def MatmulWrap(a, b):

      @function.Defun(
          func_name=func_name, *[dtypes.int32] * 2)
      def Matmul(a, b):
        return math_ops.matmul(a, b)

      return Matmul(a, b)

    with ops.Graph().as_default(), ops.device("CPU:0"):
      c = MatmulWrap(1, 2)

      with ops.device("CPU:1"):
        MatmulWrap(c, 3)

      gdef = ops.get_default_graph().as_graph_def()

      devices = []
      for node in gdef.library.function[0].node_def:
        devices.append(node.device)
      for node in gdef.library.function[1].node_def:
        devices.append(node.device)

      self.assertAllEqual(sorted(devices), ["/device:CPU:0", "/device:CPU:1"])

  def testFunctionWithName(self):
    with self.assertRaises(InvalidArgumentError) as cm:
      self._testNestedDeviceWithSameFunction("MatmulTest")
    self.assertEqual(
        cm.exception.message,
        "Cannot add function \'MatmulTest\' because a different "
        "function with the same name already exists.")

  def testFunctionWithoutName(self):
    self._testNestedDeviceWithSameFunction(None)


if __name__ == "__main__":
  test.main()
