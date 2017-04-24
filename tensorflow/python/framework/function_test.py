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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time

import numpy as np

from tensorflow.core.framework import function_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _OptimizerOptions():
  for cse in [False, True]:
    for inline in [False, True]:
      for cfold in [False, True]:
        yield config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                opt_level=config_pb2.OptimizerOptions.L0,
                do_common_subexpression_elimination=cse,
                do_function_inlining=inline,
                do_constant_folding=cfold)))


class FunctionTest(test.TestCase):

  def testDefineFunction2Args(self):

    @function.Defun(dtypes.float32, dtypes.float32, func_name="APlus2B")
    def APlus2B(a, b):
      return a + b * 2

    with ops.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([5.0], sess.run(call))

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
      dx = functional_ops._symbolic_gradient(
          input=[x, dy], Tout=[dtypes.float32], f="XSquarePlusOneFn", name="dx")
      return dx

    g = ops.Graph()
    with g.as_default():
      call_f = XSquarePlusOne([2.0])
      call_g = XSquarePlusOneGrad([2.0], [0.1])

      with session.Session() as sess:
        self.assertAllClose([5.0], sess.run(call_f))
        self.assertAllClose([0.4], sess.run(call_g))

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
    cfg = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
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
      self.assertAllClose(out, np.exp(prob - y))

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
      with self.assertRaisesRegexp(
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
      dx, dy = functional_ops._symbolic_gradient(
          input=[x, y, dz], Tout=[dtypes.float32] * 2, f="Foo")
      self.assertEqual(x.get_shape(), dx.get_shape())
      self.assertEqual(y.get_shape(), dy.get_shape())

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

    cfg = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(
            opt_level=config_pb2.OptimizerOptions.L0,
            do_common_subexpression_elimination=True,
            do_function_inlining=True,
            do_constant_folding=True)))

    with self.test_session(graph=g, config=cfg):
      self.assertAllClose(y.eval(), 6.)
      self.assertAllClose(dx.eval(), 2.)

  def testZNoDepOnY(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def Foo(x, y):  # pylint: disable=unused-argument
      return x * 2

    with ops.Graph().as_default():
      # z = Foo(x, y). z doe
      x = constant_op.constant(1.0)
      y = constant_op.constant(2.0)
      z = Foo(x, y)
      dx, dy = gradients_impl.gradients([z], [x, y])
      with session.Session() as sess:
        dx_val, dy_val = sess.run([dx, dy])
        self.assertEqual([2.0], dx_val)
        self.assertEqual([0.0], dy_val)

  def testDefineFunctionNoArgs(self):

    @function.Defun(func_name="AConstant")
    def AConstant():
      return constant_op.constant([42])

    with ops.Graph().as_default():

      call = AConstant()
      self.assertEqual("AConstant", call.op.name)
      with session.Session() as sess:
        self.assertAllEqual([42], sess.run(call))

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
      y = logging_ops.Print(x, [x], "Hello")
      with ops.control_dependencies([y]):
        z = control_flow_ops.no_op()
      with ops.control_dependencies([z]):
        return x * 2

    with ops.Graph().as_default(), self.test_session():
      z = Foo(constant_op.constant(3.0))
      self.assertAllEqual(z.eval(), 6.0)

  def testAssertOp(self):

    @function.Defun(dtypes.float32)
    def Foo(x):
      check = gen_logging_ops._assert(math_ops.greater(x, 0), [x])
      with ops.control_dependencies([check]):
        return x * 2

    g = ops.Graph()
    with g.as_default(), self.test_session():
      self.assertAllEqual(Foo(constant_op.constant(3.0)).eval(), 6.0)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "assertion failed.*-3"):
        self.assertAllEqual(Foo(constant_op.constant(-3.0)).eval(), 6.0)

  def testAssertWrapper(self):

    @function.Defun(dtypes.float32)
    def MyFn(x):
      with ops.control_dependencies(
          [control_flow_ops.Assert(math_ops.less_equal(x, 10.0), [x])]):
        return array_ops.identity(x)

    with self.test_session():
      self.assertEqual(1.0, MyFn(1.0).eval())
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "assertion"):
        _ = MyFn(100.0).eval()

  def testVar(self):

    @function.Defun(dtypes.float32)
    def Foo(x):
      return x * x + 1

    g = ops.Graph()
    with g.as_default():
      v = variables.Variable(constant_op.constant(10.0))
      z = Foo(v)

    with self.test_session(graph=g):
      variables.global_variables_initializer().run()
      self.assertAllEqual(z.eval(), 101.)

  def testResourceVarAsImplicitInput(self):
    g = ops.Graph()
    with g.as_default():
      v = variable_scope.get_variable(
          "var", (4, 4), dtypes.float32, use_resource=True)

      @function.Defun()
      def Foo():
        return array_ops.identity(v)

      y = v.value()
      z = Foo()

    with self.test_session(graph=g):
      v.initializer.run()
      self.assertAllEqual(y.eval(), z.eval())

  def testDefineErrors(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "can not return None"):

        @function.Defun()
        def NoResult():
          pass

        _ = NoResult.definition

      with self.assertRaisesRegexp(ValueError, "can not return None"):

        @function.Defun()
        def TwoNone():
          return None, None

        _ = TwoNone.definition

      with self.assertRaisesRegexp(ValueError, "are not supported"):

        @function.Defun()
        def DefaultArg(unused_a=12):
          return constant_op.constant([1])

        _ = DefaultArg.definition
      with self.assertRaisesRegexp(ValueError, "are not supported"):

        @function.Defun()
        def KwArgs(**unused_kwargs):
          return constant_op.constant([1])

        _ = KwArgs.definition
      with self.assertRaisesRegexp(ValueError, "specified input types"):

        @function.Defun(dtypes.float32)
        def PlusMinusV2(a, b):
          return a + b, b - a

        _ = PlusMinusV2.definition
      with self.assertRaisesRegexp(ValueError, "specified input types"):

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
      with self.assertRaisesRegexp(ValueError, "arguments: 0"):
        _ = Const(1)
      with self.assertRaisesRegexp(ValueError, "arguments: 0"):
        _ = Const(1, 2)

      with self.assertRaisesRegexp(ValueError, "arguments: 1"):
        _ = PlusOne()
      _ = PlusOne(1)
      with self.assertRaisesRegexp(ValueError, "arguments: 1"):
        _ = PlusOne(1, 2)

      with self.assertRaisesRegexp(ValueError, "arguments: 2"):
        _ = PlusMinus()
      with self.assertRaisesRegexp(ValueError, "arguments: 2"):
        _ = PlusMinus(1)
      _ = PlusMinus(1, 2)

      _ = PlusOne(1, name="p1")
      with self.assertRaisesRegexp(ValueError, "Unknown keyword arguments"):
        _ = PlusOne(1, device="/gpu:0")

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
        self.assertAllEqual([1], sess.run(call1))
        self.assertAllEqual([0], sess.run(call2))

  def testNestedFunction(self):

    @function.Defun(dtypes.float32)
    def Cube(x):
      return x * x * x

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):
      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.test_session():
        self.assertAllEqual(z.eval(), 25.0)

  def testNestedDefinedFunction(self):

    @function.Defun(dtypes.float32, dtypes.float32)
    def CubeXPlusY(x, y):

      @function.Defun(dtypes.float32)
      def Cube(x):
        return x * x * x

      return Cube(x) + y

    with ops.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.test_session():
        self.assertAllEqual(z.eval(), 25.0)

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
    with self.test_session(graph=g) as sess:
      vals = sess.run([y0, y1, dx0, dx1], {x: np.random.uniform(size=(3, 7))})
      self.assertAllClose(vals[0], vals[1])
      self.assertAllClose(vals[2], vals[3])

  def testDeclare(self):
    foo = function.Declare("Foo", [("x", dtypes.float32)],
                           [("y", dtypes.float32)])

    @function.Defun(dtypes.float32, func_name="Foo", out_names=["y"])
    def FooImpl(x):
      return x * x + 1

    x = array_ops.placeholder(dtypes.float32)
    y = foo(x)

    g = ops.get_default_graph()
    FooImpl.add_to_graph(g)

    with self.test_session():
      rand = np.random.uniform(size=(3, 3))
      expected = rand * rand + 1.0
      self.assertAllClose(expected, y.eval(feed_dict={x: rand}))

  def testDeclareUsedInDefun(self):
    foo = function.Declare("Foo", [("x", dtypes.float32)],
                           [("y", dtypes.float32)])

    @function.Defun()
    def Bar(x):
      return foo(x)

    @function.Defun(dtypes.float32, func_name="Foo", out_names=["y"])
    def FooImpl(x):
      return x * x + 1

    x = array_ops.placeholder(dtypes.float32)
    y = Bar(x)

    g = ops.get_default_graph()
    FooImpl.add_to_graph(g)

    with self.test_session():
      rand = np.random.uniform(size=(3, 3))
      expected = rand * rand + 1.0
      self.assertAllClose(expected, y.eval(feed_dict={x: rand}))

  def testDeclareTypeMistake(self):
    foo = function.Declare("Foo", [("x", dtypes.float32)],
                           [("y", dtypes.float32)])

    @function.Defun(dtypes.float32, func_name="Foo", out_names=["y"])
    def Foo(x):
      return x * x + 1

    g = ops.Graph()
    with g.as_default():
      y = foo(2.0)
      with self.test_session(graph=g):
        with self.assertRaisesRegexp(errors_impl.NotFoundError,
                                     "not registered"):
          _ = y.eval()

    g = ops.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      y = foo(2)
      with self.test_session(graph=g):
        with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                     "int32.*float"):
          _ = y.eval()

    g = ops.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      with self.assertRaisesRegexp(
          ValueError, "Expected number of arguments: 1, received: 2"):
        _ = foo(2.0, 2.0)

    g = ops.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      y = foo(2.0)
      with self.test_session(graph=g):
        self.assertAllEqual(y.eval(), 5.0)

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

    with self.test_session(graph=g):
      variables.global_variables_initializer().run()
      self.assertAllEqual(y.eval(), [[12.0]])

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

      with self.assertRaisesRegexp(ValueError, "not an element of this graph."):
        # NOTE: We still do not support capturing control deps.
        _ = Foo(x)

  def testStableName(self):

    @function.Defun()
    def Foo(x, y, z):
      return math_ops.tanh(math_ops.matmul(x, y) + z)

    self.assertEqual("Foo_d643acf7", Foo.instantiate([dtypes.float32] * 3).name)

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

    with self.test_session(graph=g) as sess:
      v0, v1 = sess.run([y, z])
      self.assertAllEqual(v0, 20.)
      self.assertAllEqual(v1, 20.)

  def testShapeFunction(self):
    @function.Defun(dtypes.float32,
                    shape_func=lambda op: [op.inputs[0].get_shape()])
    def Foo(x):
      return x + 1.0

    @function.Defun(
        shape_func=lambda op: [[1] + op.inputs[0].get_shape().as_list()])
    def Bar(x):
      return array_ops.stack([x])

    g = ops.Graph()
    with g.as_default():
      x = Foo([1.0, 2.0])
      self.assertEqual(x.get_shape().as_list(), [2])
      y = Bar(array_ops.zeros([1, 2, 3]))
      self.assertAllEqual(y.get_shape().as_list(), [1, 1, 2, 3])

  def testVariableReuse(self):
    def LinearWithReuse(input_tensor, reuse=None):
      size = input_tensor.shape.dims[1]
      with variable_scope.variable_scope("linear", reuse=reuse):
        w = variable_scope.get_variable("w", shape=[size, size],
                                        dtype=input_tensor.dtype)
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
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(output_op,
                            feed_dict={input_op: np.random.rand(32, 100)})
      self.assertEqual(output_val.shape, (32, 100))

  def testFunctionCallInDifferentVariableScopes(self):
    @function.Defun(dtypes.float32)
    def Foo(inputs):
      var = variable_scope.get_variable("var", shape=[10], dtype=dtypes.float32,
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
      sess.run(variables.global_variables_initializer())
      out1, out2 = sess.run([out1_op, out2_op],
                            feed_dict={input_op: np.linspace(1, 10, 10)})
      self.assertAllEqual(out1, np.linspace(2, 11, 10))
      self.assertAllEqual(out2, np.linspace(2, 11, 10))


class FunctionsFromProtos(test.TestCase):

  def expectFunctionsEqual(self, func, grad_func=None, new_func=None):
    if new_func is None:
      # Make a copy of func.definition to avoid any bugs masked by using the
      # same object
      serialized_fdef = func.definition.SerializeToString()
      # Serialize and then deserialize `func` to create `new_func`
      fdef = function_pb2.FunctionDef.FromString(serialized_fdef)
      new_func = function._from_definition(fdef, grad_func=grad_func)
    self.assertEqual(func.name, new_func.name)
    self.assertEqual(func.definition, new_func.definition)
    self.assertEqual(func.grad_func_name, new_func.grad_func_name)
    self.assertEqual(func.declared_input_types, new_func.declared_input_types)
    self.assertEqual(func.captured_inputs, new_func.captured_inputs)

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
    self.assertEqual(Foo.definition, new_func.definition)
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
    new_funcs = function._from_library(library)

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
    self.assertEqual(len(function._from_library(library)), 0)

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

    with self.assertRaisesRegexp(
        ValueError, "FunctionDefLibrary missing 'G1_........' FunctionDef"):
      function._from_library(library)

    # Create invalid function def that is missing F1 function def
    library = function_pb2.FunctionDefLibrary()
    library.gradient.extend([gradient])
    library.function.extend([G1.definition])

    with self.assertRaisesRegexp(
        ValueError, "FunctionDefLibrary missing 'F1_........' FunctionDef"):
      function._from_library(library)

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

    with self.assertRaisesRegexp(
        ValueError, "FunctionDefLibrary contains cyclic gradient functions!"):
      function._from_library(library)


class FunctionOverloadTest(test.TestCase):

  def testBasic(self):

    @function.Defun()
    def Sinh(x):
      return 1 / 2. * (math_ops.exp(x) - math_ops.exp(-x))

    g = ops.Graph()
    with g.as_default():
      x = Sinh(constant_op.constant(0.25, dtypes.float32))
      y = Sinh(constant_op.constant(0.25, dtypes.float64))

    with self.test_session(graph=g):
      self.assertAllClose(x.eval(), np.sinh(0.25))
      self.assertAllClose(y.eval(), np.sinh(0.25))

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

        with self.test_session(graph=g):
          self.assertAllClose(dx.eval(), 0.25)

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
    new_c = clip_ops.clip_by_value(new_c, -50.0, 50.0)
    new_m = math_ops.sigmoid(o_g) * math_ops.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):

    def Loop(cell, w, i):
      x = array_ops.unstack(i, self.NUM_UNROLL)
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
                          dtypes.float32)(cell)
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
        x = array_ops.unstack(inp, self.NUM_UNROLL)
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
        return sess.run(m)

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
        return sess.run(dw)

    d0 = RunForwardBackward("complete")
    for cfg in _OptimizerOptions():
      tf_logging.info("cfg = %s", cfg)
      d1 = RunForwardBackward("cell", cfg)
      d2 = RunForwardBackward("loop", cfg)
      d3 = RunForwardBackward("loop10", cfg)
      self.assertAllClose(d0, d1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(d0, d3, rtol=1e-4, atol=1e-4)


class FunctionInlineControlTest(test.TestCase):

  def testFoo(self):
    dtype = dtypes.float32
    cfg = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(
            opt_level=config_pb2.OptimizerOptions.L0,
            do_common_subexpression_elimination=True,
            do_function_inlining=True,
            do_constant_folding=True)))
    cell_func_call_pattern = re.compile(r"Cell[^/]*\(")
    for noinline in [False, True]:

      @function.Defun(dtype, noinline=noinline)
      def Cell(v):
        # If v is a vector [n, 1], x is a big square matrix.
        x = math_ops.tanh(v + array_ops.transpose(v, [1, 0]))
        return math_ops.reduce_sum(x, 1, keep_dims=True)

      @function.Defun(dtype)
      def Forward(x):
        for _ in range(10):
          # pylint: disable=cell-var-from-loop
          x = Cell(x)
        return math_ops.reduce_sum(x, [0, 1])

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
        ans = sess.run([y, dx], {x: inp},
                       run_metadata=run_metadata,
                       options=config_pb2.RunOptions(
                           trace_level=config_pb2.RunOptions.FULL_TRACE))
        print(ans[0], np.sum(ans[1]))
        self.assertAllClose(ans[0], 255.971, rtol=1e-3)
        self.assertAllClose(np.sum(ans[1]), 13.0408, rtol=1e-3)

      def MetadataHasCell(run_metadata):
        for dev_stats in run_metadata.step_stats.dev_stats:
          for node_stats in dev_stats.node_stats:
            if cell_func_call_pattern.search(node_stats.timeline_label):
              return True
        return False

      self.assertEqual(MetadataHasCell(run_metadata), noinline)


@function.Defun(*[dtypes.float32] * 3)
def Linear(w, b, x):
  return nn_ops.relu(math_ops.matmul(x, w) + b)


@function.Defun(*[dtypes.float32] * 5)
def Linear2(w1, b1, w2, b2, x):
  return Linear(w2, b2, Linear(w1, b1, x))


class ModuleFunctionTest(test.TestCase):

  def testBasic(self):
    with ops.Graph().as_default():
      a, b, c, d, e = [
          constant_op.constant(
              [[_]], dtype=dtypes.float32) for _ in range(5)
      ]
      y = Linear(a, b, c)
      z = Linear2(a, b, c, d, e)
      with session.Session() as sess:
        self.assertAllEqual([[1]], sess.run(y))
        self.assertAllEqual([[5]], sess.run(z))


class VariableHoistingTest(test.TestCase):

  def _testSimpleModel(self, use_forward_func, use_resource=False):

    def _Model(x):
      w = variable_scope.get_variable(
          "w", (64, 64),
          initializer=init_ops.random_uniform_initializer(seed=312),
          use_resource=use_resource)
      b = variable_scope.get_variable(
          "b", (64), initializer=init_ops.zeros_initializer(),
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

    with self.test_session(graph=g) as sess:
      sess.run(variables.global_variables_initializer())
      w, b, x, y0, loss, dw, db = sess.run([w, b, x, y0, loss, dw, db])

    self.assertAllEqual(w.shape, (64, 64))
    self.assertAllClose(np.sum(w), 2050.44)
    self.assertAllEqual(b.shape, (64,))
    self.assertAllClose(np.sum(b), 0.0)
    self.assertAllClose(loss, -2.27, rtol=1e-2)
    self.assertAllEqual(dw.shape, (64, 64))
    self.assertAllClose(np.sum(dw), -1.04, rtol=1e-2)
    self.assertAllEqual(db.shape, (64,))
    self.assertAllClose(np.sum(db), 0.509, rtol=1e-2)

  def testBasic(self):
    self._testSimpleModel(True)
    self._testSimpleModel(False)

  def testBasicResource(self):
    self._testSimpleModel(True, use_resource=True)
    self._testSimpleModel(False, use_resource=True)

if __name__ == "__main__":
  test.main()
