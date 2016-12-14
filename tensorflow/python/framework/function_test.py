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

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_logging_ops


def _OptimizerOptions():
  for cse in [False, True]:
    for inline in [False, True]:
      for cfold in [False, True]:
        yield tf.ConfigProto(graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                opt_level=tf.OptimizerOptions.L0,
                do_common_subexpression_elimination=cse,
                do_function_inlining=inline,
                do_constant_folding=cfold)))


class FunctionTest(tf.test.TestCase):

  def testDefineFunction2Args(self):

    @function.Defun(tf.float32, tf.float32, func_name="APlus2B")
    def APlus2B(a, b):
      return a + b * 2

    with tf.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEqual("APlus2B", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([5.0], sess.run(call))

  def testDefineFunctionDuplicateOutputs(self):

    @function.Defun(tf.float32, func_name="Duplicate")
    def Duplicate(a):
      b = a + 1.0
      return b, b

    g = tf.Graph()
    with g.as_default():
      Duplicate([3.0])
      func_sig = g.as_graph_def().library.function[0].signature
      # The names given to both outputs should be different
      # even though the same tensor is emitted to both.
      out_names = [a.name for a in func_sig.output_arg]
      self.assertEqual(2, len(out_names))
      self.assertNotEqual(out_names[0], out_names[1])

  def testGradientFunc(self):

    @function.Defun(tf.float32, func_name="XSquarePlusOneFn")
    def XSquarePlusOne(x):
      return x * x + 1.0

    @function.Defun(tf.float32, tf.float32)
    def XSquarePlusOneGrad(x, dy):
      dx = functional_ops._symbolic_gradient(
          input=[x, dy], Tout=[tf.float32], f="XSquarePlusOneFn", name="dx")
      return dx

    g = tf.Graph()
    with g.as_default():
      call_f = XSquarePlusOne([2.0])
      call_g = XSquarePlusOneGrad([2.0], [0.1])

      with tf.Session() as sess:
        self.assertAllClose([5.0], sess.run(call_f))
        self.assertAllClose([0.4], sess.run(call_g))

  def testTanhSymGrad(self):

    @function.Defun(tf.float32)
    def Forward(x):
      return tf.reduce_sum(tf.tanh(x))

    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32)
      y = Forward(x)
      dx = tf.gradients([y], [x])

    inp = np.array([-1, 1, 2, -2], dtype=np.float32)
    feed = {x: inp}
    cfg = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L1, do_function_inlining=True)))
    with tf.Session(graph=g, config=cfg) as sess:
      out, = sess.run(dx, feed)
    self.assertAllClose(1 - np.square(np.tanh(inp)), out)

  def testCustomGradient(self):
    dtype = tf.float32

    @function.Defun(dtype, dtype, dtype)
    def XentLossGrad(logits, labels, dloss):
      dlogits = tf.reshape(dloss, [-1, 1]) * (tf.nn.softmax(logits) - labels)
      dlabels = tf.zeros_like(labels)
      # Takes exp(dlogits) to differentiate it from the "correct" gradient.
      return tf.exp(dlogits), dlabels

    @function.Defun(dtype, dtype, grad_func=XentLossGrad)
    def XentLoss(logits, labels):
      return tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits)), 1)

    g = tf.Graph()
    with g.as_default():
      logits = tf.placeholder(dtype)
      labels = tf.placeholder(dtype)
      loss = XentLoss(logits, labels)
      dlogits = tf.gradients([loss], [logits])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    prob = np.exp(x) / np.sum(np.exp(x), 1, keepdims=1)
    y = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    for cfg in _OptimizerOptions():
      tf.logging.info("cfg = %s", cfg)
      with tf.Session(graph=g, config=cfg) as sess:
        out, = sess.run(dlogits, {logits: x, labels: y})
      self.assertAllClose(out, np.exp(prob - y))

  def testCustomGradientError(self):
    dtype = tf.float32

    @function.Defun(dtype, dtype, dtype)
    def Grad(x, dy, dz):
      # Should have returned 1 result.
      return x, dy + dz

    @function.Defun(dtype, grad_func=Grad)
    def Forward(x):
      return x, x

    g = tf.Graph()
    with g.as_default():
      inp = tf.placeholder(dtype)
      out = tf.add_n(Forward(inp))
      dinp = tf.gradients(out, [inp])

    x = np.random.uniform(-10., 10., size=(4, 9)).astype(np.float32)
    with tf.Session(graph=g) as sess:
      with self.assertRaisesRegexp(
          tf.errors.InvalidArgumentError,
          "SymGrad expects to return 1.*but get 2.*instead"):
        _ = sess.run(dinp, {inp: x})

  def testSymGradShape(self):
    g = tf.Graph()
    with g.as_default():
      x = tf.placeholder(tf.float32, [25, 4])
      y = tf.placeholder(tf.float32, [200, 100])
      dz = tf.placeholder(tf.float32, [1])
      # We assume Foo is a function of (x, y) -> (z) Then, Foo's
      # gradient function is (x, y, dz) -> (dx, dy).  dx's shape
      # should be the same as x's; and dy's shape should be the same
      # as y's.
      dx, dy = functional_ops._symbolic_gradient(
          input=[x, y, dz], Tout=[tf.float32] * 2, f="Foo")
      self.assertEqual(x.get_shape(), dx.get_shape())
      self.assertEqual(y.get_shape(), dy.get_shape())

  def testSymGradAttr(self):

    @function.Defun(noinline=True)
    def Foo(x):
      return x * 2

    self.assertTrue(
        Foo.instantiate([tf.float32]).definition.attr["_noinline"].b)

    g = tf.Graph()
    with g.as_default():
      x = tf.constant(3.0)
      y = Foo(x)
      dx, = tf.gradients(y, [x])

    cfg = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L0,
            do_common_subexpression_elimination=True,
            do_function_inlining=True,
            do_constant_folding=True)))

    with self.test_session(graph=g, config=cfg):
      self.assertAllClose(y.eval(), 6.)
      self.assertAllClose(dx.eval(), 2.)

  def testZNoDepOnY(self):

    @function.Defun(tf.float32, tf.float32)
    def Foo(x, y):  # pylint: disable=unused-argument
      return x * 2

    with tf.Graph().as_default():
      # z = Foo(x, y). z doe
      x = tf.constant(1.0)
      y = tf.constant(2.0)
      z = Foo(x, y)
      dx, dy = tf.gradients([z], [x, y])
      with tf.Session() as sess:
        dx_val, dy_val = sess.run([dx, dy])
        self.assertEqual([2.0], dx_val)
        self.assertEqual([0.0], dy_val)

  def testDefineFunctionNoArgs(self):

    @function.Defun(func_name="AConstant")
    def AConstant():
      return tf.constant([42])

    with tf.Graph().as_default():

      call = AConstant()
      self.assertEqual("AConstant", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([42], sess.run(call))

  def testDefineFunctionNames(self):

    @function.Defun(tf.float32, func_name="Foo")
    def Foo(a):
      return a + 1

    with tf.Graph().as_default():
      call1 = Foo([1.0])
      self.assertEqual("Foo", call1.op.name)
      call2 = Foo([1.0])
      self.assertEqual("Foo_1", call2.op.name)
      # pylint: disable=unexpected-keyword-arg
      call3 = Foo([1.0], name="mine")
      self.assertEqual("mine", call3.op.name)
      with tf.name_scope("my"):
        call4 = Foo([1.0], name="precious")
        self.assertEqual("my/precious", call4.op.name)

  def testNoOp(self):

    @function.Defun(tf.float32)
    def Foo(x):
      y = tf.Print(x, [x], "Hello")
      with tf.control_dependencies([y]):
        z = tf.no_op()
      with tf.control_dependencies([z]):
        return x * 2

    with tf.Graph().as_default(), self.test_session():
      z = Foo(tf.constant(3.0))
      self.assertAllEqual(z.eval(), 6.0)

  def testAssertOp(self):

    @function.Defun(tf.float32)
    def Foo(x):
      check = gen_logging_ops._assert(tf.greater(x, 0), [x])
      with tf.control_dependencies([check]):
        return x * 2

    g = tf.Graph()
    with g.as_default(), self.test_session():
      self.assertAllEqual(Foo(tf.constant(3.0)).eval(), 6.0)
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   "assertion failed.*-3"):
        self.assertAllEqual(Foo(tf.constant(-3.0)).eval(), 6.0)

  def testAssertWrapper(self):

    @function.Defun(tf.float32)
    def MyFn(x):
      with tf.control_dependencies([tf.Assert(tf.less_equal(x, 10.0), [x])]):
        return tf.identity(x)

    with self.test_session():
      self.assertEqual(1.0, MyFn(1.0).eval())
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "assertion"):
        _ = MyFn(100.0).eval()

  def testVar(self):

    @function.Defun(tf.float32)
    def Foo(x):
      return x * x + 1

    g = tf.Graph()
    with g.as_default():
      v = tf.Variable(tf.constant(10.0))
      z = Foo(v)

    with self.test_session(graph=g):
      tf.global_variables_initializer().run()
      self.assertAllEqual(z.eval(), 101.)

  def testDefineErrors(self):
    with tf.Graph().as_default():
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
          return tf.constant([1])

        _ = DefaultArg.definition
      with self.assertRaisesRegexp(ValueError, "are not supported"):

        @function.Defun()
        def KwArgs(**unused_kwargs):
          return tf.constant([1])

        _ = KwArgs.definition
      with self.assertRaisesRegexp(ValueError, "specified input types"):

        @function.Defun(tf.float32)
        def PlusMinusV2(a, b):
          return a + b, b - a

        _ = PlusMinusV2.definition
      with self.assertRaisesRegexp(ValueError, "specified input types"):

        @function.Defun(tf.float32, tf.float32, tf.float32)
        def PlusMinusV3(a, b):
          return a + b, b - a

        _ = PlusMinusV3.definition

  def testCallErrors(self):

    @function.Defun()
    def Const():
      return tf.constant(1)

    @function.Defun(tf.int32)
    def PlusOne(a):
      return a + 1

    @function.Defun(tf.int32, tf.int32)
    def PlusMinus(a, b):
      return a + b, b - a

    with tf.Graph().as_default():

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

    @function.Defun(tf.float32, func_name="Minus1")
    def Minus1(b):
      return b - 1.0

    with tf.Graph().as_default():
      call1 = Minus1([2.])
      self.assertTrue(isinstance(Minus1, function._DefinedFunction))
      self.assertEqual(Minus1.name, "Minus1")
      # pylint: disable=unexpected-keyword-arg
      call2 = Minus1(call1, name="next")
      # pylint: enable=unexpected-keyword-arg
      self.assertEqual("next", call2.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([1], sess.run(call1))
        self.assertAllEqual([0], sess.run(call2))

  def testNestedFunction(self):

    @function.Defun(tf.float32)
    def Cube(x):
      return x * x * x

    @function.Defun(tf.float32, tf.float32)
    def CubeXPlusY(x, y):
      return Cube(x) + y

    with tf.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.test_session():
        self.assertAllEqual(z.eval(), 25.0)

  def testNestedDefinedFunction(self):

    @function.Defun(tf.float32, tf.float32)
    def CubeXPlusY(x, y):

      @function.Defun(tf.float32)
      def Cube(x):
        return x * x * x

      return Cube(x) + y

    with tf.Graph().as_default():
      z = CubeXPlusY(3.0, -2.0)
      with self.test_session():
        self.assertAllEqual(z.eval(), 25.0)

  def testUnusedFunction(self):
    invoked = False
    # pylint: disable=unused-variable
    @function.Defun()
    def Unused():
      invoked = True
      return tf.constant(42.)

    self.assertFalse(invoked)
    g = tf.Graph()
    with g.as_default():

      @function.Defun()
      def Unused2():
        invoked = True
        return tf.constant(7.)

      tf.constant(3.)
    # pylint: enable=unused-variable
    self.assertFalse(invoked)
    gdef = g.as_graph_def()
    self.assertEqual(0, len(gdef.library.function))

  def testReduction(self):
    g = tf.Graph()

    # BN0 is computing batch normed matrix along rows.
    def BN0(x):
      mean = tf.reduce_mean(x, [0])
      var = tf.reduce_mean(tf.square(x - mean))  # biased var
      rstd = tf.rsqrt(var + 1e-8)
      return (x - mean) * rstd

    # Wraps BatchNorm in a tf function.
    @function.Defun(tf.float32)
    def BN1(x):
      return BN0(x)

    with g.as_default():
      x = tf.placeholder(tf.float32)
      y0 = BN0(x)  # A plain graph
      y1 = BN1(x)  # A tf function
      dx0, = tf.gradients([y0], [x])
      dx1, = tf.gradients([y1], [x])

    # Both should produce the same result and gradient.
    with self.test_session(graph=g) as sess:
      vals = sess.run([y0, y1, dx0, dx1], {x: np.random.uniform(size=(3, 7))})
      self.assertAllClose(vals[0], vals[1])
      self.assertAllClose(vals[2], vals[3])

  def testDeclareTypeMistake(self):
    foo = function.Declare("Foo", [("x", tf.float32)], [("y", tf.float32)])

    @function.Defun(tf.float32, func_name="Foo", out_names=["y"])
    def Foo(x):
      return x * x + 1

    g = tf.Graph()
    with g.as_default():
      y = foo(2.0)
      with self.test_session(graph=g):
        with self.assertRaisesRegexp(tf.errors.NotFoundError, "not registered"):
          _ = y.eval()

    g = tf.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      y = foo(2)
      with self.test_session(graph=g):
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                     "int32.*float"):
          _ = y.eval()

    g = tf.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      with self.assertRaisesRegexp(
          ValueError, "Expected number of arguments: 1, received: 2"):
        _ = foo(2.0, 2.0)

    g = tf.Graph()
    with g.as_default():
      Foo.add_to_graph(g)
      y = foo(2.0)
      with self.test_session(graph=g):
        self.assertAllEqual(y.eval(), 5.0)

  def testCapture(self):
    g = tf.Graph()
    with g.as_default():
      w = tf.Variable(tf.constant([[1.0]]))
      b = tf.Variable(tf.constant([2.0]))

      # Foo() captures w and b.
      @function.Defun(tf.float32)
      def Foo(x):

        # Plus() captures b.
        @function.Defun(tf.float32)
        def Plus(y):
          return y + b

        return Plus(tf.matmul(w, x))

      y = Foo(tf.constant([[10.]]))

    with self.test_session(graph=g):
      tf.global_variables_initializer().run()
      self.assertAllEqual(y.eval(), [[12.0]])

  def testCaptureControls(self):
    g = tf.Graph()
    with g.as_default():
      x = tf.constant([10.0])
      x = tf.Print(x, [x], "outer")

      @function.Defun(tf.float32)
      def Foo(y):
        with tf.control_dependencies([x]):
          y = tf.Print(y, [y], "inner")
        return y

      with self.assertRaisesRegexp(ValueError, "not an element of this graph."):
        # NOTE: We still do not support capturing control deps.
        _ = Foo(x)

  def testStableName(self):

    @function.Defun()
    def Foo(x, y, z):
      return tf.tanh(tf.matmul(x, y) + z)

    self.assertEqual("Foo_d643acf7", Foo.instantiate([tf.float32] * 3).name)

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

    g = tf.Graph()
    with g.as_default():
      x = tf.constant(10.0)
      y = Foo(x)
      z = Bar(x)

    with self.test_session(graph=g) as sess:
      v0, v1 = sess.run([y, z])
      self.assertAllEqual(v0, 20.)
      self.assertAllEqual(v1, 20.)


class FunctionOverloadTest(tf.test.TestCase):

  def testBasic(self):

    @function.Defun()
    def Sinh(x):
      return 1 / 2. * (tf.exp(x) - tf.exp(-x))

    g = tf.Graph()
    with g.as_default():
      x = Sinh(tf.constant(0.25, tf.float32))
      y = Sinh(tf.constant(0.25, tf.float64))

    with self.test_session(graph=g):
      self.assertAllClose(x.eval(), np.sinh(0.25))
      self.assertAllClose(y.eval(), np.sinh(0.25))

  def testGradient(self):

    @function.Defun(func_name="Spec")
    def G(x, dy):
      return x * dy

    @function.Defun(grad_func=G)
    def F(x):
      return tf.exp(x) - tf.exp(-x)

    for dtype in [tf.float32, tf.float64]:
      g = tf.Graph()
      with g.as_default():
        x = tf.constant(0.25, dtype)
        y = F(x)
        dx, = tf.gradients(y, x)

        with self.test_session(graph=g):
          self.assertAllClose(dx.eval(), 0.25)

  def testDocString(self):

    @function.Defun()
    def Foo(x):
      """Successor of x."""
      return x + 1

    g = tf.Graph()
    with g.as_default():
      _ = Foo(1)

    self.assertEqual(g.as_graph_def().library.function[0].signature.description,
                     "Successor of x.")


class UnrollLSTMTest(tf.test.TestCase):
  BATCH_SIZE = 16
  LSTM_DIMS = 32
  NUM_UNROLL = 20

  def _Weights(self):
    dims = self.LSTM_DIMS
    return tf.random_uniform([2 * dims, 4 * dims], -1, 1, seed=123456)

  def _Input(self):
    return tf.random_uniform(
        [self.NUM_UNROLL, self.BATCH_SIZE, self.LSTM_DIMS], seed=654321)

  # Helper to construct a LSTM cell graph.
  @classmethod
  def LSTMCell(cls, x, mprev, cprev, weights):
    xm = tf.concat_v2([x, mprev], 1)
    i_i, i_g, f_g, o_g = tf.split(
        value=tf.matmul(xm, weights), num_or_size_splits=4, axis=1)
    new_c = tf.sigmoid(f_g) * cprev + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c = tf.clip_by_value(new_c, -50.0, 50.0)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):

    def Loop(cell, w, i):
      x = tf.unstack(i, self.NUM_UNROLL)
      m = tf.zeros_like(x[0])
      c = tf.zeros_like(x[0])
      for i in range(self.NUM_UNROLL):
        m, c = cell(x[i], m, c, w)
      return m

    cell = UnrollLSTMTest.LSTMCell
    if mode == "complete":
      # Constructs the complete graph in python.
      return Loop(cell, weights, inp)

    cell = function.Defun(tf.float32, tf.float32, tf.float32, tf.float32)(cell)
    if mode == "cell":
      # Just represent the LSTM as a function.
      return Loop(cell, weights, inp)

    if mode == "loop":
      # Wraps the whole loop as a function.
      @function.Defun(tf.float32, tf.float32)
      def LSTMLoop(w, i):
        return Loop(cell, w, i)

      return LSTMLoop(weights, inp)

    if mode == "loop10":
      # Wraps 10 lstm steps into one function, and the whole loop
      # into another calling the formers.

      # Groups 10 steps at a time.
      @function.Defun(tf.float32, tf.float32, tf.float32, *([tf.float32] * 10))
      def Loop10(w, m, c, *args):
        for x in args:
          m, c = cell(x, m, c, w)
        return m, c

      @function.Defun(tf.float32, tf.float32)
      def LSTMLoop10(weights, inp):
        x = tf.unstack(inp, self.NUM_UNROLL)
        m = tf.zeros_like(x[0])
        c = tf.zeros_like(x[0])
        assert self.NUM_UNROLL % 10 == 0
        for i in range(0, self.NUM_UNROLL, 10):
          m, c = Loop10(weights, m, c, *x[i:i + 10])
        return m

      return LSTMLoop10(weights, inp)

  def testUnrollLSTM(self):
    # Run one step of the unrolled lstm graph.
    def RunForward(mode, cfg=None):
      tf.logging.info("mode = %s", mode)
      g = tf.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
      gdef = g.as_graph_def()
      finish = time.time()
      tf.logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), tf.Session(config=cfg) as sess:
        return sess.run(m)

    mv0 = RunForward("complete")
    for cfg in _OptimizerOptions():
      tf.logging.info("cfg = %s", cfg)
      mv1 = RunForward("cell", cfg)
      mv2 = RunForward("loop", cfg)
      mv3 = RunForward("loop10", cfg)
      self.assertAllClose(mv0, mv1, rtol=1e-4)
      self.assertAllClose(mv0, mv2, rtol=1e-4)
      self.assertAllClose(mv0, mv3, rtol=1e-4)

  def testUnrollLSTMGrad(self):
    # Run one step of the unrolled lstm graph.
    def RunForwardBackward(mode, cfg=None):
      tf.logging.info("mode = %s", mode)
      g = tf.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
        loss = tf.reduce_sum(tf.square(m))
        dw = tf.gradients([loss], [weights])
      gdef = g.as_graph_def()
      finish = time.time()
      tf.logging.info("time: %f txt size: %d gdef bin size: %d", finish - start,
                      len(str(gdef)), len(gdef.SerializeToString()))
      with g.as_default(), tf.Session(config=cfg) as sess:
        return sess.run(dw)

    d0 = RunForwardBackward("complete")
    for cfg in _OptimizerOptions():
      tf.logging.info("cfg = %s", cfg)
      d1 = RunForwardBackward("cell", cfg)
      d2 = RunForwardBackward("loop", cfg)
      d3 = RunForwardBackward("loop10", cfg)
      self.assertAllClose(d0, d1, rtol=1e-4)
      self.assertAllClose(d0, d2, rtol=1e-4)
      self.assertAllClose(d0, d3, rtol=1e-4)


class FunctionInlineControlTest(tf.test.TestCase):

  def testFoo(self):
    dtype = tf.float32
    cfg = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L0,
            do_common_subexpression_elimination=True,
            do_function_inlining=True,
            do_constant_folding=True)))
    for noinline in [False, True]:

      @function.Defun(dtype, noinline=noinline)
      def Cell(v):
        # If v is a vector [n, 1], x is a big square matrix.
        x = tf.tanh(v + tf.transpose(v, [1, 0]))
        return tf.reduce_sum(x, 1, keep_dims=True)

      @function.Defun(dtype)
      def Forward(x):
        for _ in range(10):
          # pylint: disable=cell-var-from-loop
          x = Cell(x)
        return tf.reduce_sum(x, [0, 1])

      self.assertEqual(noinline, Cell.definition.attr["_noinline"].b)

      g = tf.Graph()
      with g.as_default():
        x = tf.placeholder(dtype)
        y = Forward(x)
        dx, = tf.gradients([y], [x])

      np.random.seed(321)
      inp = np.random.uniform(-1, 1, [16, 1]).astype(np.float32)
      run_metadata = tf.RunMetadata()
      with tf.Session(graph=g, config=cfg) as sess:
        ans = sess.run(
            [y, dx], {x: inp},
            run_metadata=run_metadata,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        print(ans[0], np.sum(ans[1]))
        self.assertAllClose(ans[0], 255.971, rtol=1e-3)
        self.assertAllClose(np.sum(ans[1]), 13.0408, rtol=1e-3)

      def MetadataHasCell(run_metadata):
        for dev_stats in run_metadata.step_stats.dev_stats:
          for node_stats in dev_stats.node_stats:
            if "Cell" in node_stats.timeline_label:
              return True
        return False

      self.assertEqual(MetadataHasCell(run_metadata), noinline)


@function.Defun(*[tf.float32] * 3)
def Linear(w, b, x):
  return tf.nn.relu(tf.matmul(x, w) + b)


@function.Defun(*[tf.float32] * 5)
def Linear2(w1, b1, w2, b2, x):
  return Linear(w2, b2, Linear(w1, b1, x))


class ModuleFunctionTest(tf.test.TestCase):

  def testBasic(self):
    with tf.Graph().as_default():
      a, b, c, d, e = [tf.constant([[_]], dtype=tf.float32) for _ in range(5)]
      y = Linear(a, b, c)
      z = Linear2(a, b, c, d, e)
      with tf.Session() as sess:
        self.assertAllEqual([[1]], sess.run(y))
        self.assertAllEqual([[5]], sess.run(z))


class VariableHoistingTest(tf.test.TestCase):

  def _testSimpleModel(self, use_forward_func):

    def _Model(x):
      w = tf.get_variable(
          "w", (64, 64), initializer=tf.random_uniform_initializer(seed=312))
      b = tf.get_variable("b", (64), initializer=tf.zeros_initializer()),
      return tf.sigmoid(tf.matmul(x, w) + b)

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
      loss = tf.reduce_mean(tf.reduce_sum(y0 * tf.log(y), 1), 0)
      arg_w, arg_b = function.get_extra_args()
      self.assertEqual(arg_w.get_shape(), tf.TensorShape([64, 64]))
      self.assertEqual(arg_b.get_shape(), tf.TensorShape([64]))
      dw, db = tf.gradients(loss, [arg_w, arg_b])
      cvars.extend(function.get_extra_vars())
      return loss, dw, db

    g = tf.Graph()
    with g.as_default():
      x = tf.random_normal([64, 64], seed=100)
      y0 = tf.random_normal([64, 64], seed=200)
      with tf.variable_scope("Foo"):
        loss, dw, db = Grad(x, y0)

    self.assertEqual(2, len(cvars))
    w, b = cvars[:2]
    self.assertEqual("Foo/w", w.op.name)
    self.assertEqual("Foo/b", b.op.name)

    with self.test_session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
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


if __name__ == "__main__":
  tf.test.main()
