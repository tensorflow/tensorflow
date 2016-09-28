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

import collections
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
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

  def _mat(self, x):
    return np.array([x]).astype("float32").reshape([1, 1])

  def testBasic(self):
    g = tf.Graph()

    # Define a function
    #   foo(a:float, b:float, c:float)->u:float,v:float,w:float
    #     u = matmul(a, b) + c
    #     v = u^2
    #     w = u + v
    foo = tf.Graph()
    with foo.as_default():
      a = tf.placeholder(tf.float32, name="a")
      b = tf.placeholder(tf.float32, name="b")
      c = tf.placeholder(tf.float32, name="c")
      u = tf.add(tf.matmul(a, b), c, name="u")
      v = tf.square(u, name="v")
      w = tf.add_n([u, v], name="w")
    fdef = function._graph_to_function_def(foo, "foo", [a, b, c], [u, v, w])

    class Mock(function._DefinedFunction):

      def __init__(self, fdef):
        self._func_name = "foo"
        self._definition = fdef
        self._sub_functions = collections.OrderedDict()
        self._grad_func = None
        self._python_grad_func = None
        self._hash = hash(fdef.SerializeToString())

    g._add_function(Mock(fdef))

    # Compute 2 * 3 + 4 and its square.
    with g.as_default(), tf.Session() as sess:
      two = tf.constant(self._mat(2.0), name="two")
      three = tf.constant(self._mat(3.0), name="three")
      four = tf.constant(self._mat(4.0), name="four")
      # TODO(zhifengc): w/ @decorator sugar, we will just do:
      #   y, s, t = foo_func(two, three, four)

      # The graph contains two ops each of which calls foo.
      u0, v0, w0 = g.create_op(
          "foo", [two, three, four], [tf.float32, tf.float32, tf.float32],
          compute_shapes=False).outputs
      u1, v1, w1 = g.create_op(
          "foo", [four, two, three], [tf.float32, tf.float32, tf.float32],
          compute_shapes=False).outputs

      # Checks some property of the graph def.
      gdef = g.as_graph_def()
      self.assertEqual(len(gdef.node), 5)  # 5 nodes added.
      self.assertEqual(len(gdef.library.function), 1)  # 1 function is defined.

      for _ in xrange(10):
        # Run the graph, which is basically two function calls.
        ans_u0, ans_v0, ans_w0, ans_u1, ans_v1, ans_w1 = sess.run([u0, v0, w0,
                                                                   u1, v1, w1])
        self.assertAllEqual(ans_u0, self._mat(10.0))  # 2 * 3 + 4 = 10
        self.assertAllEqual(ans_v0, self._mat(100.0))  # 10^2 = 100
        self.assertAllEqual(ans_w0, self._mat(110.0))  # 100 + 10 = 110
        self.assertAllEqual(ans_u1, self._mat(11.0))  # 4 * 2 + 3 = 11
        self.assertAllEqual(ans_v1, self._mat(121.0))  # 11^2 = 121
        self.assertAllEqual(ans_w1, self._mat(132.0))  # 11 + 121 = 132

  def testDefineFunction2Args(self):

    @function.Defun(tf.float32, tf.float32)
    def APlus2B(a, b):
      return a + b * 2

    with tf.Graph().as_default():
      call = APlus2B([1.0], [2.0])
      self.assertEquals("APlus2B", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([5.0], sess.run(call))

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
      print("cfg = ", cfg)
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
      self.assertEquals(x.get_shape(), dx.get_shape())
      self.assertEquals(y.get_shape(), dy.get_shape())

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
        self.assertEquals([2.0], dx_val)
        self.assertEquals([0.0], dy_val)

  def testDefineFunctionNoArgs(self):

    @function.Defun()
    def AConstant():
      return tf.constant([42])

    with tf.Graph().as_default():

      call = AConstant()
      self.assertEquals("AConstant", call.op.name)
      with tf.Session() as sess:
        self.assertAllEqual([42], sess.run(call))

  def testDefineFunctionNames(self):

    @function.Defun(tf.float32)
    def Foo(a):
      return a + 1

    with tf.Graph().as_default():
      call1 = Foo([1.0])
      self.assertEquals("Foo", call1.op.name)
      call2 = Foo([1.0])
      self.assertEquals("Foo_1", call2.op.name)
      # pylint: disable=unexpected-keyword-arg
      call3 = Foo([1.0], name="mine")
      self.assertEquals("mine", call3.op.name)
      with tf.name_scope("my"):
        call4 = Foo([1.0], name="precious")
        self.assertEquals("my/precious", call4.op.name)

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

  def testAssert(self):

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

  def testVar(self):

    @function.Defun(tf.float32)
    def Foo(x):
      return x * x + 1

    g = tf.Graph()
    with g.as_default():
      v = tf.Variable(tf.constant(10.0))
      z = Foo(v)

    with self.test_session(graph=g):
      tf.initialize_all_variables().run()
      self.assertAllEqual(z.eval(), 101.)

  def testDefineErrors(self):
    with tf.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, "return at least one tensor"):

        @function.Defun()
        def NoResult():
          pass

        _ = NoResult.definition
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

        @function.Defun()
        def PlusMinusV1(a, b):
          return a + b, b - a

        _ = PlusMinusV1.definition
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

  def testDupDefinition(self):

    @function.Defun(tf.float32)
    def Foo(x):
      return x + 1

    @function.Defun(tf.float32, func_name="Foo")
    def Bar(x):
      return x + 1

    @function.Defun(tf.float32, func_name="Foo")
    def Baz(x):
      return x + 2

    with tf.Graph().as_default():
      y = Foo(100.0)
      z = Bar(100.0)  # OK.
      with self.test_session():
        self.assertAllEqual(y.eval(), z.eval())
      with self.assertRaisesRegexp(ValueError, "already defined"):
        z = Baz(100.0)

  def testFunctionDecorator(self):

    @function.Defun(tf.float32)
    def Minus1(b):
      return b - 1.0

    with tf.Graph().as_default():
      call1 = Minus1([2.])
      self.assertTrue(isinstance(Minus1, function._DefinedFunction))
      self.assertEqual(Minus1.name, "Minus1")
      # pylint: disable=unexpected-keyword-arg
      call2 = Minus1(call1, name="next")
      # pylint: enable=unexpected-keyword-arg
      self.assertEquals("next", call2.op.name)
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
    self.assertEquals(0, len(gdef.library.function))

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
    foo = function.Declare("Foo", [tf.float32], [tf.float32])

    @function.Defun(tf.float32)
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
      tf.initialize_all_variables().run()
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
    xm = tf.concat(1, [x, mprev])
    i_i, i_g, f_g, o_g = tf.split(1, 4, tf.matmul(xm, weights))
    new_c = tf.sigmoid(f_g) * cprev + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c = tf.clip_by_value(new_c, -50.0, 50.0)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    return new_m, new_c

  def _BuildForward(self, weights, inp, mode="cell"):

    def Loop(cell, w, i):
      x = tf.unpack(i, self.NUM_UNROLL)
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
        x = tf.unpack(inp, self.NUM_UNROLL)
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
      print("mode = ", mode)
      g = tf.Graph()
      start = time.time()
      with g.as_default():
        weights = self._Weights()
        inp = self._Input()
        m = self._BuildForward(weights, inp, mode)
      gdef = g.as_graph_def()
      finish = time.time()
      print("time: ", finish - start, " txt size: ", len(str(gdef)),
            "gdef bin size: ", len(gdef.SerializeToString()))
      with g.as_default(), tf.Session(config=cfg) as sess:
        return sess.run(m)

    mv0 = RunForward("complete")
    for cfg in _OptimizerOptions():
      print("cfg = ", cfg)
      mv1 = RunForward("cell", cfg)
      mv2 = RunForward("loop", cfg)
      mv3 = RunForward("loop10", cfg)
      self.assertAllClose(mv0, mv1, rtol=1e-4)
      self.assertAllClose(mv0, mv2, rtol=1e-4)
      self.assertAllClose(mv0, mv3, rtol=1e-4)

  def testUnrollLSTMGrad(self):
    # Run one step of the unrolled lstm graph.
    def RunForwardBackward(mode, cfg=None):
      print("mode = ", mode)
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
      print("time: ", finish - start, " txt size: ", len(str(gdef)),
            "gdef bin size: ", len(gdef.SerializeToString()))
      with g.as_default(), tf.Session(config=cfg) as sess:
        return sess.run(dw)

    d0 = RunForwardBackward("complete")
    for cfg in _OptimizerOptions():
      print("cfg = ", cfg)
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

      # pylint: disable=unexpected-keyword-arg
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

      g = tf.Graph()
      with g.as_default():
        x = tf.placeholder(dtype)
        y = Forward(x)
        dx, = tf.gradients([y], [x])

      np.random.seed(321)
      inp = np.random.uniform(-1, 1, [16, 1]).astype(np.float32)
      with tf.Session(graph=g, config=cfg) as sess:
        ans = sess.run([y, dx], {x: inp})
        print(ans[0], np.sum(ans[1]))
        self.assertAllClose(ans[0], 255.971, rtol=1e-3)
        self.assertAllClose(np.sum(ans[1]), 13.0408, rtol=1e-3)


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


if __name__ == "__main__":
  tf.test.main()
