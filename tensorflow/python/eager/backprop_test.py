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

import functools

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training


class BackpropTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testAggregateGradients(self):

    def fn(x):
      ind1 = constant_op.constant(np.array([0, 1]))
      ind2 = constant_op.constant(np.array([2, 3]))
      ind3 = constant_op.constant(np.array([1, 3]))
      # A mixture of IndexedSlices and dense tensor to aggregate.
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = embedding_ops.embedding_lookup(x, ind2)
      g3 = embedding_ops.embedding_lookup(x, ind3)
      g4 = math_ops.reduce_sum(x * constant_op.constant(2.0))
      return g1 * g2 * g3 * g4

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = constant_op.constant(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]
    grad = self.evaluate(ops.convert_to_tensor(grad))

    with context.graph_mode(), self.test_session():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_ind2 = array_ops.constant([2, 3])
      tf_ind3 = array_ops.constant([1, 3])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = embedding_ops.embedding_lookup(tf_var, tf_ind2)
      tf_g3 = embedding_ops.embedding_lookup(tf_var, tf_ind3)
      tf_g4 = math_ops.reduce_sum(tf_var * 2.0, reduction_indices=(0, 1))
      tf_y = tf_g1 * tf_g2 * tf_g3 * tf_g4
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      tf_dense_grad = math_ops.unsorted_segment_sum(
          tf_grad.values, tf_grad.indices, tf_grad.dense_shape[0])

      self.assertAllClose(grad, tf_dense_grad.eval())

  def testImplicitGradWithResourceVariable(self):
    x = resource_variable_ops.ResourceVariable(
        initial_value=constant_op.constant(1.0), name='x')

    def fn():
      tape.watch_variable(x)
      b = constant_op.constant(2.0)
      c = math_ops.add(x.value(), b)
      return math_ops.add(c, constant_op.constant(3.0))

    grads_and_vars = backprop.implicit_grad(fn)()
    self.assertAllEqual(grads_and_vars[0][0], 1.0)
    self.assertAllEqual(id(grads_and_vars[0][1]), id(x))

  def testWhereGradient(self):
    # Note: where is special because only some of its arguments are of
    # differentiable dtypes.

    def f(x):
      return array_ops.where(x < 10, x, x * x)

    g = backprop.gradients_function(f)

    self.assertAllEqual(g(5.)[0], 1.0)
    self.assertAllEqual(g(50.)[0], 100.0)

  def testTwoTargets(self):
    with backprop.GradientTape() as t:
      x = constant_op.constant(3.0)
      y = constant_op.constant(2.0)
      t.watch([x, y])
      xx = 2 * x
      yy = 3 * y
    dx, dy = t.gradient([xx, yy], [x, y])
    self.assertAllEqual(dx, 2.0)
    self.assertAllEqual(dy, 3.0)

  def testOutputGradUsedInComputation(self):
    with backprop.GradientTape() as t:
      x = constant_op.constant(3.0)
      y = constant_op.constant(2.0)
      t.watch([x, y])
      loss = x * y
    dx, = t.gradient([loss, x], [x], output_gradients=[1.0, 2.0])
    self.assertAllEqual(dx, 4.0)

  def testDy(self):

    def f(x):
      return x

    grad_fn = backprop.gradients_function(f)
    self.assertAllEqual(2., grad_fn(1., dy=2.)[0])

  def testGradientInteger(self):

    def f(x):
      return x + x

    int_tensor = constant_op.constant(1)
    self.assertEqual(backprop.gradients_function(f)(int_tensor)[0], None)

  def testErrors(self):

    @custom_gradient.custom_gradient
    def f(x):
      def grad(_):
        raise RuntimeError('x')
      return x, grad

    # TODO(apassos) raise the right error here
    with self.assertRaises(RuntimeError):
      backprop.gradients_function(f)(constant_op.constant(1.0))

  def testGradientsFunctionInCustomGradient(self):

    @custom_gradient.custom_gradient
    def f(x):
      (y,) = backprop.gradients_function(lambda x: x * x)(x)

      def grad(dy):
        return [2 * dy]

      return y, grad

    self.assertAllEqual(f(1.0), 2.0)

  def testImplicitGradOverEmbeddingLookup(self):
    batch_size = 8
    embedding_size = 512
    vocab_size = 1000
    lrn_rate = 0.1
    random_init = random_ops.random_uniform([vocab_size, embedding_size])

    x = array_ops.ones((batch_size), dtypes.int64)
    embedding = resource_variable_ops.ResourceVariable(
        initial_value=random_init, dtype=dtypes.float32, name='embedding')

    def f():
      tape.watch_variable(embedding)
      embedded_x = embedding_ops.embedding_lookup(embedding, x)
      return constant_op.constant(1.0, dtypes.float32) - embedded_x

    grad = backprop.implicit_grad(f)()[0][0]
    opt = training.GradientDescentOptimizer(lrn_rate)

    with context.graph_mode(), self.test_session():
      tf_x = array_ops.ones((batch_size), dtypes.int64)
      # TODO(ashankar,apassos): Change to ResourceVariable.
      tf_embedding = variables.Variable(
          random_init.numpy(), name='tf_embedding')
      tf_embedded_x = embedding_ops.embedding_lookup(tf_embedding, tf_x)
      tf_y = 1.0 - tf_embedded_x
      tf_grad = gradients.gradients(tf_y, [tf_embedding])[0]
      tf_opt = training.GradientDescentOptimizer(0.1)
      tf_embedding.initializer.run()

      self.assertAllClose(tf_grad.indices.eval(), grad.indices)
      self.assertAllClose(tf_grad.values.eval(), grad.values)

      tf_opt.apply_gradients([(tf_grad, tf_embedding)]).run()
      expected = tf_embedding.eval()
    opt.apply_gradients([(grad, embedding)])
    self.assertAllClose(expected, embedding.read_value())

  def testImplicitGradOrdering(self):
    v0 = resource_variable_ops.ResourceVariable(1.0)
    v1 = resource_variable_ops.ResourceVariable(2.0)

    def f():
      x = v1 * v1
      y = v0 * v0
      return x + y

    grads = backprop.implicit_grad(f)()
    ordered_variables = [x[1] for x in grads]
    self.assertTrue(ordered_variables[0] is v0)
    self.assertTrue(ordered_variables[1] is v1)

  def testTapeStopRecording(self):
    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      with t.stop_recording():
        y = x * x
    self.assertEqual(t.gradient(y, x), None)

  def testTapeReset(self):
    with backprop.GradientTape() as t:
      v = resource_variable_ops.ResourceVariable(1.0)
      loss = v * v
      t.reset()
      loss += v * v
    self.assertAllEqual(t.gradient(loss, v), 2.0)

  @test_util.assert_no_new_tensors
  def testGradientNone(self):

    def loss(x, l):
      return math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits(logits=x, labels=l),
          constant_op.constant([0]))

    logits = constant_op.constant([[0.0, 0.0]])
    labels = constant_op.constant([[1.0, 0.0]])
    # softmax_cross_entropy_with_logits returns two outputs and in this case the
    # gradient wrt the second is None.
    g, = backprop.gradients_function(loss, [0])(logits, labels)
    self.assertAllEqual(g.numpy(), [[-0.5, 0.5]])

  @test_util.run_in_graph_and_eager_modes
  def testGradientWithinTapeBlock(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v1.initializer)
    with backprop.GradientTape() as t:
      loss = 2 * v1
      grad = t.gradient(loss, v1)
    self.assertAllEqual(self.evaluate(grad), 2.0)

    with backprop.GradientTape(persistent=True) as t:
      loss = 2 * v1
      grad = t.gradient(loss, v1)
    self.assertAllEqual(self.evaluate(grad), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testNestedSelfContexts(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(v1.initializer)
    with backprop.GradientTape() as t:
      with self.assertRaises(ValueError):
        with t:
          pass

  @test_util.assert_no_new_tensors
  def testSecondGrad(self):

    def first(x):
      l = constant_op.constant([[0.0]])
      x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=x)
      x = math_ops.reduce_sum(x, constant_op.constant([0]))
      return x

    def second(x):
      grad = backprop.gradients_function(first, [0])(x)[0]
      return math_ops.reduce_sum(grad, constant_op.constant([0]))

    f = constant_op.constant([[0.1]])
    grad = backprop.gradients_function(second, [0])(f)[0]
    self.assertAllEqual([[0.0]], grad)

  @test_util.assert_no_new_tensors
  def testMakeVJP(self):

    def f(x):
      return x * x

    wrapped_fn = backprop.make_vjp(f, persistent=False)
    result, vjp = wrapped_fn(constant_op.constant(3.0))
    self.assertAllEqual(result, 9.0)
    self.assertAllEqual(vjp(2.0)[0], 12.0)

  def testPersistentMakeVJP(self):

    def f(x):
      return x * x

    wrapped_fn = backprop.make_vjp(f, persistent=True)
    _, vjp = wrapped_fn(constant_op.constant(3.0))
    vjp_result1 = vjp(2.0)[0]
    vjp_result2 = vjp(2.0)[0]
    self.assertAllEqual(vjp_result1, vjp_result2, 12.0)

  @test_util.assert_no_new_tensors
  def testGradGrad(self):

    def sq(x):
      return x * x

    def grad(x):
      value = backprop.gradients_function(sq, [0])(x)[0]
      return value

    gradgrad = backprop.gradients_function(grad, [0])

    self.assertAllEqual(gradgrad(constant_op.constant(3.0))[0], 2.0)

  @test_util.assert_no_new_tensors
  def testGradGradExp(self):

    def grad(x):
      value = backprop.gradients_function(math_ops.exp, [0])(x)[0]
      return value

    gradgrad = backprop.gradients_function(grad, [0])

    self.assertAllEqual(gradgrad(constant_op.constant(0.0))[0], 1.0)

  @test_util.assert_no_new_tensors
  def testStopGradient(self):
    grad = backprop.gradients_function(
        lambda x: array_ops.stop_gradient(math_ops.argmax(x)))
    self.assertAllEqual(grad([0.0])[0], None)

  @test_util.assert_no_new_tensors
  def testArgmax(self):
    def argmax(x):
      i = math_ops.argmax(x)
      return array_ops.stop_gradient(i)

    grad = backprop.gradients_function(argmax)
    self.assertAllEqual(grad([0.0])[0], None)

  @test_util.assert_no_new_tensors
  def testGPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    def fn(x):
      with context.device('/gpu:0'):
        b = constant_op.constant(2.0)
        c = math_ops.add(x.gpu(), b)
        # TODO(apassos): remove cpu below by making TensorVSPace aware
        # of devices.
        return math_ops.add(c, constant_op.constant(3.0)).cpu()

    grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.assert_no_new_tensors
  def testGPUImplicitGrad(self):
    if not context.context().num_gpus():
      self.skipTest('No GPU found')
    with context.device('gpu:0'):
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant(1.0), name='v')

    def f():
      with context.device('gpu:0'):
        tape.watch_variable(v)
        return v.read_value()

    self.assertEqual(
        backprop.implicit_grad(f)()[0][0].cpu().numpy(), 1.0)

  @test_util.assert_no_new_tensors
  def testCPU(self):

    def fn(x):
      b = constant_op.constant(2.0)
      c = math_ops.add(x, b)
      return math_ops.add(c, constant_op.constant(3.0))

    grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.assert_no_new_tensors
  def testTensorCopyGPU2CPU2GPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    def f(a, b):
      return a.cpu() + b.cpu()

    with context.device('/gpu:0'):
      a = constant_op.constant(1.0)
      b = constant_op.constant(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.assert_no_new_tensors
  def testEmptyParams(self):

    def fn(a, b):
      return a * b

    x = constant_op.constant(1.0)
    y = constant_op.constant(2.0)
    dx, dy = backprop.gradients_function(fn)(x, y)
    self.assertAllEqual(dx, y.numpy())
    self.assertAllEqual(dy, x.numpy())

  @test_util.assert_no_new_tensors
  def testUnconnectedNone(self):
    v = resource_variable_ops.ResourceVariable(
        1.0, name='testUnconnectedNone')

    def f():
      v.read_value()
      return constant_op.constant(1.0)

    self.assertEqual(backprop.implicit_grad(f)()[0][0], None)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeRepeatedSource(self):
    with backprop.GradientTape(persistent=False) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = 2 * x
    grad = g.gradient(target=y, sources=[x, x])
    self.assertEqual(self.evaluate(grad), [2.0, 2.0])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testPersistentGradientTapeRepeatedSource(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      y = constant_op.constant(5.0)
      g.watch(x)
      g.watch(y)
      z = x * x + x * y
    grad = g.gradient(target=z, sources=[x, x])
    self.assertEqual(self.evaluate(grad), [11.0, 11.0])
    grad = g.gradient(target=z, sources=[y, x])
    self.assertEqual(self.evaluate(grad), [3.0, 11.0])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeStructure(self):
    with backprop.GradientTape(persistent=True) as g:
      # Using different constant values because constant tensors are
      # cached, leading to a different gradient then what one might expect.
      x1 = constant_op.constant(3.0)
      x2 = constant_op.constant(3.1)
      x3 = constant_op.constant(3.2)
      g.watch(x1)
      g.watch(x2)
      g.watch(x3)
      y = x1  + 2 * x2  + 3 * x3
    self.assertEqual(self.evaluate(g.gradient(y, x1)), [1.0])
    self.assertEqual(self.evaluate(g.gradient(y, (x1,))), (1.0,))
    self.assertEqual(self.evaluate(g.gradient(y, (x1, x2))), (1.0, 2.0))
    self.assertEqual(self.evaluate(g.gradient(y, [(x1, x2), (x2, x3)])),
                     [(1.0, 2.0), (2.0, 3.0)])
    self.assertEqual(self.evaluate(g.gradient(y, (x1, x2, [x1, x3]))),
                     (1.0, 2.0, [1.0, 3.0]))
    self.assertEqual(self.evaluate(g.gradient(y, [x1, {'x2': x2, 'x3': x3}])),
                     [1.0, {'x2': 2.0, 'x3': 3.0}])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTape(self):
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      with backprop.GradientTape() as gg:
        gg.watch(y)
        z = 2 * y
      inner_grad = gg.gradient(z, [y])[0]
      self.assertEqual(self.evaluate(inner_grad), 2.0)
      y += inner_grad
    grad = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(grad), 6.0)

  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeWithCond(self):
    x = constant_op.constant(3.0)

    def true_fn():
      return x

    def false_fn():
      return x * x

    with backprop.GradientTape() as g:
      g.watch(x)
      y = control_flow_ops.cond(x < x, true_fn, false_fn)

    if not context.executing_eagerly():
      with self.assertRaisesRegexp(NotImplementedError, 'tf.gradients'):
        dy = g.gradient(y, [x])[0]
    else:
      dy = g.gradient(y, [x])[0]
      self.assertEqual(self.evaluate(dy), 6.0)

  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeWithWhileLoop(self):
    i = constant_op.constant(1)
    x = constant_op.constant(2.)

    def cond(i, _):
      return i < 3

    def body(i, x):
      return i + 1, x * 2

    with backprop.GradientTape() as g:
      g.watch([x])
      _, y = control_flow_ops.while_loop(cond, body, [i, x])

    if not context.executing_eagerly():
      with self.assertRaisesRegexp(NotImplementedError, 'tf.gradients'):
        dy = g.gradient(y, [x])[0]
    else:
      dy = g.gradient(y, [x])[0]
      self.assertEqual(self.evaluate(dy), 4.0)

  @test_util.assert_no_new_tensors
  def testGradientTapeGradientCalledMultipleTimes(self):
    with backprop.GradientTape() as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
    g.gradient(z, [x])
    with self.assertRaisesRegexp(
        RuntimeError, 'GradientTape.gradient can only be called once'):
      g.gradient(y, [x])

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testPersistentTape(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      z = y * y
    dz_dx = g.gradient(z, [x])[0]
    self.assertEqual(self.evaluate(dz_dx), 4 * 3 * 3 * 3)
    dy_dx = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(dy_dx), 2 * 3)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testHigherOrderGradient(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x ** 3                      # y       := x^3
      dy_dx = g.gradient(y, x)        # dy/dx   := 3x^2
      d2y_dx2 = g.gradient(dy_dx, x)  # d2y/dx2 := 6x
    d3y_dx3 = g.gradient(d2y_dx2, x)  # d3y/dx3 := 6
    x = 3
    self.assertEqual(self.evaluate(y), x ** 3)
    self.assertEqual(self.evaluate(dy_dx), 3 * x ** 2)
    self.assertEqual(self.evaluate(d2y_dx2), 6 * x)
    self.assertEqual(self.evaluate(d3y_dx3), 6)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testPersistentNestedTape(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = x * x
      with backprop.GradientTape(persistent=True) as gg:
        gg.watch(y)
        z = 2 * y
      for _ in range(2):
        inner_grad = gg.gradient(z, [y])[0]
        self.assertEqual(self.evaluate(inner_grad), 2.0)
      y += inner_grad
      del gg
    grad = g.gradient(y, [x])[0]
    self.assertEqual(self.evaluate(grad), 6.0)
    grad = g.gradient(z, [x])[0]
    self.assertEqual(self.evaluate(grad), 12.0)
    del g

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGradientTapeVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')
    self.evaluate(v.initializer)
    with backprop.GradientTape() as g:
      y = v * v
    grad = g.gradient(y, [v])[0]
    self.assertAllEqual(self.evaluate(grad), 2.0)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testNestedGradients(self):
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch(x)
      y = x * x
      z = y * y
    dz_dx, dz_dy = g.gradient(z, [x, y])
    self.assertEqual(self.evaluate(dz_dx), 108.0)
    self.assertEqual(self.evaluate(dz_dy), 18.0)

  @test_util.assert_no_new_tensors
  def testEmptyParamsForValueAndGradFunction(self):
    def fn(a, b):
      return a * b
    val_and_grads_fn = backprop.val_and_grad_function(fn)

    x = 2.0
    y = 3.0
    val, (dx, dy) = val_and_grads_fn(x, y)
    self.assertAllClose(val, x * y)
    self.assertAllEqual(dx, y)
    self.assertAllEqual(dy, x)

  @test_util.assert_no_new_tensors
  def testNonEmptyParamsForValueAndGradFunction(self):
    def fn(a, b):
      return a * b
    val_and_grad_fn = backprop.val_and_grad_function(fn, params=[1])

    x = 2.0
    y = 3.0
    val, grads = val_and_grad_fn(x, y)
    self.assertAllClose(val, x * y)
    self.assertEqual(1, len(grads))
    self.assertAllEqual(grads[0], x)

  @test_util.assert_no_new_tensors
  def testTensorCopyCPU2GPU2CPU(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # forward: a (cpu->gpu) -> add (gpu) -> c (gpu->cpu) -> add (cpu) -> e (cpu)
    # back: e (cpu) -> add (cpu) -> c (cpu->gpu) -> add (gpu) -> grad (gpu->cpu)
    def f(a, b):
      with context.device('/gpu:0'):
        c = math_ops.add(a.gpu(0), b.gpu(0))
      return math_ops.add(c.cpu(), constant_op.constant(3.0))

    with context.device('/cpu:0'):
      a = constant_op.constant(1.0)
      b = constant_op.constant(2.0)

    grad = backprop.gradients_function(f, [0])(a, b)[0]
    self.assertAllEqual(grad, 1.0)

  def testGetAttrType(self):
    typ = backprop.op_attr_type('Add', 'T')
    self.assertEqual(typ, pywrap_tensorflow.TF_ATTR_TYPE)

  def testGetAttrList(self):
    typ = backprop.op_attr_type('MaxPool', 'ksize')
    self.assertEqual(typ, [pywrap_tensorflow.TF_ATTR_INT])

  def testMakeAttrType(self):
    self.assertEqual(dtypes.float32,
                     backprop.make_attr(pywrap_tensorflow.TF_ATTR_TYPE, 1))

  def testMakeAttrTypeList(self):
    self.assertEqual([dtypes.float32],
                     backprop.make_attr([pywrap_tensorflow.TF_ATTR_TYPE], [1]))

  def testMulType(self):

    def mul(x):
      return math_ops._mul_dispatch(x, x)  # pylint: disable=protected-access

    self.assertAllEqual(
        backprop.gradients_function(mul)(3.0)[0].numpy(),
        6.0)

  def testMakeAttrShape(self):
    for s in ([], None, [1, 2, 3], [None, None], [1, None, 3]):
      expected = tensor_shape.TensorShape(s).as_proto()
      actual = backprop.make_attr(pywrap_tensorflow.TF_ATTR_SHAPE, s)
      self.assertEqual(
          expected,
          actual,
          msg=('For shape %r, expected %r != %r actual' % (s, expected,
                                                           actual)))

  def testMakeAttrShapeList(self):
    shape_list = [[], None, [1, 2, 3], [None, None], [1, None, 3]]
    self.assertEqual(
        [tensor_shape.TensorShape(s).as_proto() for s in shape_list],
        backprop.make_attr([pywrap_tensorflow.TF_ATTR_SHAPE], shape_list))

  def testArgsGradientFunction(self):

    def f(*args):
      return args[0] * args[0]

    grad = backprop.gradients_function(f)
    self.assertAllEqual(grad(1.0)[0], 2.0)

  def testPartial(self):

    def f(x, y):
      return x * y

    part = functools.partial(f, constant_op.constant(2.0))
    self.assertAllEqual(
        backprop.gradients_function(part)(constant_op.constant(1.0))[0],
        2.0)

  def testReturnSameThing(self):

    def f(x):
      return x, 2 * x

    self.assertAllEqual(backprop.gradients_function(f)(1.0)[0], 3.0)

  @test_util.assert_no_new_tensors
  def testExceptionSafety(self):

    def f(unused_x):
      raise ValueError()

    try:
      backprop.gradients_function(f)(1.0)
    except ValueError:
      pass

    def real_f(x):
      return x * x

    self.assertAllEqual(backprop.gradients_function(real_f)(1.0)[0], 2.0)

  @test_util.assert_no_new_tensors
  def testMultiValueConvertToTensor(self):
    x = resource_variable_ops.ResourceVariable(
        initial_value=array_ops.constant([1.0]), name='x')

    def fn():
      tape.watch_variable(x)
      a = math_ops.add(x.value(), 1.0)
      # Make sure convert_to_tensor works correctly with list of TensorNodes.
      b = array_ops.stack([a, a], axis=0)
      return math_ops.reduce_mean(b)

    grad = backprop.implicit_grad(fn)()[0][0]
    self.assertAllEqual([1.0], grad)

  def testOutput(self):

    def multiout(x):
      return x + 2, x * x

    x = constant_op.constant([0.0, 1.0, 2.0])

    grad = backprop.gradients_function(multiout)(x)[0]
    self.assertAllEqual([1.0, 3.0, 5.0], grad)

  def testMultiValuePreservesIfNotDiffedAgainst(self):

    def tfe_conv2d(timage, tkernel, conv2dstrides):
      return nn_ops.conv2d(timage, tkernel, conv2dstrides, 'SAME')

    i = constant_op.constant([[[[1.0]]]])
    k = constant_op.constant([[[[2.0]]]])
    s = [1, 1, 1, 1]

    grad = backprop.gradients_function(tfe_conv2d, params=(0,))(i, k, s)[0]
    self.assertAllEqual([[[[2.0]]]], grad)

  def testSameObjectForMultipleArguments(self):

    def f(x, y):
      return math_ops.multiply(x, y)

    g = backprop.gradients_function(f)

    def np_g(x, y):
      dx, dy = g(x, y)
      return [dx.numpy(), dy.numpy()]

    x = constant_op.constant(1.)
    self.assertAllEqual([1., 1.], np_g(x, x))
    x = 1.
    self.assertAllEqual([1., 1.], np_g(x, x))
    x = constant_op.constant([[1.]])
    self.assertAllEqual([[[1.]], [[1.]]], np_g(x, x))
    x = [[1.]]
    self.assertAllEqual([[[1.]], [[1.]]], np_g(x, x))

    v = resource_variable_ops.ResourceVariable(
        initial_value=1., name='testSameObjectForMultipleArguments.Variable')
    self.assertAllEqual([1., 1.], np_g(v, v))

  @test_util.assert_no_new_tensors
  def testImplicitGradientsCustomGradientAndCachedVariableValue(self):

    @custom_gradient.custom_gradient
    def my_square(x):
      result = math_ops.square(x)

      def grad(dr):
        return 2 * dr * x + 1

      return result, grad

    x = resource_variable_ops.ResourceVariable(
        initial_value=3., name='X.' + self.id())

    def f():
      return my_square(x)

    g = backprop.implicit_grad(f)

    grads_and_vars = g()
    self.assertEqual(1, len(grads_and_vars))
    grad, var = grads_and_vars[0]
    self.assertAllEqual(7, grad)
    self.assertAllEqual(x, var)

  @test_util.assert_no_new_tensors
  def testCustomGradient(self):

    @custom_gradient.custom_gradient
    def my_mul(x, y):
      result = x*y

      def grad(dr):
        return [dr*y, dr*x]
      return result, grad

    lr = 0.25
    x = resource_variable_ops.ResourceVariable(2., name='x')

    def loss(x):
      return my_mul(2., x.read_value())

    loss_grads_fn = backprop.implicit_val_and_grad(loss)

    losses = []
    for _ in range(5):
      loss, grads_and_vars = loss_grads_fn(x)
      losses.append(loss.numpy())
      for (grad, var) in grads_and_vars:
        var.assign_sub(lr*grad)
    self.assertAllEqual(losses, [4.0, 3., 2., 1., 0.])

  @test_util.assert_no_new_tensors
  def testCustomGradientIdentity(self):

    @custom_gradient.custom_gradient
    def my_identity(x):

      def grad(dresult):
        return [2 * dresult]

      return x, grad

    self.assertAllEqual(backprop.gradients_function(my_identity)(1.0)[0], 2.0)

  def testDifferentiatingFunctionThatReturnsNone(self):

    def fn(x, y):
      result = x*y  # pylint: disable=unused-variable

    x = constant_op.constant(1)
    y = constant_op.constant(2)

    loss_grads_fn = backprop.implicit_val_and_grad(fn)
    with self.assertRaisesRegexp(
        ValueError, 'Cannot differentiate a function that returns None; '
        'did you forget to return a value from fn?'):
      loss_grads_fn(x, y)

    val_and_grads_fn = backprop.val_and_grad_function(fn)
    with self.assertRaisesRegexp(
        ValueError, 'Cannot differentiate a function that returns None; '
        'did you forget to return a value from fn?'):
      val_and_grads_fn(x, y)

  def testZerosCacheDoesntLeakAcrossModes(self):
    with ops.Graph().as_default():
      t = random_ops.random_normal(shape=[100, 2])
      x = random_ops.random_normal(shape=[100, 4])
      dy = random_ops.random_normal(shape=[100, 4])
      with backprop.GradientTape() as gradient_tape:
        gradient_tape.watch(x)
        x1, _ = array_ops.split(x, num_or_size_splits=2, axis=1)
        y1 = x1 ** 2.
        y = array_ops.concat([y1, t], axis=1)

      dx = gradient_tape.gradient(y, x, output_gradients=dy)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(dx)

    t = random_ops.random_normal(shape=[100, 2])
    x = random_ops.random_normal(shape=[100, 4])
    dy = random_ops.random_normal(shape=[100, 4])
    with backprop.GradientTape() as gradient_tape:
      gradient_tape.watch(x)
      x1, _ = array_ops.split(x, num_or_size_splits=2, axis=1)
      y1 = x1 ** 2.
      y = array_ops.concat([y1, t], axis=1)

    dx = gradient_tape.gradient(y, x, output_gradients=dy)


if __name__ == '__main__':
  test.main()
