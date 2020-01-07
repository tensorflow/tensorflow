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

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import tape as tape_lib
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.layers.pooling import max_pooling3d
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training


class BackpropTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testAggregateGradients(self):

    def fn(x):
      ind1 = constant_op.constant(np.array([0, 1]))
      ind2 = constant_op.constant(np.array([2, 3]))
      ind3 = constant_op.constant(np.array([1, 3]))
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = embedding_ops.embedding_lookup(x, ind2)
      g3 = embedding_ops.embedding_lookup(x, ind3)
      return g1 * g2 * g3

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = constant_op.constant(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]
    grad = self.evaluate(ops.convert_to_tensor(grad))

    if not context.executing_eagerly():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_ind2 = array_ops.constant([2, 3])
      tf_ind3 = array_ops.constant([1, 3])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = embedding_ops.embedding_lookup(tf_var, tf_ind2)
      tf_g3 = embedding_ops.embedding_lookup(tf_var, tf_ind3)
      tf_y = tf_g1 * tf_g2 * tf_g3
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      tf_dense_grad = math_ops.unsorted_segment_sum(
          tf_grad.values, tf_grad.indices, tf_grad.dense_shape[0])

      self.assertAllClose(grad, self.evaluate(tf_dense_grad))

  @test_util.run_in_graph_and_eager_modes
  def testAggregateGradientsWithTensor(self):

    def fn(x):
      ind1 = constant_op.constant(np.array([0, 1]))
      # A mixture of IndexedSlices and dense tensor to aggregate.
      g1 = embedding_ops.embedding_lookup(x, ind1)
      g2 = math_ops.reduce_sum(x * constant_op.constant(2.0))
      return g1 * g2

    var_np = np.random.rand(4, 2).astype(np.float32)
    var = constant_op.constant(var_np)
    grad = backprop.gradients_function(fn, [0])(var)[0]
    grad = self.evaluate(ops.convert_to_tensor(grad))

    if not context.executing_eagerly():
      tf_var = array_ops.constant(var_np, dtypes.float32)
      tf_ind1 = array_ops.constant([0, 1])
      tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
      tf_g2 = math_ops.reduce_sum(tf_var * 2.0, axis=(0, 1))
      tf_y = tf_g1 * tf_g2
      tf_grad = gradients.gradients(tf_y, [tf_var])[0]

      self.assertAllClose(grad, tf_grad)

  def testImplicitGradWithResourceVariable(self):
    x = resource_variable_ops.ResourceVariable(
        initial_value=constant_op.constant(1.0), name='x')

    def fn():
      b = constant_op.constant(2.0)
      c = math_ops.add(x.value(), b)
      return math_ops.add(c, constant_op.constant(3.0))

    grads_and_vars = backprop.implicit_grad(fn)()
    self.assertAllEqual(grads_and_vars[0][0], 1.0)
    self.assertAllEqual(id(grads_and_vars[0][1]), id(x))

  @parameterized.named_parameters(
      [('Function', def_function.function),
       ('NoFunction', lambda f: f)])
  def testNoOpBehaviorConsistent(self, decorator):

    @decorator
    def f(x):
      # Test all different types of no-ops
      x1 = array_ops.identity(x)
      x2 = math_ops.add_v2(x, 0)
      x3 = math_ops.subtract(x, 0)
      x4 = math_ops.multiply(x, 1)
      with backprop.GradientTape() as t:
        t.watch(x)
        t.watch(x1)
        t.watch(x2)
        t.watch(x3)
        t.watch(x4)
        y1 = x * 2.
        y2 = x1 * 3.
        y3 = x2 * 3.
        y4 = x3 * 3.
        y5 = x4 * 3.
        loss = y1 + y2 + y3 + y4 + y5
      return t.gradient(loss, [x, x1, x2, x3, x4])

    self.assertAllClose([2., 3., 3., 3., 3.], f(constant_op.constant(10.)))

  def testGradientInsideLoop(self):
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)

      def body(_):
        _ = v + 1.0  # This reads the variable inside the loop context
        with backprop.GradientTape() as t:
          result = v * 2
        self.assertIsNotNone(t.gradient(result, v))
        return 1.0

      control_flow_ops.while_loop(lambda i: False, body, [1.0])

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

  def testCustomGradientEmptyError(self):

    @custom_gradient.custom_gradient
    def identity(x):
      def grad(_):
        return []  # This return value is wrong!
      return x, grad

    x = variables.Variable(1.0)
    with backprop.GradientTape() as t:
      y = identity(x)
    with self.assertRaises(ValueError):
      t.gradient(y, [x])

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
      embedded_x = embedding_ops.embedding_lookup(embedding, x)
      return constant_op.constant(1.0, dtypes.float32) - embedded_x

    grad = backprop.implicit_grad(f)()[0][0]
    opt = training.GradientDescentOptimizer(lrn_rate)

    with ops.Graph().as_default(), self.cached_session():
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
      expected = self.evaluate(tf_embedding)
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
    self.assertIs(ordered_variables[0], v0)
    self.assertIs(ordered_variables[1], v1)

  def testTapeNoOpGradient(self):
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x
    self.assertEqual(t.gradient(y, x).numpy(), 1.0)

  def testTapeIdentityGradientIsIdentity(self):
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = array_ops.identity(x)
    self.assertEqual(t.gradient(y, x).numpy(), 1.0)

  def testFunctionIndexedSlicesGradient(self):

    @def_function.function
    def f(x):
      return x + 1

    with backprop.GradientTape() as t:
      x = constant_op.constant([1.0])
      t.watch(x)
      y = f(x)
      y = array_ops.gather(y, [0])
    self.assertAllEqual(t.gradient(y, x), [1.0])

  def testTapeGradientMultiTargetOneIsSource(self):
    x = constant_op.constant(2.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x*x
    self.assertEqual(t.gradient([x, y], x).numpy(), 5.0)

  def testTapeNoOpGradientWithMultiTargetAllSource(self):
    x = constant_op.constant(3.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      y = x
    self.assertEqual(t.gradient([y, y], x).numpy(), 2.0)

  def testTapeNoOpGradientWithMultiTargetMultiSource(self):
    x = constant_op.constant(3.0)
    y = constant_op.constant(5.0)
    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(y)
      z = y * y
    self.assertAllEqual(t.gradient([x, y, z], [x, y]), [1.0, 11.0])

  def testTapeGradientStringTarget(self):
    s = constant_op.constant('unknown', dtype=dtypes.string)
    x = constant_op.constant(3.0)

    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(s)
    grads = t.gradient(s, x)
    self.assertEqual(grads, None)

  def testTapeNoOpGradientStringSourceAndTarget(self):
    s = constant_op.constant('unknown', dtype=dtypes.string)

    with backprop.GradientTape() as t:
      t.watch(s)
    grads = t.gradient(s, s)
    self.assertEqual(grads, None)

  def testTapeNoOpGradientWithMultiTargetMultiSourceIncludeString(self):
    x = constant_op.constant(3.0)
    y = constant_op.constant(5.0)
    s = constant_op.constant('unknown', dtype=dtypes.string)

    with backprop.GradientTape() as t:
      t.watch(x)
      t.watch(y)
      t.watch(s)
      z = y * y
    grads = t.gradient([x, y, z, s], [x, y, s])
    self.assertAllEqual(grads[:2], [1.0, 11.0])
    self.assertEqual(grads[2], None)

  def testTapeNoOpOnVariableIsIdentity(self):
    v0 = resource_variable_ops.ResourceVariable(1.0)
    with backprop.GradientTape() as t:
      y = v0.read_value()
    self.assertEqual(t.gradient(y, v0).numpy(), 1.0)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testTapeNoOpGradient2By2(self):
    a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
    dy_dy = tape.gradient(a_2_by_2, [a_2_by_2])[0]
    self.assertAllEqual(dy_dy.numpy(),
                        constant_op.constant(1.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTapeNoOpGradientMultiTarget2By2(self):
    a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
    dy_dy = tape.gradient([a_2_by_2, a_2_by_2], [a_2_by_2])[0]
    self.assertAllEqual(dy_dy.numpy(),
                        constant_op.constant(2.0, shape=[2, 2]).numpy())

  def testTapeStopRecording(self):
    with backprop.GradientTape() as t:
      x = resource_variable_ops.ResourceVariable(1.0)
      with t.stop_recording():
        y = x * x
    self.assertEqual(t.gradient(y, x), None)

  def testTapeStopStartRecording(self):
    with backprop.GradientTape(persistent=True) as t:
      x = resource_variable_ops.ResourceVariable(1.0)
      x2 = x * 2  # This should be differentiated through.
      with t.stop_recording():
        y = x2 * x2
      z = x2 * x2
    self.assertEqual(t.gradient(y, x2), None)

    # If the x*2 was not differentiated through, this would be 2.0, not 4.0
    self.assertEqual(t.gradient(z, x2).numpy(), 4.0)

  def testTapeReset(self):
    with backprop.GradientTape() as t:
      v = resource_variable_ops.ResourceVariable(1.0)
      loss = v * v
      t.reset()
      loss += v * v
    self.assertAllEqual(t.gradient(loss, v), 2.0)

  def testPythonMax(self):
    x = [resource_variable_ops.ResourceVariable(2.),
         resource_variable_ops.ResourceVariable(3.),
         resource_variable_ops.ResourceVariable(5.)]
    with backprop.GradientTape() as t:
      f = max(x)
    grad = t.gradient(f, x)
    self.assertAllEqual(self.evaluate(f), 5.)
    self.assertAllEqual(self.evaluate(grad), [None, None, 1.0])

  def testAutomaticWatchedVariables(self):
    with backprop.GradientTape() as t:
      self.assertEqual(0, len(t.watched_variables()))
      v = resource_variable_ops.ResourceVariable(1.0)
      loss = v * v
      self.assertAllEqual([v], t.watched_variables())

      t.reset()
      self.assertEqual(0, len(t.watched_variables()))
      loss += v * v
      self.assertAllEqual([v], t.watched_variables())

  def testExplicitWatchedVariables(self):
    with backprop.GradientTape() as t:
      self.assertEqual(0, len(t.watched_variables()))
      v = resource_variable_ops.ResourceVariable(1.0)
      t.watch(v)
      self.assertAllEqual([v], t.watched_variables())

      t.reset()
      self.assertEqual(0, len(t.watched_variables()))
      t.watch(v)
      self.assertAllEqual([v], t.watched_variables())

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

  @test_util.run_in_graph_and_eager_modes
  def testWatchingIsTapeLocal(self):
    x1 = resource_variable_ops.ResourceVariable(2.0, trainable=False)
    x2 = resource_variable_ops.ResourceVariable(2.0, trainable=False)

    with backprop.GradientTape() as tape1:
      with backprop.GradientTape() as tape2:
        tape1.watch(x1)
        tape2.watch([x1, x2])
        y = x1 ** 3
        z = x2 ** 2
        dy, dz = tape2.gradient([y, z], [x1, x2])
      d2y, d2z = tape1.gradient([dy, dz], [x1, x2])

    self.evaluate([x1.initializer, x2.initializer])
    self.assertEqual(self.evaluate(d2y), 12.0)
    self.assertIsNone(d2z)

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

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testGPU(self):
    def fn(x):
      with context.device('/gpu:0'):
        b = constant_op.constant(2.0)
        c = math_ops.add(x.gpu(), b)
        # TODO(apassos): remove cpu below by making TensorVSPace aware
        # of devices.
        return math_ops.add(c, constant_op.constant(3.0)).cpu()

    grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
    self.assertAllEqual(grad, 1.0)

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testGPUImplicitGrad(self):
    with context.device('gpu:0'):
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant(1.0), name='v')

    def f():
      with context.device('gpu:0'):
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

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testTensorCopyGPU2CPU2GPU(self):
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
  def testGradientTapeReEnterContext(self):
    g = backprop.GradientTape()
    with g:
      x = constant_op.constant(3.0)
      g.watch(x)
      y = 2*x
    with g:
      z = 2*y
    grad = g.gradient(target=z, sources=[x])
    self.assertEqual(self.evaluate(grad), [4.0])

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

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testGadientTapeCalledOnConstantTarget(self):
    with backprop.GradientTape() as g:
      x = variables.Variable([3.0])
      y = variables.Variable([2.0])
    grad = g.gradient(x, y)
    self.assertAllEqual(grad, None)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only('b/120545219')
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
  @test_util.run_v1_only('b/120545219')
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
  @test_util.run_v1_only('b/120545219')
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
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsDefault(self):
    x = constant_op.constant(1.0)
    y = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x)
    self.assertEqual(dz_dx, None)

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsZeros(self):
    x = constant_op.constant(1.0, shape=[2, 2])
    y = constant_op.constant(3.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x, unconnected_gradients='zero')
    self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

  @test_util.assert_no_new_tensors
  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsVariablesZeros(self):
    x = resource_variable_ops.ResourceVariable(
        constant_op.constant(1., shape=[2, 2]))
    self.evaluate(x.initializer)
    y = resource_variable_ops.ResourceVariable(constant_op.constant(3.))
    self.evaluate(y.initializer)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    dz_dx = g.gradient(z, x, unconnected_gradients='zero')
    self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

  @test_util.run_in_graph_and_eager_modes
  def testUnknownUnconnectedGradientsValueGiven(self):
    x = constant_op.constant(1.0)
    y = constant_op.constant(1.0)
    with backprop.GradientTape() as g:
      g.watch([x, y])
      z = y * 2
    with self.assertRaisesRegexp(
        ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
      g.gradient(z, x, unconnected_gradients='nonsense')

  @test_util.run_in_graph_and_eager_modes
  def testUnconnectedGradientsNestedDefunZeros(self):

    @function.defun
    def f(x):
      return x * x

    @function.defun
    def h(y):
      z = f(y)
      return array_ops.stop_gradient(z)

    x = constant_op.constant(1.0)
    with backprop.GradientTape() as g:
      g.watch(x)
      k = x + 2.
      y = h(k)

    dy_dx = g.gradient(y, x, unconnected_gradients='zero')
    self.assertEqual(0.0, self.evaluate(dy_dx))

  def testInvalidRecordOperationMessage(self):
    y = constant_op.constant(2.)
    x = constant_op.constant(1.)
    with backprop.GradientTape() as g:
      g.watch(x)
      tape_lib.record_operation(
          'InvalidBackprop',
          [y],
          [x],
          lambda dy: [])
    with self.assertRaisesRegexp(
        errors_impl.InternalError, 'InvalidBackprop.*too few gradients'):
      g.gradient(y, x)

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

  @test_util.run_gpu_only
  @test_util.assert_no_new_tensors
  def testTensorCopyCPU2GPU2CPU(self):
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
    self.assertEqual(typ, int(pywrap_tfe.TF_ATTR_TYPE))

  def testGetAttrList(self):
    typ = backprop.op_attr_type('MaxPool', 'ksize')
    self.assertEqual(typ, [int(pywrap_tfe.TF_ATTR_INT)])

  def testMakeAttrType(self):
    self.assertEqual(dtypes.float32,
                     backprop.make_attr(int(pywrap_tfe.TF_ATTR_TYPE), 1))

  def testMakeAttrTypeList(self):
    self.assertEqual([dtypes.float32],
                     backprop.make_attr([int(pywrap_tfe.TF_ATTR_TYPE)], [1]))

  def testMulType(self):

    def mul(x):
      return math_ops._mul_dispatch(x, x)  # pylint: disable=protected-access

    self.assertAllEqual(
        backprop.gradients_function(mul)(3.0)[0].numpy(),
        6.0)

  def testMakeAttrShape(self):
    for s in ([], None, [1, 2, 3], [None, None], [1, None, 3]):
      expected = tensor_shape.TensorShape(s).as_proto()
      actual = backprop.make_attr(int(pywrap_tfe.TF_ATTR_SHAPE), s)
      self.assertEqual(
          expected,
          actual,
          msg=('For shape %r, expected %r != %r actual' % (s, expected,
                                                           actual)))

  def testMakeAttrShapeList(self):
    shape_list = [[], None, [1, 2, 3], [None, None], [1, None, 3]]
    self.assertEqual(
        [tensor_shape.TensorShape(s).as_proto() for s in shape_list],
        backprop.make_attr([int(pywrap_tfe.TF_ATTR_SHAPE)], shape_list))

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

  def testJacobianCustomGradient(self):

    class MyCallable(object):

      def __init__(self):
        self.a = variables.Variable(1.)
        self.b = variables.Variable(2.)
        self.c = variables.Variable(3.)

      def __call__(self, x):
        return self.a * x * x + self.b * x + self.c

    @def_function.function
    def call(c, x):

      @custom_gradient.custom_gradient
      def _call():
        y = c(x)

        def grad(dy, variables=None):  # pylint: disable=redefined-outer-name
          with backprop.GradientTape(persistent=True) as g:
            g.watch(variables)
            y = c(x)
          grad_vars = [
              2 * math_ops.reduce_sum(dy * g.jacobian(y, v)) for v in variables
          ]
          del g
          return (), grad_vars

        return y, grad

      return _call()

    c = MyCallable()
    x = constant_op.constant([1., 2., 3.])
    with backprop.GradientTape(persistent=True) as g:
      g.watch([c.a, c.b, c.c])
      y = call(c, x)
    self.assertAllEqual(g.gradient(y, x), None)

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

  def testZerosCacheDoesntLeakAcrossGraphs(self):
    with ops.Graph().as_default():
      def get_grad():
        with ops.Graph().as_default(), self.cached_session():
          t = constant_op.constant(1, dtype=dtypes.float32, shape=(10, 4))
          x = constant_op.constant(2, dtype=dtypes.float32, shape=(10, 4))
          with backprop.GradientTape() as tape:
            tape.watch(x)
            x1, _ = array_ops.split(x, num_or_size_splits=2, axis=1)
            y1 = x1**2
            y = array_ops.concat([y1, t], axis=1)
          return self.evaluate(tape.gradient(y, x))

      grad1 = get_grad()
      grad2 = get_grad()

      self.assertAllEqual(grad1, grad2)

  @test_util.run_in_graph_and_eager_modes
  def testSelectivelyWatchVariables(self):
    x1 = resource_variable_ops.ResourceVariable(1.0)
    x2 = resource_variable_ops.ResourceVariable(1.0)
    with backprop.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(x2)
      y = x1**2
      z = x2**3
    self.assertTupleEqual(tape.watched_variables(), (x2,))
    dy, dz = tape.gradient([y, z], [x1, x2])
    self.evaluate([x1.initializer, x2.initializer])
    self.assertIsNone(dy)
    self.assertEqual(self.evaluate(dz), 3.0)

  @test_util.run_in_graph_and_eager_modes
  def testDifferentiatingScalarCache(self):
    # In the following test, if x2 = x1 (i.e the objects are the exact same),
    # then y is essentially, 2*x1, and dy/dx1 = 2.
    # When we had a pure scalar cache in eager, this would be the case. This
    # test prevents us from going back to that case.
    with backprop.GradientTape(persistent=False) as g:
      x1 = constant_op.constant(3.0)
      x2 = constant_op.constant(3.0)
      g.watch(x1)
      g.watch(x2)
      y = x1 + x2
    grad = g.gradient(target=y, sources=[x1])
    self.assertEqual(self.evaluate(grad), [1.0])

  def testVariablesAndConstantsProduceTheSameGradients(self):

    # In the following test, differentiating [y, z] against [a, b] gives:
    # (dy/da + dz/da, dy/db + dz/db).
    # If a and b are the same constant, dz/da will not be 0 (which it should
    # be).
    # This is solved by using variable since doing a read_value on a tensor will
    # produce a new tensor and corresponding TensorHandle, and not reuse the
    # same tensor (which would happen if we are using a cache and reusing
    # EagerTensor objects).
    def get_grads(a, b):
      with backprop.GradientTape() as tape:
        tape.watch([a, b])
        y = a**3
        z = b**2
      return tape.gradient([y, z], [a, b])

    gradients_constants = get_grads(
        constant_op.constant(2.0), constant_op.constant(2.0))
    gradients_variables = get_grads(
        resource_variable_ops.ResourceVariable(2.0),
        resource_variable_ops.ResourceVariable(2.0))
    self.assertAllEqual(gradients_constants, gradients_variables)

  def testUnknownShapes(self):
    with ops.Graph().as_default():
      with backprop.GradientTape() as tape:
        a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
        tape.watch(a)
        b = a**3

      db_da = tape.gradient(b, a)

      with self.cached_session() as sess:
        self.assertEqual((8.0, 12.0), sess.run((b, db_da), feed_dict={a: 2.0}))

  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientInEagerAndGraph(self):
    @custom_gradient.custom_gradient
    def f(x):
      y = x * x

      def grad(dy):
        return [4 * dy]

      return y, grad

    with backprop.GradientTape() as t:
      c = constant_op.constant(1.0)
      t.watch(c)
      g = f(c)
    self.assertAllEqual(self.evaluate(t.gradient(g, c)), 4.0)

  def testOverrideSecondOrderWithCustomGradient(self):
    @custom_gradient.custom_gradient
    def f(x):

      def first_order_grad(dz):
        @custom_gradient.custom_gradient
        def first_order_custom(unused_x):

          def h(ddz):
            return -2.1 * ddz

          return -1.1, h
        return dz * first_order_custom(x)

      return x + 10., first_order_grad

    c = constant_op.constant(1.)
    with backprop.GradientTape() as outer:
      outer.watch(c)
      with backprop.GradientTape() as inner:
        inner.watch(c)
        d = f(c) ** 4.
      dd = inner.gradient(d, c)
      self.assertAllClose(4. * f(c) ** 3. * -1.1, dd)
    self.assertAllClose(3. * 4. * f(c) ** 2. * -1.1 * -1.1
                        + 4. * f(c) ** 3. * -2.1,
                        outer.gradient(dd, c))

  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientForwardprop(self):
    @custom_gradient.custom_gradient
    def f(x):
      z = 2. * tensor_util.constant_value(x)
      def g(dz):
        @custom_gradient.custom_gradient
        def first_order(unused_x, unused_dz):
          def second_order_and_transpose(unused_ddz):
            return 2.2, 3.1
          return 2.1, second_order_and_transpose
        return first_order(x, dz)
      return z, g

    with backprop.GradientTape(persistent=True) as t:
      with backprop.GradientTape() as tt:
        c = constant_op.constant(1.)
        t.watch(c)
        tt.watch(c)
        output_grad = array_ops.ones([])
        t.watch(output_grad)
        output = f(c)
        self.assertAllClose(2., output)
      gc = tt.gradient(output, c, output_gradients=output_grad)
      self.assertAllClose(2.1, gc)
    ggc = t.gradient(gc, c)
    self.assertAllClose(2.2, ggc)
    # Note that executed eagerly this kind of transpose is not efficient. But
    # from a tf.function we could prune out the first-order gradient
    # computation.
    transpose = t.gradient(gc, output_grad)
    self.assertAllClose(3.1, transpose)

  @test_util.run_in_graph_and_eager_modes
  def testMaxPooling3DGradient(self):

    if test.is_built_with_rocm():
      self.skipTest('Pooling with 3D tensors is not supported in ROCm')

    def forward(a):
      r = max_pooling3d(a, pool_size=pool_size, strides=strides, padding='SAME')
      return r

    input_sizes = [1, 3, 2, 4, 1]
    pool_size = (2, 2, 1)
    strides = (1, 1, 1)

    total_size = np.prod(input_sizes)
    x = np.arange(1, total_size + 1, dtype=np.float32)
    aa = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
    da = backprop.gradients_function(forward)(aa)

    if not context.executing_eagerly():
      tf_aa = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
      tf_max = max_pooling3d(
          tf_aa, pool_size=pool_size, strides=strides, padding='SAME')
      tf_da = gradients.gradients(tf_max, [tf_aa])
      self.assertAllEqual(da[0], tf_da[0].eval())

  @test_util.run_in_graph_and_eager_modes
  def testWatchBadThing(self):
    g = backprop.GradientTape()
    with self.assertRaisesRegexp(ValueError, 'ndarray'):
      g.watch(np.array(1.))

  def testWatchedVariablesAfterNonPersistentGradientCall(self):
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
    tape.gradient(x, x)
    self.assertEqual((x,), tape.watched_variables())

  def testWatchedVariablesOnlyHasVariablesFromLastTape(self):
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
    with backprop.GradientTape(persistent=False) as tape:
      z = resource_variable_ops.ResourceVariable(2.0)
      tape.watch(z)
    tape.gradient(z, z)
    self.assertEqual((z,), tape.watched_variables())

  def testWatchedVariablesRespectReset(self):
    with backprop.GradientTape(persistent=False) as tape:
      x = resource_variable_ops.ResourceVariable(1.0)
      tape.watch(x)
      self.assertEqual((x,), tape.watched_variables())
      tape.reset()
      z = resource_variable_ops.ResourceVariable(2.0)
      tape.watch(z)
      self.assertEqual((z,), tape.watched_variables())
    tape.gradient(z, z)
    self.assertEqual((z,), tape.watched_variables())

class JacobianTest(test.TestCase):

  def _jacobian(self, experimental_use_pfor):
    persistent = context.executing_eagerly and not experimental_use_pfor
    with backprop.GradientTape(persistent=persistent) as g:
      x = constant_op.constant([1., 2.])
      y = constant_op.constant([3., 4.])
      g.watch(x)
      g.watch(y)
      z = x * x * y
    jacobian = g.jacobian(z, [x, y],
                          experimental_use_pfor=experimental_use_pfor)
    answer = [array_ops.diag(2 * x * y), array_ops.diag(x * x)]
    return jacobian, answer

  @test_util.run_v1_only('b/120545219')
  def testPfor(self):
    jacobian, answer = self._jacobian(experimental_use_pfor=True)
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testWhileLoop(self):
    jacobian, answer = self._jacobian(experimental_use_pfor=False)
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testPforDefun(self):

    @function.defun
    def _f():
      return self._jacobian(experimental_use_pfor=True)

    jacobian, answer = _f()
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testWhileLoopDefun(self):

    @function.defun
    def _f():
      return self._jacobian(experimental_use_pfor=False)

    jacobian, answer = _f()
    for j, a in zip(jacobian, answer):
      self.assertAllEqual(a, j)

  @test_util.run_v1_only('b/120545219')
  def testPersistentTape(self):
    if not context.executing_eagerly():
      return
    with backprop.GradientTape() as g:
      x = constant_op.constant([1.0, 2.0])
      g.watch(x)
      y = x * x
    with self.assertRaisesRegexp(RuntimeError, 'persistent'):
      g.jacobian(y, x, experimental_use_pfor=False)

  @test_util.run_v1_only('b/120545219')
  def testPforException(self):
    var = variables.Variable([1.])

    @custom_gradient.custom_gradient
    def op(x):
      def grad(_):
        # Note that we perform a stateful operation here that will not be
        # compatible with parallel for construct.
        with ops.control_dependencies(
            [var.assign(random_ops.random_uniform([1]))]):
          return constant_op.constant(1.)
      return x, grad

    with backprop.GradientTape() as g:
      x = constant_op.constant([1., 2.])
      g.watch(x)
      y = op(x)
    with self.assertRaisesRegexp(ValueError, 'No converter'):
      g.jacobian(y, x, experimental_use_pfor=True)

  @test_util.run_v1_only('b/120545219')
  def test_parallel_iterations(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant([[1., 2], [3, 4]])
      g.watch(x)
      y = math_ops.matmul(x, x)
    self.assertAllClose(g.jacobian(y, x, parallel_iterations=2),
                        g.jacobian(y, x, parallel_iterations=3))

  @test_util.run_in_graph_and_eager_modes
  def test_nested_jacobian(self):
    if context.executing_eagerly():
      # TODO(agarwal): b/128842926
      self.skipTest('Conversion of function calls not implemented yet.')
    x = array_ops.ones((10, 2))
    with backprop.GradientTape(persistent=False) as g:
      g.watch(x)
      with backprop.GradientTape(persistent=False) as gg:
        gg.watch(x)
        y = math_ops.reduce_sum(math_ops.square(x))
      dy_x = gg.jacobian(y, x)
    dy_xx = g.batch_jacobian(dy_x, x)
    dy_xx_answer = [[[2., 0], [0, 2.]]] * 10
    self.assertAllClose(dy_xx_answer, self.evaluate(dy_xx))

  @test_util.run_in_graph_and_eager_modes
  def test_indexed_slices(self):
    with backprop.GradientTape(persistent=True) as g:
      inp = random_ops.random_uniform([3, 2])
      g.watch(inp)
      output = nn.embedding_lookup(inp, [0, 2])
    self.assertAllClose(
        g.jacobian(output, inp, experimental_use_pfor=True),
        g.jacobian(output, inp, experimental_use_pfor=False))


@test_util.run_all_in_graph_and_eager_modes
class BatchJacobianTest(test.TestCase, parameterized.TestCase):

  def _batch_jacobian(self, experimental_use_pfor):
    persistent = context.executing_eagerly and not experimental_use_pfor
    with backprop.GradientTape(persistent=persistent) as g:
      x = constant_op.constant([[1., 2.], [3., 4.]])
      y = constant_op.constant([[3., 4.], [5., 6.]])
      g.watch(x)
      z = x * x * y
    batch_jacobian = g.batch_jacobian(
        z, x, experimental_use_pfor=experimental_use_pfor)
    answer = array_ops.stack([array_ops.diag(2 * x[0] * y[0]),
                              array_ops.diag(2 * x[1] * y[1])])
    return batch_jacobian, answer

  def testPfor(self):
    batch_jacobian, answer = self._batch_jacobian(experimental_use_pfor=True)
    self.assertAllEqual(answer, batch_jacobian)

  def testWhileLoop(self):
    batch_jacobian, answer = self._batch_jacobian(experimental_use_pfor=False)
    self.assertAllEqual(answer, batch_jacobian)

  def testPforDefun(self):

    @function.defun
    def _f():
      return self._batch_jacobian(experimental_use_pfor=True)

    batch_jacobian, answer = _f()
    self.assertAllEqual(answer, batch_jacobian)

  def testWhileLoopDefun(self):

    @function.defun
    def _f():
      return self._batch_jacobian(experimental_use_pfor=False)

    batch_jacobian, answer = _f()
    self.assertAllEqual(answer, batch_jacobian)

  def testPersistentTape(self):
    if not context.executing_eagerly():
      return
    with backprop.GradientTape() as g:
      x = constant_op.constant([[1.0, 2.0]])
      g.watch(x)
      y = x * x
    with self.assertRaisesRegexp(RuntimeError, 'persistent'):
      g.batch_jacobian(y, x, experimental_use_pfor=False)

  def testBadShape(self):
    x = random_ops.random_uniform([2, 3])
    with backprop.GradientTape() as g:
      y = array_ops.concat([x, x], axis=0)
    with self.assertRaisesRegexp(ValueError, 'Need first dimension'):
      g.batch_jacobian(y, x)

  def testBadInputRank(self):
    x = random_ops.random_uniform([2])
    with backprop.GradientTape() as g:
      y = random_ops.random_uniform([2, 2])
    with self.assertRaisesRegexp(ValueError, 'must have rank at least 2'):
      g.batch_jacobian(y, x)

  def testBadOutputRank(self):
    x = random_ops.random_uniform([2, 2])
    with backprop.GradientTape() as g:
      y = random_ops.random_uniform([2])
    with self.assertRaisesRegexp(ValueError, 'must have rank at least 2'):
      g.batch_jacobian(y, x)

  def testPforException(self):
    var = variables.Variable([1.])

    @custom_gradient.custom_gradient
    def op(x):
      def grad(_):
        # Note that we perform a stateful operation here that will not be
        # compatible with parallel for construct.
        with ops.control_dependencies(
            [var.assign(random_ops.random_uniform([1]))]):
          return constant_op.constant(1.)
      return x, grad

    with backprop.GradientTape() as g:
      x = constant_op.constant([[1.], [2.]])
      g.watch(x)
      y = op(x)
    with self.assertRaisesRegexp(ValueError, 'No converter'):
      g.batch_jacobian(y, x, experimental_use_pfor=True)

  def test_parallel_iterations(self):
    with backprop.GradientTape(persistent=True) as g:
      x = constant_op.constant([[1., 2], [3, 4]])
      g.watch(x)
      w = constant_op.constant([[1., 2, 3, 4], [5, 6, 7, 8]])
      y = math_ops.matmul(x, w)
    self.assertAllClose(g.batch_jacobian(y, x, parallel_iterations=2),
                        g.batch_jacobian(y, x, parallel_iterations=3))

  @parameterized.parameters(
      (True, True),
      (True, False),
      (False, True),
      (False, False))
  def test_degenerate_shape(self, use_function, use_pfor):

    def f(x):
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x**2
      return tape.batch_jacobian(y, x, experimental_use_pfor=use_pfor)

    if use_function:
      f = def_function.function(f)
    self.assertAllEqual([1, 0, 0], array_ops.shape(f(array_ops.zeros([1, 0]))))


class AggregateIndexedSlicesGradientsTest(test_util.TensorFlowTestCase):

  def _assert_indexed_slices_equal(self, left, right):
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  def testNoGradients(self):
    self.assertIsNone(backprop.aggregate_indexed_slices_gradients([]))

  def testOneGradient(self):
    t = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    result = backprop.aggregate_indexed_slices_gradients([t])
    self._assert_indexed_slices_equal(t, result)

  def testMultipleGradients(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop.aggregate_indexed_slices_gradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)

  def testMultipleGradientsWithNones(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(
        constant_op.constant([[0., 0.], [5, 6], [7., 8.]]))
    t3 = None
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop.aggregate_indexed_slices_gradients([t0, t1, t3])
    self._assert_indexed_slices_equal(total, result)

  def testMixedTensorAndIndexedSlices(self):
    t0 = math_ops._as_indexed_slices(
        constant_op.constant([[1., 2.], [0, 0], [3., 4.]]))
    t1 = constant_op.constant([[0., 0.], [5, 6], [7., 8.]])
    total = constant_op.constant([[1., 2.], [5, 6], [10., 12.]])
    result = backprop.aggregate_indexed_slices_gradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)


if __name__ == '__main__':
  test.main()
