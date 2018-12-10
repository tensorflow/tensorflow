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

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class FunctionGradientsTest(test.TestCase, parameterized.TestCase):

  def testGraphModeWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @def_function.function
    def step():
      def inner():
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    self.assertAllEqual(step(), 2.0)

  def testGraphGradientVariable(self):
    with ops.Graph().as_default(), self.cached_session():
      v = variables.Variable(1.0)

      @def_function.function
      def f():
        return 2.0 * v

      node = f()
      grads, = gradients_impl.gradients(node, v)
      v.initializer.run()
      self.assertAllEqual(grads.eval(), 2.0)
      self.assertEqual(grads.shape, v.shape)

  def testSymGradGatherNd(self):
    with ops.Graph().as_default(), self.cached_session() as sess:

      @def_function.function
      def f(x):
        return array_ops.gather_nd(x, [[0]])

      c = constant_op.constant([[2.]])
      f_c = f(c)
      g, = gradients_impl.gradients(f_c, c)
      self.assertAllEqual(self.evaluate(g).values, [[1.0]])

  def testNoSymGradNestedDefun(self):

    @def_function.function
    def outer():

      @def_function.function
      def f(x):
        return array_ops.gather_nd(x, [[0]])

      c = constant_op.constant([[2.]])
      f_c = f(c)
      g, = gradients_impl.gradients(f_c, c)
      self.assertIsInstance(g, ops.IndexedSlices)

    outer()

  def testGraphFunctionWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @def_function.function
    def step():
      def inner():
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    step_op = step.get_concrete_function()
    self.assertEqual(step_op.output_dtypes, dtypes.float32)
    self.assertEqual(step_op.output_shapes, tensor_shape.TensorShape([]))
    self.assertAllEqual(step_op(), 2.0)

  @test_util.run_in_graph_and_eager_modes()
  def testDefunCondGradient(self):

    @def_function.function
    def f(x):
      return control_flow_ops.cond(x > 0.5, lambda: 2 * x, lambda: 3 * x)

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 2.0)

  @test_util.run_in_graph_and_eager_modes()
  def testGraphLoopGradient(self):

    @def_function.function
    def f(x):
      return control_flow_ops.while_loop(lambda _, i: i < 2,
                                         lambda x, i: (2*x, i + 1),
                                         [x, 0])[0]

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 4.0)

  def testDefunDifferentiable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def f():
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testDefunCanBeDifferentiatedTwice(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def f():
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)
    # Ensure that v is watched again.
    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testSymbolicGradientVariableNoneNotZerosLike(self):
    with ops.Graph().as_default():
      v = variables.Variable(1.0)

      @def_function.function
      def f(x, v):
        v.read_value()
        return x * x

      x = constant_op.constant(1.0)
      l = f(x, v)
      _, dv = gradients_impl.gradients(l, [x, v])
      with self.cached_session():
        v.initializer.run()
        self.assertEqual(dv, None)

  def testDefunCallBackprop(self):

    @def_function.function
    def f(x):
      return math_ops.add(x, x)

    @def_function.function
    def g(x):
      return backprop.gradients_function(f, [0])(x)[0]

    self.assertAllEqual(2, g(constant_op.constant(2.)))

  @test_util.run_v1_only('b/120545219')
  def testGraphModeEagerGradError(self):
    with context.graph_mode():
      def f():
        x = variable_scope.get_variable(
            'v', initializer=constant_op.constant(1.0))
        return x * constant_op.constant(2.0)

      with self.assertRaisesRegexp(ValueError,
                                   'No trainable variables were accessed'):
        backprop.implicit_val_and_grad(f)()

  def testDefunCallBackpropUsingSameObjectForMultipleArguments(self):

    @def_function.function
    def g(x):
      return backprop.gradients_function(math_ops.multiply, [0, 1])(x, x)

    def np_g(x):
      return [d.numpy() for d in g(x)]

    x = constant_op.constant(1.)
    self.assertAllEqual([1., 1.], np_g(x))
    self.assertAllEqual([1., 1.], np_g(1.))

  def testGradientTensorConversionWithDefun(self):
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @def_function.function
    def f(x):
      return math_ops.add(x, three)

    def g(x):
      return f(x)

    g = backprop.implicit_grad(g)(constant_op.constant(1.0))[0][0]
    self.assertAllEqual(g, 1.0)

  def testGradient(self):
    matmul = def_function.function(math_ops.matmul)

    def sq(x):
      return matmul(x, x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

  def testGradientInFunction(self):

    @def_function.function
    def f(x):
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(constant_op.constant(1.0)), 2.0)

  def testGradientOfGatherWithDefun(self):
    v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    grad_fn = backprop.implicit_grad(sum_gather)
    gradient = grad_fn()
    defun_grad_fn = backprop.implicit_grad(def_function.function(sum_gather))
    defun_gradient = defun_grad_fn()
    self.assertEqual(len(gradient), len(defun_gradient))

    gradient = gradient[0][0]
    defun_gradient = defun_gradient[0][0]
    self.assertAllEqual(gradient.values, defun_gradient.values)
    self.assertAllEqual(gradient.indices, defun_gradient.indices)
    self.assertAllEqual(gradient.dense_shape, defun_gradient.dense_shape)

  def testDifferentiableFunctionNoneOutputs(self):

    @def_function.function
    def my_function(x):
      return x, None

    def wrapper(x):
      return my_function(x)[0]

    g = backprop.gradients_function(wrapper, [0])(constant_op.constant(0.0))
    self.assertAllEqual(g[0], 1.)

    @def_function.function
    def foo(a):
      return None, a * a

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      none, r = foo(x)
    g = tp.gradient(r, x)

    self.assertIs(none, None)
    self.assertAllEqual(r, 25.0)
    self.assertAllEqual(g, 2 * 5.0)

  @test_util.run_in_graph_and_eager_modes
  def testNestedDifferentiableFunction(self):
    @def_function.function
    def inner_fn(a, b):
      return a * math_ops.add(a, b)

    @def_function.function
    def outer_fn(x):
      return inner_fn(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunction(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def outer_fn(x):
      return middle_fn(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionWithMultipleGradCalls(self):
    @def_function.function
    def inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def middle_fn(a, b):
      return math_ops.mul(a, inner_fn(a, b))

    @def_function.function
    def outer_fn(x):
      return middle_fn(x, 3.0)

    x = constant_op.constant(5.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
    self.assertAllEqual(middle_fn(3.0, x), 3.0 * (3.0 + 5.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)

    y = constant_op.constant(4.0)
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 3.0)

    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = inner_fn(y, y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def outer_fn(x):
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInNestedDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def almost_outer_fn(x):
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    @def_function.function
    def outer_fn(x):
      return almost_outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionGradientTapeInMultNestedDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def almost_outer_fn(x):
      with backprop.GradientTape() as tp:
        tp.watch(x)
        result = middle_fn(x, 1.0)
      grad = tp.gradient(result, x)
      return grad

    @def_function.function
    def outer_fn(x):
      return almost_outer_fn(x)

    @def_function.function
    def outer_outer_fn(x):
      return outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def outer_fn(x):
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInNestedDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def almost_outer_fn(x):
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    @def_function.function
    def outer_fn(x):
      return almost_outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testDeeplyNestedDifferentiableFunctionTFGradientInMultNestedDefun(self):
    @def_function.function
    def inner_inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def inner_fn(a, b):
      return inner_inner_fn(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def almost_outer_fn(x):
      result = middle_fn(x, 1.0)
      return gradients_impl.gradients(result, [x])[0]

    @def_function.function
    def outer_fn(x):
      return almost_outer_fn(x)

    @def_function.function
    def outer_outer_fn(x):
      return outer_fn(x)

    x = constant_op.constant(5.0)
    grad = outer_outer_fn(x)
    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariable(self):
    var = variables.Variable(constant_op.constant(1.0))

    @def_function.function
    def inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def middle_fn(a, b):
      return a * inner_fn(a, b)

    @def_function.function
    def outer_fn(x):
      return middle_fn(x, var)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariableMultipleGradCalls(self):
    v = variables.Variable(constant_op.constant(3.0))

    @def_function.function
    def inner_fn(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def middle_fn(a, b):
      return math_ops.mul(a, inner_fn(a, b))

    @def_function.function
    def outer_fn(x):
      return middle_fn(x, v)

    x = constant_op.constant(5.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)
    self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
    self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))

    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 3.0)

    y = constant_op.constant(4.0)
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 3.0)

    v.assign(constant_op.constant(1.5))
    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = outer_fn(y)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 2 * 4.0 + 1.5)

    with backprop.GradientTape() as tp:
      tp.watch(y)
      result = inner_fn(y, v)
    grad = tp.gradient(result, y)

    self.assertAllEqual(grad, 1.0)

  def testDeeplyNestedDifferentiableFunctionWithVariableMultipleTFGrads(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(3.0)
      v.initializer.run()

      @def_function.function
      def inner_fn(a, b):
        return math_ops.add(a, b)

      @def_function.function
      def middle_fn(a, b):
        return math_ops.mul(a, inner_fn(a, b))

      @def_function.function
      def outer_fn(x):
        return middle_fn(x, v)

      x = constant_op.constant(5.0)
      self.assertAllEqual(outer_fn(x).eval(), 5.0 * (5.0 + 3.0))

      grad, = gradients_impl.gradients(outer_fn(x), x)

      self.assertAllEqual(grad, 2 * 5.0 + 3.0)
      self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
      self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))

      grad, = gradients_impl.gradients(outer_fn(x), x)

      self.assertAllEqual(grad, 2 * 5.0 + 3.0)

      y = constant_op.constant(4.0)
      grad, = gradients_impl.gradients(outer_fn(y), y)
      self.assertAllEqual(grad, 2 * 4.0 + 3.0)

      self.evaluate(v.assign(constant_op.constant(1.5)))
      grad, = gradients_impl.gradients(outer_fn(y), y)

      self.assertAllEqual(grad, 2 * 4.0 + 1.5)

      grad, = gradients_impl.gradients(inner_fn(y, v), y)
      self.assertAllEqual(grad, 1.0)

  def testNestedDifferentiableFunctionNoneOutputs(self):
    @def_function.function
    def foo(a, b):
      return None, a * math_ops.add(a, b), None, 2*a

    @def_function.function
    def bar(x):
      return foo(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape(persistent=True) as tp:
      tp.watch(x)
      none1, r1, none2, r2 = bar(x)
    g1 = tp.gradient(r1, x)
    g2 = tp.gradient(r2, x)

    self.assertAllEqual(r1, 30.0)
    self.assertAllEqual(r2, 10.0)
    self.assertIs(none1, None)
    self.assertIs(none2, None)
    self.assertAllEqual(g1, 2 * 5.0 + 1.0)
    self.assertAllEqual(g2, 2.0)

  def testGradientWithKeywordArguments(self):
    matmul = def_function.function(math_ops.matmul)

    def sq(x):
      return matmul(a=x, b=x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(t)
      one = matmul(t, b=t, transpose_a=True)
      two = matmul(b=t, a=t, transpose_a=True)
      three = matmul(a=t, b=t, transpose_a=True)

    for output in [one, two, three]:
      self.assertAllEqual(tape.gradient(output, t), [[6, 6], [14, 14]])

  def testGradientInFunctionWithKeywordArguments(self):

    @def_function.function
    def f(x):
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(x=constant_op.constant(1.0)), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testBackwardNone(self):
    model = variables.Variable(1.0, name='model')
    count = variables.Variable(0)

    @function.defun
    def forward_pass(value):
      count.assign_add(1)
      residuals = value - model
      loss = 0.5 * math_ops.reduce_mean(math_ops.pow(residuals, 2))
      # Note: count is an integer, so its doutput will be None
      return loss, count

    def reduce_fn(x):
      if context.executing_eagerly():
        with backprop.GradientTape() as t:
          loss, count = forward_pass(x)
        return t.gradient(loss, model), count
      loss, count = forward_pass(x)
      grad_only = gradients_impl.gradients(loss, model)
      return grad_only, count

    g, _ = reduce_fn(constant_op.constant([7.0]))

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(nest.flatten(self.evaluate(g)), [-6.0])


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
