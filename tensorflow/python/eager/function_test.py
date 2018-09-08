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

import collections
import functools
from multiprocessing.pool import ThreadPool
import sys

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as tf_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import momentum
from tensorflow.python.training import training_ops
from tensorflow.python.util import compat
from tensorflow.python.util import nest


@test_util.with_c_shapes
class FunctionTest(test.TestCase):

  def testBasic(self):
    matmul = function.defun(math_ops.matmul)
    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq = matmul(t, t, transpose_a=True)
    sq2 = matmul(sq, t, transpose_a=True)
    self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])
    self.assertAllEqual(sq2.numpy().reshape(-1), [52, 76, 74, 108])

  def testBasicGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    @function.defun
    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out = sq(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedInputsGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @function.defun
    def a_times_b(inputs):
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    out = a_times_b(pair({'a': t}, {'b': t}))
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testGraphModeWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @function.defun
    def step():
      def inner():
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    self.assertAllEqual(step(), 2.0)

  def testGraphGradientVariable(self):
    with ops.Graph().as_default(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)

      @function.defun
      def f():
        return 2.0 * v

      node = f()
      grads, = gradients_impl.gradients(node, v)
      v.initializer.run()
      self.assertAllEqual(grads.eval(), 2.0)
      self.assertEqual(grads.shape, v.shape)

  def testGraphEagerIsolation(self):

    @function.defun
    def f():
      v = resource_variable_ops.ResourceVariable(1.0)
      return v.read_value()

    self.assertAllEqual(f(), 1.0)

    with ops.Graph().as_default():
      self.assertEqual(f().shape, ())

  def testBasicGraphFunction(self):
    matmul = function.defun(math_ops.matmul)

    @function.defun
    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = sq.get_concrete_function(t)
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testExecutingStatelessDefunConcurrently(self):

    @function.defun
    def stateless(x):
      return math_ops.multiply(2.0, x)

    pool = ThreadPool()
    inputs = [constant_op.constant(1.0 * x) for x in range(100)]
    outputs = [float(out) for out in pool.map(stateless, inputs)]
    expected = [float(2.0 * x) for x in inputs]
    self.assertSequenceEqual(outputs, expected)

  def testExecutingManyStatelessDefunsConcurrently(self):

    @function.defun
    def stateless(x):
      del x
      return math_ops.multiply(2.0, 2.0)

    pool = ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    outputs = [
        float(out)
        for out in pool.map(stateless, [object() for _ in range(100)])
    ]
    expected = [4.0] * 100
    self.assertSequenceEqual(outputs, expected)

  def testExecutingStatefulDefunConcurrently(self):

    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def stateful(x):
      v.assign(x)

    pool = ThreadPool()
    inputs = [constant_op.constant(0.0)] * 100
    pool.map(stateful, inputs)
    self.assertEqual(float(v.read_value()), 0.0)

  def testExecutingManyStatefulDefunsConcurrently(self):

    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def stateful(x):
      del x
      return v.assign(0.0)

    pool = ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    pool.map(stateful, [object() for _ in range(100)])
    self.assertEqual(float(v.read_value()), 0.0)

  def disabled_testRandomSeed(self):

    @function.defun
    def f():
      return random_ops.random_normal(())

    random_seed.set_random_seed(1)
    x = f()
    self.assertNotEqual(x, f())
    random_seed.set_random_seed(1)
    self.assertAllEqual(f(), x)

  def testSymGradGatherNd(self):
    with ops.Graph().as_default(), self.test_session() as sess:

      @function.defun
      def f(x):
        return array_ops.gather_nd(x, [[0]])

      c = constant_op.constant([[2.]])
      f_c = f(c)
      g, = gradients_impl.gradients(f_c, c)
      self.assertAllEqual(sess.run(g), [[1.0]])

  def testNestedInputsGraphFunction(self):
    matmul = function.defun(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @function.defun
    def a_times_b(inputs):
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    inputs = pair({'a': t}, {'b': t})
    sq_op = a_times_b.get_concrete_function(inputs)
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(inputs)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputGraphFunction(self):
    matmul = function.defun(math_ops.matmul)

    @function.defun
    def sq(a):
      return (matmul(a, a), {'b': constant_op.constant(1.0)})

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = sq.get_concrete_function(t)
    self.assertEqual(sq_op.output_shapes,
                     (tensor_shape.TensorShape([2, 2]),
                      {'b': tensor_shape.TensorShape([])}))
    self.assertEqual(sq_op.output_dtypes,
                     (dtypes.float32, {'b': dtypes.float32}))
    (a, b) = sq_op(t)
    self.assertAllEqual(a, math_ops.matmul(t, t).numpy())
    self.assertAllEqual(b['b'].numpy(), 1.0)

  def testGraphFunctionWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    @function.defun
    def step():
      def inner():
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    step_op = step.get_concrete_function()
    self.assertEqual(step_op.output_dtypes, dtypes.float32)
    self.assertEqual(step_op.output_shapes, tensor_shape.TensorShape([]))
    self.assertAllEqual(step_op(), 2.0)

  def testGraphFunctionNoneOutput(self):
    @function.defun
    def fn(unused_a, unused_b):
      return None

    x = constant_op.constant(1)
    fn_op = fn.get_concrete_function(x, x)
    self.assertEqual(fn_op.output_dtypes, None)
    self.assertEqual(fn_op.output_shapes, None)
    self.assertAllEqual(fn_op(x, x), None)

  @test_util.run_in_graph_and_eager_modes()
  def testDefunCondGradient(self):

    @function.defun
    def f(x):
      return control_flow_ops.cond(x > 0.5, lambda: 2 * x, lambda: 3 * x)

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 2.0)

  @test_util.run_in_graph_and_eager_modes()
  def testGraphLoopGradient(self):

    @function.defun
    def f(x):
      return control_flow_ops.while_loop(lambda _, i: i < 2,
                                         lambda x, i: (2*x, i + 1),
                                         [x, 0])[0]

    with backprop.GradientTape() as t:
      x = constant_op.constant(1.0)
      t.watch(x)
      y = f(x)
    self.assertAllEqual(self.evaluate(t.gradient(y, x)), 4.0)

  def testDefunNumpyArraysConvertedToTensors(self):

    def f(x):
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    defined = function.defun(f)
    defined(x)
    self.assertEqual(len(defined._function_cache), 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertEqual(len(defined._function_cache), 1)

  def testDefunCapturedInt32(self):
    x = constant_op.constant(1, dtype=dtypes.int32)

    @function.defun
    def add_int32s():
      return x + x

    self.assertEqual(2, int(add_int32s()))

  def testDefunReadVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def f():
      return v.read_value()

    self.assertEqual(1.0, float(f()))

  def testDefunAssignAddVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    x = constant_op.constant(2.0)

    @function.defun
    def test_assign_add():
      v.assign_add(x)
      return v.read_value()

    self.assertEqual(3.0, float(test_assign_add()))

  @test_util.run_in_graph_and_eager_modes
  def testTensorInitializationInFunctionRaisesError(self):
    error_msg = ('Tensor-typed variable initializers must either be '
                 'wrapped in an init_scope or callable.*')

    @function.defun
    def tensor_init():
      with self.assertRaisesRegexp(ValueError, error_msg):
        resource_variable_ops.ResourceVariable(constant_op.constant(2.0))

    tensor_init()

  @test_util.run_in_graph_and_eager_modes
  def testCallableTensorInitializationInFunction(self):

    @function.defun
    def tensor_init():
      v = resource_variable_ops.ResourceVariable(
          lambda: constant_op.constant(2.0))
      return v.read_value()

    value = tensor_init()
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(value), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testInitScopeTensorInitializationInFunction(self):

    @function.defun
    def tensor_init():
      with ops.init_scope():
        const = constant_op.constant(2.0)
      v = resource_variable_ops.ResourceVariable(const)
      return v.read_value()

    value = tensor_init()
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(value), 2.0)

  def testDefunShapeInferenceWithCapturedResourceVariable(self):
    v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

    def f():
      x = constant_op.constant([[1, 2], [3, 4]])
      out = math_ops.matmul(v, x)
      self.assertEqual(out.get_shape(), tensor_shape.TensorShape([2, 2]))

    compiled = function.defun(f)
    compiled()

  def testVariableInLoopInFunction(self):

    @function.defun
    def test_function():

      def loop_test(_):
        return False

      def loop_body(_):
        return variable_scope.get_variable('a', shape=())

      return control_flow_ops.while_loop(loop_test, loop_body, [0.0])

    self.assertEqual(test_function().shape, [])

  def testDefunShapeInferenceWithCapturedResourceVariableInGraphMode(self):
    with context.graph_mode():
      v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

      def f():
        x = constant_op.constant([[1, 2], [3, 4]])
        out = math_ops.matmul(v, x)
        self.assertEqual(out.get_shape(), tensor_shape.TensorShape([2, 2]))

      compiled = function.defun(f)
      compiled()

  def testDefunShapeInferenceWithCapturedVariableInGraphMode(self):
    with context.graph_mode():
      v = variables.Variable([[1, 2], [3, 4]])

      def f():
        x = constant_op.constant([[1, 2], [3, 4]])
        out = math_ops.matmul(v, x)
        self.assertEqual(out.get_shape(), tensor_shape.TensorShape([2, 2]))

      # Check that shape inference works while creating the defun
      compiled = function.defun(f)
      compiled()

  @test_util.run_in_graph_and_eager_modes
  def testDefunForcesResourceVariables(self):

    def variable_creator():
      return variables.Variable(0.0).read_value()

    defined = function.defun(variable_creator)
    defined()  # Create the variable.
    self.assertEqual(len(defined.variables), 1)
    self.assertIsInstance(
        defined.variables[0], resource_variable_ops.ResourceVariable)

  def testDefunDifferentiable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def f():
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testDefunCanBeDifferentiatedTwice(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def f():
      return v * v

    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)
    # Ensure that v is watched again.
    self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

  def testGraphModeCaptureVariable(self):
    with context.graph_mode(), self.test_session() as sess:

      class HasAVar(object):

        def __init__(self):
          self.v = resource_variable_ops.ResourceVariable(1.0)

        def call(self):
          return self.v * 2

      o = HasAVar()
      variables.global_variables_initializer().run()
      call = function.defun(o.call)
      op = call()
      self.assertAllEqual(sess.run(op), 2.0)

  def testSymbolicGradientVariableZerosLike(self):
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)

      @function.defun
      def f(x, v):
        v.read_value()
        return x * x

      x = constant_op.constant(1.0)
      l = f(x, v)
      _, dv = gradients_impl.gradients(l, [x, v])
      with self.test_session():
        v.initializer.run()
        self.assertAllEqual(dv.eval(), 0.0)

  def testGraphModeManyFunctions(self):
    with context.graph_mode(), self.test_session():

      @function.defun
      def f(x):
        return x * x

      @function.defun
      def g(x):
        return f(x) + 1

      self.assertAllEqual(g(constant_op.constant(2.0)).eval(), 5.0)

  def testDict(self):

    @function.defun
    def f(x):
      return {'name': x + 1}

    self.assertAllEqual(f(constant_op.constant(1.0))['name'], 2.0)

  def testTensorConversionWithDefun(self):

    @function.defun
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    self.assertAllEqual(5, f(constant_op.constant(2)))

  def testTensorConversionCall(self):

    @function.defun
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    @function.defun
    def g(x):
      return f(f(x))

    self.assertAllEqual(8, g(constant_op.constant(2)))

  def testDefunCallBackprop(self):

    @function.defun
    def f(x):
      return math_ops.add(x, x)

    @function.defun
    def g(x):
      return backprop.gradients_function(f, [0])(x)[0]

    self.assertAllEqual(2, g(constant_op.constant(2.)))

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

    @function.defun
    def g(x):
      return backprop.gradients_function(math_ops.multiply, [0, 1])(x, x)

    def np_g(x):
      return [d.numpy() for d in g(x)]

    x = constant_op.constant(1.)
    self.assertAllEqual([1., 1.], np_g(x))
    self.assertAllEqual([1., 1.], np_g(1.))

  def testCallShape(self):

    @function.defun
    def f(x):
      return x + 1

    @function.defun
    def g(x):
      x = f(x)
      self.assertEqual(x.shape.as_list(), [])
      return None

    g(constant_op.constant(1.0))

  def testNestedDefunWithNoOutputAndTapedInput(self):
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @function.defun
    def f(x):
      # This function intentionally takes a taped variable as input,
      # but does not return any values
      math_ops.add(x, three)

    @function.defun
    def g(x):
      y = math_ops.add(x, three)
      f(y)

    g(three)

  def testGradientTensorConversionWithDefun(self):
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @function.defun
    def f(x):
      return math_ops.add(x, three)

    def g(x):
      return f(x)

    g = backprop.implicit_grad(g)(constant_op.constant(1.0))[0][0]
    self.assertAllEqual(g, 1.0)

  def testGradient(self):
    matmul = function.defun(math_ops.matmul)

    def sq(x):
      return matmul(x, x, transpose_a=True)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    grad_t, = backprop.gradients_function(sq, [0])(t)
    self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

  def testGradientInFunction(self):

    @function.defun
    def f(x):
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(constant_op.constant(1.0)), 2.0)

  def testGatherResourceWithDefun(self):
    with ops.device('cpu:0'):
      v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    defined = function.defun(sum_gather)
    self.assertAllEqual(sum_gather(), defined())

  def testGradientOfGatherWithDefun(self):
    v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    grad_fn = backprop.implicit_grad(sum_gather)
    gradient = grad_fn()
    defun_grad_fn = backprop.implicit_grad(function.defun(sum_gather))
    defun_gradient = defun_grad_fn()
    self.assertEqual(len(gradient), len(defun_gradient))

    gradient = gradient[0][0]
    defun_gradient = defun_gradient[0][0]
    self.assertAllEqual(gradient.values, defun_gradient.values)
    self.assertAllEqual(gradient.indices, defun_gradient.indices)
    self.assertAllEqual(gradient.dense_shape, defun_gradient.dense_shape)

  def testReturningIndexedSlicesWithDefun(self):

    def validate(indexed_slice):
      @function.defun
      def f():
        return indexed_slice

      output = f()
      self.assertTrue(isinstance(output, ops.IndexedSlices))
      self.assertAllEqual(indexed_slice.values, output.values)
      self.assertAllEqual(indexed_slice.indices, output.indices)
      self.assertAllEqual(indexed_slice.dense_shape, output.dense_shape)

      self.assertEqual(
          f.get_concrete_function().output_shapes,
          indexed_slice.values.shape)

    arg = ops.IndexedSlices(
        values=constant_op.constant([1, 2]),
        indices=constant_op.constant([0, 1]),
        dense_shape=constant_op.constant([2]))
    validate(arg)

    arg = ops.IndexedSlices(
        values=constant_op.constant([1, 2]),
        indices=constant_op.constant([0, 1]),
        dense_shape=None)
    validate(arg)

  def testIndexedSliceAsArgumentWithDefun(self):

    @function.defun
    def f(indexed_slice):
      return indexed_slice

    def validate(arg):
      output = f(arg)
      self.assertTrue(isinstance(output, ops.IndexedSlices))
      self.assertAllEqual(arg.values, output.values)
      self.assertAllEqual(arg.indices, output.indices)
      self.assertAllEqual(arg.dense_shape, output.dense_shape)

    indexed_slice = ops.IndexedSlices(
        values=constant_op.constant([1]),
        indices=constant_op.constant([0]),
        dense_shape=constant_op.constant([1]))
    validate(indexed_slice)

    # Test that `f` works even when `dense_shape` is None.
    indexed_slice = ops.IndexedSlices(
        values=constant_op.constant([1]),
        indices=constant_op.constant([0]),
        dense_shape=None)
    validate(indexed_slice)

  def testFunctionOnDevice(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    x = constant_op.constant([1.]).gpu()
    f = function.defun(math_ops.add)
    y = f(x, x).cpu()
    self.assertAllEqual(y, [2.])

  @test_util.run_in_graph_and_eager_modes
  def testFunctionWithResourcesOnDifferentDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found.')

    with ops.device('/cpu:0'):
      v_cpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    with ops.device('/gpu:0'):
      v_gpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      cpu_result = math_ops.reduce_sum(array_ops.gather(v_cpu, [1, 2]))
      gpu_result = math_ops.reduce_sum(array_ops.gather(v_gpu, [1, 2]))
      return cpu_result, gpu_result

    defined = function.defun(sum_gather)
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    expected = self.evaluate(sum_gather())
    self.assertAllEqual(expected, self.evaluate(defined()))

  @test_util.run_in_graph_and_eager_modes
  def testOpInFunctionWithConflictingResourceInputs(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found.')

    with ops.device('/cpu:0'):
      v_cpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='cpu')
      v_also_cpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='also_cpu')

    with ops.device('/gpu:0'):
      v_gpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='gpu')

    @function.defun
    def resource_apply_adam():
      training_ops.resource_apply_adam(
          v_cpu.handle,
          v_gpu.handle,
          v_also_cpu.handle,
          1.0,  # beta1_power
          1.0,  # beta2_power
          1.0,  # learning_rate
          1.0,  # beta1
          1.0,  # beta2
          1.0,  # epsilon,
          [1.0, 1.0, 1.0],  # grad
          False)  # use_locking
      return None

    with self.assertRaisesRegexp(
        errors.InvalidArgumentError, 'Could not colocate node with its '
        'resource and reference inputs.*'):
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      self.evaluate(resource_apply_adam())

  def testFunctionHandlesInputsOnDifferentDevices(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = function.defun(array_ops.reshape)
    value = constant_op.constant([1., 2.]).gpu()
    shape = constant_op.constant([2, 1])
    reshaped = reshape(value, shape).cpu()
    self.assertAllEqual(reshaped, [[1], [2]])

  def testFunctionHandlesInputsPlacedOnTheWrongDeviceGracefully(self):
    if not context.context().num_gpus():
      self.skipTest('No GPUs found')

    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = function.defun(array_ops.reshape)
    value = constant_op.constant([1., 2.])
    shape = constant_op.constant([2, 1]).gpu()
    reshape(value, shape)  # No error is raised

  def testDifferentiableFunctionNoneOutputs(self):

    @function.defun
    def my_function(x):
      return x, None

    def wrapper(x):
      return my_function(x)[0]

    g = backprop.gradients_function(wrapper, [0])(constant_op.constant(0.0))
    self.assertAllEqual(g[0], 1.)

    @function.defun
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

  def testNestedDifferentiableFunction(self):
    @function.defun
    def inner_fn(a, b):
      return a * math_ops.add(a, b)

    @function.defun
    def outer_fn(x):
      return inner_fn(x, 1.0)

    x = constant_op.constant(5.0)
    with backprop.GradientTape() as tp:
      tp.watch(x)
      result = outer_fn(x)
    grad = tp.gradient(result, x)

    self.assertAllEqual(grad, 2 * 5.0 + 1.0)

  def testNestedDifferentiableFunctionNoneOutputs(self):
    @function.defun
    def foo(a, b):
      return None, a * math_ops.add(a, b), None, 2*a

    @function.defun
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

  def testNoneOutput(self):

    @function.defun
    def my_function(_):
      return None

    self.assertAllEqual(my_function(1), None)

  def testNestedFunctions(self):
    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @tf_function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    @function.defun
    def add_one(x):
      return add(x, 1)

    self.assertAllEqual(3, add_one(constant_op.constant(2)))

  def testVariableCaptureInNestedFunctions(self):
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int32)

    @function.defun
    def inner_read():
      return v.read_value()

    @function.defun
    def outer():
      return inner_read()

    self.assertEqual(1, int(outer()))

  def testReturnCapturedEagerTensor(self):
    t = constant_op.constant(1)

    @function.defun
    def read():
      return t

    self.assertEqual(1, int(read()))

  def testReturnCapturedGraphTensor(self):
    with context.graph_mode(), self.test_session():
      t = constant_op.constant(1)

      @function.defun
      def read():
        return t

      self.assertEqual(1, int(self.evaluate(read())))

  def testSequenceInputs(self):
    clip_by_global_norm = function.defun(clip_ops.clip_by_global_norm)
    t_list = [constant_op.constant(1.0), constant_op.constant(2.0)]
    clipped_list, global_norm = clip_by_global_norm(t_list,
                                                    constant_op.constant(.2))
    for t in clipped_list:
      self.assertTrue(isinstance(t, ops.Tensor))
    self.assertTrue(isinstance(global_norm, ops.Tensor))

  def testNestedSequenceInputs(self):

    def my_op(inputs):
      a, b, c = inputs
      e, f = b
      g, h = e
      return [a + a, [tuple([f + f, g + g]), h + h], c + c], a + f + g + h + c

    my_eager_op = function.defun(my_op)
    ret = my_eager_op([
        constant_op.constant(1), [(constant_op.constant(2),
                                   constant_op.constant(3)),
                                  constant_op.constant(4)],
        constant_op.constant(5)
    ])
    self.assertEqual(len(ret), 2)
    self.assertAllEqual(ret[0][0], 2)
    self.assertAllEqual(ret[0][1][0][0], 8)
    self.assertAllEqual(ret[0][1][0][1], 4)
    self.assertTrue(isinstance(ret[0][1][0], tuple))
    self.assertAllEqual(ret[0][1][1], 6)
    self.assertAllEqual(ret[0][2], 10)
    self.assertAllEqual(ret[1], 15)

  def testVariableNamesRespectNameScopesWithDefun(self):
    @function.defun
    def create_variable():
      with ops.name_scope('foo'):
        v = resource_variable_ops.ResourceVariable(0.0, name='bar')
      self.assertEqual(v.name, 'foo/bar:0')

    create_variable()

  def testVariableNamesRespectNameScopesWithDefunInGraph(self):
    with context.graph_mode():
      @function.defun
      def create_variable():
        with ops.name_scope('foo'):
          v = resource_variable_ops.ResourceVariable([1.0, 2.0], name='bar')
        self.assertEqual(v.name, 'foo/bar:0')

      with ops.get_default_graph().as_default():
        create_variable()

  def testLayerInDefun(self):
    conv = convolutional.Conv2D(
        filters=1,
        kernel_size=2,
        kernel_initializer=init_ops.ones_initializer(),
        bias_initializer=init_ops.zeros_initializer())

    @function.defun
    def model(x):
      return conv(x)

    x = array_ops.ones([1, 2, 2, 1])
    y = model(x)
    self.assertAllEqual([[[[4.0]]]], y.numpy())

  # Note: The ConfigProto below unfortunately only configures graph
  # construction. Eager's configuration is controlled in `__main__`.
  @test_util.run_in_graph_and_eager_modes(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  def testDeviceAnnotationsRespected(self):

    def multi_device_fn():
      with ops.device('/cpu:0'):
        s0 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      with ops.device('/cpu:1'):
        s1 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      with ops.device('/cpu:2'):
        s2 = iterator_ops.Iterator.from_structure(
            (dtypes.float32,)).string_handle()
      s3 = iterator_ops.Iterator.from_structure(
          (dtypes.float32,)).string_handle()
      return s0, s1, s2, s3

    defined = function.defun(multi_device_fn)
    outputs = self.evaluate(defined())
    self.assertEqual(len(defined._function_cache), 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])

    with ops.device('/cpu:3'):
      outputs = self.evaluate(defined())
    self.assertEqual(len(defined._function_cache), 2)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
    self.assertIn(compat.as_bytes('CPU:3'), outputs[3])

    # This should retrieve the call-site-device agnostic function
    defined()
    self.assertEqual(len(defined._function_cache), 2)

    # And this should retrieve the function created for '/cpu:3'
    with ops.device('/cpu:3'):
      defined()
    self.assertEqual(len(defined._function_cache), 2)

  @test_util.run_in_graph_and_eager_modes(
      config=config_pb2.ConfigProto(device_count={'CPU': 2}))
  def testCallingGraphFunctionOnIncompatibleDeviceRaisesError(self):

    def func():
      return constant_op.constant(0)

    defined = function.defun(func)
    with ops.device('cpu:0'):
      cpu_graph_function = defined.get_concrete_function()

    with ops.device('cpu:0'):
      self.assertEqual(
          self.evaluate(cpu_graph_function()), self.evaluate(func()))

    with self.assertRaisesRegexp(
        ValueError,
        'The current device stack does not match the device stack under '
        'which the TensorFlow function \'.*func.*\' was created.\n'
        'Current device stack: .*\n.*func.* device stack.*'):
      with ops.device('cpu:1'):
        cpu_graph_function()

    with self.assertRaisesRegexp(
        ValueError,
        'The current device stack does not match the device stack under '
        'which the TensorFlow function \'.*func.*\' was created.\n'
        'Current device stack: .*\n.*func.* device stack.*'):
      with ops.device(None):
        cpu_graph_function()

    default_graph_function = defined.get_concrete_function()
    self.assertEqual(
        self.evaluate(default_graph_function()), self.evaluate(func()))

    with self.assertRaisesRegexp(
        ValueError,
        'The current device stack does not match the device stack under '
        'which the TensorFlow function \'.*func.*\' was created.\n'
        'Current device stack: .*\n.*func.* device stack.*'):
      with ops.device('cpu:1'):
        default_graph_function()

  @test_util.run_in_graph_and_eager_modes
  def testColocateWithRespected(self):
    # TODO(b/113291792): Use multiple CPUs instead of a GPU.
    if not context.context().num_gpus():
      self.skipTest('No GPUs found.')

    with ops.device('cpu:0'):
      x = constant_op.constant(1.0)

    with ops.device('gpu:0'):
      y = constant_op.constant(1.0)

    @function.defun
    def foo():
      return iterator_ops.Iterator.from_structure(
          (dtypes.float32,)).string_handle()

    with ops.colocate_with(x):
      self.assertIn(compat.as_bytes('CPU:0'), self.evaluate(foo()))

    with ops.colocate_with(y):
      self.assertIn(compat.as_bytes('GPU:0'), self.evaluate(foo()))

  def testVariablesAreTracked(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    def foo(x):
      return v * x

    defined = function.defun(foo)

    x = constant_op.constant([1.0])
    self.assertAllEqual(defined.variables, [])
    _ = defined(x)
    self.assertAllEqual(defined.variables, [v])

    x = constant_op.constant([1.0, 2.0])
    _ = defined(x)  # ensure the variables list remains the same
    self.assertAllEqual(defined.variables, [v])

  def testPythonFunctionWithDefaultArgs(self):

    def func(foo, bar=1, baz=2):
      del foo
      del bar
      del baz
      return

    defined = function.defun(func)
    defined(0, baz=20)

    def cache_keys():
      """Sanitizes cache keys of non-input metadata."""
      return tuple(key[:3] for key in defined._function_cache)

    # `True` corresponds to the fact that we're executing eagerly
    self.assertIn((0, 1, 20), cache_keys())

    defined(1)  # bar=1, baz=2
    self.assertIn((1, 1, 2), cache_keys())

    # This matches the previous call.
    defined(foo=1)
    self.assertEqual(len(defined._function_cache), 2)

    defined(1, 2, 3)
    self.assertIn((1, 2, 3), cache_keys())

    # This matches the previous call.
    defined(1, bar=2, baz=3)
    self.assertEqual(len(defined._function_cache), 3)

    # This matches the previous call.
    defined(1, baz=3, bar=2)
    self.assertEqual(len(defined._function_cache), 3)

  def testFunctoolsPartialUnwrappedCorrectly(self):

    def full_function(a, b, c=3):
      return a, b, c

    partial = functools.partial(full_function, 1, c=3)
    a, b, c = partial(2)

    defined = function.defun(partial)
    func_a, func_b, func_c = defined(2)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureWithCompatibleInputs(self):

    def foo(a):
      self.assertEqual(a.shape, (2,))
      return a

    signature = [tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = function.defun(foo, input_signature=signature)
    a = array_ops.ones([2])
    out = defined(a)
    self.assertEqual(len(defined._function_cache), 1)
    self.assertAllEqual(out, a)

    def bar(a):
      self.assertEqual(a._shape_tuple(), (2, None))
      return a

    signature = [tensor_spec.TensorSpec((2, None), dtypes.float32)]
    defined = function.defun(bar, input_signature=signature)
    a = array_ops.ones([2, 1])
    out = defined(a)
    self.assertEqual(len(defined._function_cache), 1)
    self.assertAllEqual(out, a)

    # Changing the second dimension shouldn't create a new function.
    b = array_ops.ones([2, 3])
    out = defined(b)
    self.assertEqual(len(defined._function_cache), 1)
    self.assertAllEqual(out, b)

  def testNestedInputSignatures(self):

    def foo(a, b):
      self.assertEqual(a[0]._shape_tuple(), (2, None))
      self.assertEqual(a[1]._shape_tuple(), (2, None))
      self.assertEqual(b._shape_tuple(), (1,))
      return [a, b]

    signature = [[tensor_spec.TensorSpec((2, None), dtypes.float32)] * 2,
                 tensor_spec.TensorSpec((1,), dtypes.float32)]
    defined = function.defun(foo, input_signature=signature)
    a = array_ops.ones([2, 1])
    b = array_ops.ones([1])
    out = defined([a, a], b)
    self.assertEqual(len(defined._function_cache), 1)
    nest.assert_same_structure(out, [[a, a], b])
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], a)
    self.assertAllEqual(out[1], b)

    # Changing the unspecified dimensions shouldn't create a new function.
    a = array_ops.ones([2, 3])
    b = array_ops.ones([2, 5])
    c = array_ops.ones([1])
    out = defined([a, b], c)
    self.assertEqual(len(defined._function_cache), 1)
    nest.assert_same_structure(out, [[a, b], c])
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], b)
    self.assertAllEqual(out[1], c)

    def bar(a):
      self.assertEqual(a['a']._shape_tuple(), (2, None))
      self.assertEqual(a['b']._shape_tuple(), (2, None))
      self.assertEqual(a['c']._shape_tuple(), (1,))
      return a

    signature = [{
        'a': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'b': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'c': tensor_spec.TensorSpec((1,), dtypes.float32)
    }]
    a = array_ops.ones([2, 3])
    b = array_ops.ones([1])
    inputs = {'a': a, 'b': a, 'c': b}
    defined = function.defun(bar, input_signature=signature)
    out = defined(inputs)
    nest.assert_same_structure(out, inputs)
    self.assertAllEqual(out['a'], inputs['a'])
    self.assertAllEqual(out['b'], inputs['b'])
    self.assertAllEqual(out['c'], inputs['c'])

  def testInputSignatureMustBeSequenceOfTensorSpecs(self):

    def foo(a, b):
      del a
      del b

    # Signatures must consist exclusively of `TensorSpec` objects.
    signature = [(2, 3), tensor_spec.TensorSpec([2, 3], dtypes.float32)]
    with self.assertRaisesRegexp(TypeError, 'Invalid input_signature.*'):
      function.defun(foo, input_signature=signature)

    # Signatures must be either lists or tuples on their outermost levels.
    signature = {'t1': tensor_spec.TensorSpec([], dtypes.float32)}
    with self.assertRaisesRegexp(TypeError, 'input_signature must be either a '
                                 'tuple or a list.*'):
      function.defun(foo, input_signature=signature)

  def testInputsIncompatibleWithSignatureRaisesError(self):

    def foo(a):
      return a

    signature = [tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = function.defun(foo, input_signature=signature)

    # Invalid shapes.
    with self.assertRaisesRegexp(ValueError, 'Python inputs incompatible.*'):
      defined(array_ops.ones([3]))

    with self.assertRaisesRegexp(ValueError, 'Python inputs incompatible.*'):
      defined(array_ops.ones([2, 1]))

    # Wrong number of arguments.
    with self.assertRaisesRegexp(ValueError,
                                 'Structure of Python function inputs.*'):
      defined(array_ops.ones([2]), array_ops.ones([2]))
    with self.assertRaisesRegexp(ValueError,
                                 'Structure of Python function inputs.*'):
      defined()

  def testInputSignatureForFunctionWithNonTensorInputsNotAllowed(self):

    def foo(a, training=True):
      if training:
        return a
      else:
        return -1.0 * a

    signature = [tensor_spec.TensorSpec([], dtypes.float32)] * 2
    defined = function.defun(foo, input_signature=signature)
    a = constant_op.constant(1.0)
    with self.assertRaisesRegexp(
        ValueError, 'When input_signature is provided, '
        'all inputs to the Python function must be Tensors.'):
      defined(a, training=True)

  def testInputSignatureWithKeywordPositionalArgs(self):

    @function.defun(input_signature=[
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.int64)
    ])
    def foo(flt, integer):
      return flt, integer

    flt = constant_op.constant(1.0)
    integer = constant_op.constant(2, dtypes.int64)

    out1, out2 = foo(flt, integer)
    self.assertEqual(len(foo._function_cache), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt=flt, integer=integer)
    self.assertEqual(len(foo._function_cache), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(integer=integer, flt=flt)
    self.assertEqual(len(foo._function_cache), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt, integer=integer)
    self.assertEqual(len(foo._function_cache), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

  def testInputSignatureWithKeywordArgsFails(self):

    def foo(a, **kwargs):
      del a
      del kwargs

    with self.assertRaisesRegexp(
        ValueError, 'Cannot define a TensorFlow function from a Python '
        'function with keyword arguments when input_signature.*'):
      function.defun(
          foo,
          input_signature=[
              tensor_spec.TensorSpec([], dtypes.float32),
              tensor_spec.TensorSpec([], dtypes.int64)
          ])

  def testTensorKeywordArguments(self):

    def foo(a, b):
      del a
      return b

    defined = function.defun(foo)
    a = constant_op.constant(2.0)
    b = constant_op.constant([1.0, 2.0])
    one = defined(a, b)
    self.assertEqual(len(defined._function_cache), 1)

    two = defined(a=a, b=b)
    self.assertEqual(len(defined._function_cache), 1)

    three = defined(b=b, a=a)
    self.assertEqual(len(defined._function_cache), 1)

    four = defined(a, b=b)
    self.assertEqual(len(defined._function_cache), 1)

    # The next call corresponds to a new input signature, hence
    # we expect another function to be defined.
    five = defined(b, a)
    self.assertEqual(len(defined._function_cache), 2)

    six = defined(a=b, b=a)
    self.assertEqual(len(defined._function_cache), 2)

    seven = defined(b=a, a=b)
    self.assertEqual(len(defined._function_cache), 2)

    self.assertAllEqual(one, [1.0, 2.0])
    self.assertAllEqual(two, [1.0, 2.0])
    self.assertAllEqual(three, [1.0, 2.0])
    self.assertAllEqual(four, [1.0, 2.0])
    self.assertAllEqual(five, 2.0)
    self.assertAllEqual(six, 2.0)
    self.assertAllEqual(seven, 2.0)

  def testGradientWithKeywordArguments(self):
    matmul = function.defun(math_ops.matmul)

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

    @function.defun
    def f(x):
      return backprop.gradients_function(lambda y: y * y, [0])(x)[0]

    self.assertAllEqual(f(x=constant_op.constant(1.0)), 2.0)

  def testDefuningInstanceMethod(self):

    integer = constant_op.constant(2, dtypes.int64)

    class Foo(object):

      def one(self, tensor):
        return tensor

      @function.defun
      def two(self, tensor, other=integer):
        return self.one(tensor), other

    foo = Foo()
    t = constant_op.constant(1.0)
    one, two = foo.two(t)
    self.assertEqual(one.numpy(), 1.0)
    self.assertEqual(two.numpy(), 2)

  def testDefuningInstanceMethodWithDefaultArgument(self):

    integer = constant_op.constant(2, dtypes.int64)

    class Foo(object):

      @function.defun
      def func(self, other=integer):
        return other

    foo = Foo()
    self.assertEqual(foo.func().numpy(), int(integer))

  def testPythonCallWithSideEffects(self):
    state = []

    @function.defun
    def side_effecting_function():
      state.append(0)

    side_effecting_function()
    self.assertAllEqual(state, [0])

    # The second invocation should call the graph function, which shouldn't
    # trigger the list append.
    side_effecting_function()
    self.assertAllEqual(state, [0])

    # Whereas calling the python function directly should create a side-effect.
    side_effecting_function.python_function()
    self.assertAllEqual(state, [0, 0])


@test_util.with_c_shapes
class AutomaticControlDependenciesTest(test.TestCase):

  def testBasic(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      with function.AutomaticControlDependencies() as c:
        v.assign(v + 1)
        v.assign(2 * v)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(), 4.0)

  def testCondMustRun(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1)
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 6.0)

  def testCondMustRunSeparateRead(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1)
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        one = constant_op.constant(1.0)
        one = c.mark_as_return(one)
      one.eval(feed_dict={p: False})
      self.assertAllEqual(v.read_value().eval(), 5.0)
      one.eval(feed_dict={p: True})
      self.assertAllEqual(v.read_value().eval(), 6.0)

  def testCondNested(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      q = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1, name='true')
          return 1.0

        def false_fn():

          def inner_true_fn():
            v.assign(v * 2, name='false_true')
            return 2.0

          def inner_false_fn():
            v.assign(v * 3, name='false_false')
            return 3.0

          control_flow_ops.cond(q, inner_true_fn, inner_false_fn)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        with ops.name_scope('final'):
          val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: False}), 3.0)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: True}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: True}), 7.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: False}), 8.0)

  def testCondOneBranch(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 5.0)

  def testCondOneBranchUpdateBefore(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:
        v.assign(v * 2)

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 12.0)

  def testCondOneBranchUpdateAfter(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()
      p = array_ops.placeholder(dtype=dtypes.bool)
      with function.AutomaticControlDependencies() as c:

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        v.assign(v * 2)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 10.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 20.0)

  def testDefunWhileLoopWithCapturedLoopVars(self):
    n = 3
    x = constant_op.constant(list(range(n)))

    @function.defun
    def loop():
      c = lambda i, x: i < n
      b = lambda i, x: (i + 1, x + 1)
      i, out = control_flow_ops.while_loop(c, b, (0, x))
      return i, out

    i, out = loop()
    self.assertEqual(int(i), 3)
    self.assertAllEqual(out, [3, 4, 5])

  def testDecorator(self):
    with context.graph_mode(), self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()

      @function.automatic_control_dependencies
      def f():
        v.assign(v + 1)
        v.assign(2 * v)
        return v.read_value()

      self.assertAllEqual(f().eval(), 4.0)

  def testOptimizerInDefun(self):
    def loss(v):
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @function.defun
    def train():
      v = resource_variable_ops.ResourceVariable(1.0)
      grad = backprop.implicit_grad(loss)(v)
      optimizer.apply_gradients(grad)
      return v.read_value()

    value = train()
    self.assertEqual(value.numpy(), -1.0)

  def testReturningNonTensorRaisesError(self):
    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)
    optimizer.apply_gradients = function.defun(optimizer.apply_gradients)
    v = resource_variable_ops.ResourceVariable(1.0)
    grad = backprop.implicit_grad(lambda v: v**2)(v)

    with self.assertRaisesRegexp(TypeError,
                                 '.*must return zero or more Tensors.*'):
      # TODO(akshayka): We might want to allow defun-ing Python functions
      # that return operations (and just execute the op instead of running it).
      optimizer.apply_gradients(grad)

  # TODO(b/111663004): This should work when the outer context is graph
  # building.
  def testOptimizerNonSlotVarsInDefunNoError(self):
    def loss(v):
      return v**2

    optimizer = adam.AdamOptimizer(learning_rate=1.0)

    @function.defun
    def train():
      v = resource_variable_ops.ResourceVariable(1.0)
      grad = backprop.implicit_grad(loss)(v)
      optimizer.apply_gradients(grad)
      return v.read_value()

    train()

  def testOptimizerInDefunWithCapturedVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    def loss():
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @function.defun
    def train():
      grad = backprop.implicit_grad(loss)()
      optimizer.apply_gradients(grad)

    train()
    self.assertEqual(v.numpy(), -1.0)

  def testFunctionModifiesInputList(self):
    # Tests on `list` methods that do in place modification, except `list.sort`
    # since it cannot even be "defunned" in the first place

    def get_list():
      return [constant_op.constant(0.), constant_op.constant(1.)]

    expected_msg = (
        'Function to be traced should not modify structure of input '
        'arguments. Check if your function has list and dictionary '
        'operations that alter input arguments, '
        'such as `list.pop`, `list.append`')

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def append(l):
        l.append(constant_op.constant(0.))

      append(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def extend(l):
        l.extend([constant_op.constant(0.)])

      extend(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def insert(l):
        l.insert(0, constant_op.constant(0.))

      insert(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def pop(l):
        l.pop()

      pop(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def reverse(l):
        l.reverse()

      reverse(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def remove(l):
        l.remove(l[0])

      remove(get_list())

    # `list.clear` is a method that is in Py3 but not Py2
    if sys.version.startswith('3'):

      with self.assertRaisesRegexp(ValueError, expected_msg):

        @function.defun
        def clear(l):
          l.clear()

        clear(get_list())

    # One last test for keyword arguments
    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def kwdappend(**kwargs):
        l = kwargs['l']
        l.append(constant_op.constant(0.))

      kwdappend(l=get_list())

  def testFunctionModifiesInputDict(self):

    def get_dict():
      return {'t1': constant_op.constant(0.), 't2': constant_op.constant(1.)}

    expected_msg = (
        'Function to be traced should not modify structure of input '
        'arguments. Check if your function has list and dictionary '
        'operations that alter input arguments, '
        'such as `list.pop`, `list.append`')

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def clear(m):
        m.clear()

      clear(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def pop(m):
        m.pop('t1')

      pop(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def popitem(m):
        m.popitem()

      popitem(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def update(m):
        m.update({'t1': constant_op.constant(3.)})

      update(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def setdefault(m):
        m.setdefault('t3', constant_op.constant(3.))

      setdefault(get_dict())

  def testFunctionModifiesInputNest(self):
    # Test on functions that modify structure of nested input arguments
    expected_msg = (
        'Function to be traced should not modify structure of input '
        'arguments. Check if your function has list and dictionary '
        'operations that alter input arguments, '
        'such as `list.pop`, `list.append`')

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @function.defun
      def modify(n):
        n[0]['t1'].append(constant_op.constant(1.))

      nested_input = [{
          't1': [constant_op.constant(0.),
                 constant_op.constant(1.)],
      },
                      constant_op.constant(2.)]

      modify(nested_input)

    with self.assertRaisesRegexp(ValueError, expected_msg):

      # The flat list doesn't change whereas the true structure changes
      @function.defun
      def modify_same_flat(n):
        n[0].append(n[1].pop(0))

      nested_input = [[constant_op.constant(0.)],
                      [constant_op.constant(1.),
                       constant_op.constant(2.)]]

      modify_same_flat(nested_input)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
