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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import tape
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as tf_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent


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

  def testBasicDefunOpGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = function.make_defun_op(sq, t)

    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedInputsDefunOpGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    def a_times_b(inputs):
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    inputs = pair({'a': t}, {'b': t})
    sq_op = function.make_defun_op(a_times_b, inputs)

    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(inputs)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputDefunOpGraphMode(self):
    matmul = function.defun(math_ops.matmul)

    def sq(a):
      return (matmul(a, a), {'b': constant_op.constant(1.0)})

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = function.make_defun_op(sq, t)

    self.assertEqual(sq_op.output_shapes,
                     (tensor_shape.TensorShape([2, 2]),
                      {'b': tensor_shape.TensorShape([])}))
    self.assertEqual(sq_op.output_dtypes,
                     (dtypes.float32, {'b': dtypes.float32}))
    (a, b) = sq_op(t)
    self.assertAllEqual(a, math_ops.matmul(t, t).numpy())
    self.assertAllEqual(b['b'].numpy(), 1.0)

  def testDefunOpGraphModeWithGradients(self):
    v = resource_variable_ops.ResourceVariable(1.0, name='v')

    def step():
      def inner():
        return v * v

      return backprop.implicit_grad(inner)()[0][0]

    step_op = function.make_defun_op(step)

    self.assertEqual(step_op.output_dtypes, dtypes.float32)
    self.assertEqual(step_op.output_shapes, tensor_shape.TensorShape([]))
    self.assertAllEqual(step_op(), 2.0)

  def testDefunOpGraphModeNoneOutput(self):
    def fn(unused_a, unused_b):
      return None

    x = constant_op.constant(1)
    fn_op = function.make_defun_op(fn, x, x)

    self.assertEqual(fn_op.output_dtypes, None)
    self.assertEqual(fn_op.output_shapes, None)
    self.assertAllEqual(fn_op(x, x), None)

  def testDefunReadVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def f():
      return v.read_value()

    self.assertEqual(1.0, float(f()))

  def testDefunAssignAddVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @function.defun
    def f():
      v.assign_add(2.0)
      return v.read_value()

    self.assertEqual(3.0, float(f()))

  def testDefunShapeInferenceWithCapturedResourceVariable(self):
    v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

    def f():
      x = constant_op.constant([[1, 2], [3, 4]])
      out = math_ops.matmul(v, x)
      self.assertEqual(out.get_shape(), tensor_shape.TensorShape([2, 2]))

    compiled = function.defun(f)
    compiled()

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

  def testGradientTensorConversionWithDefun(self):
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @function.defun
    def f(x):
      return math_ops.add(x, three)

    def g(x):
      tape.watch_variable(three)
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

  def testGradientOfGatherWithDefun(self):
    with ops.device('cpu:0'):
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
      def f():
        return indexed_slice

      output = function.defun(f)()
      self.assertTrue(isinstance(output, ops.IndexedSlices))
      self.assertAllEqual(indexed_slice.values, output.values)
      self.assertAllEqual(indexed_slice.indices, output.indices)
      self.assertAllEqual(indexed_slice.dense_shape, output.dense_shape)

      self.assertEqual(
          function.make_defun_op(f).output_shapes, indexed_slice.values.shape)

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
    v = resource_variable_ops.ResourceVariable(1)

    @function.defun
    def read():
      return v.read_value()

    @function.defun
    def outer():
      return read()

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

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

    @function.defun
    def train():
      v = resource_variable_ops.ResourceVariable(1.0)
      grad = backprop.implicit_grad(loss)(v)
      optimizer.apply_gradients(grad)
      return v.read_value()

    value = train()
    self.assertEqual(value.numpy(), -1.0)

  def testOptimizerInDefunWithCapturedVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    def loss():
      return v**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)

    @function.defun
    def train():
      grad = backprop.implicit_grad(loss)()
      optimizer.apply_gradients(grad)

    train()
    self.assertEqual(v.numpy(), -1.0)


if __name__ == '__main__':
  test.main()
