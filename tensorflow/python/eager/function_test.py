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
import itertools
import multiprocessing.pool
import sys
import time
import weakref

from absl.testing import parameterized
import numpy

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function as tf_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import training_ops
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def total_function_cache(defined):
  # pylint: disable=protected-access
  return (set(defined._function_cache.primary)
          | set(defined._function_cache.arg_relaxed))
  # pylint: enable=protected-access


class MiniModel(keras_training.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name='')
    self.fc = keras.layers.Dense(1, name='fc', kernel_initializer='ones',
                                 bias_initializer='ones')

  def call(self, inputs, training=True):
    return self.fc(inputs)


class DefunnedMiniModel(MiniModel):

  @function.defun
  def call(self, inputs, training=True):
    return super(DefunnedMiniModel, self).call(inputs, training=training)


def _example_indexed_slices_with_dense_shape():
  return indexed_slices.IndexedSlices(
      constant_op.constant([1, 2]), constant_op.constant([0, 1]),
      constant_op.constant([2]))


def _example_indexed_slices_without_dense_shape():
  return indexed_slices.IndexedSlices(
      constant_op.constant([1, 2]), constant_op.constant([0, 1]))


class FunctionTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FunctionTest, self).setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 4 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  def testBasic(self):
    matmul = def_function.function(math_ops.matmul)
    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq = matmul(t, t, transpose_a=True)
    sq2 = matmul(sq, t, transpose_a=True)
    self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])
    self.assertAllEqual(sq2.numpy().reshape(-1), [52, 76, 74, 108])

  def testOnExitCallback(self):
    values = []
    def append_1():
      values.append(1)

    def append_2():
      values.append(2)

    def g(x):
      old_values = list(values)
      ops.add_exit_callback_to_default_func_graph(append_1)
      self.assertEqual(old_values, values)
      return x + 1

    tf_g = def_function.function(g)

    def f(x):
      old_values = list(values)
      ops.add_exit_callback_to_default_func_graph(append_2)
      self.assertEqual(old_values, values)
      return tf_g(x)

    tf_f = def_function.function(f)
    self.assertEmpty(values)
    tf_f(constant_op.constant(1.0))
    self.assertEqual(values, [1, 2])  # Once for g, once for f.
    tf_f(constant_op.constant([1.0]))  # force a retrace
    self.assertEqual(values, [1, 2, 1, 2])  # And again.

  def testCannotAddExitCallbackWhenNotInFunctionScope(self):
    with self.assertRaisesRegexp(RuntimeError, 'when not building a function.'):
      ops.add_exit_callback_to_default_func_graph(lambda: None)

  def testVariable(self):
    v1 = variables.Variable(1.0)
    add = def_function.function(lambda x, v: x + v1 + v)
    v2 = variables.Variable(1.0)
    x = constant_op.constant(1.0)
    r = add(x, v2)
    self.assertEqual(3.0, self.evaluate(r))

  def testVariableOnly(self):
    v = variables.Variable(1.0)
    add = def_function.function(lambda x: x.assign_add(1.0))
    r1 = add(v)
    self.assertEqual(2.0, self.evaluate(r1))
    c = constant_op.constant(1.0)
    with self.assertRaisesRegexp(AttributeError, 'no attribute'):
      add(c)

  def testImplementsAttributeBasic(self):
    v = def_function.function(
        experimental_implements='func')(lambda x, y: x + y)
    with context.graph_mode(), self.cached_session():
      a = array_ops.placeholder(dtypes.float32, ())
      b = array_ops.placeholder(dtypes.float32, ())
      v(a, b)
      gradients_impl.gradients(v(a, b), [a, b])
      fdefs = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(fdefs, 3)
      not_present = 0
      present = 0
      for f in fdefs:
        name = f.signature.name
        if 'forward' in name or 'backward' in name:
          not_present += 1
          self.assertNotIn(function.IMPLEMENTS_ATTRIBUTE_NAME, f.attr, f)
        else:
          present += 1
          self.assertEqual(f.attr[function.IMPLEMENTS_ATTRIBUTE_NAME].s,
                           'func'.encode('ascii'), f)
      self.assertEqual(not_present, 2, fdefs)
      self.assertEqual(present, 1, fdefs)

  def testImplementsAttributeAssertsOnSideInput(self):
    with context.graph_mode(), self.cached_session():
      z = array_ops.zeros(0)
      v = def_function.function(
          experimental_implements='func')(lambda x, y: x + y + z)
      a = array_ops.ones((1.0,))
      b = array_ops.ones((1.0,))
      with self.assertRaisesRegexp(AssertionError,
                                   'variables are always captured'):
        v(a, b)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertEmpty(functions)

  def testImplementsAttributeWorksOnVariables(self):
    with context.graph_mode(), self.cached_session():
      v = def_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable((1.0,))
      b = variables.Variable((1.0,))
      r1 = v(a, b)
      _ = v(a, a)
      functions = ops.get_default_graph().as_graph_def().library.function
      # Verify that we created only one function
      self.assertLen(functions, 1)
      # Verify that eval() reads the current values.
      a.initializer.run()
      b.initializer.run()
      self.assertEqual(r1.eval(), 2)

      a.assign_add([1]).eval()
      self.assertEqual(r1.eval(), 3)

  def testImplementsAttributeWorksOnConstants(self):
    with context.graph_mode(), self.cached_session():
      v = def_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable(1.0)
      r1 = v(a, 2.)
      r2 = v(2., a)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(functions, 1)
      self.assertLen(functions[0].signature.input_arg, 2)
      # Verify that eval() reads the current values.
      a.initializer.run()
      self.assertEqual(r1.eval(), 3)
      self.assertEqual(r2.eval(), 3)

  def testImplementsAttributeSpecializes(self):
    with context.graph_mode(), self.cached_session():
      v = def_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable(1.0)
      r1 = v(a, [2.])
      r2 = v([2., 2], a)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(functions, 2)
      # Ensure that all parameters are still there and haven't been inlined!

      self.assertLen(functions[0].signature.input_arg, 2)
      self.assertLen(functions[1].signature.input_arg, 2)
      # Verify that eval() reads the current values.
      a.initializer.run()
      numpy.testing.assert_equal(r1.eval(), [3.])
      numpy.testing.assert_equal(r2.eval(), [3., 3.])

  def testExternalControlDependency(self):
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.0)
      v.initializer.run()

      op = v.assign_add(1.0)

      @function.defun
      def f():
        with ops.control_dependencies([op]):
          return 1.0

      self.evaluate(f())
      self.assertAllEqual(self.evaluate(v), 2.0)

  def testInputShapeFunctionRelaxation(self):
    unknown_dim = [False]

    @function.defun(experimental_relax_shapes=True)
    def func(a):
      if a._shape_tuple()[0] is None:
        unknown_dim[0] = True
      return a + 1

    func(constant_op.constant([]))
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 1)

    func(constant_op.constant([1.0]))
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 2)

    func(constant_op.constant([1.0, 2.0]))
    self.assertTrue(unknown_dim[0])
    self.assertLen(total_function_cache(func), 2)

  def testInputShapeRelaxationOnInstanceMethod(self):
    # Test that experimental_relax_shapes is passed during
    # instance method bounding.
    unknown_dim = [False]

    class Foo(object):

      @def_function.function(experimental_relax_shapes=True)
      def func(self, a):
        if a._shape_tuple()[0] is None:
          unknown_dim[0] = True
        return a + 1

    foo = Foo()
    foo.func(constant_op.constant([]))
    self.assertFalse(unknown_dim[0])

    foo.func(constant_op.constant([1.0]))
    self.assertFalse(unknown_dim[0])

    foo.func(constant_op.constant([1.0, 2.0]))
    self.assertTrue(unknown_dim[0])

  def testCapturesVariables(self):
    a = variables.Variable(1.0, trainable=False)
    b = variables.Variable(1.0)
    cc = [None]

    @def_function.function
    def f():
      c = cc[0]
      if c is None:
        c = cc[0] = variables.Variable(1.)
      return a + b + c + 1

    cf = f.get_concrete_function()
    c = cc[0]

    captured_variables = {v.experimental_ref() for v in (a, b, c)}
    trainable_variables = {v.experimental_ref() for v in (b, c)}
    self.assertEqual({v.experimental_ref() for v in cf.variables},
                     captured_variables)
    self.assertEqual({v.experimental_ref() for v in cf.trainable_variables},
                     trainable_variables)
    self.assertEqual(cf.variables, cf.graph.variables)
    self.assertEqual(cf.trainable_variables, cf.graph.trainable_variables)

  def testNestedInputShapeFunctionRelaxation(self):
    unknown_dim = [False]

    @function.defun(experimental_relax_shapes=True)
    def func(a_, b_=None):
      del a_  # Only used to check which cache is used.
      self.assertEqual(b_[0]._shape_tuple(), ())
      if b_[1]._shape_tuple()[0] is None:
        unknown_dim[0] = True
      return b_[0] + 1

    a = 'hi'
    b0 = constant_op.constant(1.0)
    func(a, b_=[b0, constant_op.constant([])])
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 1)

    func(a, b_=[b0, constant_op.constant([1.0])])
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 2)

    func(a, b_=[b0, constant_op.constant([1.0, 1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(total_function_cache(func), 2)

    unknown_dim[0] = False

    # Now do the same except with a new a which is not a tensor; this should
    # change the cache key.
    a = 'bye'
    func(a, b_=[b0, constant_op.constant([])])
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 3)

    # Since we already marked a cache miss for a function with the same
    # non-input signatures, here we will immediately start relaxing shapes.
    func(a, b_=[b0, constant_op.constant([1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(total_function_cache(func), 3)

  def testFunctionRelaxationLosesInnerDimWithKerasLayer(self):
    layer = keras.layers.Dense(1)
    fn = def_function.function(experimental_relax_shapes=True)(layer)

    with self.captureWritesToStream(sys.stderr) as printed:
      fn(array_ops.ones((3, 2)))
      self.assertNotIn('ValueError', printed.contents())
    with self.captureWritesToStream(sys.stderr) as printed:
      # Use batch size 2 to trigger a second cache miss on the shape.
      fn(array_ops.ones((2, 2)))
      self.assertNotIn('ValueError', printed.contents())

    # Shape relaxation passes TensorShape([None, None]), which causes layer
    # matmul to fail, due to incompatible dims.  What would have been a graph
    # build time error (layer would complain about the inner dim being 4).
    with self.captureWritesToStream(sys.stderr) as printed:
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   r'Matrix size-incompatible'):
        fn(array_ops.ones((3, 4)))

  def testNestedShapeFunctionRelaxation(self):

    got_shape = [None]

    # The inner function will go through shape relaxation because the shapes it
    # receives will be [1], [2], [3], ...
    @def_function.function(experimental_relax_shapes=True)
    def bar(x_shape):
      got_shape[0] = x_shape._shape_tuple()
      return x_shape

    # The outer function will not go through shape relaxation because the shapes
    # it receives will be [1], [[1]], [[[1]]], ...
    @def_function.function(experimental_relax_shapes=True)
    def foo(ones):
      return bar(array_ops.shape(ones))

    for rank in range(1, 6):
      x_shape = self.evaluate(foo(array_ops.ones([1] * rank)))
      self.assertAllEqual(x_shape, [1] * rank)
      if rank < 3:
        self.assertEqual(got_shape[0], (rank,))
      else:
        self.assertEqual(got_shape[0], (None,))

  def testNoHash(self):

    @def_function.function()
    def f(_):
      return 1.0

    with self.assertRaisesRegexp(AttributeError, 'set'):
      f(set([]))

  def testFuncName(self):

    @function.defun_with_attributes(attributes={'func_name': 'multiply'})
    def add(x, y):
      _ = x * y
      return x + y

    @function.defun
    def add_2(x, y):
      _ = x * y
      return x + y

    self.assertEqual(add._name, 'multiply')
    self.assertEqual(add_2._name, 'add_2')

  def testBasicGraphMode(self):
    matmul = def_function.function(math_ops.matmul)

    @def_function.function
    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out = sq(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedInputsGraphMode(self):
    matmul = def_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @def_function.function
    def a_times_b(inputs):
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    out = a_times_b(pair({'a': t}, {'b': t}))
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputsGraphMode(self):
    matmul = def_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @def_function.function()
    def pairs_mul(pair_a, pair_b):
      return pair(matmul(pair_a.a, pair_b.a), matmul(pair_a.b, pair_b.b))

    a = constant_op.constant([[1.0, 2.0], [1.0, 2.0]])
    b = constant_op.constant([[3.0, 4.0], [3.0, 4.0]])

    out = pairs_mul(pair(a, b), pair(b, a))
    expected = pair(math_ops.matmul(a, b).numpy(),
                    math_ops.matmul(b, a).numpy())
    self.assertAllClose(out, expected)

  @parameterized.named_parameters(
      dict(testcase_name='Defun',
           function_decorator=function.defun),
      dict(testcase_name='DefFunction',
           function_decorator=def_function.function))
  def testNestedFunctionGraphNotOutOfDate(self, function_decorator):
    @function_decorator
    def f():
      return constant_op.constant(1.)

    class _Model(object):

      @function_decorator
      def g(self):
        self.f = f.get_concrete_function()

    model = _Model()
    model.g()
    concrete = model.f
    weak_g_graph = weakref.ref(model.g.get_concrete_function().graph)
    self.assertIs(weak_g_graph(), concrete.graph.outer_graph)
    weak_g = weakref.ref(model.g)
    del model
    self.assertIsNone(weak_g())
    self.assertIsNone(weak_g_graph())
    self.assertIsNotNone(concrete.graph.outer_graph)
    self.assertIs(ops.get_default_graph(), concrete.graph.outer_graph)

  def testGraphEagerIsolation(self):

    @function.defun
    def f():
      self.v = variables.Variable(1.0)
      return self.v.read_value()

    self.assertAllEqual(f(), 1.0)

    with ops.Graph().as_default():
      self.assertEqual(f().shape, ())

  def testBasicGraphFunction(self):
    matmul = def_function.function(math_ops.matmul)

    @def_function.function
    def sq(a):
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = sq.get_concrete_function(t)
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testGetConcreteFunctionThreadSafety(self):

    @def_function.function
    def sq():
      t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      return math_ops.matmul(t, t)

    concrete_functions = []

    def thread_func(_):
      cf = sq.get_concrete_function()
      concrete_functions.append(cf)

    num_threads = 100
    pool = multiprocessing.pool.ThreadPool(num_threads)
    _ = pool.map(thread_func, list(range(num_threads)))

    self.assertLen(set(concrete_functions), 1)

  def testGetConcreteFunctionThreadSafetyWithArgs(self):
    @def_function.function
    def add_100(*args):
      return math_ops.add_n(args)

    p = multiprocessing.pool.ThreadPool(2)
    args = (constant_op.constant(1.),) * 100
    f1, f2 = p.map(add_100.get_concrete_function, [args] * 2)
    # I see about len(args) + max(0, len(args) - 3) arguments expected.
    f1(*args)
    del f2

  def testInputSpecGraphFunction(self):
    matmul = def_function.function(math_ops.matmul)

    @def_function.function
    def sq(a):
      return matmul(a, a)

    sq_op = sq.get_concrete_function(
        tensor_spec.TensorSpec((None, None), dtypes.float32))
    self.assertEqual([None, None], sq_op.output_shapes.as_list())

    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out1 = sq_op(t1)
    self.assertAllEqual(out1, math_ops.matmul(t1, t1).numpy())

    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out2 = sq_op(t2)
    self.assertAllEqual(out2, math_ops.matmul(t2, t2).numpy())

  def testNestedInputSpecGraphFunction(self):
    matmul = def_function.function(math_ops.matmul)

    @def_function.function
    def sq(mats):
      ((a, b),) = mats
      return matmul(a, b)

    sq_op_autonamed = sq.get_concrete_function(
        [(tensor_spec.TensorSpec((None, None), dtypes.float32),
          tensor_spec.TensorSpec((None, None), dtypes.float32))])
    self.assertEqual([None, None], sq_op_autonamed.output_shapes.as_list())

    sq_op = sq.get_concrete_function(
        [(tensor_spec.TensorSpec((None, None), dtypes.float32,
                                 name='first_mat'),
          tensor_spec.TensorSpec((None, None), dtypes.float32,
                                 name='second_mat'))])
    self.assertEqual([None, None], sq_op.output_shapes.as_list())

    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.4, 2.4], [3.4, 4.4]])
    out = sq_op(first_mat=t1, second_mat=t2)
    self.assertAllEqual(out, math_ops.matmul(t1, t2).numpy())
    self.assertAllEqual(sq_op_autonamed(t1, t2),
                        math_ops.matmul(t1, t2).numpy())

  def testExecutingStatelessDefunConcurrently(self):

    @def_function.function
    def stateless(x):
      return math_ops.multiply(2.0, x)

    pool = multiprocessing.pool.ThreadPool()
    inputs = [constant_op.constant(1.0 * x) for x in range(100)]
    outputs = [float(out) for out in pool.map(stateless, inputs)]
    expected = [float(2.0 * x) for x in inputs]
    self.assertSequenceEqual(outputs, expected)

  def testExecutingManyStatelessDefunsConcurrently(self):

    @def_function.function
    def stateless(x):
      del x
      return math_ops.multiply(2.0, 2.0)

    pool = multiprocessing.pool.ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    objects = [object() for _ in range(100)]
    outputs = [float(out) for out in pool.map(stateless, objects)]
    expected = [4.0] * 100
    self.assertSequenceEqual(outputs, expected)

  def testExecutingStatefulDefunConcurrently(self):

    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def stateful(x):
      v.assign(x)

    pool = multiprocessing.pool.ThreadPool()
    inputs = [constant_op.constant(0.0)] * 100
    pool.map(stateful, inputs)
    self.assertEqual(float(v.read_value()), 0.0)

  def testExecutingManyStatefulDefunsConcurrently(self):

    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def stateful(x):
      del x
      return v.assign(0.0)

    pool = multiprocessing.pool.ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    pool.map(stateful, [object() for _ in range(100)])
    self.assertEqual(float(v.read_value()), 0.0)

  def disabled_testRandomSeed(self):

    @def_function.function
    def f():
      return random_ops.random_normal(())

    random_seed.set_random_seed(1)
    x = f()
    self.assertNotEqual(x, f())
    random_seed.set_random_seed(1)
    self.assertAllEqual(f(), x)

  def testNestedInputsGraphFunction(self):
    matmul = def_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @def_function.function
    def a_times_b(inputs):
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq_op = a_times_b.get_concrete_function(
        pair(dict(a=tensor_spec.TensorSpec([2, 2], dtypes.float32, 'a')),
             dict(b=tensor_spec.TensorSpec([2, 2], dtypes.float32, 'b'))))
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(a=t, b=t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputGraphFunction(self):
    matmul = def_function.function(math_ops.matmul)

    @def_function.function
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

  def testGraphFunctionNoneOutput(self):
    @def_function.function
    def fn(unused_a, unused_b):
      return None

    x = constant_op.constant(1)
    fn_op = fn.get_concrete_function(x, x)
    self.assertEqual(fn_op.output_dtypes, None)
    self.assertEqual(fn_op.output_shapes, None)
    self.assertAllEqual(fn_op(x, x), None)

  def testDefunNumpyArraysConvertedToTensors(self):

    def f(x):
      self.assertIsInstance(x, ops.Tensor)
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    defined = function.defun(f)
    defined(x)
    self.assertLen(total_function_cache(defined), 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertLen(total_function_cache(defined), 1)

    # Test that the numpy array is properly an argument to the graph function.
    self.assertEqual(1., defined(numpy.ones([])).numpy())
    self.assertEqual(0., defined(numpy.zeros([])).numpy())
    self.assertEqual(1., defined(array_ops.ones([])).numpy())
    self.assertEqual(0., defined(array_ops.zeros([])).numpy())

  def testDefunCapturedInt32(self):
    x = constant_op.constant(1, dtype=dtypes.int32)

    @def_function.function
    def add_int32s():
      return x + x

    self.assertEqual(2, int(add_int32s()))

  def testDefunReadVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def f():
      return v.read_value()

    self.assertEqual(1.0, float(f()))

  def testDefunAssignAddVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    x = constant_op.constant(2.0)

    @def_function.function
    def test_assign_add():
      v.assign_add(x)
      return v.read_value()

    self.assertEqual(3.0, float(test_assign_add()))

  @test_util.run_in_graph_and_eager_modes
  def testTensorInitializationInFunctionRaisesError(self):
    error_msg = ('Tensor-typed variable initializers must either be '
                 'wrapped in an init_scope or callable.*')

    @def_function.function
    def tensor_init():
      with self.assertRaisesRegexp(ValueError, error_msg):
        resource_variable_ops.ResourceVariable(constant_op.constant(2.0))

    tensor_init()

  @test_util.run_in_graph_and_eager_modes
  def testCallableTensorInitializationInFunction(self):

    @def_function.function
    def tensor_init():
      self.v = resource_variable_ops.ResourceVariable(
          lambda: constant_op.constant(2.0))
      return self.v.read_value()

    value = tensor_init()
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(value), 2.0)

  @test_util.also_run_as_tf_function
  def testInitScopeTensorInitializationInFunction(self):

    @def_function.function
    def tensor_init():
      with ops.init_scope():
        const = constant_op.constant(2.0)
      # Note: this variable bypasses tf.function's variable creation
      # requirements by bypassing variable_creator_scope by using
      # ResourceVariable instead of Variable.
      self.v = resource_variable_ops.ResourceVariable(const)
      return self.v.read_value()

    value = tensor_init()
    self.assertAllEqual(value, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testGetConcreteFunctionCreatesVariables(self):

    v_holder = []

    @def_function.function
    def tensor_init():
      if not v_holder:
        v_holder.append(variables.Variable(5.))
      return v_holder[0].read_value()

    concrete = tensor_init.get_concrete_function()
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(5., self.evaluate(concrete()))
    self.assertAllEqual(5., self.evaluate(tensor_init()))

  def testFuncGraphCaptureByValue(self):
    v = variables.Variable(1.0)

    def trivial_function():
      return v.read_value()

    graph_function = function.Function(
        trivial_function, 'test', capture_by_value=True)

    self.assertAllEqual(graph_function(), 1.0)
    v.assign(2.0)
    self.assertAllEqual(graph_function(), 1.0)

  def testFuncGraphCaptureByValueNested(self):
    v = variables.Variable(1.0)

    def trivial_function():
      return control_flow_ops.cond(
          array_ops.placeholder_with_default(True, ()),
          v.read_value, v.read_value)

    graph_function = function.Function(
        trivial_function, 'test', capture_by_value=True)

    self.assertAllEqual(graph_function(), 1.0)
    v.assign(2.0)
    self.assertAllEqual(graph_function(), 1.0)

  def testDefunShapeInferenceWithCapturedResourceVariable(self):
    v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

    def f():
      x = constant_op.constant([[1, 2], [3, 4]])
      out = math_ops.matmul(v, x)
      self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))
      # We do not return v directly since the tensor conversion function of
      # ResourceVariable returns the read value and not the resource itself.
      return v._handle

    compiled = def_function.function(f)
    var_handle = compiled()
    self.assertEqual(var_handle.dtype, dtypes.resource)
    self.assertEqual(var_handle.shape, tensor_shape.TensorShape([]))
    var_t = resource_variable_ops.read_variable_op(var_handle, dtype=v.dtype)
    self.assertEqual(var_t.shape, tensor_shape.TensorShape([2, 2]))

  def testShapeInferenceForMoreSpecificInput(self):

    def f(a):
      return array_ops.reshape(a, [-1, 3])

    signature = [tensor_spec.TensorSpec(None, dtypes.float32)]
    compiled = def_function.function(f, input_signature=signature)

    @def_function.function
    def use_f():
      inputs = array_ops.zeros([10, 10, 3])
      self.assertAllEqual(f(inputs).shape, compiled(inputs).shape)

    use_f()

  def testFuncListAttr(self):

    @function.defun
    def test_function(val):

      def fn1():
        return array_ops.ones([10])

      fn2 = lambda: array_ops.ones([10]) * 2

      def fn3(x=3):
        return array_ops.ones([10]) * x
      fn4 = functools.partial(fn3, x=4)
      fn5 = functools.partial(fn3, 5)

      return gen_functional_ops.case(val, [], [dtypes.float32],
                                     [function.defun(f).get_concrete_function()
                                      for f in (fn1, fn2, fn3, fn4, fn5)])

    ones = array_ops.ones([10])
    self.assertAllEqual([ones], test_function(0))
    self.assertAllEqual([ones * 2], test_function(1))
    self.assertAllEqual([ones * 3], test_function(2))
    self.assertAllEqual([ones * 4], test_function(3))
    self.assertAllEqual([ones * 5], test_function(4))
    self.assertAllEqual([ones * 5], test_function(22))  # default branch

  @test_util.enable_control_flow_v2
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
        self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))
        # We do not return v directly since the tensor conversion function of
        # ResourceVariable returns the read value and not the resource itself.
        return v._handle

      compiled = def_function.function(f)
      var_handle = compiled()
      self.assertEqual(var_handle.dtype, dtypes.resource)
      self.assertEqual(var_handle.shape, tensor_shape.TensorShape([]))
      var_t = resource_variable_ops.read_variable_op(var_handle, dtype=v.dtype)
      self.assertEqual(var_t.shape, tensor_shape.TensorShape([2, 2]))

  def testDefunShapeInferenceWithCapturedVariableInGraphMode(self):
    with context.graph_mode():
      v = variables.Variable([[1, 2], [3, 4]])

      def f():
        x = constant_op.constant([[1, 2], [3, 4]])
        out = math_ops.matmul(v, x)
        self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))

      # Check that shape inference works while creating the defun
      compiled = def_function.function(f)
      compiled()

  def testDefunShapeInferenceWithCapturedTensorListInGraphMode(self):
    with context.graph_mode():
      tensor_list = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      tensor_list = list_ops.tensor_list_push_back(tensor_list,
                                                   constant_op.constant(1.0))
      tensor_list = list_ops.tensor_list_push_back(tensor_list,
                                                   constant_op.constant(2.0))

      def f():
        tl, value = list_ops.tensor_list_pop_back(
            tensor_list, element_dtype=dtypes.float32)
        self.assertEqual(value.shape, tensor_shape.TensorShape([]))
        return tl

      compiled = def_function.function(f)
      output_tensor_list = compiled()
      _, value = list_ops.tensor_list_pop_back(
          output_tensor_list, element_dtype=dtypes.float32)
      self.assertEqual(value.shape, tensor_shape.TensorShape([]))

  @test_util.run_in_graph_and_eager_modes
  def testDefunForcesResourceVariables(self):

    def variable_creator():
      self.v = variables.Variable(0.0)
      return self.v.read_value()

    self.v = None
    defined = function.defun(variable_creator)
    defined()  # Create the variable.
    self.assertIsInstance(
        self.v, resource_variable_ops.ResourceVariable)

  def testRunMetadata(self):

    @def_function.function
    def f(x):
      return x * x

    with ops.device('cpu:0'):
      context.enable_run_metadata()
      f(constant_op.constant(1.0))
    run_metadata = context.export_run_metadata()
    context.disable_run_metadata()
    self.assertLen(run_metadata.partition_graphs, 1)

  def testGraphModeCaptureVariable(self):
    with context.graph_mode(), self.cached_session():

      class HasAVar(object):

        def __init__(self):
          self.v = resource_variable_ops.ResourceVariable(1.0)

        def call(self):
          return self.v * 2

      o = HasAVar()
      self.evaluate(variables.global_variables_initializer())
      call = def_function.function(o.call)
      op = call()
      self.assertAllEqual(self.evaluate(op), 2.0)

  def testGraphModeManyFunctions(self):
    with ops.Graph().as_default(), self.cached_session():

      @def_function.function
      def f(x):
        return x * x

      @def_function.function
      def g(x):
        return f(x) + 1

      self.assertAllEqual(g(constant_op.constant(2.0)).eval(), 5.0)

  def testDict(self):

    @def_function.function
    def f(x):
      return {'name': x + 1}

    self.assertAllEqual(f(constant_op.constant(1.0))['name'], 2.0)

  def testTensorConversionWithDefun(self):

    @def_function.function
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    self.assertAllEqual(5, f(constant_op.constant(2)))

  def testTensorConversionCall(self):

    @def_function.function
    def f(x):
      return math_ops.add(x, constant_op.constant(3))

    @def_function.function
    def g(x):
      return f(f(x))

    self.assertAllEqual(8, g(constant_op.constant(2)))

  def testCallShape(self):

    @def_function.function
    def f(x):
      return x + 1

    @def_function.function
    def g(x):
      x = f(x)
      self.assertEqual(x.shape.as_list(), [])
      return None

    g(constant_op.constant(1.0))

  def testNestedDefunWithNoOutputAndTapedInput(self):
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @def_function.function
    def f(x):
      # This function intentionally takes a taped variable as input,
      # but does not return any values
      math_ops.add(x, three)

    @def_function.function
    def g(x):
      y = math_ops.add(x, three)
      f(y)

    g(three)

  def testGatherResourceWithDefun(self):
    with ops.device('cpu:0'):
      v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    defined = def_function.function(sum_gather)
    self.assertAllEqual(sum_gather(), defined())

  @parameterized.named_parameters([
      ('IndexedSlicesWithDenseShape',
       _example_indexed_slices_with_dense_shape,),
      ('IndexedSlicesWithoutDenseShape',
       _example_indexed_slices_without_dense_shape,),
      ('RaggedTensorRaggedRank1', ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]}),
      ('RaggedTensorRaggedRank2',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]}),
      ('SparseTensor', sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]}),
  ])  # pyformat: disable
  def testReturnCompositeTensorWithDefun(self,
                                         factory_fn,
                                         factory_kwargs={},
                                         input_signature=None):
    input_ct = factory_fn(**factory_kwargs)

    @def_function.function(input_signature=input_signature)
    def f():
      return input_ct

    output_ct = f()
    self.assertIsInstance(output_ct, type(input_ct))
    nest.assert_same_structure(input_ct, output_ct, expand_composites=True)

    input_flat = nest.flatten(input_ct, expand_composites=True)
    output_flat = nest.flatten(output_ct, expand_composites=True)
    for (input_component, output_component) in zip(input_flat, output_flat):
      self.assertAllEqual(input_component, output_component)

  @parameterized.named_parameters([
      ('IndexedSlicesWithDenseShape',
       _example_indexed_slices_with_dense_shape,),
      ('IndexedSlicesWithoutDenseShape',
       _example_indexed_slices_without_dense_shape,),
      ('RaggedTensorRaggedRank1',
       ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]}),
      ('RaggedTensorRaggedRank2',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]}),
      ('SparseTensor',
       sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]}),
      ('RaggedTensorRaggedRank1WithSignature',
       ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]},
       [ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)]),
      ('RaggedTensorRaggedRank2WithSignature',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]},
       [ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32)]),
      ('SparseTensorWithSignature',
       sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]},
       [sparse_tensor.SparseTensorSpec([None], dtypes.int32)]),
  ])  # pyformat: disable
  def testCompositeAsArgumentTensorWithDefun(self,
                                             factory_fn,
                                             factory_kwargs={},
                                             input_signature=None):
    input_ct = factory_fn(**factory_kwargs)

    @def_function.function(input_signature=input_signature)
    def f(x):
      return x

    output_ct = f(input_ct)
    self.assertIsInstance(output_ct, type(input_ct))
    nest.assert_same_structure(input_ct, output_ct, expand_composites=True)

    input_flat = nest.flatten(input_ct, expand_composites=True)
    output_flat = nest.flatten(output_ct, expand_composites=True)
    for (input_component, output_component) in zip(input_flat, output_flat):
      self.assertAllEqual(input_component, output_component)

  def testTracedCompositeDiscardsShapeInfo(self):
    # SparseTensorSpec intentionally excludes info about the number of elements
    # that are in a sparse tensor (which is recorded as st.indices.shape[0] and
    # st.values.shape[0]).  Similarly, RaggedTensorSpec intentionally excludes
    # info about the total number of values in a RaggedTensor (stored as
    # rt.values.shape[0]).  This test checks that the placeholders created by
    # tf.function() properly mask this shape info.
    @def_function.function
    def f(rt, st):
      self.assertEqual(st.indices.shape.as_list()[:1], [None])
      self.assertEqual(st.values.shape.as_list(), [None])
      return (rt, st)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    st = sparse_tensor.SparseTensor([[0]], [0], [10])
    f(rt, st)

  @test_util.run_gpu_only
  def testFunctionOnDevice(self):
    x = constant_op.constant([1.]).gpu()
    f = def_function.function(math_ops.add)
    y = f(x, x).cpu()
    self.assertAllEqual(y, [2.])

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testFunctionWithResourcesOnDifferentDevices(self):
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

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testOpInFunctionWithConflictingResourceInputs(self):
    with ops.device('/cpu:0'):
      v_cpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='cpu')
      v_also_cpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='also_cpu')

    with ops.device('/gpu:0'):
      v_gpu = resource_variable_ops.ResourceVariable(
          [0.0, 1.0, 2.0], name='gpu')

    @def_function.function
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
        errors.InvalidArgumentError,
        'Cannot place the graph because a reference or resource edge connects '
        'colocation groups with incompatible assigned devices'):
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      self.evaluate(resource_apply_adam())

  @test_util.run_gpu_only
  def testFunctionHandlesInputsOnDifferentDevices(self):
    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = def_function.function(array_ops.reshape)
    value = constant_op.constant([1., 2.]).gpu()
    shape = constant_op.constant([2, 1])
    reshaped = reshape(value, shape).cpu()
    self.assertAllEqual(reshaped, [[1], [2]])

  @test_util.run_gpu_only
  def testFunctionHandlesInputsPlacedOnTheWrongDeviceGracefully(self):
    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = def_function.function(array_ops.reshape)
    value = constant_op.constant([1., 2.])
    shape = constant_op.constant([2, 1]).gpu()
    reshape(value, shape)  # No error is raised

  def testNoneOutput(self):

    @def_function.function
    def my_function(_):
      return None

    self.assertAllEqual(my_function(1), None)

  def testNestedFunctions(self):
    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @tf_function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      return math_ops.add(a, b)

    @def_function.function
    def add_one(x):
      return add(x, 1)

    self.assertAllEqual(3, add_one(constant_op.constant(2)))

  def testVariableCaptureInNestedFunctions(self):
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int32)

    @def_function.function
    def inner_read():
      return v.read_value()

    @def_function.function
    def outer():
      return inner_read()

    self.assertEqual(1, int(outer()))

  def testReturnCapturedEagerTensor(self):
    t = constant_op.constant(1)

    @def_function.function
    def read():
      return t

    self.assertEqual(1, int(read()))

  def testReturnCapturedGraphTensor(self):
    with context.graph_mode(), self.cached_session():
      t = constant_op.constant(1)

      @def_function.function
      def read():
        return t

      self.assertEqual(1, int(self.evaluate(read())))

  def testSequenceInputs(self):
    clip_by_global_norm = def_function.function(clip_ops.clip_by_global_norm)
    t_list = [constant_op.constant(1.0), constant_op.constant(2.0)]
    clipped_list, global_norm = clip_by_global_norm(t_list,
                                                    constant_op.constant(.2))
    for t in clipped_list:
      self.assertIsInstance(t, ops.Tensor)
    self.assertIsInstance(global_norm, ops.Tensor)

  def testNestedSequenceInputs(self):

    def my_op(inputs):
      a, b, c = inputs
      e, f = b
      g, h = e
      return [a + a, [tuple([f + f, g + g]), h + h], c + c], a + f + g + h + c

    my_eager_op = def_function.function(my_op)
    ret = my_eager_op([
        constant_op.constant(1), [(constant_op.constant(2),
                                   constant_op.constant(3)),
                                  constant_op.constant(4)],
        constant_op.constant(5)
    ])
    self.assertLen(ret, 2)
    self.assertAllEqual(ret[0][0], 2)
    self.assertAllEqual(ret[0][1][0][0], 8)
    self.assertAllEqual(ret[0][1][0][1], 4)
    self.assertIsInstance(ret[0][1][0], tuple)
    self.assertAllEqual(ret[0][1][1], 6)
    self.assertAllEqual(ret[0][2], 10)
    self.assertAllEqual(ret[1], 15)

  def testVariableNamesRespectNameScopesWithDefun(self):
    @def_function.function
    def create_variable():
      with ops.name_scope('foo', skip_on_eager=False):
        v = resource_variable_ops.ResourceVariable(0.0, name='bar')
      self.assertEqual(v.name, 'foo/bar:0')

    create_variable()

  def testVariableNamesRespectNameScopesWithDefunInGraph(self):
    with context.graph_mode():
      @def_function.function
      def create_variable():
        with ops.name_scope('foo', skip_on_eager=False):
          v = resource_variable_ops.ResourceVariable([1.0, 2.0], name='bar')
        self.assertEqual(v.name, 'foo/bar:0')

      with ops.get_default_graph().as_default():
        create_variable()

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
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

    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    self.assertAllClose([[[[4.0]]]], self.evaluate(y))

  # Variable lifting is somewhat different between defun/tf.function, so testing
  # device placement on both makes sense.
  @parameterized.named_parameters(
      dict(testcase_name='Defun',
           function_decorator=function.defun),
      dict(testcase_name='DefFunction',
           function_decorator=def_function.function))
  @test_util.run_in_graph_and_eager_modes
  def testVariablesPlacedOnOutsideDevice(self, function_decorator):

    class _Obj(object):

      def __init__(self):
        self.v = None

      @function_decorator
      def f(self):
        if self.v is None:
          self.v = variables.Variable(1.)
        return self.v + 1.

    has_device = _Obj()
    with ops.device('cpu:0'):
      has_device.f()
    self.assertIn('CPU', has_device.v.device)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testDefunKerasModelCall(self):
    model = MiniModel()
    model.call = function.defun(model.call)

    x = array_ops.ones([1, 2])
    y = model(x)

    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    self.assertAllEqual([[3.0]], self.evaluate(y))

    # Break the reference cycle between the MiniModel and the defun:
    # `MiniModel` --(through its `call` method)--> `Function`
    # `Function` --(instancemethod on `MiniModel`)--> `MiniModel`
    del model.call

  @test_util.run_in_graph_and_eager_modes
  def testDeviceAnnotationsRespected(self):

    def multi_device_fn():
      with ops.device('/cpu:0'):
        s0 = test_ops.device_placement_op()
      with ops.device('/cpu:1'):
        s1 = test_ops.device_placement_op()
      with ops.device('/cpu:2'):
        s2 = test_ops.device_placement_op()
      s3 = test_ops.device_placement_op()
      return s0, s1, s2, s3

    defined = function.defun(multi_device_fn)
    outputs = self.evaluate(defined())
    self.assertLen(total_function_cache(defined), 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])

    with ops.device('/cpu:3'):
      outputs = self.evaluate(defined())
    # All function definitions are agnostic to call site devices.
    self.assertLen(total_function_cache(defined), 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
    self.assertIn(compat.as_bytes('CPU:3'), outputs[3])

    with ops.device('/cpu:0'):
      outputs = self.evaluate(defined())
    self.assertLen(total_function_cache(defined), 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
    self.assertIn(compat.as_bytes('CPU:0'), outputs[3])

  @test_util.run_in_graph_and_eager_modes
  def testCallingGraphFunctionOnDifferentDevice(self):

    def func():
      return constant_op.constant(0)

    defined = def_function.function(func)
    with ops.device('cpu:0'):
      cpu_graph_function = defined.get_concrete_function()

    with ops.device('cpu:0'):
      self.assertEqual(
          self.evaluate(cpu_graph_function()), self.evaluate(func()))

    with ops.device('cpu:1'):
      self.assertEqual(0., self.evaluate(cpu_graph_function()))

    with ops.device(None):
      self.assertEqual(0., self.evaluate(cpu_graph_function()))

    default_graph_function = defined.get_concrete_function()
    self.assertEqual(
        self.evaluate(default_graph_function()), self.evaluate(func()))

    with ops.device('cpu:1'):
      self.assertEqual(0., self.evaluate(default_graph_function()))

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testColocateWithRespected(self):
    # TODO(b/113291792): Use multiple CPUs instead of a GPU.
    with ops.device('cpu:0'):
      x = constant_op.constant(1.0)

    with ops.device('gpu:0'):
      y = constant_op.constant(1.0)

    @def_function.function
    def foo():
      return test_ops.device_placement_op()

    with ops.colocate_with(x):
      self.assertIn(compat.as_bytes('CPU:0'), self.evaluate(foo()))

    with ops.colocate_with(y):
      self.assertIn(compat.as_bytes('GPU:0'), self.evaluate(foo()))

  def testVariablesAreTracked(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    def foo(x):
      return v * x

    defined = def_function.function(foo)

    x = constant_op.constant([1.0])
    self.assertEqual(1., self.evaluate(defined(x)))
    v.assign(2.)

    x = constant_op.constant([1.0, 2.0])
    self.assertAllEqual([2., 4.], self.evaluate(defined(x)))

  def testCacheObjectHashCollisions(self):

    class Foo(object):

      def __hash__(self):
        return 42

    def func(foo):
      del foo
      return

    defined = function.defun(func)
    defined(Foo())
    self.assertLen(total_function_cache(defined), 1)

    defined(Foo())
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorDtypeCollision(self):

    def func(t):
      return t + t

    defined = function.defun(func)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 1)

    t = constant_op.constant([[1.0]], dtype=dtypes.complex128)
    defined(t)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorShapeCollision(self):

    def func(t):
      return t + t

    defined = function.defun(func)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 1)

    t = constant_op.constant([1.0], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorShapeDtypeCollision(self):

    def func(t):
      return t + t

    defined = function.defun(func)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 1)

    t = constant_op.constant([1.0], dtype=dtypes.complex128)
    defined(t)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorUnknownShapesCollisionRelaxedShapes(self):

    def func(t):
      return t + t

    with context.graph_mode(), self.cached_session():
      defined = function.defun(func, experimental_relax_shapes=True)

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      defined(p)
      self.assertLen(total_function_cache(defined), 1)

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
      defined(p)
      self.assertLen(total_function_cache(defined), 2)

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[2])
      defined(p)
      # Gradual shape relaxation is performed; and the common shape between
      # [1] and [2] is one containing unknown dimensions.
      self.assertLen(total_function_cache(defined), 2)

      # pylint: disable=protected-access
      self.assertLen(defined._function_cache.arg_relaxed_shapes, 1)
      relaxed_shapes = (
          list(defined._function_cache.arg_relaxed_shapes.values())[0])
      self.assertLen(relaxed_shapes, 1)
      relaxed_shape = relaxed_shapes[0]
      # pylint: enable=protected-access
      self.assertEqual(relaxed_shape.rank, 1)
      self.assertEqual(tensor_shape.dimension_value(relaxed_shape[0]), None)

      t = constant_op.constant([1.0, 1.0, 1.0], dtype=dtypes.float32)
      defined(t)
      # Shape (3,) matches the relaxed shape TensorShape([None])
      self.assertLen(total_function_cache(defined), 2)

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
      return tuple(key[0] for key in total_function_cache(defined))

    # `True` corresponds to the fact that we're executing eagerly
    self.assertIn(('URRRu', (0, 1, 20)), cache_keys())

    defined(1)  # bar=1, baz=2
    self.assertIn(('URRRu', (1, 1, 2)), cache_keys())

    # This matches the previous call.
    defined(foo=1)
    self.assertLen(total_function_cache(defined), 2)

    defined(1, 2, 3)
    self.assertLen(total_function_cache(defined), 3)
    self.assertIn(('URRRu', (1, 2, 3)), cache_keys())

    # This matches the previous call.
    defined(1, bar=2, baz=3)
    self.assertLen(total_function_cache(defined), 3)

    # This matches the previous call.
    defined(1, baz=3, bar=2)
    self.assertLen(total_function_cache(defined), 3)

  def testFunctoolsPartialUnwrappedCorrectly(self):

    def full_function(a, b, c=3):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    a, b, c = partial(2)

    defined = function.defun(partial)
    func_a, func_b, func_c = defined(2)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureWithMatchingInputs(self):

    def foo(a):
      self.assertEqual(a.shape, (2,))
      return a

    signature = [tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = function.defun(foo, input_signature=signature)
    a = array_ops.ones([2])
    self.assertAllEqual(a, defined(a))
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(a, defined.get_concrete_function()(a))
    self.assertAllEqual(a, defined.get_concrete_function(a)(a))
    self.assertAllEqual(a, defined.get_concrete_function(
        tensor_spec.TensorSpec((2,), dtype=dtypes.float32))(a))
    self.assertLen(total_function_cache(defined), 1)

    def bar(a):
      self.assertEqual(a._shape_tuple(), (2, None))
      return a

    signature = [tensor_spec.TensorSpec((2, None), dtypes.float32)]
    defined = function.defun(bar, input_signature=signature)
    a = array_ops.ones([2, 1])
    out = defined(a)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out, a)

    # Changing the second dimension shouldn't create a new function.
    b = array_ops.ones([2, 3])
    out = defined(b)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out, b)

  def testInputSignatureWithCompatibleInputs(self):

    rank2_spec = tensor_spec.TensorSpec(shape=(None, None),
                                        dtype=dtypes.float32)

    @function.defun(input_signature=[rank2_spec])
    def func(a):
      self.assertEqual([None, None], a.shape.as_list())
      return array_ops.shape(a)

    self.assertAllEqual([3, 1], func([[0], [1.0], [1]]))
    self.assertAllEqual([2, 2], func(numpy.array([[1, 1], [2, 2]])))

    with self.assertRaisesRegexp(ValueError, 'incompatible'):
      func([0.0, 1.0, 2.0])  # Wrong shape.

    with self.assertRaisesRegexp(ValueError, 'incompatible'):
      func([['wrong dtype']])

  def testNestedInputSignatures(self):

    def expected_foo(a, b):
      return [a, b]

    @function.defun(input_signature=[
        [tensor_spec.TensorSpec((2, None), dtypes.float32)] * 2,
        tensor_spec.TensorSpec((1,), dtypes.float32),
    ])
    def foo(a, b):
      self.assertEqual(a[0]._shape_tuple(), (2, None))
      self.assertEqual(a[1]._shape_tuple(), (2, None))
      self.assertEqual(b._shape_tuple(), (1,))
      return [a, b]

    a = array_ops.ones([2, 1])
    b = array_ops.ones([1])
    expected = expected_foo([a, a], b)
    out = foo([a, a], b)
    self.assertLen(total_function_cache(foo), 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], a)
    self.assertAllEqual(out[1], b)

    # Changing the unspecified dimensions shouldn't create a new function.
    a = array_ops.ones([2, 3])
    b = array_ops.ones([2, 5])
    c = array_ops.ones([1])
    expected = expected_foo([a, b], c)
    out = foo([a, b], c)
    self.assertLen(total_function_cache(foo), 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], b)
    self.assertAllEqual(out[1], c)

    # Passing compatible inputs should work.
    a = a.numpy().tolist()
    b = b.numpy().tolist()
    c = c.numpy().tolist()
    out = foo([a, b], c)
    self.assertLen(total_function_cache(foo), 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], b)
    self.assertAllEqual(out[1], c)

  def testNestedInputSignaturesWithDict(self):
    def expected_bar(a):
      return a

    @function.defun(input_signature=[{
        'a': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'b': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'c': tensor_spec.TensorSpec((1,), dtypes.float32)}])
    def bar(a):
      self.assertEqual(a['a']._shape_tuple(), (2, None))
      self.assertEqual(a['b']._shape_tuple(), (2, None))
      self.assertEqual(a['c']._shape_tuple(), (1,))
      return a

    a = array_ops.ones([2, 3])
    b = array_ops.ones([1])
    inputs = {'a': a, 'b': a, 'c': b}
    expected = expected_bar(inputs)
    out = bar(inputs)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out['a'], expected['a'])
    self.assertAllEqual(out['b'], expected['b'])
    self.assertAllEqual(out['c'], expected['c'])

    # Passing compatible inputs should work.
    a = a.numpy().tolist()
    b = b.numpy().tolist()
    inputs = {'a': a, 'b': a, 'c': b}
    out = bar(inputs)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out['a'], expected['a'])
    self.assertAllEqual(out['b'], expected['b'])
    self.assertAllEqual(out['c'], expected['c'])

  def testInputSignatureMustBeSequenceOfTensorSpecs(self):

    def foo(a, b):
      del a
      del b

    # Signatures must consist exclusively of `TensorSpec` objects.
    signature = [(2, 3), tensor_spec.TensorSpec([2, 3], dtypes.float32)]
    with self.assertRaisesRegexp(TypeError, 'Invalid input_signature.*'):
      def_function.function(foo, input_signature=signature)

    # Signatures must be either lists or tuples on their outermost levels.
    signature = {'t1': tensor_spec.TensorSpec([], dtypes.float32)}
    with self.assertRaisesRegexp(TypeError, 'input_signature must be either a '
                                 'tuple or a list.*'):
      function.defun(foo, input_signature=signature)

  @test_util.run_in_graph_and_eager_modes
  def testInputsIncompatibleWithSignatureRaisesError(self):

    def foo(a):
      return a

    signature = [tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = def_function.function(foo, input_signature=signature)

    # Invalid shapes.
    with self.assertRaisesRegexp(ValueError, 'Python inputs incompatible.*'):
      defined(array_ops.ones([3]))

    with self.assertRaisesRegexp(ValueError, 'Python inputs incompatible.*'):
      defined(array_ops.ones([2, 1]))

    # Wrong number of arguments.
    with self.assertRaisesRegexp(TypeError, r'Received 2 argument\(s\)'):
      defined(array_ops.ones([2]), array_ops.ones([2]))
    with self.assertRaisesRegexp(ValueError,
                                 'Structure of Python function inputs.*'):
      defined()

    with self.assertRaisesRegexp(ValueError,
                                 'inputs incompatible with input_signature'):
      defined.get_concrete_function(
          tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.float32))

  def testInputsIncompatibleWithNestedSignatureRaisesError(self):

    def foo(a, b):
      return [a, b]

    signature = [[tensor_spec.TensorSpec((1,), dtypes.float32)] * 2,
                 [tensor_spec.TensorSpec((1,), dtypes.float32)] * 2]
    defined = function.defun(foo, input_signature=signature)
    a = array_ops.ones([1])

    with self.assertRaisesRegexp(ValueError,
                                 'Structure of Python function inputs.*'):
      defined([a, a, a], [a])

    with self.assertRaisesRegexp(ValueError,
                                 'Structure of Python function inputs.*'):
      defined([a], [a, a, a])
    defined([a, a], [a, a])

  def testUnderspecifiedInputSignature(self):
    @function.defun(input_signature=[
        tensor_spec.TensorSpec([], dtypes.float32),
    ])
    def foo(a, training=True):
      if training:
        return a
      else:
        return -1.0 * a

    x = constant_op.constant(1.0)
    with self.assertRaisesRegexp(TypeError, 'only pass arguments'):
      foo(x, training=True)

    with self.assertRaisesRegexp(TypeError, 'only pass arguments'):
      foo(x, training=False)

    self.assertAllEqual(x.numpy(), foo(x).numpy())

  def testInputSignatureWithPartialFunction(self):
    def full_function(a, b, c=3.0):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    a, b, c = partial(2.0)
    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    defined = function.defun(partial, input_signature=signature)
    x = constant_op.constant(2.0)
    func_a, func_b, func_c = defined(x)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureConversionWithDefaultArg(self):

    def foo(a, training=True):
      if training:
        return a
      else:
        return -1.0 * a

    signature = [
        tensor_spec.TensorSpec([], dtypes.float32),
        tensor_spec.TensorSpec([], dtypes.bool),
    ]
    defined = def_function.function(foo, input_signature=signature)
    a = constant_op.constant(1.0)
    self.assertAllEqual(a.numpy(), defined(a))
    self.assertAllEqual(a.numpy(), defined(a, training=True))
    self.assertAllEqual(-a.numpy(), defined(a, training=False))

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
    self.assertLen(total_function_cache(foo), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt=flt, integer=integer)
    self.assertLen(total_function_cache(foo), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(integer=integer, flt=flt)
    self.assertLen(total_function_cache(foo), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt, integer=integer)
    self.assertLen(total_function_cache(foo), 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

  def testInputSignatureWithKeywordArgs(self):
    def foo(a, b, **kwargs):
      del kwargs
      return a, b

    x = function.defun(
        foo,
        input_signature=[
            tensor_spec.TensorSpec([], dtypes.float32),
            tensor_spec.TensorSpec([], dtypes.int32)
        ]).get_concrete_function()
    result = x(constant_op.constant(5.0), constant_op.constant(5))
    self.assertAllEqual(result, [5.0, 5])

  def testInputSignatureWithCompositeTensors(self):
    def f(rt):
      self.assertEqual(rt.values.shape.as_list(), [None])
      self.assertEqual(rt.row_splits.shape.as_list(), [4])
      return rt

    signature = [ragged_tensor.RaggedTensorSpec(
        shape=[3, None], dtype=dtypes.int32)]
    defined = function.defun(f, input_signature=signature)
    rt1 = ragged_factory_ops.constant([[1], [], [2, 3, 4]])
    out1 = defined(rt1)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out1.values, rt1.values)
    self.assertAllEqual(out1.row_splits, rt1.row_splits)

    # Changing the row lengths shouldn't create a new function.
    rt2 = ragged_factory_ops.constant([[1, 2], [3, 4], [5]])
    out2 = defined(rt2)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out2.values, rt2.values)
    self.assertAllEqual(out2.row_splits, rt2.row_splits)

    # Different number of rows
    rt3 = ragged_factory_ops.constant([[1, 2], [3, 4], [5], [6]])
    with self.assertRaisesRegexp(ValueError, 'incompatible'):
      defined(rt3)

    # Different dtype
    rt4 = ragged_factory_ops.constant([[1.0, 2.0], [], [3.0]])
    with self.assertRaisesRegexp(ValueError, 'Structure .* does not match'):
      defined(rt4)

    # Different rank
    rt5 = ragged_factory_ops.constant([[[1]], [[2]], [[3]]])
    with self.assertRaisesRegexp(ValueError, 'does not match'):
      defined(rt5)

  def testInputSignatureWithVariableArgs(self):

    def f(v):
      v.assign_add(1)

    signature = [
        resource_variable_ops.VariableSpec(shape=[], dtype=dtypes.int32)
    ]
    defined = function.defun(f, input_signature=signature)

    v1 = variables.Variable(0)
    v2 = variables.Variable(0)

    defined(v1)
    self.assertEqual(v1.numpy(), 1)
    self.assertEqual(v2.numpy(), 0)

    defined(v=v2)
    self.assertEqual(v1.numpy(), 1)
    self.assertEqual(v2.numpy(), 1)

  def testTensorKeywordArguments(self):

    def foo(a, b):
      del a
      return b

    defined = function.defun(foo)
    a = constant_op.constant(2.0)
    b = constant_op.constant([1.0, 2.0])
    one = defined(a, b)
    self.assertLen(total_function_cache(defined), 1)

    two = defined(a=a, b=b)
    self.assertLen(total_function_cache(defined), 1)

    three = defined(b=b, a=a)
    self.assertLen(total_function_cache(defined), 1)

    four = defined(a, b=b)
    self.assertLen(total_function_cache(defined), 1)

    # The next call corresponds to a new input signature, hence
    # we expect another function to be defined.
    five = defined(b, a)
    self.assertLen(total_function_cache(defined), 2)

    six = defined(a=b, b=a)
    self.assertLen(total_function_cache(defined), 2)

    seven = defined(b=a, a=b)
    self.assertLen(total_function_cache(defined), 2)

    self.assertAllEqual(one, [1.0, 2.0])
    self.assertAllEqual(two, [1.0, 2.0])
    self.assertAllEqual(three, [1.0, 2.0])
    self.assertAllEqual(four, [1.0, 2.0])
    self.assertAllEqual(five, 2.0)
    self.assertAllEqual(six, 2.0)
    self.assertAllEqual(seven, 2.0)

  def testDefuningInstanceMethod(self):

    integer = constant_op.constant(2, dtypes.int64)

    class Foo(object):

      def one(self, tensor):
        return tensor

      @def_function.function
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

      @def_function.function
      def func(self, other=integer):
        return other

    foo = Foo()
    self.assertEqual(foo.func().numpy(), int(integer))

  def testPythonCallWithSideEffects(self):
    state = []

    @def_function.function
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

  def testFunctionWithNestedFunctionCallAndSideEffects(self):
    v1 = variables.Variable(1.0)
    v2 = variables.Variable(1.0)

    @def_function.function
    def add_one(a):
      a.assign_add(1.0)

    # Grappler will inline calls to `add_one` into the function body, we check
    # that all side-effects were executed.
    @def_function.function
    def side_effecting_function(a, b):
      add_one(a)
      add_one(b)
      return a + b

    result = side_effecting_function(v1, v2)
    self.assertEqual(result.numpy(), 4.0)

  def testFunctionWithExtraAttributes(self):
    @function.defun_with_attributes(attributes={'experimental_1': 'value1',
                                                'experimental_2': 2})
    def matmul(x, y):
      return math_ops.matmul(x, y)

    def add(x, y):
      return math_ops.add(x, y)
    defun_add = function.defun_with_attributes(
        add, attributes={'experimental_3': True, 'experimental_4': 1.0})

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        sq = matmul(t, t)
        double = defun_add(t, t)
        self.assertAllEqual(sq.eval().reshape(-1), [7, 10, 15, 22])
        self.assertAllEqual(double.eval().reshape(-1), [2, 4, 6, 8])

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 2)
        functions = list(graph._functions.values())
        self.assertRegexpMatches(
            functions[0].definition.signature.name, '.*matmul.*')
        attrs = functions[0].definition.attr
        self.assertLen(attrs, 2)
        self.assertEqual(attrs['experimental_1'].s, b'value1')
        self.assertEqual(attrs['experimental_2'].i, 2)

        self.assertRegexpMatches(
            functions[1].definition.signature.name, '.*add.*')
        attrs = functions[1].definition.attr
        self.assertLen(attrs, 2)
        self.assertEqual(attrs['experimental_3'].b, True)
        self.assertEqual(attrs['experimental_4'].f, 1.0)
        # pylint: enable=protected-access

  def testFunctionWithInvalidAttribute(self):
    @function.defun_with_attributes(attributes={'experimental_1': ['value1']})
    def add(x, y):
      return math_ops.add(x, y)

    with self.assertRaisesRegexp(ValueError,
                                 '.*Unsupported attribute type.*'):
      with context.graph_mode(), self.cached_session():
        with ops.get_default_graph().as_default():
          t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
          add(t, t)

  def testRegisterFunction(self):

    @function.defun
    def add(x, y):
      return math_ops.add(x, y)

    def matmul(x, y):
      return math_ops.matmul(x, y)
    defun_matmul = function.defun(matmul)

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        function.register(defun_matmul, t, t)
        function.register(add, t, t)

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 6)
        # two sets of functions, each of them are (inference, forward, backward)
        functions = list(graph._functions.values())
        captured_function_names = [
            f.definition.signature.name for f in functions
        ]
        expected_func_name_regex = [
            '.*inference.*matmul.*',
            '.*forward.*matmul.*',
            '.*inference.*backward.*matmul.*',
            '.*inference.*add.*',
            '.*forward.*add.*',
            '.*inference.*backward.*add.*',
        ]
        for i in range(len(functions)):
          self.assertRegexpMatches(captured_function_names[i],
                                   expected_func_name_regex[i])

        # Check the forward and backward function has the correct attributes.
        self.assertEqual(
            functions[1].definition.attr['backward_function_name'].s,
            functions[2].name)
        self.assertEqual(
            functions[2].definition.attr['forward_function_name'].s,
            functions[1].name)

        self.assertEqual(
            functions[4].definition.attr['backward_function_name'].s,
            functions[5].name)
        self.assertEqual(
            functions[5].definition.attr['forward_function_name'].s,
            functions[4].name)

        sq = defun_matmul(t, t)
        double = add(t, t)
        self.assertAllEqual(sq.eval().reshape(-1), [7, 10, 15, 22])
        self.assertAllEqual(double.eval().reshape(-1), [2, 4, 6, 8])
        # Make sure the pre registered function is used, and no other function
        # is added.
        self.assertLen(graph._functions, 6)
        functions = list(graph._functions.values())
        for i in range(len(functions)):
          self.assertEqual(captured_function_names[i],
                           functions[i].definition.signature.name)

  @parameterized.named_parameters(
      dict(testcase_name='Defun',
           function_decorator=function.defun),
      dict(testcase_name='DefFunction',
           function_decorator=def_function.function))
  def testRegisterConcreteFunction(self, function_decorator):
    @function_decorator
    def py_add(x, y):
      return math_ops.add(x, y)

    py_add(array_ops.ones([]), array_ops.ones([]))
    add = py_add.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32))

    @function_decorator
    def py_composite(x, y):
      return x, add(x, y)

    py_composite(array_ops.ones([]), array_ops.ones([]))
    composite = py_composite.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32))

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        composite.add_to_graph()
        composite.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 6)
        # two sets of functions, each of them are (inference, forward, backward)
        functions = list(graph._functions.values())
        captured_function_names = [
            f.definition.signature.name for f in functions
        ]
        expected_func_name_regex = [
            '.*inference.*py_composite.*',
            '.*inference.*py_add.*',
            '.*forward.*py_composite.*',
            '.*forward.*py_add.*',
            '.*inference.*backward.*py_composite.*',
            '.*inference.*backward.*py_add.*',
        ]
        for expected, found in zip(
            expected_func_name_regex,
            captured_function_names):
          self.assertRegexpMatches(found, expected)

        composite_t, composite_double = composite(t, t)
        double = add(t, t)
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(double))
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(composite_double))
        self.assertAllEqual([[1, 2], [3, 4]], self.evaluate(composite_t))
        # Make sure the pre registered function is used, and no other function
        # is added.
        self.assertLen(graph._functions, 6)

  @parameterized.named_parameters(
      dict(testcase_name='Defun',
           function_decorator=function.defun),
      dict(testcase_name='DefFunction',
           function_decorator=def_function.function))
  def testEagerCaptures(self, function_decorator):
    with context.eager_mode():
      large_tensor = array_ops.ones(shape=(256,))
      self.assertGreater(256, func_graph._EAGER_CONST_THRESHOLD)

      small_tensor = array_ops.ones(shape=(4,))
      self.assertLessEqual(4, func_graph._EAGER_CONST_THRESHOLD)

      v = resource_variable_ops.ResourceVariable(0.0)

    for captured, op_type in [(large_tensor, 'Placeholder'),
                              (small_tensor, 'Const'), (v, 'Placeholder')]:
      @function_decorator
      def test_fn():
        return captured + 1  # pylint: disable=cell-var-from-loop

      g = test_fn.get_concrete_function().graph
      internal_captures = g.internal_captures
      self.assertLen(internal_captures, 1)
      self.assertEqual(internal_captures[0].op.type, op_type)

  def testRegisterFunctionWithInputSignature(self):
    def matmul(x, y):
      return math_ops.matmul(x, y)
    defun_matmul = function.defun(
        matmul,
        input_signature=[
            tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32)
        ])
    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        function.register(defun_matmul, t, t)

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

        # Test register function with cache, note inputs are ignored.
        function.register(defun_matmul)
        graph = ops.get_default_graph()
        self.assertLen(graph._functions, 3)

  def testRegisterFunctionWithCache(self):
    def matmul(x, y):
      return math_ops.matmul(x, y)
    defun_matmul = function.defun(matmul)

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        t2 = constant_op.constant([[2.0, 3.0], [4.0, 5.0]])
        function.register(defun_matmul, t, t)
        function.register(defun_matmul, t2, t2)

        graph = ops.get_default_graph()
        # Only one function is registered since the input param are in same type
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

  def testCallingFunctionWithDifferentVariables(self):

    @function.defun
    def foo(v):
      v.assign_add(1.0)
      return v.read_value()

    v = resource_variable_ops.ResourceVariable(0.0)
    graph_function = foo.get_concrete_function(v)
    self.assertLen(graph_function.inputs, 1)
    self.assertEmpty(graph_function.captured_inputs)

    self.assertEqual(float(graph_function(v)), 1.0)
    self.assertEqual(float(graph_function(v)), 2.0)

    w = resource_variable_ops.ResourceVariable(0.0)

    @function.defun
    def bar(v):
      del v
      return constant_op.constant(1.0)

    graph_function = bar.get_concrete_function(v)
    self.assertEqual(float(graph_function(v)), 1.0)
    self.assertEqual(float(graph_function(w)), 1.0)

  def testCallingFunctionWithNonTensorsFails(self):

    @function.defun
    def foo(x):
      return x

    graph_function = foo.get_concrete_function(constant_op.constant(1.0))
    with self.assertRaisesRegexp(
        ValueError, 'All inputs to `ConcreteFunction`s must be Tensors;.*'):
      graph_function('Not a Tensor.')

  def testSwapImplementationWithGrapplerPlugin(self):
    # Set the min_graph_nodes to -1 since the graph in this test is too small,
    # and will be ignored by grappler if don't set this.
    rewrites = rewriter_config_pb2.RewriterConfig()
    rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
    rewrites.min_graph_nodes = -1
    graph_options = config_pb2.GraphOptions(
        rewrite_options=rewrites, build_cost_model=1)
    config_proto = config_pb2.ConfigProto(graph_options=graph_options)

    with context.graph_mode(), self.cached_session(
        config=config_proto, graph=ops.Graph(), use_gpu=True):

      @function.defun_with_attributes(
          attributes={
              'api_implements': 'random_boost',
              'api_preferred_device': 'CPU'
          })
      def cpu_boost(x):
        return math_ops.add(x, 2.0)

      @function.defun_with_attributes(
          attributes={
              'api_implements': 'random_boost',
              'api_preferred_device': 'GPU'
          })
      def gpu_boost(x):
        return math_ops.add(x, 4.0)

      x = constant_op.constant(1.0)

      function.register(cpu_boost, x)
      y = gpu_boost(x)
      y_value = self.evaluate(y)

      if test.is_gpu_available():
        self.assertEqual(y_value, 5.0)
      else:
        # Grappler fallback to use the CPU impl even called with GPU function.
        self.assertEqual(y_value, 3.0)

  def testSwapImplementationInEager(self):
    if not context.executing_eagerly():
      self.skipTest('eager only')

    context.context().set_optimizer_experimental_options(
        {'min_graph_nodes': -1, 'implementation_selector': True})

    @function.defun_with_attributes(
        attributes={'api_implements': 'foo',
                    'api_preferred_device': 'CPU'})
    def on_cpu(x):
      return x + 2

    @function.defun_with_attributes(
        attributes={'api_implements': 'foo',
                    'api_preferred_device': 'GPU'})
    def on_gpu(x):
      return x + 4

    @function.defun
    def run_on_cpu(t):
      function.register(on_cpu, t)
      with ops.device('CPU:0'):
        return on_gpu(t)

    # Expect to run the on_cpu branch, regardless whether gpu is available.
    self.assertEqual(run_on_cpu(constant_op.constant(1)).numpy(), 3)

  def testDefunFunctionSeparateGraphs(self):
    with context.graph_mode():

      @function.defun
      def add(x):
        return x + 5

      @function.defun
      def maybe_add(x, should_add):
        if should_add:
          return add(x)
        else:
          return x

      with ops.Graph().as_default():
        x = constant_op.constant(11)
        maybe_add(x, True)
        self.assertLen(total_function_cache(maybe_add), 1)
        self.assertLen(total_function_cache(add), 1)

        maybe_add(x, False)
        self.assertLen(total_function_cache(maybe_add), 2)
        self.assertLen(total_function_cache(add), 1)

      with ops.Graph().as_default():
        x = constant_op.constant(11)
        maybe_add(x, True)
        self.assertLen(total_function_cache(maybe_add), 3)
        self.assertLen(total_function_cache(add), 2)

  def testCacheKeyOverlappingShapes(self):
    @function.defun
    def defined(t):
      return t

    defined(array_ops.zeros([12, 1]))
    self.assertLen(total_function_cache(defined), 1)

    defined(array_ops.zeros([1, 21]))
    self.assertLen(total_function_cache(defined), 2)

  def testCacheKeyNestedLists(self):
    @function.defun
    def defined(l):
      return l

    a = constant_op.constant(1.)
    b = constant_op.constant(2.)
    c = constant_op.constant(3.)
    defined([[a], b, c])
    self.assertLen(total_function_cache(defined), 1)

    defined([[a, b], c])
    self.assertLen(total_function_cache(defined), 2)

  def testCacheKeyAttrsClass(self):
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class TestClass(object):
      a = attr.ib()
      b = attr.ib()

    @function.defun
    def defined(l):
      return l

    defined(
        TestClass(
            constant_op.constant(1.),
            [constant_op.constant(2.),
             constant_op.constant(3.)]))
    self.assertLen(total_function_cache(defined), 1)
    defined(
        TestClass(
            constant_op.constant(1.),
            [constant_op.constant(2.),
             constant_op.constant(3.)]))
    self.assertLen(total_function_cache(defined), 1)

    defined(
        TestClass([constant_op.constant(1.),
                   constant_op.constant(2.)], constant_op.constant(3.)))
    self.assertLen(total_function_cache(defined), 2)

  def testCacheKeyVariables(self):
    @function.defun
    def defined(a, b, c):
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)

    # If tensor equality is not enabled, we always get a cache miss if the
    # function is called with different variables. With equality enabled we
    # should only get a miss if the aliasing changed.
    defined(x, y, z)
    self.assertLen(total_function_cache(defined), 1)

    # Calling again is a cache hit
    defined(x, y, z)
    self.assertLen(total_function_cache(defined), 1)

    # Re-arranging arguments doesn't change signature
    defined(z, y, x)
    self.assertLen(total_function_cache(defined),
                   1 if ops.Tensor._USE_EQUALITY else 2)

    # Aliasing causes cache miss
    defined(x, x, z)
    self.assertLen(total_function_cache(defined),
                   2 if ops.Tensor._USE_EQUALITY else 3)

    # Re-arranging arguments doesn't change signature
    defined(y, y, z)
    self.assertLen(total_function_cache(defined),
                   2 if ops.Tensor._USE_EQUALITY else 4)

    # Different alias positions causes cache miss
    defined(z, y, y)
    self.assertLen(total_function_cache(defined),
                   3 if ops.Tensor._USE_EQUALITY else 5)

  def testDecoratedMethod(self):
    m = DefunnedMiniModel()
    instance_call_one = m.call(array_ops.ones([1, 2]), training=True)
    instance_call_two = m.call(
        inputs=array_ops.ones([1, 2]), training=True)
    class_call = DefunnedMiniModel.call(m, array_ops.ones([1, 2]),
                                        training=True)
    self.assertAllEqual(instance_call_one, instance_call_two)
    self.assertAllEqual(instance_call_one, class_call)

  def testDecoratedMethodUniqueFunctionPerInstance(self):
    m = DefunnedMiniModel()
    n = DefunnedMiniModel()

    class_method_one = DefunnedMiniModel.call
    class_method_two = DefunnedMiniModel.call

    m_method_one = m.call
    m_method_two = m.call

    n_method_one = n.call
    n_method_two = n.call

    self.assertEqual(class_method_one, class_method_two)
    self.assertEqual(m_method_one, m_method_two)
    self.assertEqual(n_method_one, n_method_two)
    self.assertNotEqual(m.call, n.call)

  def testDecoratedMethodInspect(self):
    m = DefunnedMiniModel()
    fullargspec = tf_inspect.getfullargspec(m.call)
    self.assertIn('training', fullargspec.args)

  def testDecoratedMethodGetConcreteFunction(self):
    m = DefunnedMiniModel()
    instance_call_one = m.call.get_concrete_function(
        array_ops.ones([1, 2]), training=False)
    instance_call_two = m.call.get_concrete_function(
        inputs=array_ops.ones([1, 2]), training=False)
    self.assertAllEqual(instance_call_one(array_ops.ones([1, 2])),
                        instance_call_two(array_ops.ones([1, 2])))

    # Also make sure get_concrete_function works on the class method
    DefunnedMiniModel.call.get_concrete_function(
        m, array_ops.ones([1, 2]), training=False)
    DefunnedMiniModel.call.get_concrete_function(
        m, inputs=array_ops.ones([1, 2]), training=True)

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

      @def_function.function
      def append(l):
        l.append(constant_op.constant(0.))

      append(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def extend(l):
        l.extend([constant_op.constant(0.)])

      extend(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def insert(l):
        l.insert(0, constant_op.constant(0.))

      insert(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def pop(l):
        l.pop()

      pop(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def reverse(l):
        l.reverse()

      reverse(get_list())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def remove(l):
        l.remove(l[0])

      remove(get_list())

    # `list.clear` is a method that is in Py3 but not Py2
    if sys.version.startswith('3'):

      with self.assertRaisesRegexp(ValueError, expected_msg):

        @def_function.function
        def clear(l):
          l.clear()

        clear(get_list())

    # One last test for keyword arguments
    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
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

      @def_function.function
      def clear(m):
        m.clear()

      clear(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def pop(m):
        m.pop('t1')

      pop(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def popitem(m):
        m.popitem()

      popitem(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
      def update(m):
        m.update({'t1': constant_op.constant(3.)})

      update(get_dict())

    with self.assertRaisesRegexp(ValueError, expected_msg):

      @def_function.function
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

      @def_function.function
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
      @def_function.function
      def modify_same_flat(n):
        n[0].append(n[1].pop(0))

      nested_input = [[constant_op.constant(0.)],
                      [constant_op.constant(1.),
                       constant_op.constant(2.)]]

      modify_same_flat(nested_input)

  def testDecoratedMethodVariableCleanup(self):
    m = DefunnedMiniModel()
    m(array_ops.ones([1, 2]))
    variable_refs = list({v.experimental_ref() for v in m.variables})
    self.assertLen(variable_refs, 2)
    del m

    # Verifying if the variables are only referenced from variable_refs.
    # We expect the reference counter to be 1, but `sys.getrefcount` reports
    # one higher reference counter because a temporary is created when we call
    # sys.getrefcount().  Hence check if the number returned is 2.
    # https://docs.python.org/3/library/sys.html#sys.getrefcount
    self.assertEqual(sys.getrefcount(variable_refs[0].deref()), 2)
    self.assertEqual(sys.getrefcount(variable_refs[1].deref()), 2)

  def testExecutorType(self):
    @function.defun
    def add_five(x):
      return x + 5

    self.assertEqual(
        5,
        add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())

    with self.assertRaisesRegexp(errors.NotFoundError, 'NON_EXISTENT_EXECUTOR'):
      with context.function_executor_type('NON_EXISTENT_EXECUTOR'):
        add_five(constant_op.constant(0, dtype=dtypes.int32))

    for executor_type in ('', 'DEFAULT', None):
      with context.function_executor_type(executor_type):
        self.assertAllEqual(
            5,
            add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())

  @test_util.assert_no_garbage_created
  def testReferenceCycles(self):

    fn = function.defun(lambda x: 2. * x)

    fn(constant_op.constant(4.0))
    weak_fn = weakref.ref(fn)
    del fn
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  def testFunctionStackInErrorMessage(self):
    if context.executing_eagerly():
      # TODO(b/122736651): Remove this skipTest once fixed.
      self.skipTest('Error interpolation is not working when function is '
                    'invoked without PartitionedCallOp.')

    @def_function.function()
    def fn3(x):
      return x + 2

    @def_function.function()
    def fn2(x):
      check_ops.assert_equal(fn3(x), 3)
      return 2

    @def_function.function()
    def fn(x):
      return fn2(x)

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      fn(2)
    e = cm.exception
    self.assertIn('fn -> fn2', e.message)
    self.assertIn('node assert_equal/Assert/Assert (defined at', e.message)
    self.assertNotIn('fn3', e.message)

  @test_util.run_gpu_only
  def testFunctionIsNotPinned(self):
    """Tests that functions aren't pinned to the CPU by the eager runtime."""
    seed1, seed2 = 79, 25
    shape = constant_op.constant([4, 7])
    dtype = dtypes.float32

    @def_function.function
    def func():
      with ops.device('GPU:0'):
        return gen_random_ops.random_standard_normal(
            shape, dtype=dtype, seed=seed1, seed2=seed2)

    with ops.device('GPU:0'):
      x = func()
      self.assertRegexpMatches(x.device, 'GPU')

  @test_util.run_in_graph_and_eager_modes
  def testShapeCaching(self):

    @function.defun
    def func(x):
      return array_ops.shape(x)

    @function.defun(
        input_signature=[tensor_spec.TensorSpec([None, None], dtypes.float32)])
    def calls_func(x):
      return func(x)

    self.assertAllEqual([1, 1], self.evaluate(func(array_ops.zeros([1, 1]))))
    self.assertAllEqual([2, 2], self.evaluate(func(array_ops.zeros([2, 2]))))
    self.assertAllEqual(
        [3, 3],
        self.evaluate(calls_func(array_ops.zeros([3, 3]))))

  def testLimitedRetracing(self):
    trace_count = [0]
    @function.defun
    def func(x):
      trace_count[0] += 1
      return x

    for _ in range(50):
      func(constant_op.constant(3.))
      func(constant_op.constant(4.))
      func(constant_op.constant([[1., 2.]]))
      func(constant_op.constant([[]]))
      func(constant_op.constant([[3., 4.], [5., 6.]]))
      func(constant_op.constant([[3., 4.], [5., 6.], [7., 8.]]))
    # Tracing more than twice per input doesn't make sense.
    self.assertLess(trace_count[0], 13)

  def testLimitedRetracingWithCompositeTensors(self):
    trace_count = [0]

    @def_function.function
    def f(x):
      trace_count[0] += 1
      return x

    for i in range(10):
      f(ragged_factory_ops.constant([[1, 2], [i]]))
      f(ragged_factory_ops.constant([[1, 2], [], [3, 4, 5]]))
      f(ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5, 6]]]))
      self.assertEqual(trace_count[0], 3)

  def test_concrete_function_shape_mismatch(self):

    @def_function.function
    def f(argument_name):
      return argument_name + 1.

    f_concrete = f.get_concrete_function(constant_op.constant([1.]))

    # Calling a function from eager doesn't do any shape checking above what
    # kernels do while executing.
    self.assertAllEqual(
        [2., 3.],
        f_concrete(constant_op.constant([1., 2.])).numpy())

    @def_function.function
    def g():
      f_concrete(constant_op.constant([1., 2.]))

    with self.assertRaisesRegexp(ValueError, 'argument_name'):
      g()

  @test_util.run_in_graph_and_eager_modes
  def test_shape_inference_with_symbolic_shapes(self):

    @def_function.function
    def _uses_symbolic_shapes(w, x, y):
      x = array_ops.identity(x, name='name_collision')
      x = array_ops.transpose(x, [1, 0, 2])
      x_batch = array_ops.shape(x)[0]
      y_batch = array_ops.shape(y)[0]
      y *= w
      n = y_batch // x_batch
      return array_ops.reshape(y, [n, x_batch, -1])

    conc = _uses_symbolic_shapes.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32))

    @def_function.function
    def _call_concrete():
      c = constant_op.constant(1.)
      array_ops.identity(c, name='name_collision')
      output1 = conc(array_ops.ones([2]),
                     array_ops.ones([5, 4, 2]),
                     array_ops.ones([20, 2]))
      self.assertEqual([5, 4, 2], output1.shape)
      output2 = conc(array_ops.ones([3]),
                     array_ops.ones([5, 4, 3]),
                     array_ops.ones([40, 3]))
      self.assertEqual([10, 4, 3], output2.shape)
      return output1, output2

    output1, output2 = _call_concrete()
    self.assertEqual((5, 4, 2), self.evaluate(output1).shape)
    self.assertEqual((10, 4, 3), self.evaluate(output2).shape)

  def testAutoGraphContext(self):

    @def_function.function
    def test_fn():
      self.assertEqual(
          ag_ctx.control_status_ctx().status, ag_ctx.Status.ENABLED)

    prev_status = ag_ctx.control_status_ctx().status
    test_fn()
    self.assertEqual(ag_ctx.control_status_ctx().status, prev_status)

  def testCancelBeforeFunctionExecution(self):
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @def_function.function
    def f():
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    c_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      cancelable_func()

  def testCancelBlockedFunctionExecution(self):
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @def_function.function
    def f():
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    def cancel_thread():
      time.sleep(0.5)
      c_mgr.start_cancel()

    t = self.checkedThread(cancel_thread)
    t.start()
    with self.assertRaises(errors.CancelledError):
      cancelable_func()
    t.join()

  def testCancelAfterFunctionExecution(self):
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)
    q.enqueue(37)

    @def_function.function
    def f():
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    self.assertAllEqual(37, cancelable_func().numpy())

    # Cancellation after the function executes is a no-op.
    c_mgr.start_cancel()

  def testAddFunctionCallback(self):
    functions = []
    def function_callback(f):
      functions.append(f)

    @def_function.function
    def plus_one(x):
      return x + 1

    try:
      function.add_function_callback(function_callback)
      x_float32 = numpy.array(3.0, dtype=numpy.float32)
      self.assertAllClose(plus_one(x_float32), 4.0)
      self.assertLen(functions, 1)
      # Function is already created. Executing it again should not invoke the
      # function callback.
      self.assertAllClose(plus_one(x_float32), 4.0)
      self.assertLen(functions, 1)
      # Signature change leads to a new Function being built.
      x_float64 = numpy.array(3.0, dtype=numpy.float64)
      self.assertAllClose(plus_one(x_float64), 4.0)
      self.assertLen(functions, 2)
    finally:
      function.clear_function_callbacks()

  def testRemoveFunctionCallback(self):
    functions_1 = []
    def function_callback_1(f):
      functions_1.append(f)

    functions_2 = []
    def function_callback_2(f):
      functions_2.append(f)

    @def_function.function
    def plus_one(x):
      return x + 1

    try:
      function.add_function_callback(function_callback_1)
      function.add_function_callback(function_callback_2)
      self.assertAllClose(plus_one(numpy.array(3.0, dtype=numpy.float32)), 4.0)
      self.assertLen(functions_1, 1)
      self.assertLen(functions_2, 1)
      function.remove_function_callback(function_callback_1)
      # The 1st callback should not be invokved after remove_function_callback()
      # is called.
      self.assertAllClose(plus_one(numpy.array(3.0, dtype=numpy.float64)), 4.0)
      self.assertLen(functions_1, 1)
      self.assertLen(functions_2, 2)
    finally:
      function.clear_function_callbacks()

  def testClearFunctionCallbacks(self):
    function.add_function_callback(lambda f: None)
    function.add_function_callback(lambda f: None)
    self.assertLen(function._function_callbacks, 2)
    function.clear_function_callbacks()
    self.assertEmpty(function._function_callbacks)  # pylint:disable=protected-access


class MultiDeviceTest(test.TestCase, parameterized.TestCase):

  @test_util.run_gpu_only
  def testMultiDeviceOutput(self):
    """Tests that functions can produce outputs on multiple devices."""
    @function.defun
    def func(a, b, transpose_a):
      with ops.device('/device:CPU:0'):
        m1 = math_ops.matmul(a, b, transpose_a=transpose_a)
      with ops.device('/device:GPU:0'):
        m2 = math_ops.matmul(a, b, transpose_a=transpose_a)
      return m1, m2

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    m1, m2 = func(t, t, transpose_a=True)
    self.assertAllEqual(m1.numpy(), [[10, 14], [14, 20]])
    self.assertRegexpMatches(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), [[10, 14], [14, 20]])
    self.assertRegexpMatches(m2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testEmptyBody(self):
    @function.defun
    def func(a, b):
      return b, a

    with ops.device('/device:CPU:0'):
      a = constant_op.constant(3.0)
    with ops.device('/device:GPU:0'):
      b = constant_op.constant(5.0)

    m1, m2 = func(a, b)
    self.assertAllEqual(m1.numpy(), 5.0)
    self.assertRegexpMatches(m1.backing_device, 'GPU')
    self.assertAllEqual(m2.numpy(), 3.0)
    self.assertRegexpMatches(m2.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testMultiDeviceInt32(self):
    """Tests that multi-device functions can take and output INT32s.

    When an INT32 device tensor is fed into a function, it is copied to CPU
    by the eager runtime. The function sees all INT32 inputs on CPU.

    We set allocator attribute 'on_host' for INT32 outputs. They can be
    partitioned into the GPU component function, but will be allocated on
    CPU nevertheless.

    There is experimental support for `ints_on_device` in
    FunctionLibraryRuntime now. We can try that.

    """
    with ops.device('/device:CPU:0'):
      int_cpu = constant_op.constant(3, dtype=dtypes.int32)
      resource = resource_variable_ops.ResourceVariable(5, dtype=dtypes.int32)
    with ops.device('/device:GPU:0'):
      int_gpu = constant_op.constant(7, dtype=dtypes.int32)

    @function.defun
    def func(int_cpu, resource, int_gpu):
      with ops.device('/device:CPU:0'):
        m1 = int_cpu * resource + int_gpu
      with ops.device('/device:GPU:0'):
        # This computation will happen on GPU but m2 will be copied to CPU.
        m2 = int_gpu * resource + int_cpu + 1
      return m1, m2

    m1, m2 = func(int_cpu, resource, int_gpu)
    self.assertAllEqual(m1.numpy(), 22)
    self.assertRegexpMatches(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 39)
    self.assertRegexpMatches(m2.backing_device, 'CPU')

    # flip arguments
    m1, m2 = func(int_gpu, resource, int_cpu)
    self.assertAllEqual(m1.numpy(), 38)
    self.assertRegexpMatches(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 23)
    self.assertRegexpMatches(m2.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testMultiDeviceColocateWith(self):
    """Tests that function's outputs respect colocation constraints."""
    @function.defun
    def func(a, b):
      with ops.colocate_with(a):
        ra = 2 * a
      with ops.colocate_with(b):
        rb = 3 * b
      return ra, rb

    devices = ['/device:CPU:0', '/device:GPU:0']
    for dev1, dev2 in itertools.product(devices, devices):
      with ops.device(dev1):
        a = constant_op.constant(1.0)
      with ops.device(dev2):
        b = constant_op.constant(10.0)

      ra, rb = func(a, b)
      self.assertEqual(ra.numpy(), 2.0)
      self.assertRegexpMatches(ra.backing_device, dev1)
      self.assertEqual(rb.numpy(), 30.0)
      self.assertRegexpMatches(rb.backing_device, dev2)

  @test_util.run_gpu_only
  def testMultiDeviceResources(self):
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
      c2 = resource_variable_ops.ResourceVariable(7.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)
      g2 = resource_variable_ops.ResourceVariable(5.0)

    @function.defun
    def func(resource1, resource2):
      with ops.device('/device:CPU:0'):
        result1 = resource1 * g2
      with ops.device('/device:GPU:0'):
        result2 = resource2 * c2
      return result1, result2

    r1, r2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegexpMatches(r2.backing_device, 'GPU')

    # Call with flipped inputs. Check that we look at resource's
    # device and reinstantiates the function when inputs' devices change.
    r1, r2 = func(g1, c1)
    self.assertEqual(r1.numpy(), 15.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegexpMatches(r2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testOutputResources(self):
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @function.defun
    def func(resource1, resource2):
      with ops.device('/device:CPU:0'):
        result1 = resource1 * 5
      with ops.device('/device:GPU:0'):
        result2 = resource2 * 7
      return result1, resource1.handle, result2, resource2.handle

    r1, res1, r2, res2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegexpMatches(r2.backing_device, 'GPU')

    def check_handle(handle, expected_value):
      self.assertRegexpMatches(handle.backing_device, 'CPU')
      tensor = gen_resource_variable_ops.read_variable_op(
          handle, dtypes.float32)
      self.assertEqual(tensor.numpy(), expected_value)

    # Check that handles returned from functions are on CPU and an op using
    # the resource handle is correctly placed on the device backing the
    # resource.
    check_handle(res1, 2.0)
    check_handle(res2, 3.0)

    # Call with flipped inputs to make sure the same the function is
    # reinstantiated and eager runtime does not mess up the device assignment
    # for ops consuming handles returned from defuns.
    r1, res1, r2, res2 = func(g1, c1)
    self.assertEqual(r1.numpy(), 15.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegexpMatches(r2.backing_device, 'GPU')
    check_handle(res1, 3.0)
    check_handle(res2, 2.0)

  @test_util.run_gpu_only
  def testPassResourceThroughNestedFunctionCall(self):
    """Test passing GPU resource to noinline function call placed on CPU.

    PartitionedCallOp must not enforce any particular device assignment for the
    resource output. Inner function marked as `_nospecialize`, so Grappler would
    not prune unused function output.
    """

    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @function.defun_with_attributes(attributes={
        '_noinline': True,
        '_nospecialize': True
    })
    def inner(resource1):
      return resource1 * 2, resource1.handle

    @function.defun
    def outer(resource1):
      with ops.device('/device:CPU:0'):
        r1, _ = inner(resource1)
      return r1

    r1 = outer(g1)

    self.assertEqual(r1.numpy(), 6.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testReturnResourceFromNestedFunctionCall(self):
    """Test returning GPU resource from noinline function call placed on CPU.

    When inferring output devices for the return value, do not set a device for
    returns of DT_RESOURCE data type based on the device assignment of the node
    that produced that resource. As an example function call placed on CPU can
    return resources on GPU.
    """

    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @function.defun_with_attributes(attributes={
        '_noinline': True
    })
    def inner(resource1):
      resource1.assign_add(2.0)
      return resource1 * 2, resource1.handle

    @function.defun
    def outer(resource1):
      with ops.device('/device:CPU:0'):
        r1, res1 = inner(resource1)
      return r1, res1

    r1, res1 = outer(g1)

    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegexpMatches(r1.backing_device, 'CPU')

    def check_handle(handle, expected_value):
      self.assertRegexpMatches(handle.backing_device, 'CPU')
      tensor = gen_resource_variable_ops.read_variable_op(
          handle, dtypes.float32)
      self.assertEqual(tensor.numpy(), expected_value)

    # Check that handles returned from functions are on CPU and an op using
    # the resource handle is correctly placed on the device backing the
    # resource.
    check_handle(res1, 5.0)

  @test_util.run_gpu_only
  def testComplexInputOutputDevicePattern(self):
    """Tests input/output mapping logic in partitioning."""
    with ops.device('/device:CPU:0'):
      rc0 = resource_variable_ops.ResourceVariable(2.0)
      rc1 = resource_variable_ops.ResourceVariable(3.0)
      cc0 = constant_op.constant(5.0)
      cc1 = constant_op.constant(7.0)
    with ops.device('/device:GPU:0'):
      rg0 = resource_variable_ops.ResourceVariable(11.0)
      rg1 = resource_variable_ops.ResourceVariable(13.0)
      cg0 = constant_op.constant(17.0)
      cg1 = constant_op.constant(19.0)

    # Make sure tensors are on expected devices.
    for tensor in [cc0, cc1]:
      self.assertRegexpMatches(tensor.backing_device, 'CPU:0')
    for tensor in [cg0, cg1]:
      self.assertRegexpMatches(tensor.backing_device, 'GPU:0')

    @function.defun
    def func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1):
      with ops.device('/device:CPU:0'):
        m1 = rc0 * cg0
      with ops.device('/device:GPU:0'):
        m2 = rg0 * cc0

      with ops.device('/device:CPU:0'):
        r1 = 1000.0 * m2 + rc1 * cg1
      with ops.device('/device:GPU:0'):
        r2 = 1000.0 * m1 + rg1 * cc1

      return r1, r2, m2, m1

    r1, r2, m2, m1 = func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1)
    self.assertRegexpMatches(m1.backing_device, 'CPU')
    self.assertRegexpMatches(r1.backing_device, 'CPU')
    self.assertRegexpMatches(m2.backing_device, 'GPU')
    self.assertRegexpMatches(r2.backing_device, 'GPU')
    self.assertEqual(m1.numpy(), 34.0)
    self.assertEqual(r1.numpy(), 55000.0 + 3.0 * 19.0)
    self.assertEqual(m2.numpy(), 55.0)
    self.assertEqual(r2.numpy(), 34000.0 + 13.0 * 7.0)

  @test_util.run_gpu_only
  def testArgumentPrunning(self):
    """Tests functions taking unnecessary arguments."""
    with ops.device('/device:CPU:0'):
      c1 = constant_op.constant(5.0)
      c2 = constant_op.constant(7.0)

    with ops.device('/device:GPU:0'):
      g1 = constant_op.constant(11.0)
      g2 = constant_op.constant(13.0)
      g3 = constant_op.constant(17.0)

    @function.defun
    def func(g1, g2, c1, g3, c2):  # pylint: disable=unused-argument
      # arguments g1 and g2 are unused and can be pruned by grappler.
      return c1 * g3 * c2

    result = func(g1, g2, c1, g3, c2)
    self.assertEqual(result.numpy(), 5.0 * 7.0 * 17.0)

  def testNestedCallWatchedVariables(self):

    v = variables.Variable(4.)

    @def_function.function
    def f():
      return v ** 2.

    with backprop.GradientTape() as tape:
      f()

    self.assertEqual((v,), tape.watched_variables())

    @def_function.function
    def g():
      return f()

    with backprop.GradientTape() as tape:
      g()

    self.assertEqual((v,), tape.watched_variables())

    # f() can rely on the variable being read during its trace. g() checks that
    # variables from a function which knows about them are recorded on the
    # tape. h() tests that functions forward knowledge of variables to callers.

    @def_function.function
    def h():
      return g()

    with backprop.GradientTape() as tape:
      h()

    self.assertEqual((v,), tape.watched_variables())

  def testStandardTrainingLoopInFunction(self):
    layer = core.Dense(2)
    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            (array_ops.ones([784]), array_ops.ones([], dtypes.int32)))
        .map(lambda x, y: (x, y))
        .repeat(10)
        .batch(32))
    optimizer = adam.Adam()

    @def_function.function
    def train():
      for x, y in dataset:
        with backprop.GradientTape() as tape:
          out = layer(x)
          loss = math_ops.reduce_mean(
              nn_ops.sparse_softmax_cross_entropy_with_logits(
                  logits=out, labels=y))
        layer_variables = layer.trainable_variables
        gradients = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(gradients, layer_variables))

    train()

  def testEarlyStoppingTrainingLoopInFunction(self):
    layer = core.Dense(2)
    dataset = (
        dataset_ops.DatasetV2.from_tensors(
            (array_ops.ones([784]), array_ops.ones([], dtypes.int32)))
        .map(lambda x, y: (x, y))
        .repeat(10)
        .batch(32))
    optimizer = adam.Adam()

    @def_function.function
    def train():
      for x, y in dataset:
        with backprop.GradientTape() as tape:
          out = layer(x)
          loss = math_ops.reduce_mean(
              nn_ops.sparse_softmax_cross_entropy_with_logits(
                  logits=out, labels=y))
        layer_variables = layer.trainable_variables
        gradients = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(gradients, layer_variables))
        if optimizer.iterations > 3:
          break

    train()

  def testDeferredCapture(self):
    value = 1.0

    @def_function.function
    def lazy_capture(x):
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_spec.TensorSpec(None))
      return x + y

    self.assertAllEqual(lazy_capture(2.0), 3.0)
    # After changing the value of `value` the function call should return a
    # different result.
    value = 2.0
    self.assertAllEqual(lazy_capture(2.0), 4.0)

  def testDeferredCaptureWithKey(self):
    value0 = 1.0
    value1 = 2.0

    @def_function.function
    def lazy_capture(x):
      w = ops.get_default_graph().capture_call_time_value(
          lambda: value0, tensor_spec.TensorSpec(None), key=0)
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value1, tensor_spec.TensorSpec(None), key=1)
      def bad_closure():
        raise ValueError('Should not run')
      z = ops.get_default_graph().capture_call_time_value(
          bad_closure, tensor_spec.TensorSpec(None), key=1)
      return x + y + w + z

    self.assertAllEqual(lazy_capture(2.0), 7.0)
    value0 = 2.0
    value1 = 3.0
    self.assertAllEqual(lazy_capture(2.0), 10.0)

  def testDeferredCaptureTypeError(self):
    value = constant_op.constant(1.0)

    @def_function.function
    def lazy_capture(x):
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_spec.TensorSpec(()))
      return x + y

    self.assertAllEqual(lazy_capture(2.0), 3.0)

    # dtype mismatch
    value = constant_op.constant(1)
    with self.assertRaisesRegexp(ValueError, 'Value .* to a tensor with dtype'):
      lazy_capture(2.0)

    # shape mismatch
    value = constant_op.constant([1.0])
    with self.assertRaisesRegexp(ValueError, 'Value .* shape'):
      lazy_capture(2.0)

  def testDeferredCaptureReturnNestWithCompositeTensor(self):
    i_s = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64),
        constant_op.constant([2]))
    r_t = ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5, 6]]])
    s_t = sparse_tensor.SparseTensor(
        values=[1, 2, 3], indices=[[0], [8], [10]], dense_shape=[20])

    @def_function.function
    def lazy_capture():
      y = ops.get_default_graph().capture_call_time_value(
          lambda: {'i': i_s, 't': (r_t, s_t)},
          {'i': indexed_slices.IndexedSlicesSpec(
              dtype=dtypes.int32, dense_shape_dtype=dtypes.int32),
           't': (ragged_tensor.RaggedTensorSpec([2, None, None], dtypes.int32),
                 sparse_tensor.SparseTensorSpec([None], dtypes.int32))})
      return y['i'], y['t']

    i, (r, s) = lazy_capture()
    self.assertAllEqual(i_s.values, i.values)
    self.assertAllEqual(i_s.indices, i.indices)
    self.assertAllEqual(i_s.dense_shape, i.dense_shape)
    self.assertAllEqual(r_t, r)
    self.assertAllEqual(s_t.indices, s.indices)
    self.assertAllEqual(s_t.values, s.values)
    self.assertAllEqual(s_t.dense_shape, s.dense_shape)

  def testDeferredCaptureCompositeTensorSpecTypeMismatch(self):
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64))

    @def_function.function
    def lazy_capture():
      return ops.get_default_graph().capture_call_time_value(
          lambda: value,
          indexed_slices.IndexedSlicesSpec(dtype=dtypes.int32))

    # Type matches spec.
    lazy_capture()

    # Extra dense shape component.
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64),
        constant_op.constant([2]))
    with self.assertRaises(ValueError):
      lazy_capture()

    # Index dtype mismatch int32 vs. int64.
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1]))
    with self.assertRaises(ValueError):
      lazy_capture()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
