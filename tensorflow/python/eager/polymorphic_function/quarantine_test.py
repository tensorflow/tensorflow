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

import copy
import functools
import itertools
import os
import weakref

from absl.testing import parameterized
import numpy

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import quarantine
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def total_function_cache(defined):
  return defined._list_all_concrete_functions()  # pylint: disable=protected-access


# TODO(b/258247871): Do not delete these tests, migrate to use tf.function or
# TracingCompiler.
class DefunTest(test.TestCase, parameterized.TestCase):

  def testExternalControlDependency(self):
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.0)
      v.initializer.run()

      op = v.assign_add(1.0)

      @quarantine.defun_with_attributes
      def f():
        with ops.control_dependencies([op]):
          return 1.0

      self.evaluate(f())
      self.assertAllEqual(self.evaluate(v), 2.0)

  def testInputShapeFunctionRelaxation(self):
    unknown_dim = [False]

    @quarantine.defun_with_attributes(reduce_retracing=True)
    def func(a):
      if a._shape_tuple()[0] is None:
        unknown_dim[0] = True
      return a + 1

    func(constant_op.constant([]))
    self.assertFalse(unknown_dim[0])
    self.assertLen(total_function_cache(func), 1)

    func(constant_op.constant([1.0]))
    self.assertTrue(unknown_dim[0])
    self.assertLen(total_function_cache(func), 2)

    func(constant_op.constant([1.0, 2.0]))
    self.assertTrue(unknown_dim[0])

  def testNestedInputShapeFunctionRelaxation(self):
    unknown_dim = [False]

    @quarantine.defun_with_attributes(reduce_retracing=True)
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
    self.assertTrue(unknown_dim[0])
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

    # We relax the type traced previously.
    func(a, b_=[b0, constant_op.constant([1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(total_function_cache(func), 4)

  def testFuncName(self):

    @quarantine.defun_with_attributes(attributes={'func_name': 'multiply'})
    def add(x, y):
      _ = x * y
      return x + y

    @quarantine.defun_with_attributes
    def add_2(x, y):
      _ = x * y
      return x + y

    self.assertEqual(add._name, 'multiply')
    self.assertEqual(add_2._name, 'add_2')

  def testNestedFunctionGraphNotOutOfDate(self):

    @quarantine.defun_with_attributes
    def f():
      return constant_op.constant(1.)

    class _Model(object):

      @quarantine.defun_with_attributes
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

    @quarantine.defun_with_attributes
    def f():
      self.v = variables.Variable(1.0)
      return self.v.read_value()

    self.assertAllEqual(f(), 1.0)

    with ops.Graph().as_default():
      self.assertEqual(f().shape, ())

  def testDefunNumpyArraysConvertedToTensors(self):

    def f(x):
      self.assertIsInstance(x, ops.Tensor)
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    defined = quarantine.defun_with_attributes(f)
    defined(x)
    self.assertLen(total_function_cache(defined), 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertLen(total_function_cache(defined), 1)

    np_ones = numpy.ones([], numpy.float32)
    np_zeros = numpy.zeros([], numpy.float32)
    tf_ones = array_ops.ones([])
    tf_zeros = array_ops.zeros([])

    # Test that the numpy array is properly an argument to the graph function.
    self.assertEqual(1., defined(np_ones).numpy())
    self.assertLen(total_function_cache(defined), 2)
    self.assertEqual(0., defined(np_zeros).numpy())
    self.assertEqual(1., defined(tf_ones).numpy())
    self.assertEqual(0., defined(tf_zeros).numpy())
    self.assertLen(total_function_cache(defined), 2)

    # Test that mutable inputs are supported.
    mutable = numpy.ones([], numpy.float32)
    self.assertEqual(1., defined(mutable).numpy())
    mutable.fill(0)
    self.assertEqual(0., defined(mutable).numpy())

    class MyNdarray(numpy.ndarray):
      pass

    # Test that the subclasses of ndarray are converted too.
    self.assertEqual(1., defined(np_ones.view(MyNdarray)).numpy())
    self.assertEqual(0., defined(np_zeros.view(MyNdarray)).numpy())

    # We should not have triggered any re-tracing of the python function.
    self.assertLen(total_function_cache(defined), 2)

  def testNumpyDtypeInputSupported(self):

    @quarantine.defun_with_attributes
    def f(x, dtype):
      return constant_op.constant(dtype(x))

    self.assertEqual(f(1, numpy.float32).numpy(), numpy.float32(1))
    self.assertEqual(f(2, numpy.float32).numpy(), numpy.float32(2))
    self.assertEqual(f(1, numpy.int32).numpy(), numpy.int32(1))
    self.assertEqual(f(2, numpy.int32).numpy(), numpy.int32(2))

  def testDefunNumpyArraysConvertedToTensorsInKwargs(self):

    def f(**kwargs):
      x = kwargs.pop('x')
      self.assertIsInstance(x, ops.Tensor)
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    defined = quarantine.defun_with_attributes(f)
    defined(x=x)
    self.assertLen(total_function_cache(defined), 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x=x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertLen(total_function_cache(defined), 1)

    # Test that the numpy array is properly an argument to the graph function.
    self.assertEqual(1., defined(x=numpy.ones([])).numpy())
    self.assertEqual(0., defined(x=numpy.zeros([])).numpy())
    self.assertEqual(1., defined(x=array_ops.ones([])).numpy())
    self.assertEqual(0., defined(x=array_ops.zeros([])).numpy())

  def testFuncListAttr(self):

    @quarantine.defun_with_attributes
    def test_function(val):

      def fn1():
        return array_ops.ones([10])

      fn2 = lambda: array_ops.ones([10]) * 2

      def fn3(x=3):
        return array_ops.ones([10]) * x

      fn4 = functools.partial(fn3, x=4)
      fn5 = functools.partial(fn3, 5)

      return gen_functional_ops.case(val, [], [dtypes.float32], [
          quarantine.defun_with_attributes(f).get_concrete_function()
          for f in (fn1, fn2, fn3, fn4, fn5)
      ])

    ones = array_ops.ones([10])
    self.assertAllEqual([ones], test_function(0))
    self.assertAllEqual([ones * 2], test_function(1))
    self.assertAllEqual([ones * 3], test_function(2))
    self.assertAllEqual([ones * 4], test_function(3))
    self.assertAllEqual([ones * 5], test_function(4))
    self.assertAllEqual([ones * 5], test_function(22))  # default branch

  @test_util.enable_control_flow_v2
  def testVariableInLoopInFunction(self):

    @quarantine.defun_with_attributes
    def test_function():

      def loop_test(_):
        return False

      def loop_body(_):
        return variable_scope.get_variable('a', shape=())

      return while_loop.while_loop(loop_test, loop_body, [0.0])

    self.assertEqual(test_function().shape, [])

  @test_util.run_in_graph_and_eager_modes
  def testDefunForcesResourceVariables(self):

    def variable_creator():
      self.v = variables.Variable(0.0)
      return self.v.read_value()

    self.v = None
    defined = quarantine.defun_with_attributes(variable_creator)
    defined()  # Create the variable.
    self.assertIsInstance(self.v, resource_variable_ops.ResourceVariable)

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

    defined = quarantine.defun_with_attributes(sum_gather)
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    expected = self.evaluate(sum_gather())
    self.assertAllEqual(expected, self.evaluate(defined()))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testCallOptionsMemory(self):

    @quarantine.defun_with_attributes
    def model(x):
      return x + constant_op.constant(1.)

    # This happens with a lot of option toggles, e.g. soft device placement
    context.context().function_call_options = None
    model(constant_op.constant(2.))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testLayerInDefun(self):
    conv = convolutional.Conv2D(
        filters=1,
        kernel_size=2,
        kernel_initializer=init_ops.ones_initializer(),
        bias_initializer=init_ops.zeros_initializer())

    @quarantine.defun_with_attributes
    def model(x):
      return conv(x)

    x = array_ops.ones([1, 2, 2, 1])
    y = model(x)

    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    self.assertAllClose([[[[4.0]]]], self.evaluate(y))

  @test_util.run_in_graph_and_eager_modes
  def testVariablesPlacedOnOutsideDevice(self):

    class _Obj(object):

      def __init__(self):
        self.v = None

      @quarantine.defun_with_attributes
      def f(self):
        if self.v is None:
          self.v = variables.Variable(1.)
        return self.v + 1.

    has_device = _Obj()
    with ops.device('cpu:0'):
      has_device.f()
    self.assertIn('CPU', has_device.v.device)

  def testCacheObjectHashCollisions(self):

    class Foo:

      def __hash__(self):
        return 42

    def func(foo):
      return constant_op.constant([id(foo)])

    defined = quarantine.defun_with_attributes(func)
    foo_1 = Foo()
    defined(foo_1)
    self.assertLen(total_function_cache(defined), 1)

    foo_2 = Foo()
    defined(foo_2)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorDtypeCollision(self):

    def func(t):
      return t + t

    defined = quarantine.defun_with_attributes(func)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 1)

    t = constant_op.constant([[1.0]], dtype=dtypes.complex128)
    defined(t)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorShapeCollision(self):

    def func(t):
      return t + t

    defined = quarantine.defun_with_attributes(func)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 1)

    t = constant_op.constant([1.0], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(total_function_cache(defined), 2)

  def testCacheTensorShapeDtypeCollision(self):

    def func(t):
      return t + t

    defined = quarantine.defun_with_attributes(func)
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
      defined = quarantine.defun_with_attributes(func, reduce_retracing=True)

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

    defined = quarantine.defun_with_attributes(func)
    defined(0, baz=20)
    self.assertLen(total_function_cache(defined), 1)

    defined(1)  # bar=1, baz=2
    self.assertLen(total_function_cache(defined), 2)

    # This matches the previous call.
    defined(foo=1)
    self.assertLen(total_function_cache(defined), 2)

    defined(1, 2, 3)
    self.assertLen(total_function_cache(defined), 3)

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

    defined = quarantine.defun_with_attributes(partial)
    func_a, func_b, func_c = defined(2)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureWithMatchingInputs(self):

    def foo(a):
      self.assertEqual(a.shape, (2,))
      return a

    signature = [tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = quarantine.defun_with_attributes(foo, input_signature=signature)
    a = array_ops.ones([2])
    self.assertAllEqual(a, defined(a))
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(a, defined.get_concrete_function()(a))
    self.assertAllEqual(a, defined.get_concrete_function(a)(a))
    self.assertAllEqual(
        a,
        defined.get_concrete_function(
            tensor_spec.TensorSpec((2,), dtype=dtypes.float32))(a))
    self.assertLen(total_function_cache(defined), 1)

    def bar(a):
      self.assertEqual(a._shape_tuple(), (2, None))
      return a

    signature = [tensor_spec.TensorSpec((2, None), dtypes.float32)]
    defined = quarantine.defun_with_attributes(bar, input_signature=signature)
    a = array_ops.ones([2, 1])
    out = defined(a)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out, a)

    # Changing the second dimension shouldn't create a new function.
    b = array_ops.ones([2, 3])
    out = defined(b)
    self.assertLen(total_function_cache(defined), 1)
    self.assertAllEqual(out, b)

  def testInputSignatureWithDictInPositionalArgs(self):

    @quarantine.defun_with_attributes
    def f(*_args, **_kwargs):
      return None

    f(1, x=2)
    self.assertLen(total_function_cache(f), 1)
    f(1, x=2)
    self.assertLen(total_function_cache(f), 1)
    f(1, {'x': 2})
    self.assertLen(total_function_cache(f), 2)

  def testInputSignatureWithCompatibleInputs(self):

    rank2_spec = tensor_spec.TensorSpec(
        shape=(None, None), dtype=dtypes.float32)

    @quarantine.defun_with_attributes(input_signature=[rank2_spec])
    def func(a):
      self.assertEqual([None, None], a.shape.as_list())
      return array_ops.shape(a)

    self.assertAllEqual([3, 1], func([[0], [1.0], [1]]))
    self.assertAllEqual([2, 2], func(numpy.array([[1, 1], [2, 2]])))

    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `func` failed'
    ):
      func([0.0, 1.0, 2.0])  # Wrong shape.

    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `func` failed'
    ):
      func([['wrong dtype']])

  def testNestedInputSignatures(self):

    def expected_foo(a, b):
      return [a, b]

    @quarantine.defun_with_attributes(input_signature=[
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

    @quarantine.defun_with_attributes(input_signature=[{
        'a': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'b': tensor_spec.TensorSpec((2, None), dtypes.float32),
        'c': tensor_spec.TensorSpec((1,), dtypes.float32)
    }])
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

    # Signatures must be either lists or tuples on their outermost levels.
    signature = {'t1': tensor_spec.TensorSpec([], dtypes.float32)}
    with self.assertRaisesRegex(
        TypeError, 'input_signature must be either a '
        'tuple or a list.*'):
      quarantine.defun_with_attributes(foo, input_signature=signature)

  def testInputsIncompatibleWithNestedSignatureRaisesError(self):

    def foo(a, b):
      return [a, b]

    signature = [[tensor_spec.TensorSpec((1,), dtypes.float32)] * 2,
                 [tensor_spec.TensorSpec((1,), dtypes.float32)] * 2]
    defined = quarantine.defun_with_attributes(foo, input_signature=signature)
    a = array_ops.ones([1])

    with self.assertRaisesRegex(TypeError,
                                'Binding inputs to tf.function `foo` failed'):
      defined([a, a, a], [a])

    with self.assertRaisesRegex(TypeError,
                                'Binding inputs to tf.function `foo` failed'):
      defined([a], [a, a, a])
    defined([a, a], [a, a])

  def testUnderspecifiedInputSignature(self):

    @quarantine.defun_with_attributes(input_signature=[
        tensor_spec.TensorSpec([], dtypes.float32),
    ])
    def foo(a, training=True):
      if training:
        return a
      else:
        return -1.0 * a

    x = constant_op.constant(1.0)
    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `foo` failed'):
      foo(x, training=False)

    self.assertAllEqual(x.numpy(), foo(x).numpy())

  def testInputSignatureWithPartialFunction(self):

    def full_function(a, b, c=3.0):
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    a, b, c = partial(2.0)
    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    defined = quarantine.defun_with_attributes(
        partial, input_signature=signature)
    x = constant_op.constant(2.0)
    func_a, func_b, func_c = defined(x)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureWithKeywordPositionalArgs(self):

    @quarantine.defun_with_attributes(input_signature=[
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

    x = quarantine.defun_with_attributes(
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

    signature = [
        ragged_tensor.RaggedTensorSpec(shape=[3, None], dtype=dtypes.int32)
    ]
    defined = quarantine.defun_with_attributes(f, input_signature=signature)
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
    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `f` failed'
    ):
      defined(rt3)

    # Different dtype
    rt4 = ragged_factory_ops.constant([[1.0, 2.0], [], [3.0]])
    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `f` failed'
    ):
      defined(rt4)

    # Different rank
    rt5 = ragged_factory_ops.constant([[[1]], [[2]], [[3]]])
    with self.assertRaisesRegex(
        TypeError, 'Binding inputs to tf.function `f` failed'
    ):
      defined(rt5)

  def testInputSignatureWithKeywordOnlyArgs(self):

    def f(a, b, c=3, *, d=4):
      self.assertIsInstance(a, ops.Tensor)
      self.assertIsInstance(b, ops.Tensor)
      self.assertIsInstance(c, int)
      self.assertIsInstance(d, (int, ops.Tensor))
      return a + b + c + d

    signature = [
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
    ]
    defined = quarantine.defun_with_attributes(f, input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 10)

    defined = quarantine.defun_with_attributes(
        functools.partial(f, c=4), input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 11)

    defined = quarantine.defun_with_attributes(
        functools.partial(f, d=5), input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 11)

    defined = quarantine.defun_with_attributes(
        functools.partial(f, d=array_ops.constant(5)),
        input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 11)

    mod = module.Module()
    save(mod, '/tmp/kwonlyf', defined.get_concrete_function(*signature))
    loaded = load('/tmp/kwonlyf')
    result = loaded.signatures['serving_default'](
        a=array_ops.constant(1),
        b=array_ops.constant(2),
        d=array_ops.constant(5))
    self.assertEqual(result['output_0'].numpy(), 11)

  def testInputSignatureWithKeywordOnlyArgsNoDefaults(self):
    signature = [
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
    ]

    def test_func(a, *, b):
      return a + b

    with self.assertRaisesRegex(
        TypeError,
        (
            'Since input_signature is defined, keyword-only parameter `b` must'
            ' have a default value'
        ),
    ):
      quarantine.defun_with_attributes(test_func, input_signature=signature)

    test_func_lambda = lambda a, *, b: a + b
    with self.assertRaisesRegex(
        TypeError,
        (
            'Since input_signature is defined, keyword-only parameter `b` must'
            ' have a default value'
        ),
    ):
      quarantine.defun_with_attributes(
          test_func_lambda, input_signature=signature
      )

  def testTensorKeywordArguments(self):

    def foo(a, b):
      del a
      return b

    defined = quarantine.defun_with_attributes(foo)
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

  def testFunctionWithInvalidAttribute(self):
    def add(x, y):
      return math_ops.add(x, y)

    with self.assertRaisesRegex(
        ValueError,
        'TracingCompiler does not support `experimental_1` as an attribute.',
    ):
      quarantine.defun_with_attributes(
          add, attributes={'experimental_1': 'value1'}
      )

  def testRegisterFunction(self):

    @quarantine.defun_with_attributes
    def add(x, y):
      return math_ops.add(x, y)

    def matmul(x, y):
      return math_ops.matmul(x, y)

    defun_matmul = quarantine.defun_with_attributes(matmul)

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        concrete_func_matmul = defun_matmul.get_concrete_function(t, t)
        concrete_func_matmul.add_to_graph()
        concrete_func_matmul.add_gradient_functions_to_graph()

        concrete_func_add = add.get_concrete_function(t, t)
        concrete_func_add.add_to_graph()
        concrete_func_add.add_gradient_functions_to_graph()

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
          self.assertRegex(captured_function_names[i],
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

  def testRegisterConcreteFunction(self):

    @quarantine.defun_with_attributes
    def py_add(x, y):
      return math_ops.add(x, y)

    py_add(array_ops.ones([]), array_ops.ones([]))
    add = py_add.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32),
        tensor_spec.TensorSpec(None, dtypes.float32))

    @quarantine.defun_with_attributes
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
        for expected, found in zip(expected_func_name_regex,
                                   captured_function_names):
          self.assertRegex(found, expected)

        composite_t, composite_double = composite(t, t)
        double = add(t, t)
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(double))
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(composite_double))
        self.assertAllEqual([[1, 2], [3, 4]], self.evaluate(composite_t))
        # Make sure the pre registered function is used, and no other function
        # is added.
        self.assertLen(graph._functions, 6)

  def testEagerCaptures(self):
    with context.eager_mode():
      large_tensor = array_ops.ones(shape=(256,))
      self.assertGreater(256, func_graph._EAGER_CONST_THRESHOLD)

      small_tensor = array_ops.ones(shape=(4,))
      self.assertLessEqual(4, func_graph._EAGER_CONST_THRESHOLD)

      v = resource_variable_ops.ResourceVariable(0.0)

    for captured, op_type in [(large_tensor, 'Placeholder'),
                              (small_tensor, 'Const'), (v, 'Placeholder')]:

      @quarantine.defun_with_attributes
      def test_fn():
        return captured + 1  # pylint: disable=cell-var-from-loop

      g = test_fn.get_concrete_function().graph
      internal_captures = g.internal_captures
      self.assertLen(internal_captures, 1)
      self.assertEqual(internal_captures[0].op.type, op_type)

  def testRegisterFunctionWithInputSignature(self):

    def matmul(x, y):
      return math_ops.matmul(x, y)

    defun_matmul = quarantine.defun_with_attributes(
        matmul,
        input_signature=[
            tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(2, 2), dtype=dtypes.float32)
        ])
    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        concrete_func = defun_matmul.get_concrete_function(t, t)
        concrete_func.add_to_graph()
        concrete_func.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

        # Test register function with cache, note inputs are ignored.
        concrete_func = defun_matmul.get_concrete_function()
        concrete_func.add_to_graph()
        concrete_func.add_gradient_functions_to_graph()
        graph = ops.get_default_graph()
        self.assertLen(graph._functions, 3)

  def testRegisterFunctionWithCache(self):

    def matmul(x, y):
      return math_ops.matmul(x, y)

    defun_matmul = quarantine.defun_with_attributes(matmul)

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        t2 = constant_op.constant([[2.0, 3.0], [4.0, 5.0]])
        concrete_func_t = defun_matmul.get_concrete_function(t, t)
        concrete_func_t.add_to_graph()
        concrete_func_t.add_gradient_functions_to_graph()

        concrete_func_t2 = defun_matmul.get_concrete_function(t2, t2)
        concrete_func_t2.add_to_graph()
        concrete_func_t2.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # Only one function is registered since the input param are in same type
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

  def testCallingFunctionWithDifferentVariables(self):

    @quarantine.defun_with_attributes
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

    @quarantine.defun_with_attributes
    def bar(v):
      del v
      return constant_op.constant(1.0)

    graph_function = bar.get_concrete_function(v)
    self.assertEqual(float(graph_function(v)), 1.0)
    self.assertEqual(float(graph_function(w)), 1.0)

  def testCallingFunctionWithNonTensorsFails(self):

    @quarantine.defun_with_attributes
    def foo(x):
      return x

    graph_function = foo.get_concrete_function(constant_op.constant(1.0))
    with self.assertRaises((TypeError, ValueError)):
      graph_function('Not a Tensor.')

  @parameterized.parameters([
      (
          quarantine.defun_with_attributes(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'CPU',
              }
          ),
          quarantine.defun_with_attributes(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'GPU',
              }
          ),
      ),
      (
          polymorphic_function.function(
              experimental_attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'CPU',
              }
          ),
          polymorphic_function.function(
              experimental_attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'GPU',
              }
          ),
      ),
  ])
  def testSwapImplementationWithGrapplerPlugin(
      self, cpu_decorator, gpu_decorator
  ):
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

      @cpu_decorator
      def cpu_boost(x):
        return math_ops.add(x, 2.0)

      @gpu_decorator
      def gpu_boost(x):
        return math_ops.add(x, 4.0)

      x = constant_op.constant(1.0)

      concrete_func = cpu_boost.get_concrete_function(x)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
      y = gpu_boost(x)
      y_value = self.evaluate(y)

      if test.is_gpu_available():
        self.assertEqual(y_value, 5.0)
      else:
        # Grappler fallback to use the CPU impl even called with GPU function.
        self.assertEqual(y_value, 3.0)

  @test_util.disable_tfrt('b/174712583: TFRT doesn\'t support behavior '
                          'equivalent to implementation_selector for function')
  def testSwapImplementationInEager(self):
    if not context.executing_eagerly():
      self.skipTest('eager only')

    # testSharedRendezvous sets the disable_meta_optimizer flag to True
    # if that subtest runs before this one, then having that set to True
    # will cause this subtest to fail. To avoid that scenario, explicitly
    # set the disable_meta_optimizer flag to false here
    context.context().set_optimizer_experimental_options({
        'min_graph_nodes': -1,
        'implementation_selector': True,
        'disable_meta_optimizer': False
    })

    @quarantine.defun_with_attributes(attributes={
        'api_implements': 'foo',
        'api_preferred_device': 'CPU'
    })
    def on_cpu(x):
      return x + 2

    @quarantine.defun_with_attributes(attributes={
        'api_implements': 'foo',
        'api_preferred_device': 'GPU'
    })
    def on_gpu(x):
      return x + 4

    @quarantine.defun_with_attributes
    def run_on_cpu(t):
      concrete_func = on_cpu.get_concrete_function(t)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
      with ops.device('CPU:0'):
        return on_gpu(t)

    # Expect to run the on_cpu branch, regardless whether gpu is available.
    self.assertEqual(run_on_cpu(constant_op.constant(1)).numpy(), 3)

  def testDefunFunctionSeparateGraphs(self):
    with context.graph_mode():

      @quarantine.defun_with_attributes
      def add(x):
        return x + 5

      @quarantine.defun_with_attributes
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

    @quarantine.defun_with_attributes
    def defined(t):
      return t

    defined(array_ops.zeros([12, 1]))
    self.assertLen(total_function_cache(defined), 1)
    defined(array_ops.zeros([1, 21]))
    self.assertLen(total_function_cache(defined), 2)

    @quarantine.defun_with_attributes
    def defined_again(t):
      return defined(t)

    defined_again.get_concrete_function(array_ops.zeros([12, 1]))
    self.assertLen(total_function_cache(defined_again), 1)
    defined_again.get_concrete_function(array_ops.zeros([1, 21]))
    self.assertLen(total_function_cache(defined_again), 2)

  def testCacheTensorSpecIdenticalToTensor(self):

    @quarantine.defun_with_attributes
    def defined(t):
      return t

    z = array_ops.zeros([2, 2])
    z_spec = tensor_spec.TensorSpec.from_tensor(z)
    self.assertIs(
        defined.get_concrete_function(z_spec), defined.get_concrete_function(z))

  def testCacheKeyNestedLists(self):

    @quarantine.defun_with_attributes
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
    class TestClass:
      a = attr.ib()
      b = attr.ib()

    @quarantine.defun_with_attributes
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

  def testDistinctVariablesNoRetracing(self):

    @quarantine.defun_with_attributes
    def defined(a, b, c):
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)

    # We generate cache keys based on unique combinations of resource ids.
    defined(x, y, z)
    self.assertLen(total_function_cache(defined), 1)

    # Re-arranging arguments should not cause cache miss
    # because the three inputs are still distinct
    defined(z, y, x)
    self.assertLen(total_function_cache(defined), 1)

  def testRetracingOnDifferentVaribleCombinationPatterns(self):

    @quarantine.defun_with_attributes
    def defined(a, b, c):
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)

    defined(x, y, z)
    self.assertLen(total_function_cache(defined), 1)

    # Retracing because the first two arguments are the same
    defined(x, x, z)
    self.assertLen(total_function_cache(defined), 2)

    # Replacing x with y does not cause cache miss
    # because the combination stays the same as (x, x, z)
    defined(y, y, z)
    self.assertLen(total_function_cache(defined), 2)

    # A different combination pattern causes cache miss
    defined(z, y, y)
    self.assertLen(total_function_cache(defined), 3)
    defined(z, y, y)
    self.assertLen(total_function_cache(defined), 3)

  def testDeepcopyVariableNoRetracing(self):

    @quarantine.defun_with_attributes
    def defined(a, b, c):
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)
    defined(x, y, z)
    self.assertLen(total_function_cache(defined), 1)

    x_copy = copy.deepcopy(x)
    defined(x_copy, y, z)
    self.assertLen(total_function_cache(defined), 1)

  def testDecoratedMethodInspect(self):

    class DefunnedMiniModel:

      @quarantine.defun_with_attributes
      def call(self, inputs, training=True):
        pass

    m = DefunnedMiniModel()
    fullargspec = tf_inspect.getfullargspec(m.call)
    self.assertIn('training', fullargspec.args)

  @test_util.disable_tfrt('b/173429686')
  def testExecutorType(self):

    @quarantine.defun_with_attributes
    def add_five(x):
      return x + 5

    self.assertEqual(
        5,
        add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())

    with self.assertRaisesRegex(errors.NotFoundError, 'NON_EXISTENT_EXECUTOR'):
      with context.function_executor_type('NON_EXISTENT_EXECUTOR'):
        add_five(constant_op.constant(0, dtype=dtypes.int32))

    for executor_type in ('', 'DEFAULT', None):
      with context.function_executor_type(executor_type):
        self.assertAllEqual(
            5,
            add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy())

  @test_util.assert_no_garbage_created
  def testReferenceCycles(self):

    fn = quarantine.defun_with_attributes(lambda x: 2. * x)

    fn(constant_op.constant(4.0))
    weak_fn = weakref.ref(fn)
    del fn
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  @test_util.run_in_graph_and_eager_modes
  def testShapeCaching(self):

    @quarantine.defun_with_attributes
    def func(x):
      return array_ops.shape(x)

    @quarantine.defun_with_attributes(
        input_signature=[tensor_spec.TensorSpec([None, None], dtypes.float32)])
    def calls_func(x):
      return func(x)

    self.assertAllEqual([1, 1], self.evaluate(func(array_ops.zeros([1, 1]))))
    self.assertAllEqual([2, 2], self.evaluate(func(array_ops.zeros([2, 2]))))
    self.assertAllEqual([3, 3],
                        self.evaluate(calls_func(array_ops.zeros([3, 3]))))

  def testLimitedRetracing(self):
    trace_count = [0]

    @quarantine.defun_with_attributes
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


class DefunCollectionTest(test.TestCase):

  def testCollectionValueAccess(self):
    """Read values from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection('x', x)
        ops.add_to_collection('y', y)

        @quarantine.defun_with_attributes
        def fn():
          x_const = constant_op.constant(ops.get_collection('x')[0])
          y_const = constant_op.constant(ops.get_collection('y')[0])
          z = math_ops.add(x_const, y_const)
          ops.add_to_collection('z', 7)
          return z

        self.assertEqual(7, int(self.evaluate(fn())))
        self.assertEqual(ops.get_collection('x'), [2])
        self.assertEqual(ops.get_collection('y'), [5])
        self.assertEqual(ops.get_collection('z'), [])

  def testCollectionVariableValueAccess(self):
    """Read variable value from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        v = resource_variable_ops.ResourceVariable(1.0)

        @quarantine.defun_with_attributes
        def f():
          return v.read_value()

        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(1.0, float(self.evaluate(f())))
        self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)

  def testCollectionVariableValueWrite(self):
    """Write variable value inside defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        @quarantine.defun_with_attributes
        def f():
          v = resource_variable_ops.ResourceVariable(2.0)
          return v

        _ = f.get_concrete_function()
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(2.0, float(self.evaluate(f())))
        self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)


class MultiDeviceDefunTest(test.TestCase, parameterized.TestCase):

  @test_util.run_gpu_only
  def testMultiDeviceOutput(self):
    """Tests that functions can produce outputs on multiple devices."""

    @quarantine.defun_with_attributes
    def func(a, b, transpose_a):
      with ops.device('/device:CPU:0'):
        m1 = math_ops.matmul(a, b, transpose_a=transpose_a)
      with ops.device('/device:GPU:0'):
        m2 = math_ops.matmul(a, b, transpose_a=transpose_a)
      return m1, m2

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    m1, m2 = func(t, t, transpose_a=True)
    self.assertAllEqual(m1.numpy(), [[10, 14], [14, 20]])
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), [[10, 14], [14, 20]])
    self.assertRegex(m2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testEmptyBody(self):

    @quarantine.defun_with_attributes
    def func(a, b):
      return b, a

    with ops.device('/device:CPU:0'):
      a = array_ops.identity(3.0)
    with ops.device('/device:GPU:0'):
      b = array_ops.identity(5.0)

    m1, m2 = func(a, b)
    self.assertAllEqual(m1.numpy(), 5.0)
    self.assertRegex(m1.backing_device, 'GPU')
    self.assertAllEqual(m2.numpy(), 3.0)
    self.assertRegex(m2.backing_device, 'CPU')

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

    @quarantine.defun_with_attributes
    def func(int_cpu, resource, int_gpu):
      with ops.device('/device:CPU:0'):
        m1 = int_cpu * resource + int_gpu
      with ops.device('/device:GPU:0'):
        # This computation will happen on GPU but m2 will be copied to CPU.
        m2 = int_gpu * resource + int_cpu + 1
      return m1, m2

    m1, m2 = func(int_cpu, resource, int_gpu)
    self.assertAllEqual(m1.numpy(), 22)
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 39)
    self.assertRegex(m2.backing_device, 'CPU')

    # flip arguments
    m1, m2 = func(int_gpu, resource, int_cpu)
    self.assertAllEqual(m1.numpy(), 38)
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 23)
    self.assertRegex(m2.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testMultiDeviceColocateWith(self):
    """Tests that function's outputs respect colocation constraints."""

    @quarantine.defun_with_attributes
    def func(a, b):
      with ops.colocate_with(a):
        ra = 2 * a
      with ops.colocate_with(b):
        rb = 3 * b
      return ra, rb

    devices = ['/device:CPU:0', '/device:GPU:0']
    for dev1, dev2 in itertools.product(devices, devices):
      with ops.device(dev1):
        a = array_ops.identity(1.0)
      with ops.device(dev2):
        b = array_ops.identity(10.0)

      ra, rb = func(a, b)
      self.assertEqual(ra.numpy(), 2.0)
      self.assertRegex(ra.backing_device, dev1)
      self.assertEqual(rb.numpy(), 30.0)
      self.assertRegex(rb.backing_device, dev2)

  @test_util.run_gpu_only
  def testMultiDeviceResources(self):
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
      c2 = resource_variable_ops.ResourceVariable(7.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)
      g2 = resource_variable_ops.ResourceVariable(5.0)

    @quarantine.defun_with_attributes
    def func(resource1, resource2):
      with ops.device('/device:CPU:0'):
        result1 = resource1 * g2
      with ops.device('/device:GPU:0'):
        result2 = resource2 * c2
      return result1, result2

    r1, r2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegex(r2.backing_device, 'GPU')

    # Call with flipped inputs. Check that we look at resource's
    # device and reinstantiates the function when inputs' devices change.
    r1, r2 = func(g1, c1)
    self.assertEqual(r1.numpy(), 15.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegex(r2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testOutputResources(self):
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @quarantine.defun_with_attributes
    def func(resource1, resource2):
      with ops.device('/device:CPU:0'):
        result1 = resource1 * 5
      with ops.device('/device:GPU:0'):
        result2 = resource2 * 7
      return result1, resource1.handle, result2, resource2.handle

    r1, res1, r2, res2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegex(r2.backing_device, 'GPU')

    def check_handle(handle, expected_value):
      self.assertRegex(handle.backing_device, 'CPU')
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
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegex(r2.backing_device, 'GPU')
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

    @quarantine.defun_with_attributes(attributes={
        '_noinline': True,
        '_nospecialize': True
    })
    def inner(resource1):
      return resource1 * 2, resource1.handle

    @quarantine.defun_with_attributes
    def outer(resource1):
      with ops.device('/device:CPU:0'):
        r1, _ = inner(resource1)
      return r1

    r1 = outer(g1)

    self.assertEqual(r1.numpy(), 6.0)
    self.assertRegex(r1.backing_device, 'CPU')

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

    @quarantine.defun_with_attributes(attributes={'_noinline': True})
    def inner(resource1):
      resource1.assign_add(2.0)
      return resource1 * 2, resource1.handle

    @quarantine.defun_with_attributes
    def outer(resource1):
      with ops.device('/device:CPU:0'):
        r1, res1 = inner(resource1)
      return r1, res1

    r1, res1 = outer(g1)

    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')

    def check_handle(handle, expected_value):
      self.assertRegex(handle.backing_device, 'CPU')
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
      cc0 = array_ops.identity(5.0)
      cc1 = array_ops.identity(7.0)
    with ops.device('/device:GPU:0'):
      rg0 = resource_variable_ops.ResourceVariable(11.0)
      rg1 = resource_variable_ops.ResourceVariable(13.0)
      cg0 = array_ops.identity(17.0)
      cg1 = array_ops.identity(19.0)

    # Make sure tensors are on expected devices.
    for tensor in [cc0, cc1]:
      self.assertRegex(tensor.backing_device, 'CPU:0')
    for tensor in [cg0, cg1]:
      self.assertRegex(tensor.backing_device, 'GPU:0')

    @quarantine.defun_with_attributes
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
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertRegex(m2.backing_device, 'GPU')
    self.assertRegex(r2.backing_device, 'GPU')
    self.assertEqual(m1.numpy(), 34.0)
    self.assertEqual(r1.numpy(), 55000.0 + 3.0 * 19.0)
    self.assertEqual(m2.numpy(), 55.0)
    self.assertEqual(r2.numpy(), 34000.0 + 13.0 * 7.0)

  @test_util.run_gpu_only
  def testArgumentPruning(self):
    """Tests functions taking unnecessary arguments."""
    with ops.device('/device:CPU:0'):
      c1 = constant_op.constant(5.0)
      c2 = constant_op.constant(7.0)

    with ops.device('/device:GPU:0'):
      g1 = constant_op.constant(11.0)
      g2 = constant_op.constant(13.0)
      g3 = constant_op.constant(17.0)

    @quarantine.defun_with_attributes
    def func(g1, g2, c1, g3, c2):  # pylint: disable=unused-argument
      # arguments g1 and g2 are unused and can be pruned by grappler.
      return c1 * g3 * c2

    result = func(g1, g2, c1, g3, c2)
    self.assertEqual(result.numpy(), 5.0 * 7.0 * 17.0)


class FunctionCallbackTest(test.TestCase, parameterized.TestCase):

  def testAddFunctionCallback(self):
    functions = []

    def function_callback(f, name, graph, inputs, outputs):
      del name, graph, inputs, outputs
      functions.append(f)

    @polymorphic_function.function
    def plus_one(x):
      return x + 1

    try:
      quarantine.add_function_callback(function_callback)
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
      quarantine.clear_function_callbacks()

  def testFunctionCallbackAddOps(self):
    file_name = os.path.join(self.get_temp_dir(), 'test')

    def function_callback(f, name, graph, inputs, outputs):
      del f, name, inputs

      with graph.as_default():
        printer = logging_ops.print_v2(
            'hello', output_stream='file://' + file_name)
        outputs[0].op._add_control_input(printer)

    @polymorphic_function.function
    def plus_one(x):
      return x + 1

    self.addCleanup(quarantine.clear_function_callbacks)
    quarantine.add_function_callback(function_callback)
    x_float32 = numpy.array(3.0, dtype=numpy.float32)

    self.assertAllClose(plus_one(x_float32), 4.0)

    with open(file_name, 'r') as f:
      self.assertEqual(f.read().strip(), 'hello')

  def testRemoveFunctionCallback(self):
    functions_1 = []

    def function_callback_1(f, name, graph, inputs, outputs):
      del name, graph, inputs, outputs
      functions_1.append(f)

    functions_2 = []

    def function_callback_2(f, name, graph, inputs, outputs):
      del name, graph, inputs, outputs
      functions_2.append(f)

    @polymorphic_function.function
    def plus_one(x):
      return x + 1

    try:
      quarantine.add_function_callback(function_callback_1)
      quarantine.add_function_callback(function_callback_2)
      self.assertAllClose(plus_one(numpy.array(3.0, dtype=numpy.float32)), 4.0)
      self.assertLen(functions_1, 1)
      self.assertLen(functions_2, 1)
      quarantine.remove_function_callback(function_callback_1)
      # The 1st callback should not be invokved after remove_function_callback()
      # is called.
      self.assertAllClose(plus_one(numpy.array(3.0, dtype=numpy.float64)), 4.0)
      self.assertLen(functions_1, 1)
      self.assertLen(functions_2, 2)
    finally:
      quarantine.clear_function_callbacks()

  def testClearFunctionCallbacks(self):
    quarantine.add_function_callback(lambda f: None)
    quarantine.add_function_callback(lambda f: None)
    self.assertLen(atomic_function.function_callbacks, 2)
    quarantine.clear_function_callbacks()
    self.assertEmpty(atomic_function.function_callbacks)

  @test_util.run_in_graph_and_eager_modes
  def testBackwardNoneGradient(self):
    model = variables.Variable(1.0, name='model')
    count = variables.Variable(0)

    @quarantine.defun_with_attributes
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


class DefunArgumentNamingTest(test.TestCase, parameterized.TestCase):
  """Tests for recognizable export signatures from concrete functions."""

  def testBasic(self):
    @quarantine.defun_with_attributes
    def fn(a, b):
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertLen(fn_op.graph.structured_outputs, 2)
    self.assertAllClose(
        [3., 2.],
        fn_op(constant_op.constant(1.), constant_op.constant(2.)))
    self.assertAllClose(
        [3., 2.],
        fn_op(a=constant_op.constant(1.), b=constant_op.constant(2.)))

  def testVariable(self):
    @quarantine.defun_with_attributes
    def fn(a, b):
      return a + b, a * b
    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        variables.Variable(1.))
    self.assertEqual(
        ['a', 'b'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertLen(fn_op.graph.structured_outputs, 2)

  def testDictReturned(self):
    @quarantine.defun_with_attributes
    def fn(x, z=(1., 2.), y=3.):
      z1, z2 = z
      return {'alpha': x + y + z1, 'beta': x * y + z2}
    # Call the function to make def_function happy
    fn(array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        x=tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x', 'y'],
        [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'x', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs])
    self.assertEqual({'alpha', 'beta'},
                     set(fn_op.graph.structured_outputs.keys()))

    fn_op2 = fn.get_concrete_function(
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z_first'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                                  name='z_second')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        x=4.)
    self.assertEqual(
        ['z_first', 'z_second', 'custom'],
        [inp.op.name for inp in fn_op2.inputs])
    self.assertEqual(
        [b'z_first', b'z_second', b'custom'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs])

    fn_op3 = fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        z=(tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32,
                                  name='z1'),
           tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z2')),
        y=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['custom', 'z1', 'z2', 'y'],
        [inp.op.name for inp in fn_op3.inputs])
    self.assertEqual(
        [b'custom', b'z1', b'z2', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs])

  def testMethod(self):
    class HasMethod(object):

      @quarantine.defun_with_attributes
      def method(self, x):
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    HasMethod.method(has_method, array_ops.ones([]))
    class_op = HasMethod.method.get_concrete_function(
        has_method, tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in class_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs])
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([]))
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32))
    self.assertEqual(
        ['x'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    # TODO(allenl): It should be possible to override names when exporting. Do
    # TensorSpec names need to go in cache keys? Or maybe get_concrete_function
    # should always retrace?
    self.skipTest('Not working')
    method_op = has_method.method.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='y'))
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])

  def testMethodSignature(self):

    class HasMethod(object):

      @quarantine.defun_with_attributes(
          input_signature=(tensor_spec.TensorSpec(
              shape=None, dtype=dtypes.float64, name='y'),))
      def method(self, x):
        hash(self)  # No weak proxies passed as `self`
        return x

    has_method = HasMethod()
    # Call the function to make def_function happy
    has_method.method(array_ops.ones([], dtype=dtypes.float64))
    method_op = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs])
    method_op2 = has_method.method.get_concrete_function()
    self.assertEqual(
        ['y'],
        [inp.op.name for inp in method_op2.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs])

  def testVariadic(self):
    @quarantine.defun_with_attributes
    def variadic_fn(x, *args, **kwargs):
      return x + math_ops.add_n(list(args) + list(kwargs.values()))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32,
                               name='second_variadic'),
        z=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
        zz=tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'))
    self.assertEqual(
        ['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'],
        [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs])

  def testVariadicInputSignature(self):
    @quarantine.defun_with_attributes(
        input_signature=(
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32, name='z'),
        ))
    def variadic_fn(x, *args):
      return x + math_ops.add_n(list(args))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]),
                array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function()
    self.assertIn(b'variadic_fn', variadic_op.name)
    self.assertEqual(
        ['x', 'y', 'args_1', 'z'],
        [inp.op.name for inp in variadic_op.inputs])
    self.assertEqual(
        [b'x', b'y', b'args_1', b'z'],
        [inp.op.get_attr('_user_specified_name')
         for inp in variadic_op.inputs])


class DevicePlacementTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testMultipleDeviceCheck(self):

    def f():
      with ops.device('cpu'):
        return test_ops.device_placement_op()

    func = quarantine.defun_with_attributes(f)
    with ops.device('cpu:0'):
      output = self.evaluate(func())
      self.assertIn(compat.as_bytes('CPU:0'), output)

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

    defined = quarantine.defun_with_attributes(multi_device_fn)
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


def setUpModule():
  ops.enable_eager_execution()
  cpus = config.list_physical_devices('CPU')
  # Set 4 virtual CPUs
  config.set_logical_device_configuration(cpus[0], [
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration()
  ])


if __name__ == '__main__':
  test.main()
