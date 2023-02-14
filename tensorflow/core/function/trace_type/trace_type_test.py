# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for the trace_type module."""

import collections
import timeit

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.trace_type import default_types
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class TestAttr:
  """Helps test attrs collections."""

  def __init__(self, name):
    self.name = name


class TestAttrsClass:
  """Helps test attrs collections."""

  __attrs_attrs__ = (TestAttr('a'), TestAttr('b'))

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __eq__(self, other):
    return isinstance(
        other, TestAttrsClass) and self.a == other.a and self.b == other.b


class DummyGenericClass:
  """Helps test memory leaks for GenericType."""


class TraceTypeBuilderTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCompositeAndSpec(self):
    composite_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        values=[1, 2, 3], row_splits=[0, 2, 3])
    spec = ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32)

    self.assertEqual(
        trace_type.from_value(composite_tensor), trace_type.from_value(spec))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testVariableAliasing(self):
    v1 = resource_variable_ops.ResourceVariable([1])
    v2 = resource_variable_ops.ResourceVariable([1])
    v3 = resource_variable_ops.ResourceVariable([1])
    all_unique = trace_type.from_value((v1, v2, v3))
    all_same = trace_type.from_value((v1, v1, v1))
    self.assertNotEqual(all_unique, all_same)

    v3 = resource_variable_ops.ResourceVariable([2])
    v4 = resource_variable_ops.ResourceVariable([2])
    v5 = resource_variable_ops.ResourceVariable([2])
    all_unique_again = trace_type.from_value((v3, v4, v5))
    all_same_again = trace_type.from_value((v4, v4, v4))
    self.assertEqual(all_unique, all_unique_again)
    self.assertEqual(all_same, all_same_again)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorEquality(self):
    context = trace_type.InternalTracingContext()
    tensor_a = array_ops.zeros([11, 3, 5],
                               dtype=dtypes.int32).__tf_tracing_type__(context)
    tensor_b = array_ops.zeros([11, 4, 5],
                               dtype=dtypes.int32).__tf_tracing_type__(context)
    tensor_c = array_ops.zeros(
        [11, 3, 5], dtype=dtypes.float32).__tf_tracing_type__(context)
    tensor_d = array_ops.ones([11, 3, 5],
                              dtype=dtypes.int32).__tf_tracing_type__(context)

    self.assertNotEqual(tensor_a, tensor_b)
    self.assertNotEqual(tensor_a, tensor_c)
    self.assertNotEqual(tensor_b, tensor_c)
    self.assertEqual(tensor_a, tensor_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorAndSpecEquality(self):
    context = trace_type.InternalTracingContext()
    tensor = array_ops.zeros([11, 3, 5],
                             dtype=dtypes.int32).__tf_tracing_type__(context)
    spec = tensor_spec.TensorSpec(
        [11, 3, 5], dtype=dtypes.int32).__tf_tracing_type__(context)
    spec_with_name = tensor_spec.TensorSpec(
        [11, 3, 5], dtype=dtypes.int32,
        name='name').__tf_tracing_type__(context)

    self.assertEqual(tensor, spec)
    self.assertNotEqual(tensor, spec_with_name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorShapeUnknown(self):
    context = trace_type.InternalTracingContext()
    spec_1 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32).__tf_tracing_type__(context)
    spec_2 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32).__tf_tracing_type__(context)
    self.assertEqual(spec_1, spec_2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testAttrsTraceTypeGeneration(self):
    trace_a = trace_type.from_value(TestAttrsClass(1, 2))
    expected = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass, (default_types.Literal(1), default_types.Literal(2)))
    self.assertEqual(trace_a, expected)
    self.assertTrue(trace_a.is_subtype_of(trace_a))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTupleEquality(self):
    trace_a = trace_type.from_value((1, 2, 3, 4))
    trace_b = trace_type.from_value((1, 2, 2, 4))
    trace_c = trace_type.from_value((1, 2, 3))
    trace_d = trace_type.from_value((1, 2, 3, 4))
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testListEquality(self):
    trace_a = trace_type.from_value([1, 2, 3, 4])
    trace_b = trace_type.from_value([1, 2, 2, 4])
    trace_c = trace_type.from_value([1, 2, 3])
    trace_d = trace_type.from_value([1, 2, 3, 4])
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDictEquality(self):
    trace_a = trace_type.from_value({1: 2, 3: 4})
    trace_b = trace_type.from_value({1: 2, 3: 2})
    trace_c = trace_type.from_value({1: 2, 3: 0})
    trace_d = trace_type.from_value({3: 4, 1: 2})
    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testComplexStruct(self):
    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}
    trace_a = trace_type.from_value(struct)
    trace_b = trace_type.from_value(struct)
    self.assertEqual(trace_a, trace_b)
    self.assertTrue(trace_a.is_subtype_of(trace_b))
    self.assertTrue(trace_b.is_subtype_of(trace_a))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCustomUnequableTypeSucceeds(self):

    class CustomUnequable:

      def __eq__(self, o):
        raise ValueError

      def __hash__(self):
        return 0

    object_a = CustomUnequable()
    object_b = CustomUnequable()
    trace_a_1 = trace_type.from_value(object_a)
    trace_a_2 = trace_type.from_value(object_a)
    trace_b = trace_type.from_value(object_b)
    self.assertEqual(trace_a_1, trace_a_2)

    with self.assertRaises(ValueError):
      trace_a_1.__eq__(trace_b)

    del object_a
    self.assertNotEqual(trace_a_1, trace_a_2)
    self.assertNotEqual(trace_a_2, trace_a_1)

    del object_b
    self.assertNotEqual(trace_a_1, trace_a_2)
    self.assertNotEqual(trace_a_2, trace_a_1)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCustomUnhashableTypeFailsGracefully(self):

    class CustomUnhashable:

      def __eq__(self, o):
        return True

    obj = CustomUnhashable()
    with self.assertRaisesRegex(
        TypeError,
        r'Could not generate a generic TraceType for'):
      trace_type.from_value(obj)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testGetDefaultPlaceholderValue(self):
    placeholder_context = trace_type.InternalPlaceholderContext()
    composite_value = [1, 2, (3, [4, 5]), {6: [7]}, TestAttrsClass(8, (10, 11))]
    composite_type = trace_type.from_value(composite_value)
    placeholder_value = composite_type.placeholder_value(placeholder_context)
    self.assertEqual(composite_value, placeholder_value)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testWrappedNamedTuple(self):
    ActualType = collections.namedtuple('ActualType', ['a', 'b', 'c'])

    class MockWrapper(tuple):
      # Generated through trackable data structures:
      # //tensorflow/python/training/tracking/data_structures.py
      # With design pattern similar to Python functools:
      # https://docs.python.org/3/library/functools.html?highlight=__wrapped__#functools.update_wrapper
      __wrapped__ = ActualType(1, 2, 3)

    self.assertEqual(
        trace_type.from_value(MockWrapper()),
        trace_type.from_value(ActualType(1, 2, 3)))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBadReturnType(self):
    class MyClass:

      def __tf_tracing_type__(self, _):
        return 1

    with self.assertRaises(TypeError):
      trace_type.from_value(MyClass())


class CastDefaultTypesTest(test.TestCase, parameterized.TestCase):

  def testLiteral(self):
    trace_float = default_types.Literal(1.5)
    ctx = trace_type.InternalCastContext()
    value = trace_float._cast(1.5, ctx)
    self.assertEqual(value, 1.5)
    with self.assertRaises(AssertionError):
      _ = trace_float._cast(1, ctx)

  @parameterized.parameters(list, tuple)
  def testTupleAndList(self, container_type):
    foo = (
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    foo = container_type(foo)
    trace_foo = trace_type.from_value(foo)
    bar = (1, 2)
    bar = container_type(bar)
    ctx = trace_type.InternalCastContext()
    value = trace_foo._cast(bar, ctx)

    self.assertIsInstance(value, container_type)
    self.assertLen(value, len(bar))
    self.assertSequenceEqual(value, bar)
    self.assertEqual(value[0].dtype, dtypes.float32)
    self.assertEqual(value[1].dtype, dtypes.float32)

  @parameterized.parameters(
      (list, tuple),
      (tuple, list))
  def testTupleAndListCannotBeCasted(self, type_a, type_b):
    foo = (
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    foo = type_a(foo)
    trace_foo = trace_type.from_value(foo)
    bar = (1, 2)
    bar = type_b(bar)
    ctx = trace_type.InternalCastContext()
    with self.assertRaises(AssertionError):
      _ = trace_foo._cast(bar, ctx)

  def testNamedTuple(self):
    Foo = collections.namedtuple('Foo', ['x', 'y'])
    foo = Foo(
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32))
    trace_foo = trace_type.from_value(foo)
    bar = Foo(1, 2)
    ctx = trace_type.InternalCastContext()
    value = trace_foo._cast(bar, ctx)

    self.assertIsInstance(value, Foo)
    self.assertLen(value, len(bar))
    self.assertSequenceEqual(value, bar)
    self.assertEqual(value[0].dtype, dtypes.float32)
    self.assertEqual(value[1].dtype, dtypes.float32)

  def testAttrs(self):
    foo = TestAttrsClass(
        constant_op.constant(1.0, dtypes.float32),
        constant_op.constant(2.0, dtypes.float32),)
    trace_foo = trace_type.from_value(foo)
    bar = TestAttrsClass(1, 2)
    ctx = trace_type.InternalCastContext()
    value = trace_foo._cast(bar, ctx)

    self.assertIsInstance(value, TestAttrsClass)
    self.assertEqual(value.a.dtype, dtypes.float32)
    self.assertEqual(value.b.dtype, dtypes.float32)

  def testDict(self):
    foo = {'x': constant_op.constant(1.0, dtypes.float32),
           'y': constant_op.constant(2.0, dtypes.float32)}
    trace_foo = trace_type.from_value(foo)
    bar = {'x': 1, 'y': 2}
    ctx = trace_type.InternalCastContext()
    value = trace_foo._cast(bar, ctx)

    self.assertIsInstance(value, dict)
    self.assertSequenceEqual(
        set(value.keys()),
        set(bar.keys()))
    self.assertIn('x', value)
    self.assertIn('y', value)
    self.assertEqual(value['x'].dtype, dtypes.float32)
    self.assertEqual(value['y'].dtype, dtypes.float32)


class SignatureToTraceTypeTest(test.TestCase):

  def testTensorSpecs(self):
    self.assertEqual(
        trace_type.from_value(
            tensor_spec.TensorSpec(shape=None),
            trace_type.InternalTracingContext(is_legacy_signature=True)),
        tensor_spec.TensorSpec(shape=None))

  def testListofTensorSpecs(self):
    self.assertEqual(
        trace_type.from_value([
            tensor_spec.TensorSpec(shape=None),
            tensor_spec.TensorSpec(shape=None)
        ], trace_type.InternalTracingContext(is_legacy_signature=True)),
        default_types.List(
            tensor_spec.TensorSpec(shape=None),
            tensor_spec.TensorSpec(shape=None)))

  def testDictofTensorSpecs(self):
    self.assertEqual(
        trace_type.from_value(
            {
                'a': tensor_spec.TensorSpec(shape=None),
                'b': tensor_spec.TensorSpec(shape=None)
            }, trace_type.InternalTracingContext(is_legacy_signature=True)),
        default_types.Dict({
            'a': tensor_spec.TensorSpec(shape=None),
            'b': tensor_spec.TensorSpec(shape=None)
        }))


class TraceTypeMemoryTest(test.TestCase):

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testGeneric(self):
    trace_type.from_value(1)
    trace_type.from_value(DummyGenericClass())

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTensor(self):
    tensor = array_ops.zeros([10])
    trace_type.from_value(tensor)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTuple(self):
    trace_type.from_value((1, 2, 3))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testDict(self):
    trace_type.from_value({1: 1, 2: 2, 3: 3})

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testList(self):
    trace_type.from_value([1, 2, 3])

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testAttrs(self):
    trace_type.from_value(TestAttrsClass(1, 2))


class TraceTypeGenerationBenchmark(test.Benchmark):

  def benchmarkTensor(self):
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensors = []
    for s in shapes:
      tensors.append(array_ops.zeros(s))

    def encode_tensors(tensors):
      trace_type.from_value(tensors)

    iterations = 100000
    t = timeit.timeit(lambda: encode_tensors(tensors), number=iterations)
    self.report_benchmark(
        name='tensor_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkTensorSpec(self):
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensor_specs = []
    for s in shapes:
      tensor_specs.append(tensor_spec.TensorSpec(s, dtypes.int32))

    def encode_tensor_specs(tensor_specs):
      trace_type.from_value(tensor_specs)

    iterations = 100000
    t = timeit.timeit(
        lambda: encode_tensor_specs(tensor_specs), number=iterations)
    self.report_benchmark(
        name='tensor_spec_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_spec_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkVariable(self):
    var_list = [
        variables.Variable(1.0),
        variables.Variable(1),
        variables.Variable([1])
    ]

    def encode_variables(var_list):
      trace_type.from_value(var_list)

    iterations = 10000
    t = timeit.timeit(lambda: encode_variables(var_list), number=iterations)
    self.report_benchmark(
        name='variable_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'variable_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkTraceTypeLookup(self):

    @def_function.function
    def defined(t):
      return t

    call_arg_list = [
        1,
        array_ops.zeros([5, 13]),
        array_ops.zeros([9, 22, 24]),
        array_ops.zeros([5, 13, 2])
    ]

    for c in call_arg_list:
      defined(c)

    lookup_call_arg = array_ops.zeros([5, 13])

    iterations = 10000
    t = timeit.timeit(stmt=lambda: defined(lookup_call_arg), number=iterations)

    self.report_benchmark(
        name='cache_key_lookup',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'cache_key_lookup_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkNestedStruct(self):
    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}

    def encode_struct(struct):
      trace_type.from_value(struct)

    iterations = 100000
    t = timeit.timeit(lambda: encode_struct(struct), number=iterations)
    self.report_benchmark(
        name='nested_struct_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'nested_struct_cache_key_generation_avg_ms',
            'value': t / iterations * 1000
        }])

  def benchmarkFunctionInvocation(self):
    struct = (variables.Variable(1.0), array_ops.zeros([5, 13]), {
        'tensor': array_ops.zeros([5, 20]),
        'variable': variables.Variable(1.0)
    })

    @def_function.function
    def defined(t):
      return t

    defined(struct)  # Get it traced and cached.

    iterations = 10000
    t = timeit.timeit(lambda: defined(struct), number=iterations)
    self.report_benchmark(
        name='function_invocation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'function_invocation_time_avg_ms',
            'value': t / iterations * 1000
        }])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
