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
"""Tests for function_trace_type."""

import timeit

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import function
from tensorflow.python.eager import function_trace_type
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.types import trace

# TODO(b/201533914): Simulate an attrs class so that the import is not needed.
try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None

if attr is not None:
  @attr.s
  class TestAttrsClass(object):
    """Helps test memory leaks for attrs collection."""
    a = attr.ib()
    b = attr.ib()


class DummyGenericClass:
  """Helps test memory leaks for GenericType."""
  pass


class CacheKeyGenerationTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['eager']))
  def testIteratorAliasing(self):
    it1 = iter(dataset_ops.DatasetV2.from_tensor_slices([1, 2, 3]))
    it2 = iter(dataset_ops.DatasetV2.from_tensor_slices([1, 2, 3]))

    self.assertEqual(
        function_trace_type.get_arg_spec((it1, it1), False, False, True),
        function_trace_type.get_arg_spec((it2, it2), False, False, True))
    self.assertEqual(
        function_trace_type.get_arg_spec((it1, it2), False, False, True),
        function_trace_type.get_arg_spec((it2, it1), False, False, True))
    self.assertNotEqual(
        function_trace_type.get_arg_spec((it1, it1), False, False, True),
        function_trace_type.get_arg_spec((it1, it2), False, False, True))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testIteratorTypesImplementTracing(self):
    self.assertTrue(
        issubclass(iterator_ops.OwnedIterator, trace.SupportsTracingType))
    self.assertTrue(
        issubclass(iterator_ops.IteratorSpec, trace.SupportsTracingType))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testCompositeAndSpec(self):
    composite_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        values=[1, 2, 3], row_splits=[0, 2, 3])
    spec = ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32)

    self.assertEqual(
        function_trace_type.get_arg_spec(composite_tensor, False, False, True),
        function_trace_type.get_arg_spec(spec, False, False, True))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testVariableAliasing(self):
    v1 = resource_variable_ops.ResourceVariable([1])
    v2 = resource_variable_ops.ResourceVariable([1])
    v3 = resource_variable_ops.ResourceVariable([1])
    all_unique = function_trace_type.get_arg_spec((v1, v2, v3), False, True,
                                                  True)
    all_same = function_trace_type.get_arg_spec((v1, v1, v1), False, True, True)
    self.assertNotEqual(all_unique, all_same)

    v3 = resource_variable_ops.ResourceVariable([2])
    v4 = resource_variable_ops.ResourceVariable([2])
    v5 = resource_variable_ops.ResourceVariable([2])
    all_unique_again = function_trace_type.get_arg_spec((v3, v4, v5), False,
                                                        True, True)
    all_same_again = function_trace_type.get_arg_spec((v4, v4, v4), False, True,
                                                      True)
    self.assertEqual(all_unique, all_unique_again)
    self.assertEqual(all_same, all_same_again)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorEquality(self):
    context = function_trace_type.SignatureContext()
    tensor_a = array_ops.zeros([11, 3, 5],
                               dtype=dtypes.int32)._tf_tracing_type(context)
    tensor_b = array_ops.zeros([11, 4, 5],
                               dtype=dtypes.int32)._tf_tracing_type(context)
    tensor_c = array_ops.zeros([11, 3, 5],
                               dtype=dtypes.float32)._tf_tracing_type(context)
    tensor_d = array_ops.ones([11, 3, 5],
                              dtype=dtypes.int32)._tf_tracing_type(context)

    self.assertNotEqual(tensor_a, tensor_b)
    self.assertNotEqual(tensor_a, tensor_c)
    self.assertNotEqual(tensor_b, tensor_c)
    self.assertEqual(tensor_a, tensor_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorAndSpecEquality(self):
    context = function_trace_type.SignatureContext()
    tensor = array_ops.zeros([11, 3, 5],
                             dtype=dtypes.int32)._tf_tracing_type(context)
    spec = tensor_spec.TensorSpec([11, 3, 5],
                                  dtype=dtypes.int32)._tf_tracing_type(context)
    spec_with_name = tensor_spec.TensorSpec(
        [11, 3, 5], dtype=dtypes.int32, name='name')._tf_tracing_type(context)

    self.assertEqual(tensor, spec)
    self.assertNotEqual(tensor, spec_with_name)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTensorShapeUnknown(self):
    context = function_trace_type.SignatureContext()
    spec_1 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32)._tf_tracing_type(context)
    spec_2 = tensor_spec.TensorSpec(
        None, dtype=dtypes.int32)._tf_tracing_type(context)
    self.assertEqual(spec_1, spec_2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testTupleEquality(self):
    trace_a = function_trace_type.get_arg_spec((1, 2, 3, 4), False, False, True)
    trace_b = function_trace_type.get_arg_spec((1, 2, 2, 4), False, False, True)
    trace_c = function_trace_type.get_arg_spec((1, 2, 3), False, False, True)
    trace_d = function_trace_type.get_arg_spec((1, 2, 3, 4), False, False, True)

    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testListEquality(self):
    trace_a = function_trace_type.get_arg_spec([1, 2, 3, 4], False, False, True)
    trace_b = function_trace_type.get_arg_spec([1, 2, 2, 4], False, False, True)
    trace_c = function_trace_type.get_arg_spec([1, 2, 3], False, False, True)
    trace_d = function_trace_type.get_arg_spec([1, 2, 3, 4], False, False, True)

    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDictEquality(self):
    trace_a = function_trace_type.get_arg_spec({1: 2, 3: 4}, False, False, True)
    trace_b = function_trace_type.get_arg_spec({1: 2, 3: 2}, False, False, True)
    trace_c = function_trace_type.get_arg_spec({1: 2, 3: 0}, False, False, True)
    trace_d = function_trace_type.get_arg_spec({3: 4, 1: 2}, False, False, True)

    self.assertNotEqual(trace_a, trace_b)
    self.assertNotEqual(trace_a, trace_c)
    self.assertNotEqual(trace_b, trace_c)
    self.assertEqual(trace_a, trace_d)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testComplexStruct(self):
    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}
    trace_a = function_trace_type.get_arg_spec(struct, False, False, True)
    trace_b = function_trace_type.get_arg_spec(struct, False, False, True)
    self.assertEqual(trace_a, trace_b)
    self.assertTrue(trace_a.is_subtype_of(trace_b))
    self.assertTrue(trace_b.is_subtype_of(trace_a))


class CacheKeyMemoryTest(test.TestCase):

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testGeneric(self):
    function_trace_type.get_arg_spec(1, False, True, True)
    function_trace_type.get_arg_spec(DummyGenericClass(), False, True, True)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTensor(self):
    tensor = array_ops.zeros([10])
    function_trace_type.get_arg_spec(tensor, False, True, True)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testTuple(self):
    function_trace_type.get_arg_spec((1, 2, 3), False, True, True)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testDict(self):
    function_trace_type.get_arg_spec({1: 1, 2: 2, 3: 3}, False, True, True)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testList(self):
    function_trace_type.get_arg_spec([1, 2, 3], False, True, True)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testAttrs(self):
    if attr is None:
      self.skipTest('attr module is unavailable.')

    function_trace_type.get_arg_spec(TestAttrsClass(1, 2), False, True, True)


class CacheKeyGenerationBenchmark(test.Benchmark):

  def benchmarkTensor(self):
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensors = []
    for s in shapes:
      tensors.append(array_ops.zeros(s))

    def encode_tensors(tensors):
      function_trace_type.get_arg_spec(tensors, False, False,
                                       function.USE_FULL_TRACE_TYPE)

    iterations = 100000
    t = timeit.timeit(lambda: encode_tensors(tensors), number=iterations)
    self.report_benchmark(
        name='tensor_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_cache_key_generation_time',
            'value': t
        }])

  def benchmarkTensorSpec(self):
    shapes = [[1], [2, 19], [5, 11, 24], [4, 5, 9, 23]]
    tensor_specs = []
    for s in shapes:
      tensor_specs.append(tensor_spec.TensorSpec(s, dtypes.int32))

    def encode_tensor_specs(tensor_specs):
      function_trace_type.get_arg_spec(tensor_specs, False, False,
                                       function.USE_FULL_TRACE_TYPE)

    iterations = 100000
    t = timeit.timeit(
        lambda: encode_tensor_specs(tensor_specs), number=iterations)
    self.report_benchmark(
        name='tensor_spec_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'tensor_spec_cache_key_generation_time',
            'value': t
        }])

  def benchmarkVariable(self):
    var_list = [
        variables.Variable(1.0),
        variables.Variable(1),
        variables.Variable([1])
    ]

    def encode_variables(var_list):
      function_trace_type.get_arg_spec(var_list, False, False,
                                       function.USE_FULL_TRACE_TYPE)

    iterations = 1000000
    t = timeit.timeit(lambda: encode_variables(var_list), number=iterations)
    self.report_benchmark(
        name='variable_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'variable_cache_key_generation_time',
            'value': t
        }])

  def benchmarkKerasModel(self):
    inputs = keras.Input(shape=(3,))
    x = keras.layers.Dense(4, activation=nn_ops.relu)(inputs)
    outputs = keras.layers.Dense(5, activation=nn_ops.softmax)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    def encode_model(model):
      function_trace_type.get_arg_spec(model, False, False,
                                       function.USE_FULL_TRACE_TYPE)

    iterations = 100000
    t = timeit.timeit(lambda: encode_model(model), number=iterations)
    self.report_benchmark(
        name='keras_model_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'keras_model_cache_key_generation_time',
            'value': t
        }])

  def benchmarkCacheKeyLookup(self):

    @function.defun
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
            'name': 'cache_key_lookup_time',
            'value': t
        }])

  def benchmarkNestedStruct(self):

    struct = {(1, 2, 3): {(1, 2): {12: 2}}, (3, 2, 3): (2, {2: 3})}

    def encode_struct(struct):
      function_trace_type.get_arg_spec(struct, False, False,
                                       function.USE_FULL_TRACE_TYPE)

    iterations = 100000
    t = timeit.timeit(lambda: encode_struct(struct), number=iterations)
    self.report_benchmark(
        name='nested_truct_cache_key_generation',
        iters=iterations,
        wall_time=t,
        metrics=[{
            'name': 'nested_struct_cache_key_generation_time',
            'value': t
        }])


class TraceTypeEncodingTest(test.TestCase):

  def testKeyCollisionTypeFailsGracefully(self):
    class CustomSameTrace:

      def __tf_tracing_type__(self, _):
        return function_trace_type.GenericType(1)

      def __eq__(self, o):
        return self is o

      def __hash__(self):
        return 0

    dictionary = {CustomSameTrace(): 1, CustomSameTrace(): 2}

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Multiple keys produce the same TraceType for the mapping collection'):
      function_trace_type.get_arg_spec(dictionary, False, True, True)

  def testCustomUnhashableTypeFailsGracefully(self):

    class CustomUnhashable:

      def __eq__(self, o):
        return True

    obj = CustomUnhashable()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'could not be represented through the generic tracing type'):
      function_trace_type.get_arg_spec(obj, False, True, True)

  def testOrderedCollectionTypeEquality(self):
    collection = function_trace_type.OrderedCollectionType
    generic = function_trace_type.GenericType
    collection_a = collection(generic(1), generic(2), generic(3))
    collection_b = collection(generic(1), generic(2), generic(1))
    collection_c = collection(generic(1), generic(2), generic(3))

    self.assertNotEqual(collection_a, collection_b)
    self.assertEqual(collection_a, collection_c)
    self.assertEqual(hash(collection_a), hash(collection_c))

  def testOrderedCollectionTypeSubtype(self):

    class Subtypable(function_trace_type.GenericType):

      def is_subtype_of(self, other):
        return self._object == 2 or other._object == 3

    collection = function_trace_type.OrderedCollectionType
    collection_a = collection(Subtypable(1), Subtypable(2), Subtypable(3))
    collection_b = collection(Subtypable(2), Subtypable(1), Subtypable(2))
    collection_c = collection(Subtypable(1), Subtypable(3), Subtypable(3))

    self.assertTrue(collection_b.is_subtype_of(collection_c))
    self.assertFalse(collection_a.is_subtype_of(collection_b))
    self.assertFalse(collection_c.is_subtype_of(collection_a))

  def testOrderedCollectionTypeSupertype(self):

    class Supertypable(function_trace_type.GenericType):

      def most_specific_common_supertype(self, others):
        if self._object == 2 and isinstance(others[0]._object, int):
          return Supertypable(3)
        else:
          return None

    collection = function_trace_type.OrderedCollectionType
    collection_a = collection(Supertypable(1), Supertypable(2), Supertypable(3))
    collection_b = collection(Supertypable(2), Supertypable(2), Supertypable(2))

    self.assertIsNone(
        collection_a.most_specific_common_supertype([collection_b]))
    self.assertEqual(
        collection_b.most_specific_common_supertype([collection_a]),
        collection(Supertypable(3), Supertypable(3), Supertypable(3)))

  def testListTupleInequality(self):
    generic = function_trace_type.GenericType

    list_a = function_trace_type.ListType(generic(1), generic(2), generic(3))
    list_b = function_trace_type.ListType(generic(1), generic(2), generic(3))

    tuple_a = function_trace_type.TupleType(generic(1), generic(2), generic(3))
    tuple_b = function_trace_type.TupleType(generic(1), generic(2), generic(3))

    self.assertEqual(list_a, list_b)
    self.assertEqual(tuple_a, tuple_b)
    self.assertNotEqual(list_a, tuple_a)
    self.assertNotEqual(tuple_a, list_a)

  def testDictTypeEquality(self):
    dict_type = function_trace_type.DictType
    generic = function_trace_type.GenericType

    dict_a = dict_type({generic(1): generic(2), generic(3): generic(4)})
    dict_b = dict_type({generic(1): generic(2)})
    dict_c = dict_type({generic(3): generic(4), generic(1): generic(2)})

    self.assertEqual(dict_a, dict_c)
    self.assertNotEqual(dict_a, dict_b)


if __name__ == '__main__':
  test.main()
