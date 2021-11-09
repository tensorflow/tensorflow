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
"""Tests for function_cache."""

from tensorflow.python.eager import function_cache
from tensorflow.python.eager import function_trace_type
from tensorflow.python.platform import test


class MockSubtypeOf2(function_trace_type.GenericType):

  def is_subtype_of(self, other):
    return other._object == 2


class MockSupertypes2With3(function_trace_type.GenericType):

  def most_specific_common_supertype(self, others):
    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None


class FunctionCacheTest(test.TestCase):

  def testConcreteFunctionDictRetainsInsertedKeys(self):
    cache = function_cache.FunctionCache()

    key_1 = function_cache.make_cache_key(1)
    self.assertIsNone(cache.lookup(key_1))

    key_2 = function_cache.make_cache_key(2)
    key_3 = function_cache.make_cache_key(3)

    cache.add(key_1, "test_1")
    cache.add(key_2, "test_2")

    self.assertEqual(cache.lookup(key_1), "test_1")
    self.assertEqual(cache.lookup(key_2), "test_2")
    self.assertIsNone(cache.lookup(key_3))

  def testExecutionContextSetRetainsInsertedElements(self):
    cache = function_cache.FunctionCache()

    ctx_1 = function_cache.ExecutionContext(1, 1, 1, 1, 1, 1)
    self.assertFalse(cache.has_call_context(ctx_1))
    cache.add_call_context(ctx_1)
    self.assertTrue(cache.has_call_context(ctx_1))

    ctx_2 = function_cache.ExecutionContext(1, 1, 1, 1, 1, 1)
    self.assertTrue(cache.has_call_context(ctx_2))

    ctx_3 = function_cache.ExecutionContext(1, 1, 1, 1, 1, None)
    self.assertFalse(cache.has_call_context(ctx_3))
    cache.add_call_context(ctx_3)
    self.assertTrue(cache.has_call_context(ctx_3))

  def testFunctionCacheKeyRespectsEquality(self):
    ctx = function_cache.ExecutionContext(1, 1, 1, 1, 1, 1)
    generic = function_trace_type.GenericType
    key_a = function_cache.FunctionCacheKey(generic(1), ctx)
    key_b = function_cache.FunctionCacheKey(generic(2), ctx)
    key_c = function_cache.FunctionCacheKey(generic(1), ctx)

    self.assertNotEqual(key_a, key_b)
    self.assertEqual(key_a, key_c)
    self.assertEqual(hash(key_a), hash(key_c))

  def testFunctionCacheKeyRespectsSubtype(self):
    ctx = function_cache.ExecutionContext(1, 1, 1, 1, 1, 1)
    key_a = function_cache.FunctionCacheKey(MockSubtypeOf2(1), ctx)
    key_b = function_cache.FunctionCacheKey(MockSubtypeOf2(2), ctx)
    key_c = function_cache.FunctionCacheKey(MockSubtypeOf2(1), ctx)

    self.assertTrue(key_b.is_subtype_of(key_a))
    self.assertFalse(key_a.is_subtype_of(key_b))
    self.assertFalse(key_c.is_subtype_of(key_a))

  def testFunctionCacheKeyRespectsSupertype(self):
    ctx = function_cache.ExecutionContext(1, 1, 1, 1, 1, 1)
    key_a = function_cache.FunctionCacheKey(MockSupertypes2With3(1), ctx)
    key_b = function_cache.FunctionCacheKey(MockSupertypes2With3(2), ctx)

    self.assertEqual(
        key_b.most_specific_common_subtype([key_a]),
        function_cache.FunctionCacheKey(MockSupertypes2With3(3), ctx))
    self.assertIsNone(
        key_a.most_specific_common_subtype([key_b]))

if __name__ == "__main__":
  test.main()
