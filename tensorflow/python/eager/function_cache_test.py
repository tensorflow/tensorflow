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

import timeit

from tensorflow.python.eager import function_cache
from tensorflow.python.eager import function_trace_type
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class DummyClass:
  """Helps test Weakref deletion."""
  pass


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

    key_1, deletion_observer_1 = function_cache.make_cache_key(1)
    self.assertIsNone(cache.lookup(key_1, False))

    key_2, deletion_observer_2 = function_cache.make_cache_key(2)
    key_3, _ = function_cache.make_cache_key(3)

    cache.add(key_1, deletion_observer_1, "test_1")
    cache.add(key_2, deletion_observer_2, "test_2")

    self.assertEqual(cache.lookup(key_1, False), "test_1")
    self.assertEqual(cache.lookup(key_2, False), "test_2")
    self.assertIsNone(cache.lookup(key_3, False))

  def testClearRemovesAllConcreteFunctions(self):
    cache = function_cache.FunctionCache()

    key_1, deletion_observer_1 = function_cache.make_cache_key(1)
    key_2, deletion_observer_2 = function_cache.make_cache_key(2)
    key_3, _ = function_cache.make_cache_key(3)

    cache.add(key_1, deletion_observer_1, "test_1")
    cache.add(key_2, deletion_observer_2, "test_2")

    self.assertEqual(cache.lookup(key_1, False), "test_1")
    self.assertEqual(cache.lookup(key_2, False), "test_2")
    self.assertIsNone(cache.lookup(key_3, False))

    cache.clear()

    self.assertIsNone(cache.lookup(key_1, False))
    self.assertIsNone(cache.lookup(key_2, False))
    self.assertIsNone(cache.lookup(key_3, False))

  def testDeleteRemovesConcreteFunctions(self):
    cache = function_cache.FunctionCache()
    key_1, deletion_observer_1 = function_cache.make_cache_key(1)
    cache.add(key_1, deletion_observer_1, "test_1")
    self.assertEqual(cache.lookup(key_1, False), "test_1")
    cache.delete(key_1)
    self.assertIsNone(cache.lookup(key_1, False))

    key_2 = function_cache.FunctionCacheKey(MockSubtypeOf2(2), None)
    cache.add(key_2, function_trace_type.WeakrefDeletionObserver(),
              "test_2")
    self.assertEqual(cache.lookup(key_2, False), "test_2")

    key_3 = function_cache.FunctionCacheKey(MockSubtypeOf2(3), None)
    self.assertEqual(cache.lookup(key_3, True), "test_2")

    cache.delete(key_2)
    self.assertIsNone(cache.lookup(key_2, False))
    self.assertIsNone(cache.lookup(key_3, True))

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
    self.assertIsNone(key_a.most_specific_common_subtype([key_b]))

  def testWeakRefDeletionAlsoDeletesConcreteFunction(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    dummy_object = DummyClass()
    key, deletion_observer = function_cache.make_cache_key(dummy_object)

    cache = function_cache.FunctionCache()
    cache.add(key, deletion_observer, "testing")
    self.assertEqual(cache.lookup(key, False), "testing")

    del dummy_object
    self.assertIsNone(cache.lookup(key, False))

  def testMultipleObjectsWeakRefDeletion(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    dummy_object_1 = DummyClass()
    dummy_object_2 = DummyClass()
    key, deletion_observer = function_cache.make_cache_key(
        (dummy_object_1, dummy_object_2))

    cache = function_cache.FunctionCache()
    cache.add(key, deletion_observer, "testing")
    self.assertEqual(cache.lookup(key, False), "testing")

    del dummy_object_1
    self.assertIsNone(cache.lookup(key, False))

    del dummy_object_2
    self.assertIsNone(cache.lookup(key, False))

  def testObjectDeletedDuringFunctionCallDoesntAddConcreteFunction(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    def second(o):
      return function_cache.make_cache_key(o)

    def first():
      return second(DummyClass())

    key, deletion_observer = first()
    cache = function_cache.FunctionCache()
    cache.add(key, deletion_observer, "testing")
    self.assertIsNone(cache.lookup(key, False))


class FunctionCacheBenchmark(test.Benchmark):

  def benchmarkCacheHit50thKeyMiss(self):
    # If there are 50 keys and we get a new key that the cache has no concrete
    # functions for.

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(function_cache.make_cache_key(args))

    for key in keys[:-1]:
      cache.add(*key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1], True), number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1], False), number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_key_miss",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[{
            "name": "cache_hit_50th_key_miss_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_key_miss_equality_avg_ms",
            "value": equality_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_key_miss_subtype_over_equality_ratio",
            "value": subtyping_time / equality_time
        }])

  def benchmarkCacheHit50thKeyEqual(self):
    # If there are 50 keys and we get a new key that is equal to a key that is
    # in the cache.

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(function_cache.make_cache_key(args))

    for key in keys:
      cache.add(*key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1], True), number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1], False), number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_key_equal",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[{
            "name": "cache_hit_50th_key_equal_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_key_equal_equality_avg_ms",
            "value": equality_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_key_subtype_over_equality_ratio",
            "value": subtyping_time / equality_time
        }])

  def benchmarkCacheHit50thKeyKnownSubtype(self):
    # If there are 50 keys and we get a key that has a subtype in cache and
    # the cache has observed the key before (to memorize the subtype).

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks - 1):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(function_cache.make_cache_key(args))

    for key in keys:
      cache.add(*key, "testing")
    cache.add(
        function_cache.FunctionCacheKey(MockSubtypeOf2(2), None),
        function_trace_type.WeakrefDeletionObserver(), "testing")
    cache.lookup(function_cache.FunctionCacheKey(MockSubtypeOf2(3), None), True)

    iterations = 10000
    lookup_key = function_cache.FunctionCacheKey(MockSubtypeOf2(2), None)
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(lookup_key, True), number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_key_known_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_key_known_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }])

  def benchmarkCacheHit50thKeyUnknownSubtype(self):
    # If there are 50 keys and we get a key that has a subtype in cache but
    # the cache has never observed the key before (no memory for the subtype).

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks - 1):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(function_cache.make_cache_key(args))

    def setup():
      cache.clear()
      for key in keys:
        cache.add(*key, "testing")
      cache.add(
          function_cache.FunctionCacheKey(MockSubtypeOf2(3), None),
          function_trace_type.WeakrefDeletionObserver(), "testing")

    iterations = 10000
    lookup_key = function_cache.FunctionCacheKey(MockSubtypeOf2(2), None)
    subtyping_time = sum(
        timeit.repeat(
            stmt=lambda: cache.lookup(lookup_key, True),
            setup=setup,
            repeat=iterations,
            number=1))

    self.report_benchmark(
        name="cache_hit_50th_key_unknown_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_key_unknown_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }])


if __name__ == "__main__":
  test.main()
