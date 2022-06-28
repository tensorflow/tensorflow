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

import itertools
import timeit
from typing import Optional

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import function_context
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class DummyClass:
  """Helps test Weakref deletion."""
  pass


class MockGenericType(trace.TraceType):

  def __init__(self, obj):
    self._object = obj

  def is_subtype_of(self, other):
    return self == other

  def most_specific_common_supertype(self, others):
    return None

  def __eq__(self, other):
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, MockGenericType) and self._object == other._object

  def __hash__(self):
    return hash(self._object)


class MockIntGenericType(MockGenericType):

  def most_specific_common_supertype(self, others):
    if all([self._object == other._object for other in others]):
      return MockIntGenericType(self._object)
    else:
      return None


class MockSubtypeOf2(MockGenericType):

  def is_subtype_of(self, other):
    return other._object == 2


class MockSupertypes2With3(MockGenericType):

  def most_specific_common_supertype(self, others):
    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None


class MockShape(trace.TraceType):

  def __init__(self, *shape: Optional[int]):
    self.shape = shape

  def is_subtype_of(self, other: "MockShape") ->bool:
    if len(self.shape) != len(other.shape):
      return False

    if any(o is not None and s != o for s, o in zip(self.shape, other.shape)):
      return False

    return True

  def most_specific_common_supertype(self, _):
    raise NotImplementedError

  def __str__(self):
    return str(self.shape)

  def __repr__(self):
    return str(self)

  def __hash__(self) -> int:
    return hash(self.shape)

  def __eq__(self, other: "MockShape") -> bool:
    return self.shape == other.shape


class MockEmptyCaptureSnapshot(function_cache.CaptureSnapshot):

  def __init__(self, _=None):
    self.mapping = {}


class FunctionCacheTest(test.TestCase):

  def testConcreteFunctionDictRetainsInsertedKeys(self):
    cache = function_cache.FunctionCache()

    key_1, deletion_observer_1 = function_context.make_cache_key(1, {})
    self.assertIsNone(cache.lookup(key_1, False))

    key_2, deletion_observer_2 = function_context.make_cache_key(2, {})
    key_3, _ = function_context.make_cache_key(3, {})

    cache.add(key_1, deletion_observer_1, "test_1")
    cache.add(key_2, deletion_observer_2, "test_2")

    self.assertEqual(cache.lookup(key_1, False), "test_1")
    self.assertEqual(cache.lookup(key_2, False), "test_2")
    self.assertIsNone(cache.lookup(key_3, False))

  def testClearRemovesAllConcreteFunctions(self):
    cache = function_cache.FunctionCache()

    key_1, deletion_observer_1 = function_context.make_cache_key(1, {})
    key_2, deletion_observer_2 = function_context.make_cache_key(2, {})
    key_3, _ = function_context.make_cache_key(3, {})

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
    key_1, deletion_observer_1 = function_context.make_cache_key(1)
    cache.add(key_1, deletion_observer_1, "test_1")
    self.assertEqual(cache.lookup(key_1, False), "test_1")
    cache.delete(key_1)
    self.assertIsNone(cache.lookup(key_1, False))

    key_2 = function_cache.FunctionCacheKey(MockSubtypeOf2(2),
                                            MockEmptyCaptureSnapshot(), None)
    cache.add(key_2, trace_type.WeakrefDeletionObserver(),
              "test_2")
    self.assertEqual(cache.lookup(key_2, False), "test_2")

    key_3 = function_cache.FunctionCacheKey(MockSubtypeOf2(3),
                                            MockEmptyCaptureSnapshot(), None)
    self.assertEqual(cache.lookup(key_3, True), "test_2")

    cache.delete(key_2)
    self.assertIsNone(cache.lookup(key_2, False))
    self.assertIsNone(cache.lookup(key_3, True))

  def testFunctionCacheKeyRespectsEquality(self):
    ctx = function_cache.FunctionContext(0)
    generic = MockGenericType
    key_a = function_cache.FunctionCacheKey(generic(1),
                                            MockEmptyCaptureSnapshot(), ctx)
    key_b = function_cache.FunctionCacheKey(generic(2),
                                            MockEmptyCaptureSnapshot(), ctx)
    key_c = function_cache.FunctionCacheKey(generic(1),
                                            MockEmptyCaptureSnapshot(), ctx)

    self.assertNotEqual(key_a, key_b)
    self.assertEqual(key_a, key_c)
    self.assertEqual(hash(key_a), hash(key_c))

  def testFunctionCacheKeyRespectsSubtype(self):
    ctx = function_cache.FunctionContext(0)
    key_a = function_cache.FunctionCacheKey(MockSubtypeOf2(1),
                                            MockEmptyCaptureSnapshot(), ctx)
    key_b = function_cache.FunctionCacheKey(MockSubtypeOf2(2),
                                            MockEmptyCaptureSnapshot(), ctx)
    key_c = function_cache.FunctionCacheKey(MockSubtypeOf2(1),
                                            MockEmptyCaptureSnapshot(), ctx)

    self.assertTrue(key_a.is_subtype_of(key_b))
    self.assertFalse(key_b.is_subtype_of(key_a))
    self.assertFalse(key_a.is_subtype_of(key_c))

  def testFunctionCacheKeyRespectsSupertype(self):
    ctx = function_cache.FunctionContext(0)
    key_a = function_cache.FunctionCacheKey(MockSupertypes2With3(1),
                                            MockEmptyCaptureSnapshot(), ctx)
    key_b = function_cache.FunctionCacheKey(MockSupertypes2With3(2),
                                            MockEmptyCaptureSnapshot(), ctx)

    self.assertEqual(
        key_b.most_specific_common_supertype([key_a]),
        function_cache.FunctionCacheKey(MockSupertypes2With3(3),
                                        MockEmptyCaptureSnapshot(), ctx))
    self.assertIsNone(key_a.most_specific_common_supertype([key_b]))

  def testMostSpecificFunctionCacheKeyIsLookedUp(self):
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(
        function_cache.FunctionCacheKey(MockShape(1, 2, None),
                                        MockEmptyCaptureSnapshot(), ctx),
        trace_type.WeakrefDeletionObserver(), "a")
    cache.add(
        function_cache.FunctionCacheKey(MockShape(1, 2, 3),
                                        MockEmptyCaptureSnapshot(), ctx),
        trace_type.WeakrefDeletionObserver(), "b")

    self.assertEqual(
        cache.lookup(
            function_cache.FunctionCacheKey(MockShape(1, 2, 3),
                                            MockEmptyCaptureSnapshot(),
                                            ctx), True),
        "b")

  def testFirstMostSpecificFunctionCacheKeyIsLookedUp(self):
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(
        function_cache.FunctionCacheKey(MockShape(1, 2, None),
                                        MockEmptyCaptureSnapshot(), ctx),
        trace_type.WeakrefDeletionObserver(), "a")
    cache.add(
        function_cache.FunctionCacheKey(MockShape(1, None, 3),
                                        MockEmptyCaptureSnapshot(), ctx),
        trace_type.WeakrefDeletionObserver(), "b")

    self.assertEqual(
        cache.lookup(
            function_cache.FunctionCacheKey(
                MockShape(1, 2, 3), MockEmptyCaptureSnapshot(), ctx), True),
        "a")

  def testMostSpecificFunctionCacheKeyIsOrderAgnostic(self):
    ctx = function_cache.FunctionContext(0)
    keys = [(function_cache.FunctionCacheKey(MockShape(1, 1, 1),
                                             MockEmptyCaptureSnapshot(),
                                             ctx), "a"),
            (function_cache.FunctionCacheKey(MockShape(1, None, 1),
                                             MockEmptyCaptureSnapshot(),
                                             ctx), "b"),
            (function_cache.FunctionCacheKey(MockShape(None, None, 1),
                                             MockEmptyCaptureSnapshot(),
                                             ctx), "c"),
            (function_cache.FunctionCacheKey(MockShape(None, None, None),
                                             MockEmptyCaptureSnapshot(),
                                             ctx), "d")]

    for permutation in itertools.permutations(keys):
      cache = function_cache.FunctionCache()
      cache.add(permutation[0][0], trace_type.WeakrefDeletionObserver(),
                permutation[0][1])
      cache.add(permutation[1][0], trace_type.WeakrefDeletionObserver(),
                permutation[1][1])
      cache.add(permutation[2][0], trace_type.WeakrefDeletionObserver(),
                permutation[2][1])
      cache.add(permutation[3][0], trace_type.WeakrefDeletionObserver(),
                permutation[3][1])

      self.assertEqual(
          cache.lookup(
              function_cache.FunctionCacheKey(MockShape(1, 1, 1),
                                              MockEmptyCaptureSnapshot(),
                                              ctx), True),
          "a")
      self.assertEqual(
          cache.lookup(
              function_cache.FunctionCacheKey(MockShape(1, 2, 1),
                                              MockEmptyCaptureSnapshot(),
                                              ctx), True),
          "b")
      self.assertEqual(
          cache.lookup(
              function_cache.FunctionCacheKey(MockShape(2, 2, 1),
                                              MockEmptyCaptureSnapshot(),
                                              ctx), True),
          "c")
      self.assertEqual(
          cache.lookup(
              function_cache.FunctionCacheKey(MockShape(2, 2, 2),
                                              MockEmptyCaptureSnapshot(),
                                              ctx), True),
          "d")

  def testWeakRefDeletionAlsoDeletesConcreteFunction(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    dummy_object = DummyClass()
    key, deletion_observer = function_context.make_cache_key(dummy_object)

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
    key, deletion_observer = function_context.make_cache_key(
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
      return function_context.make_cache_key(o)

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
      keys.append(function_context.make_cache_key(args))

    for key in keys[:-1]:
      cache.add(*key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1][0], True), number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1][0], False), number=iterations)

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
      keys.append(function_context.make_cache_key(args))

    for key in keys:
      cache.add(*key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1][0], True), number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1][0], False), number=iterations)

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
      keys.append(function_context.make_cache_key(args))

    for key in keys:
      cache.add(*key, "testing")
    cache.add(
        function_cache.FunctionCacheKey(MockSubtypeOf2(2),
                                        MockEmptyCaptureSnapshot(), None),
        trace_type.WeakrefDeletionObserver(), "testing")
    cache.lookup(function_cache.FunctionCacheKey(MockSubtypeOf2(3),
                                                 MockEmptyCaptureSnapshot(),
                                                 None), True)

    iterations = 10000
    lookup_key = function_cache.FunctionCacheKey(MockSubtypeOf2(2),
                                                 MockEmptyCaptureSnapshot(),
                                                 None)
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
      keys.append(function_context.make_cache_key(args))

    def setup():
      cache.clear()
      for key in keys:
        cache.add(*key, "testing")
      cache.add(
          function_cache.FunctionCacheKey(MockSubtypeOf2(3),
                                          MockEmptyCaptureSnapshot(), None),
          trace_type.WeakrefDeletionObserver(), "testing")

    iterations = 10000
    lookup_key = function_cache.FunctionCacheKey(MockSubtypeOf2(2),
                                                 MockEmptyCaptureSnapshot(),
                                                 None)
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


class CaptureSnapshotTest(test.TestCase):

  def setUp(self):
    super(CaptureSnapshotTest, self).setUp()
    snapshot_type = function_cache.CaptureSnapshot
    self.snapshot_a = snapshot_type({
        "a": MockIntGenericType(1),
        "b": MockIntGenericType(1)
    })
    self.snapshot_b = snapshot_type({
        "a": MockIntGenericType(1),
        "b": MockIntGenericType(1),
        "c": MockIntGenericType(1)
    })
    self.snapshot_c = snapshot_type({
        "a": MockIntGenericType(2),
        "b": MockIntGenericType(2),
        "c": MockIntGenericType(2)
    })
    self.snapshot_d = snapshot_type({
        "a": MockIntGenericType(1),
        "b": MockIntGenericType(1),
        "c": MockIntGenericType(2)
    })
    self.snapshot_e = snapshot_type({
        "d": MockIntGenericType(1)
    })

  def testCaptureSnapshotSubtype(self):
    self.assertFalse(self.snapshot_a.is_subtype_of(self.snapshot_b))
    self.assertTrue(self.snapshot_b.is_subtype_of(self.snapshot_a))
    self.assertFalse(self.snapshot_b.is_subtype_of(self.snapshot_c))
    self.assertFalse(self.snapshot_b.is_subtype_of(self.snapshot_c))
    self.assertFalse(self.snapshot_e.is_subtype_of(self.snapshot_a))

  def testCaptureSnapshotSupertype(self):
    supertype_1 = self.snapshot_b.most_specific_common_supertype(
        [self.snapshot_b])
    self.assertLen(supertype_1.mapping, 3)
    supertype_2 = self.snapshot_a.most_specific_common_supertype(
        [self.snapshot_b, self.snapshot_c])
    self.assertIsNone(supertype_2)
    supertype_3 = self.snapshot_a.most_specific_common_supertype(
        [self.snapshot_d])
    self.assertLen(supertype_3.mapping, 2)
    supertype_4 = self.snapshot_b.most_specific_common_supertype(
        [self.snapshot_d])
    self.assertIsNone(supertype_4)
    supertype_5 = self.snapshot_b.most_specific_common_supertype(
        [self.snapshot_e])
    self.assertEmpty(supertype_5.mapping)


if __name__ == "__main__":
  test.main()
