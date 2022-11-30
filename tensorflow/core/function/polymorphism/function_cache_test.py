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
from tensorflow.core.function.polymorphism import function_type
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
    if not isinstance(other, MockGenericType):
      return False

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

  def is_subtype_of(self, other: "MockShape") -> bool:
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


def make_single_param_type(type_constraint):
  return function_type.FunctionType([
      function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                              False, type_constraint)
  ])


def make_type_and_deleter(value):
  typing_context = trace_type.InternalTracingContext()
  value_type = trace_type.from_value(value, typing_context)
  f_type = make_single_param_type(value_type)
  return f_type, typing_context.deletion_observer


def make_none_context():
  return function_cache.FunctionContext(None)


class FunctionCacheTest(test.TestCase):

  def testConcreteFunctionDictRetainsInsertedKeys(self):
    cache = function_cache.FunctionCache()

    f_type_1, deletion_observer_1 = make_type_and_deleter(1)
    self.assertIsNone(cache.lookup(make_none_context(), f_type_1))

    f_type_2, deletion_observer_2 = make_type_and_deleter(2)
    f_type_3, _ = make_type_and_deleter(3)

    cache.add(make_none_context(), f_type_1, deletion_observer_1, "test_1")
    cache.add(make_none_context(), f_type_2, deletion_observer_2, "test_2")

    self.assertEqual(
        cache.lookup(make_none_context(), f_type_1), "test_1")
    self.assertEqual(
        cache.lookup(make_none_context(), f_type_2), "test_2")
    self.assertIsNone(cache.lookup(make_none_context(), f_type_3))

  def testClearRemovesAllConcreteFunctions(self):
    cache = function_cache.FunctionCache()

    f_type_1, deletion_observer_1 = make_type_and_deleter(1)
    f_type_2, deletion_observer_2 = make_type_and_deleter(2)
    f_type_3, _ = make_type_and_deleter(3)

    cache.add(make_none_context(), f_type_1, deletion_observer_1, "test_1")
    cache.add(make_none_context(), f_type_2, deletion_observer_2, "test_2")

    self.assertEqual(
        cache.lookup(make_none_context(), f_type_1), "test_1")
    self.assertEqual(
        cache.lookup(make_none_context(), f_type_2), "test_2")
    self.assertIsNone(cache.lookup(make_none_context(), f_type_3))

    cache.clear()

    self.assertIsNone(cache.lookup(make_none_context(), f_type_1))
    self.assertIsNone(cache.lookup(make_none_context(), f_type_2))
    self.assertIsNone(cache.lookup(make_none_context(), f_type_3))

  def testDeleteRemovesConcreteFunctions(self):
    cache = function_cache.FunctionCache()
    f_type_1, deletion_observer_1 = make_type_and_deleter(1)
    cache.add(make_none_context(), f_type_1, deletion_observer_1, "test_1")
    self.assertEqual(
        cache.lookup(make_none_context(), f_type_1), "test_1")
    cache.delete(make_none_context(), f_type_1)
    self.assertIsNone(cache.lookup(make_none_context(), f_type_1))

    f_type_2 = make_single_param_type(MockSubtypeOf2(2))
    cache.add(make_none_context(), f_type_2,
              trace_type.WeakrefDeletionObserver(), "test_2")
    self.assertEqual(
        cache.lookup(make_none_context(), f_type_2), "test_2")

    f_type_3 = make_single_param_type(MockSubtypeOf2(3))
    self.assertEqual(
        cache.lookup(make_none_context(), f_type_3), "test_2")

    cache.delete(make_none_context(), f_type_2)
    self.assertIsNone(cache.lookup(make_none_context(), f_type_2))
    self.assertIsNone(cache.lookup(make_none_context(), f_type_3))

  def testMostSpecificFunctionCacheKeyIsLookedUp(self):
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(ctx, make_single_param_type(MockShape(1, 2, None)),
              trace_type.WeakrefDeletionObserver(), "a")
    cache.add(ctx, make_single_param_type(MockShape(1, 2, 3)),
              trace_type.WeakrefDeletionObserver(), "b")

    self.assertEqual(
        cache.lookup(ctx, make_single_param_type(MockShape(1, 2, 3))),
        "b")

  def testFirstMostSpecificFunctionCacheKeyIsLookedUp(self):
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(ctx, make_single_param_type(MockShape(1, 2, None)),
              trace_type.WeakrefDeletionObserver(), "a")
    cache.add(ctx, make_single_param_type(MockShape(1, None, 3)),
              trace_type.WeakrefDeletionObserver(), "b")

    self.assertEqual(
        cache.lookup(ctx, make_single_param_type(MockShape(1, 2, 3))),
        "a")

  def testMostSpecificFunctionCacheKeyIsOrderAgnostic(self):
    ctx = function_cache.FunctionContext(0)
    keys = [(ctx, make_single_param_type(MockShape(1, 1, 1)), "a"),
            (ctx, make_single_param_type(MockShape(1, None, 1)), "b"),
            (ctx, make_single_param_type(MockShape(None, None, 1)), "c"),
            (ctx, make_single_param_type(MockShape(None, None, None)), "d")]

    for permutation in itertools.permutations(keys):
      cache = function_cache.FunctionCache()
      cache.add(permutation[0][0], permutation[0][1],
                trace_type.WeakrefDeletionObserver(), permutation[0][2])
      cache.add(permutation[1][0], permutation[1][1],
                trace_type.WeakrefDeletionObserver(), permutation[1][2])
      cache.add(permutation[2][0], permutation[2][1],
                trace_type.WeakrefDeletionObserver(), permutation[2][2])
      cache.add(permutation[3][0], permutation[3][1],
                trace_type.WeakrefDeletionObserver(), permutation[3][2])

      self.assertEqual(
          cache.lookup(ctx, make_single_param_type(MockShape(1, 1, 1))), "a")
      self.assertEqual(
          cache.lookup(ctx, make_single_param_type(MockShape(1, 2, 1))), "b")
      self.assertEqual(
          cache.lookup(ctx, make_single_param_type(MockShape(2, 2, 1))), "c")
      self.assertEqual(
          cache.lookup(ctx, make_single_param_type(MockShape(2, 2, 2))), "d")

  def testWeakRefDeletionAlsoDeletesConcreteFunction(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    dummy_object = DummyClass()
    key, deletion_observer = make_type_and_deleter(dummy_object)

    cache = function_cache.FunctionCache()
    cache.add(make_none_context(), key, deletion_observer, "testing")
    self.assertEqual(cache.lookup(make_none_context(), key), "testing")

    del dummy_object
    self.assertIsNone(cache.lookup(make_none_context(), key))

  def testMultipleObjectsWeakRefDeletion(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    dummy_object_1 = DummyClass()
    dummy_object_2 = DummyClass()
    key, deletion_observer = make_type_and_deleter(
        (dummy_object_1, dummy_object_2))

    cache = function_cache.FunctionCache()
    cache.add(make_none_context(), key, deletion_observer, "testing")
    self.assertEqual(cache.lookup(make_none_context(), key), "testing")

    del dummy_object_1
    self.assertIsNone(cache.lookup(make_none_context(), key))

    del dummy_object_2
    self.assertIsNone(cache.lookup(make_none_context(), key))

  def testObjectDeletedDuringFunctionCallDoesntAddConcreteFunction(self):
    if not function_cache.DELETE_WITH_WEAKREF:
      self.skipTest("Weakref-Based Deletion is disabled")

    def second(o):
      return make_type_and_deleter(o)

    def first():
      return second(DummyClass())

    key, deletion_observer = first()
    cache = function_cache.FunctionCache()
    cache.add(make_none_context(), key, deletion_observer, "testing")
    self.assertIsNone(cache.lookup(make_none_context(), key))


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
      keys.append(make_type_and_deleter(args))

    for key in keys[:-1]:
      cache.add(make_none_context(), *key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(make_none_context(), keys[-1][0]),
        number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(make_none_context(), keys[-1][0]),
        number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_f_type_miss",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_miss_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_f_type_miss_equality_avg_ms",
            "value": equality_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_f_type_miss_subtype_over_equality_ratio",
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
      keys.append(make_type_and_deleter(args))

    for key in keys:
      cache.add(make_none_context(), *key, "testing")

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(make_none_context(), keys[-1][0]),
        number=iterations)
    equality_time = timeit.timeit(
        lambda: cache.lookup(make_none_context(), keys[-1][0]),
        number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_f_type_equal",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_equal_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_f_type_equal_equality_avg_ms",
            "value": equality_time / iterations * 1000
        }, {
            "name": "cache_hit_50th_f_type_subtype_over_equality_ratio",
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
      keys.append(make_type_and_deleter(args))

    for key in keys:
      cache.add(make_none_context(), *key, "testing")
    cache.add(make_none_context(),
              make_single_param_type(MockSubtypeOf2(2)),
              trace_type.WeakrefDeletionObserver(), "testing")
    cache.lookup(make_none_context(),
                 make_single_param_type(MockSubtypeOf2(3)))

    iterations = 10000
    lookup_key = make_single_param_type(MockSubtypeOf2(2))
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(make_none_context(), lookup_key),
        number=iterations)

    self.report_benchmark(
        name="cache_hit_50th_f_type_known_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_known_subtype_avg_ms",
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
      keys.append(make_type_and_deleter(args))

    def setup():
      cache.clear()
      for key in keys:
        cache.add(make_none_context(), *key, "testing")
      cache.add(make_none_context(),
                make_single_param_type(MockSubtypeOf2(3)),
                trace_type.WeakrefDeletionObserver(), "testing")

    iterations = 10000
    lookup_key = make_single_param_type(MockSubtypeOf2(2))
    subtyping_time = sum(
        timeit.repeat(
            stmt=lambda: cache.lookup(make_none_context(), lookup_key),
            setup=setup,
            repeat=iterations,
            number=1))

    self.report_benchmark(
        name="cache_hit_50th_f_type_unknown_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_unknown_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000
        }])


if __name__ == "__main__":
  test.main()
