# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import multiprocessing.dummy
import pickle
import time
import timeit

import numpy as np

from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import tracking


_PICKLEABLE_CALL_COUNT = collections.Counter()


class MyPickleableObject(tracking.AutoTrackable):
  """Needed for InterfaceTests.test_property_cache_serialization.

  This class must be at the top level. This is a constraint of pickle,
  unrelated to `cached_per_instance`.
  """

  @property
  @layer_utils.cached_per_instance
  def my_id(self):
    _PICKLEABLE_CALL_COUNT[self] += 1
    return id(self)


class LayerUtilsTest(test.TestCase):

  def test_property_cache(self):
    test_counter = collections.Counter()

    class MyObject(tracking.AutoTrackable):

      def __init__(self):
        super(MyObject, self).__init__()
        self._frozen = True

      def __setattr__(self, key, value):
        """Enforce that cache does not set attribute on MyObject."""
        if getattr(self, "_frozen", False):
          raise ValueError("Cannot mutate when frozen.")
        return super(MyObject, self).__setattr__(key, value)

      @property
      @layer_utils.cached_per_instance
      def test_property(self):
        test_counter[id(self)] += 1
        return id(self)

    first_object = MyObject()
    second_object = MyObject()

    # Make sure the objects return the correct values
    self.assertEqual(first_object.test_property, id(first_object))
    self.assertEqual(second_object.test_property, id(second_object))

    # Make sure the cache does not share across objects
    self.assertNotEqual(first_object.test_property, second_object.test_property)

    # Check again (Now the values should be cached.)
    self.assertEqual(first_object.test_property, id(first_object))
    self.assertEqual(second_object.test_property, id(second_object))

    # Count the function calls to make sure the cache is actually being used.
    self.assertAllEqual(tuple(test_counter.values()), (1, 1))

  def test_property_cache_threaded(self):
    call_count = collections.Counter()

    class MyObject(tracking.AutoTrackable):

      @property
      @layer_utils.cached_per_instance
      def test_property(self):
        # Random sleeps to ensure that the execution thread changes
        # mid-computation.
        call_count["test_property"] += 1
        time.sleep(np.random.random() + 1.)

        # Use a RandomState which is seeded off the instance's id (the mod is
        # because numpy limits the range of seeds) to ensure that an instance
        # returns the same value in different threads, but different instances
        # return different values.
        return int(np.random.RandomState(id(self) % (2 ** 31)).randint(2 ** 16))

      def get_test_property(self, _):
        """Function provided to .map for threading test."""
        return self.test_property

    # Test that multiple threads return the same value. This requires that
    # the underlying function is repeatable, as cached_property makes no attempt
    # to prioritize the first call.
    test_obj = MyObject()
    with contextlib.closing(multiprocessing.dummy.Pool(32)) as pool:
      # Intentionally make a large pool (even when there are only a small number
      # of cpus) to ensure that the runtime switches threads.
      results = pool.map(test_obj.get_test_property, range(64))
    self.assertEqual(len(set(results)), 1)

    # Make sure we actually are testing threaded behavior.
    self.assertGreater(call_count["test_property"], 1)

    # Make sure new threads still cache hit.
    with contextlib.closing(multiprocessing.dummy.Pool(2)) as pool:
      start_time = timeit.default_timer()  # Don't time pool instantiation.
      results = pool.map(test_obj.get_test_property, range(4))
    total_time = timeit.default_timer() - start_time

    # Note(taylorrobie): The reason that it is safe to time a unit test is that
    #                    a cache hit will be << 1 second, and a cache miss is
    #                    guaranteed to be >= 1 second. Empirically confirmed by
    #                    100,000 runs with no flakes.
    self.assertLess(total_time, 0.95)

  def test_property_cache_serialization(self):
    # Reset call count. .keys() must be wrapped in a list, because otherwise we
    # would mutate the iterator while iterating.
    for k in list(_PICKLEABLE_CALL_COUNT.keys()):
      _PICKLEABLE_CALL_COUNT.pop(k)

    first_instance = MyPickleableObject()
    self.assertEqual(id(first_instance), first_instance.my_id)

    # Test that we can pickle and un-pickle
    second_instance = pickle.loads(pickle.dumps(first_instance))

    self.assertEqual(id(second_instance), second_instance.my_id)
    self.assertNotEqual(first_instance.my_id, second_instance.my_id)

    # Make sure de-serialized object uses the cache.
    self.assertEqual(_PICKLEABLE_CALL_COUNT[second_instance], 1)

    # Make sure the decorator cache is not being serialized with the object.
    expected_size = len(pickle.dumps(second_instance))
    for _ in range(5):
      # Add some more entries to the cache.
      _ = MyPickleableObject().my_id
    self.assertEqual(len(_PICKLEABLE_CALL_COUNT), 7)
    size_check_instance = MyPickleableObject()
    _ = size_check_instance.my_id
    self.assertEqual(expected_size, len(pickle.dumps(size_check_instance)))


if __name__ == "__main__":
  test.main()
