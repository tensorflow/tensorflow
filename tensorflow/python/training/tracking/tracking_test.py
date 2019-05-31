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
import contextlib
import multiprocessing.dummy
import os
import pickle
import time
import timeit

import numpy as np
import six

from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest


_PICKLEABLE_CALL_COUNT = collections.Counter()


class MyPickleableObject(tracking.AutoTrackable):
  """Needed for InterfaceTests.test_property_cache_serialization.

  This class must be at the top level. This is a constraint of pickle,
  unrelated to `cached_per_instance`.
  """

  @property
  @tracking.cached_per_instance
  def my_id(self):
    _PICKLEABLE_CALL_COUNT[self] += 1
    return id(self)


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = tracking.AutoTrackable()
    root.leaf = tracking.AutoTrackable()
    root.leaf = root.leaf
    duplicate_name_dep = tracking.AutoTrackable()
    with self.assertRaisesRegexp(ValueError, "already declared"):
      root._track_trackable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    self.assertIs(duplicate_name_dep, root._lookup_dependency("leaf"))
    (_, dep_object), = root._checkpoint_dependencies
    self.assertIs(duplicate_name_dep, dep_object)

  def testNoDependency(self):
    root = tracking.AutoTrackable()
    hasdep = tracking.AutoTrackable()
    root.hasdep = hasdep
    nodep = tracking.AutoTrackable()
    root.nodep = data_structures.NoDependency(nodep)
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertIs(root._checkpoint_dependencies[0].ref, root.hasdep)
    self.assertIs(root.hasdep, hasdep)
    self.assertIs(root.nodep, nodep)

    class NoDependencyModel(training.Model):

      @base.no_automatic_dependency_tracking
      def __init__(self):
        super(NoDependencyModel, self).__init__()
        self.a = []
        self.b = tracking.AutoTrackable()

    nodeps = NoDependencyModel()
    self.assertEqual([nodeps], util.list_objects(nodeps))

  def testRemoveDependency(self):
    root = tracking.AutoTrackable()
    root.a = tracking.AutoTrackable()
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._checkpoint_dependencies[0].ref)
    del root.a
    self.assertFalse(hasattr(root, "a"))
    self.assertEqual(0, len(root._checkpoint_dependencies))
    self.assertEqual(0, len(root._unconditional_checkpoint_dependencies))
    root.a = tracking.AutoTrackable()
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertEqual(1, len(root._unconditional_checkpoint_dependencies))
    self.assertIs(root.a, root._checkpoint_dependencies[0].ref)

  def testListBasic(self):
    a = tracking.AutoTrackable()
    b = tracking.AutoTrackable()
    a.l = [b]
    c = tracking.AutoTrackable()
    a.l.append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    direct_a_dep, = a._checkpoint_dependencies
    self.assertEqual("l", direct_a_dep.name)
    self.assertIn(b, direct_a_dep.ref)
    self.assertIn(c, direct_a_dep.ref)

  @test_util.run_in_graph_and_eager_modes
  def testMutationDirtiesList(self):
    a = tracking.AutoTrackable()
    b = tracking.AutoTrackable()
    a.l = [b]
    c = tracking.AutoTrackable()
    a.l.insert(0, c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testOutOfBandEditDirtiesList(self):
    a = tracking.AutoTrackable()
    b = tracking.AutoTrackable()
    held_reference = [b]
    a.l = held_reference
    c = tracking.AutoTrackable()
    held_reference.append(c)
    checkpoint = util.Checkpoint(a=a)
    with self.assertRaisesRegexp(ValueError, "The wrapped list was modified"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testNestedLists(self):
    a = tracking.AutoTrackable()
    a.l = []
    b = tracking.AutoTrackable()
    a.l.append([b])
    c = tracking.AutoTrackable()
    a.l[0].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    a.l[0].append(1)
    d = tracking.AutoTrackable()
    a.l[0].append(d)
    a_deps = util.list_objects(a)
    self.assertIn(d, a_deps)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertNotIn(1, a_deps)
    e = tracking.AutoTrackable()
    f = tracking.AutoTrackable()
    a.l1 = [[], [e]]
    a.l1[0].append(f)
    a_deps = util.list_objects(a)
    self.assertIn(e, a_deps)
    self.assertIn(f, a_deps)
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l[0].append(data_structures.NoDependency([]))
    a.l[0][-1].append(5)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    # Dirtying the inner list means the root object is unsaveable.
    a.l[0][1] = 2
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testDictionariesBasic(self):
    a = training.Model()
    b = training.Model()
    a.attribute = {"b": b}
    c = training.Model()
    a.attribute["c"] = []
    a.attribute["c"].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertIs(b, a.attribute["b"])
    six.assertCountEqual(
        self,
        ["b", "c"],
        [dep.name for dep in a.attribute._checkpoint_dependencies])
    self.assertEqual([b, c], a.layers)
    self.assertEqual([b, c], a.attribute.layers)
    self.assertEqual([c], a.attribute["c"].layers)
    checkpoint = util.Checkpoint(a=a)
    save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    with self.cached_session():
      checkpoint.restore(save_path).assert_consumed().initialize_or_restore()

  @test_util.run_in_graph_and_eager_modes
  def testNoDepList(self):
    a = training.Model()
    a.l1 = data_structures.NoDependency([])
    a.l1.insert(1, 0)
    self.assertTrue(isinstance(a.l1, list))
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l2 = []
    a.l2.insert(1, 0)
    with self.assertRaisesRegexp(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  @test_util.run_in_graph_and_eager_modes
  def testAssertions(self):
    a = tracking.AutoTrackable()
    a.l = {"k": [np.zeros([2, 2])]}
    self.assertAllEqual(nest.flatten({"k": [np.zeros([2, 2])]}),
                        nest.flatten(a.l))
    self.assertAllClose({"k": [np.zeros([2, 2])]}, a.l)
    nest.map_structure(self.assertAllClose, a.l, {"k": [np.zeros([2, 2])]})
    a.tensors = {"k": [array_ops.ones([2, 2]), array_ops.zeros([3, 3])]}
    self.assertAllClose({"k": [np.ones([2, 2]), np.zeros([3, 3])]},
                        self.evaluate(a.tensors))

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
      @tracking.cached_per_instance
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
      @tracking.cached_per_instance
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
    #                    guaranteed to be >= 1 second. Emperically confirmed by
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


class _DummyResource(tracking.TrackableResource):

  def __init__(self, handle_name):
    self._handle_name = handle_name
    super(_DummyResource, self).__init__()

  def _create_resource(self):
    return self._handle_name


class ResourceTrackerTest(test.TestCase):

  def testBasic(self):
    resource_tracker = tracking.ResourceTracker()
    with tracking.resource_tracker_scope(resource_tracker):
      dummy_resource1 = _DummyResource("test1")
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)

  def testTwoScopes(self):
    resource_tracker1 = tracking.ResourceTracker()
    with tracking.resource_tracker_scope(resource_tracker1):
      dummy_resource1 = _DummyResource("test1")

    resource_tracker2 = tracking.ResourceTracker()
    with tracking.resource_tracker_scope(resource_tracker2):
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test1", resource_tracker1.resources[0].resource_handle)
    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)

  def testNestedScopesScopes(self):
    resource_tracker = tracking.ResourceTracker()
    with tracking.resource_tracker_scope(resource_tracker):
      resource_tracker1 = tracking.ResourceTracker()
      with tracking.resource_tracker_scope(resource_tracker1):
        dummy_resource1 = _DummyResource("test1")

      resource_tracker2 = tracking.ResourceTracker()
      with tracking.resource_tracker_scope(resource_tracker2):
        dummy_resource2 = _DummyResource("test2")

    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test1", resource_tracker1.resources[0].resource_handle)
    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)
    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)


if __name__ == "__main__":
  test.main()
