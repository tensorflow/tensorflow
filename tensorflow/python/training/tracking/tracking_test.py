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

import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest


def run_inside_wrap_function_in_eager_mode(graph_function):
  """Decorator to execute the same graph code in eager and graph modes.

  In graph mode, we just execute the graph_function passed as argument. In eager
  mode, we wrap the function using wrap_function and then execute the wrapped
  result.

  Args:
    graph_function: python function containing graph code to be wrapped

  Returns:
    decorated function
  """
  def wrap_and_execute(self):
    if context.executing_eagerly():
      wrapped = wrap_function.wrap_function(graph_function, [self])
      # use the wrapped graph function
      wrapped()
    else:
      # use the original function
      graph_function(self)
  return wrap_and_execute


class InterfaceTests(test.TestCase):

  def testMultipleAssignment(self):
    root = tracking.AutoTrackable()
    root.leaf = tracking.AutoTrackable()
    root.leaf = root.leaf
    duplicate_name_dep = tracking.AutoTrackable()
    with self.assertRaisesRegex(ValueError, "already declared"):
      root._track_trackable(duplicate_name_dep, name="leaf")
    # No error; we're overriding __setattr__, so we can't really stop people
    # from doing this while maintaining backward compatibility.
    root.leaf = duplicate_name_dep
    root._track_trackable(duplicate_name_dep, name="leaf", overwrite=True)
    self.assertIs(duplicate_name_dep, root._lookup_dependency("leaf"))
    (_, dep_object), = root._checkpoint_dependencies
    self.assertIs(duplicate_name_dep, dep_object)

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
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
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
    with self.assertRaisesRegex(ValueError, "The wrapped list was modified"):
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
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
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


class _DummyResource(tracking.TrackableResource):

  def __init__(self, handle_name):
    self._handle_name = handle_name
    super(_DummyResource, self).__init__()

  def _create_resource(self):
    return self._handle_name


class _DummyResource1(tracking.TrackableResource):

  def __init__(self, handle_name):
    self._handle_name = handle_name
    self._value = 0
    super(_DummyResource1, self).__init__()

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
    self.assertEqual(1, len(resource_tracker2.resources))
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
    self.assertEqual(1, len(resource_tracker2.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)
    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)


class ResourceCreatorScopeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreator(self):
    def resource_creator_fn(next_creator, *a, **kwargs):
      kwargs["handle_name"] = "forced_name"
      return next_creator(*a, **kwargs)

    # test that two resource classes use the same creator function
    with ops.resource_creator_scope(["_DummyResource", "_DummyResource1"],
                                    resource_creator_fn):
      dummy_0 = _DummyResource(handle_name="fake_name_0")
      dummy_1 = _DummyResource1(handle_name="fake_name_1")

    self.assertEqual(dummy_0._handle_name, "forced_name")
    self.assertEqual(dummy_1._handle_name, "forced_name")

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreatorNestingError(self):

    def creator(next_creator, *a, **kwargs):
      return next_creator(*a, **kwargs)

    # Save the state so we can clean up at the end.
    graph = ops.get_default_graph()
    old_creator_stack = graph._resource_creator_stack["_DummyResource"]

    try:
      scope = ops.resource_creator_scope(creator, "_DummyResource")
      scope.__enter__()
      with ops.resource_creator_scope(creator, "_DummyResource"):
        with self.assertRaises(RuntimeError):
          scope.__exit__(None, None, None)
    finally:
      graph._resource_creator_stack["_DummyResource"] = old_creator_stack

  @test_util.run_in_graph_and_eager_modes
  @run_inside_wrap_function_in_eager_mode
  def testResourceCreatorNesting(self):

    def resource_creator_fn_0(next_creator, *a, **kwargs):
      instance = next_creator(*a, **kwargs)
      instance._value = 1
      return instance

    def resource_creator_fn_1(next_creator, *a, **kwargs):
      kwargs["handle_name"] = "forced_name1"
      return next_creator(*a, **kwargs)

    with ops.resource_creator_scope(["_DummyResource1"], resource_creator_fn_0):
      with ops.resource_creator_scope(["_DummyResource1"],
                                      resource_creator_fn_1):
        dummy_0 = _DummyResource1(handle_name="fake_name")

    self.assertEqual(dummy_0._handle_name, "forced_name1")
    self.assertEqual(dummy_0._value, 1)


if __name__ == "__main__":
  test.main()
