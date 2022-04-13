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

from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import resource


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


class _DummyResource(resource.TrackableResource):

  def __init__(self, handle_name):
    self._handle_name = handle_name
    super(_DummyResource, self).__init__()

  def _create_resource(self):
    return self._handle_name


class _DummyResource1(resource.TrackableResource):

  def __init__(self, handle_name):
    self._handle_name = handle_name
    self._value = 0
    super(_DummyResource1, self).__init__()

  def _create_resource(self):
    return self._handle_name


class ResourceTrackerTest(test.TestCase):

  def testBasic(self):
    resource_tracker = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker):
      dummy_resource1 = _DummyResource("test1")
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(2, len(resource_tracker.resources))
    self.assertEqual("test1", resource_tracker.resources[0].resource_handle)
    self.assertEqual("test2", resource_tracker.resources[1].resource_handle)

  def testTwoScopes(self):
    resource_tracker1 = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker1):
      dummy_resource1 = _DummyResource("test1")

    resource_tracker2 = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker2):
      dummy_resource2 = _DummyResource("test2")

    self.assertEqual(1, len(resource_tracker1.resources))
    self.assertEqual("test1", resource_tracker1.resources[0].resource_handle)
    self.assertEqual(1, len(resource_tracker2.resources))
    self.assertEqual("test2", resource_tracker2.resources[0].resource_handle)

  def testNestedScopesScopes(self):
    resource_tracker = resource.ResourceTracker()
    with resource.resource_tracker_scope(resource_tracker):
      resource_tracker1 = resource.ResourceTracker()
      with resource.resource_tracker_scope(resource_tracker1):
        dummy_resource1 = _DummyResource("test1")

      resource_tracker2 = resource.ResourceTracker()
      with resource.resource_tracker_scope(resource_tracker2):
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
