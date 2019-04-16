"""Dependency tracking for trackable objects."""
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

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import tf_contextlib


# global _RESOURCE_TRACKER_STACK
_RESOURCE_TRACKER_STACK = []


class NotTrackable(object):
  """Marks instances of child classes as unsaveable using an object-based API.

  Useful for marking objects which would otherwise look trackable because
  of inheritance (e.g. through `Layer`) as not trackable. Inheriting from
  `NotTrackable` does not prevent an object from being assigned to any
  attributes, but will throw an error on save/restore.
  """
  pass


class AutoTrackable(base.Trackable):
  """Manages dependencies on other objects.

  `Trackable` objects may have dependencies: other `Trackable` objects
  which should be saved if the object declaring the dependency is saved. A
  correctly saveable program has a dependency graph such that if changing a
  global variable affects an object (e.g. changes the behavior of any of its
  methods) then there is a chain of dependencies from the influenced object to
  the variable.

  Dependency edges have names, and are created implicitly when a
  `Trackable` object is assigned to an attribute of another
  `Trackable` object. For example:

  ```
  obj = Trackable()
  obj.v = ResourceVariable(0.)
  ```

  The `Trackable` object `obj` now has a dependency named "v" on a
  variable.

  `Trackable` objects may specify `Tensor`s to be saved and restored
  directly (e.g. a `Variable` indicating how to save itself) rather than through
  dependencies on other objects. See
  `Trackable._gather_saveables_for_checkpoint` for details.
  """

  def __setattr__(self, name, value):
    """Support self.foo = trackable syntax."""
    if getattr(self, "_self_setattr_tracking", True):
      value = data_structures.sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    super(AutoTrackable, self).__setattr__(name, value)

  def __delattr__(self, name):
    self._maybe_initialize_trackable()
    if name in self._unconditional_dependency_names:
      del self._unconditional_dependency_names[name]
      for index, (dep_name, _) in enumerate(
          self._unconditional_checkpoint_dependencies):
        if dep_name == name:
          del self._unconditional_checkpoint_dependencies[index]
          break
    super(AutoTrackable, self).__delattr__(name)

  def _no_dependency(self, value):
    """Override to allow TrackableBase to disable dependency tracking."""
    return data_structures.NoDependency(value)

  def _list_functions_for_serialization(self):
    """Return a dict of `Function`s of a trackable."""
    functions = {}
    for attribute_name in dir(self):
      try:
        attribute_value = getattr(self, attribute_name, None)
      except Exception:  # pylint: disable=broad-except
        # We really don't want to throw an exception just because some object's
        # attribute accessor is broken.
        attribute_value = None
      if isinstance(attribute_value, (def_function.Function,
                                      defun.ConcreteFunction)):
        functions[attribute_name] = attribute_value
    return functions


class ResourceTracker(object):
  """An object that tracks a list of resources."""

  def __init__(self):
    self._resources = []

  @property
  def resources(self):
    return self._resources

  def add_resource(self, resource):
    self._resources.append(resource)


@tf_contextlib.contextmanager
def resource_tracker_scope(resource_tracker):
  """A context to manage resource trackers.

  Use this in order to collect up all resources created within a block of code.
  Example usage:

  ```python
  resource_tracker = ResourceTracker()
  with resource_tracker_scope(resource_tracker):
    resource = TrackableResource()

  assert resource_tracker.resources == [resource]

  Args:
    resource_tracker: The passed in ResourceTracker object

  Yields:
    A scope in which the resource_tracker is active.
  """
  global _RESOURCE_TRACKER_STACK
  old = list(_RESOURCE_TRACKER_STACK)
  _RESOURCE_TRACKER_STACK.append(resource_tracker)
  try:
    yield
  finally:
    _RESOURCE_TRACKER_STACK = old


class TrackableResource(base.Trackable):
  """Base class for all resources that need to be tracked."""

  def __init__(self):
    global _RESOURCE_TRACKER_STACK
    for resource_tracker in _RESOURCE_TRACKER_STACK:
      resource_tracker.add_resource(self)

    self._resource_handle = None

  def _create_resource(self):
    """A function that creates a resource handle."""
    raise NotImplementedError("TrackableResource._create_resource not "
                              "implemented.")

  def _initialize(self):
    """A function that initializes the resource. Optional."""
    pass

  @property
  def resource_handle(self):
    """Returns the resource handle associated with this Resource."""
    if self._resource_handle is None:
      self._resource_handle = self._create_resource()
    return self._resource_handle

  def _list_functions_for_serialization(self):
    @def_function.function(input_signature=[], autograph=False)
    def _creator():
      resource = self._create_resource()
      return resource

    @def_function.function(input_signature=[], autograph=False)
    def _initializer():
      self._initialize()
      return 1  # Dummy return

    return {
        "_create_resource": _creator,
        "_initialize": _initializer,
    }


class TrackableAsset(base.Trackable):
  """Base class for asset files which need to be tracked."""

  def __init__(self, path):
    """Record the full path to the asset."""
    # The init_scope prevents functions from capturing `path` in an
    # initialization graph, since it is transient and should not end up in a
    # serialized function body.
    with ops.init_scope():
      self._path = ops.internal_convert_to_tensor(path, dtype=dtypes.string,
                                                  name="asset_path")

  @property
  def asset_path(self):
    """Fetch the current asset path."""
    return self._path

ops.register_tensor_conversion_function(
    TrackableAsset,
    lambda asset, **kw: ops.internal_convert_to_tensor(asset.asset_path, **kw))
