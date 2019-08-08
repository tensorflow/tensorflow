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

import functools
import weakref

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


class CapturableResource(base.Trackable):
  """Holds a Tensor which a tf.function can capture.

  `CapturableResource`s are discovered by traversing the graph of object
  attributes, e.g. during `tf.saved_model.save`. They are excluded from the
  scope-based tracking of `TrackableResource`; generally things that require
  initialization should inherit from `TrackableResource` instead of
  `CapturableResource` directly.
  """

  def __init__(self, device=""):
    """Initialize the `CapturableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
    """
    self._resource_handle = None
    self._resource_device = device

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
      with ops.device(self._resource_device):
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


class TrackableResource(CapturableResource):
  """Adds scope tracking to CapturableResource."""

  def __init__(self, device=""):
    """Initialize the `TrackableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
    """
    global _RESOURCE_TRACKER_STACK
    for resource_tracker in _RESOURCE_TRACKER_STACK:
      resource_tracker.add_resource(self)
    super(TrackableResource, self).__init__(device=device)


class TrackableAsset(base.Trackable):
  """Base class for asset files which need to be tracked."""

  def __init__(self, path):
    """Record the full path to the asset."""
    # The init_scope prevents functions from capturing `path` in an
    # initialization graph, since it is transient and should not end up in a
    # serialized function body.
    with ops.init_scope(), ops.device("CPU"):
      self._path = ops.internal_convert_to_tensor(path, dtype=dtypes.string,
                                                  name="asset_path")

  @property
  def asset_path(self):
    """Fetch the current asset path."""
    return self._path


def cached_per_instance(f):
  """Lightweight decorator for caching lazily constructed properties.

  When to use:
  This decorator provides simple caching with minimal overhead. It is designed
  for properties which are expensive to compute and static over the life of a
  class instance, and provides no mechanism for cache invalidation. Thus it is
  best suited for lazily exposing derived properties of other static data.

  For classes with custom getattr / setattr behavior (such as trackable
  objects), storing cache results as object attributes is not performant.
  Instead, a specialized cache can significantly reduce property lookup
  overhead. (While still allowing the decorated property to be lazily computed.)
  Consider the following class:

  ```
  class MyClass(object):
    def __setattr__(self, key, value):
      # Some expensive class specific code
      # ...
      # ...

      super(MyClass, self).__setattr__(key, value)

    @property
    def thing(self):
      # `thing` is expensive to compute (and may not even be requested), so we
      # want to lazily compute it and then cache it.
      output = getattr(self, '_thing', None)
      if output is None:
        self._thing = output = compute_thing(self)
      return output
  ```

  It's also worth noting that ANY overriding of __setattr__, even something as
  simple as:
  ```
    def __setattr__(self, key, value):
      super(MyClass, self).__setattr__(key, value)
  ```

  Slows down attribute assignment by nearly 10x.

  By contrast, replacing the definition of `thing` with the following sidesteps
  the expensive __setattr__ altogether:

  '''
  @property
  @tracking.cached_per_instance
  def thing(self):
    # `thing` is expensive to compute (and may not even be requested), so we
    # want to lazily compute it and then cache it.
    return compute_thing(self)
  '''

  Performance:
  The overhead for this decorator is ~0.4 us / call. A much lower overhead
  implementation (~0.085 us / call) can be achieved by using a custom dict type:

  ```
  def dict_based_cache(f):
    class Cache(dict):
      __slots__ = ()
      def __missing__(self, key):
        self[key] = output = f(key)
        return output

    return property(Cache().__getitem__)
  ```

  However, that implementation holds class instances as keys, and as a result
  blocks garbage collection. (And modifying it to use weakref's as keys raises
  the lookup overhead to ~0.4 us) As a result, the WeakKeyDictionary
  implementation below turns out to be more prudent.

  Args:
    f: The function to cache.

  Returns:
    f decorated with simple caching behavior.
  """

  cache = weakref.WeakKeyDictionary()

  @functools.wraps(f)
  def wrapped(item):
    output = cache.get(item)
    if output is None:
      cache[item] = output = f(item)
    return output
  return wrapped


ops.register_tensor_conversion_function(
    TrackableAsset,
    lambda asset, **kw: ops.internal_convert_to_tensor(asset.asset_path, **kw))
