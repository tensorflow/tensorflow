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

import copy
import warnings

from absl import logging

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


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


@tf_export("__internal__.tracking.AutoTrackable", v1=[])
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
    try:
      if getattr(self, name) is value:
        # Short circuit for `self.$x = self.$x`.
        return
    except AttributeError:
      pass

    if getattr(self, "_self_setattr_tracking", True):
      value = data_structures.sticky_attribute_assignment(
          trackable=self, value=value, name=name)
    super(AutoTrackable, self).__setattr__(name, value)

  def __delattr__(self, name):
    self._maybe_initialize_trackable()
    delete_tracking(self, name)
    super(AutoTrackable, self).__delattr__(name)

  def _no_dependency(self, value):
    """Override to allow TrackableBase to disable dependency tracking."""
    return data_structures.NoDependency(value)

  def _list_functions_for_serialization(self, unused_serialization_cache):
    """Return a dict of `Function`s of a trackable."""
    functions = {}
    for attribute_name in dir(self):
      # We get the attributes, suppressing warnings and exceptions.
      logging_verbosity = logging.get_verbosity()
      try:
        logging.set_verbosity(logging.FATAL)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          attribute_value = getattr(self, attribute_name, None)
      except Exception:  # pylint: disable=broad-except
        # We really don't want to throw an exception just because some object's
        # attribute accessor is broken.
        attribute_value = None
      finally:
        # We reset the verbosity setting in a `finally` block, to make
        # sure it always happens, even if we make the exception catching above
        # be less broad.
        logging.set_verbosity(logging_verbosity)
      if isinstance(attribute_value, (def_function.Function,
                                      defun.ConcreteFunction)):
        functions[attribute_name] = attribute_value
    return functions


def delete_tracking(obj, name):
  """Removes the tracking of name from object."""
  # pylint: disable=protected-access
  if name in obj._unconditional_dependency_names:
    del obj._unconditional_dependency_names[name]
    for index, (dep_name, _) in enumerate(
        obj._unconditional_checkpoint_dependencies):
      if dep_name == name:
        del obj._unconditional_checkpoint_dependencies[index]
        break
  # pylint: enable=protected-access


class ResourceTracker(object):
  """An object that tracks a list of resources."""

  __slots__ = ["_resources"]

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


class CapturableResourceDeleter(object):
  """Deleter to destroy CapturableResource without overriding its __del__()."""

  __slots__ = ["_destruction_context", "_destroy_resource"]

  def __init__(self, destroy_resource_fn=None):
    if destroy_resource_fn:
      self._destroy_resource = destroy_resource_fn
      self._destruction_context = (
          context.eager_mode if context.executing_eagerly()
          else ops.get_default_graph().as_default)
    else:
      self._destroy_resource = None

  def destroy_resource(self):
    if self._destroy_resource:
      return self._destroy_resource()

  def __del__(self):
    if self._destroy_resource:
      with self._destruction_context():
        self._destroy_resource()


class CapturableResource(base.Trackable):
  """Holds a Tensor which a tf.function can capture.

  `CapturableResource`s are discovered by traversing the graph of object
  attributes, e.g. during `tf.saved_model.save`. They are excluded from the
  scope-based tracking of `TrackableResource`; generally things that require
  initialization should inherit from `TrackableResource` instead of
  `CapturableResource` directly.
  """

  def __init__(self, device="", deleter=None):
    """Initialize the `CapturableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
      deleter: A CapturableResourceDeleter that will destroy the created
        resource during destruction.
    """
    self._resource_handle = None
    self._resource_device = device
    self._resource_deleter = deleter or CapturableResourceDeleter()

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

  def _map_resources(self, _):
    """For implementing `Trackable`."""
    new_obj = copy.copy(self)
    # pylint: disable=protected-access
    with ops.device(self._resource_device):
      new_resource = new_obj._create_resource()
    new_obj._resource_handle = new_resource
    # pylint: enable=protected-access
    obj_map = {self: new_obj}
    resource_map = {self.resource_handle: new_resource}
    return obj_map, resource_map

  def _list_functions_for_serialization(self, unused_functions):
    @def_function.function(input_signature=[], autograph=False)
    def _creator():
      resource = self._create_resource()
      return resource

    @def_function.function(input_signature=[], autograph=False)
    def _initializer():
      self._initialize()
      return 1  # Dummy return

    @def_function.function(input_signature=[], autograph=False)
    def _destroyer():
      self._resource_deleter.destroy_resource()
      return 1  # Dummy return

    return {
        "_create_resource": _creator,
        "_initialize": _initializer,
        "_destroy_resource": _destroyer,
    }


class TrackableResource(CapturableResource):
  """Adds scope tracking to CapturableResource."""

  def __init__(self, device="", deleter=None):
    """Initialize the `TrackableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
      deleter: A CapturableResourceDeleter that will destroy the created
        resource during destruction.
    """
    global _RESOURCE_TRACKER_STACK
    for resource_tracker in _RESOURCE_TRACKER_STACK:
      resource_tracker.add_resource(self)
    super(TrackableResource, self).__init__(device=device, deleter=deleter)


@tf_export("saved_model.Asset")
class Asset(base.Trackable):
  """Represents a file asset to hermetically include in a SavedModel.

  A SavedModel can include arbitrary files, called assets, that are needed
  for its use. For example a vocabulary file used initialize a lookup table.

  When a trackable object is exported via `tf.saved_model.save()`, all the
  `Asset`s reachable from it are copied into the SavedModel assets directory.
  Upon loading, the assets and the serialized functions that depend on them
  will refer to the correct filepaths inside the SavedModel directory.

  Example:

  ```
  filename = tf.saved_model.Asset("file.txt")

  @tf.function(input_signature=[])
  def func():
    return tf.io.read_file(filename)

  trackable_obj = tf.train.Checkpoint()
  trackable_obj.func = func
  trackable_obj.filename = filename
  tf.saved_model.save(trackable_obj, "/tmp/saved_model")

  # The created SavedModel is hermetic, it does not depend on
  # the original file and can be moved to another path.
  tf.io.gfile.remove("file.txt")
  tf.io.gfile.rename("/tmp/saved_model", "/tmp/new_location")

  reloaded_obj = tf.saved_model.load("/tmp/new_location")
  print(reloaded_obj.func())
  ```

  Attributes:
    asset_path: A 0-D `tf.string` tensor with path to the asset.
  """

  def __init__(self, path):
    """Record the full path to the asset."""
    # The init_scope prevents functions from capturing `path` in an
    # initialization graph, since it is transient and should not end up in a
    # serialized function body.
    with ops.init_scope(), ops.device("CPU"):
      self._path = ops.convert_to_tensor(
          path, dtype=dtypes.string, name="asset_path")

  @property
  def asset_path(self):
    """Fetch the current asset path."""
    return self._path


ops.register_tensor_conversion_function(
    Asset, lambda asset, **kw: ops.convert_to_tensor(asset.asset_path, **kw))
