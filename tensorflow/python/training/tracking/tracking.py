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

import contextlib
import copy
import warnings

from absl import logging
import six

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
    self._delete_tracking(name)
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

  def _delete_tracking(self, name):
    """Removes the tracking of name."""
    self._maybe_initialize_trackable()
    if name in self._unconditional_dependency_names:
      del self._unconditional_dependency_names[name]
      for index, (dep_name, _) in enumerate(
          self._unconditional_checkpoint_dependencies):
        if dep_name == name:
          del self._unconditional_checkpoint_dependencies[index]
          break


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


def _make_getter(captured_getter, captured_previous):
  """To avoid capturing loop variables."""

  def getter(*args, **kwargs):
    return captured_getter(captured_previous, *args, **kwargs)

  return getter


class ResourceMetaclass(type):
  """Metaclass for CapturableResource."""

  def __call__(cls, *args, **kwargs):

    def default_resource_creator(next_creator, *a, **kw):
      assert next_creator is None
      obj = cls.__new__(cls, *a, **kw)
      obj.__init__(*a, **kw)
      return obj

    previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
    resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    for getter in resource_creator_stack[cls._resource_type()]:
      previous_getter = _make_getter(getter, previous_getter)

    return previous_getter(*args, **kwargs)


class CapturableResource(six.with_metaclass(ResourceMetaclass, base.Trackable)):
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
    self._self_destruction_context = (
        context.eager_mode if context.executing_eagerly()
        else ops.get_default_graph().as_default)

  @classmethod
  def _resource_type(cls):
    return cls.__name__

  @property
  def _destruction_context(self):
    return getattr(self, "_self_destruction_context",
                   # no-op context
                   contextlib.suppress)

  @_destruction_context.setter
  def _destruction_context(self, destruction_context):
    self._self_destruction_context = destruction_context

  def _create_resource(self):
    """A function that creates a resource handle."""
    raise NotImplementedError("TrackableResource._create_resource not "
                              "implemented.")

  def _initialize(self):
    """A function that initializes the resource. Optional."""
    pass

  def _destroy_resource(self):
    """A function that destroys the resource. Optional."""
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
      self._destroy_resource()
      return 1  # Dummy return

    return {
        "_create_resource": _creator,
        "_initialize": _initializer,
        "_destroy_resource": _destroyer,
    }

  def __del__(self):
    try:
      # Outer race condition: on program exit, the destruction context may be
      # deleted before this __del__ is called. At this point we can safely
      # exit without calling _destroy_resource() and let Python handle things.
      with self._destruction_context():
        # Inner race condition: possible between this and `ScopedTFFunction`
        # whereby if an entire garbage collection chain containing both
        # objects is moved to unreachable during the same garbage collection
        # cycle, the __del__ for `ScopedTFFunction` can be collected before
        # this method is called. In that case, we can't do much but
        # continue.
        self._destroy_resource()
    except Exception:  # pylint: disable=broad-except
      # Silence all error logs that occur when attempting to destroy this
      # resource.
      pass


@tf_export("saved_model.experimental.TrackableResource")
class TrackableResource(CapturableResource):
  """Holds a Tensor which a tf.function can capture.

  A TrackableResource is most useful for stateful Tensors that require
  initialization, such as `tf.lookup.StaticHashTable`. `TrackableResource`s
  are discovered by traversing the graph of object attributes, e.g. during
  `tf.saved_model.save`.

  A TrackableResource has three methods to override:

  * `_create_resource` should create the resource tensor handle.
  * `_initialize` should initialize the resource held at `self.resource_handle`.
  * `_destroy_resource` is called upon a `TrackableResource`'s destruction
    and should decrement the resource's ref count. For most resources, this
    should be done with a call to `tf.raw_ops.DestroyResourceOp`.

  Example usage:

  >>> class DemoResource(tf.saved_model.experimental.TrackableResource):
  ...   def __init__(self):
  ...     super().__init__()
  ...     self._initialize()
  ...   def _create_resource(self):
  ...     return tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=[2])
  ...   def _initialize(self):
  ...     tf.raw_ops.AssignVariableOp(
  ...         resource=self.resource_handle, value=tf.ones([2]))
  ...   def _destroy_resource(self):
  ...     tf.raw_ops.DestroyResourceOp(resource=self.resource_handle)
  >>> class DemoModule(tf.Module):
  ...   def __init__(self):
  ...     self.resource = DemoResource()
  ...   def increment(self, tensor):
  ...     return tensor + tf.raw_ops.ReadVariableOp(
  ...         resource=self.resource.resource_handle, dtype=tf.float32)
  >>> demo = DemoModule()
  >>> demo.increment([5, 1])
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 2.], dtype=float32)>
  """

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
