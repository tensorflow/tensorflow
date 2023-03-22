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
"""Definitions for resource-type trackable object classes."""

import contextlib
import copy
import weakref

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

# global _RESOURCE_TRACKER_STACK
_RESOURCE_TRACKER_STACK = []


class ResourceTracker:
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


class _ResourceMetaclass(type):
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


class CapturableResource(base.Trackable, metaclass=_ResourceMetaclass):
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
    self._resource_handle_value = None
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

  @property
  def _resource_handle(self):
    return self._resource_handle_value

  @_resource_handle.setter
  def _resource_handle(self, value):
    if isinstance(value, (ops.Tensor, ops.EagerTensor)):
      value._parent_trackable = weakref.ref(self)  # pylint: disable=protected-access
    self._resource_handle_value = value

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

  def _export_to_saved_model_graph(
      self, object_map, tensor_map, **unused_kwargs):
    """For implementing `Trackable`."""
    new_obj = copy.copy(self)
    # pylint: disable=protected-access
    with ops.device(self._resource_device):
      new_resource = new_obj._create_resource()
    new_obj._resource_handle = new_resource
    # pylint: enable=protected-access
    object_map[self] = new_obj
    tensor_map[self.resource_handle] = new_resource
    return [self.resource_handle]

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    children = super()._trackable_children(save_type, **kwargs)
    if save_type == "savedmodel":
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

      children.update({
          "_create_resource": _creator,
          "_initialize": _initializer,
          "_destroy_resource": _destroyer,
      })
    return children

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
    super().__init__(device=device)


# TODO(b/124205571,b/124092991): Solve destruction of resources.
class RestoredResource(TrackableResource):
  """Restored SavedResource."""

  def __init__(self, device=""):
    super().__init__(device=device)

  @classmethod
  def _deserialize_from_proto(cls, object_proto, dependencies, **unused_kwargs):
    obj = cls(device=object_proto.resource.device)
    resource_creator = dependencies.get("_create_resource")
    if resource_creator is not None:
      obj._create_resource = resource_creator  # pylint: disable=protected-access
    return obj

  def _add_trackable_child(self, name, value):
    setattr(self, name, value)
    if (isinstance(value, base.Trackable) and
        not isinstance(value, def_function.Function)):
      self._track_trackable(value, name)
