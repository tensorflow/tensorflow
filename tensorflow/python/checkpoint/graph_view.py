"""Manages a graph of Trackable objects."""
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
import copy
import weakref

from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.checkpoint import util
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.tracking.ObjectGraphView", v1=[])
class ObjectGraphView(trackable_view.TrackableView):
  """Gathers and serializes an object graph."""

  def __init__(self, root, attached_dependencies=None):
    """Configure the graph view.

    Args:
      root: A `Trackable` object whose variables (including the variables of
        dependencies, recursively) should be saved. May be a weak reference.
      attached_dependencies: List of dependencies to attach to the root object.
        Used when saving a Checkpoint with a defined root object. To avoid
        reference cycles, this should use the WeakTrackableReference class.
    """
    trackable_view.TrackableView.__init__(self, root)
    # ObjectGraphView should never contain a strong reference to root, since it
    # may result in a cycle:
    #   root -> deferred dependencies -> CheckpointPosition
    #   -> CheckpointRestoreCoordinator -> ObjectGraphView -> root
    self._root_ref = (root if isinstance(root, weakref.ref)
                      else weakref.ref(root))
    self._attached_dependencies = attached_dependencies

  def __deepcopy__(self, memo):
    # By default, weak references are not copied, which leads to surprising
    # deepcopy behavior. To fix, we first we copy the object itself, then we
    # make a weak reference to the copy.
    strong_root = self._root_ref()
    if strong_root is not None:
      strong_copy = copy.deepcopy(strong_root, memo)
      memo[id(self._root_ref)] = weakref.ref(strong_copy)
    # super() does not have a __deepcopy__, so we need to re-implement it
    copied = super().__new__(type(self))
    memo[id(self)] = copied
    for key, value in vars(self).items():
      setattr(copied, key, copy.deepcopy(value, memo))
    return copied

  def list_children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
    """Returns list of all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      List of all children attached to the object.
    """
    children = []
    for name, ref in super(ObjectGraphView,
                           self).children(obj, save_type, **kwargs).items():
      children.append(base.TrackableReference(name, ref))

    # GraphView objects may define children of the root object that are not
    # actually attached, e.g. a Checkpoint object's save_counter.
    if obj is self.root and self._attached_dependencies:
      children.extend(self._attached_dependencies)
    return children

  def children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
    """Returns all child trackables attached to obj.

    Args:
      obj: A `Trackable` object.
      save_type: A string, can be 'savedmodel' or 'checkpoint'.
      **kwargs: kwargs to use when retrieving the object's children.

    Returns:
      Dictionary of all children attached to the object with name to trackable.
    """
    children = {}
    for name, ref in self.list_children(obj, **kwargs):
      children[name] = ref
    return children

  @property
  def attached_dependencies(self):
    """Returns list of dependencies that should be saved in the checkpoint.

    These dependencies are not tracked by root, but are in the checkpoint.
    This is defined when the user creates a Checkpoint with both root and kwargs
    set.

    Returns:
      A list of TrackableReferences.
    """
    return self._attached_dependencies

  @property
  def root(self):
    if isinstance(self._root_ref, weakref.ref):
      derefed = self._root_ref()
      assert derefed is not None
      return derefed
    else:
      return self._root_ref

  def breadth_first_traversal(self):
    return self._breadth_first_traversal()

  def _breadth_first_traversal(self):
    """Find shortest paths to all dependencies of self.root."""
    return super(ObjectGraphView, self)._all_nodes_with_paths()

  def serialize_object_graph(self, saveables_cache=None):
    """Determine checkpoint keys for variables and build a serialized graph.

    Non-slot variables are keyed based on a shortest path from the root saveable
    to the object which owns the variable (i.e. the one which called
    `Trackable._add_variable` to create it).

    Slot variables are keyed based on a shortest path to the variable being
    slotted for, a shortest path to their optimizer, and the slot name.

    Args:
      saveables_cache: An optional cache storing previously created
        SaveableObjects created for each Trackable. Maps Trackables to a
        dictionary of attribute names to Trackable.

    Returns:
      A tuple of (named_variables, object_graph_proto, feed_additions):
        named_variables: A dictionary mapping names to variable objects.
        object_graph_proto: A TrackableObjectGraph protocol buffer
          containing the serialized object graph and variable references.
        feed_additions: A dictionary mapping from Tensors to values which should
          be fed when saving.

    Raises:
      ValueError: If there are invalid characters in an optimizer's slot names.
    """
    named_saveable_objects, object_graph_proto, feed_additions, _ = (
        util.serialize_object_graph_with_registered_savers(self,
                                                           saveables_cache))
    return named_saveable_objects, object_graph_proto, feed_additions

  def frozen_saveable_objects(self,
                              object_map=None,
                              to_graph=None,
                              call_with_mapped_captures=None):
    """Creates SaveableObjects with the current object graph frozen."""
    return util.frozen_saveables_and_savers(
        self, object_map, to_graph, call_with_mapped_captures)[0]
