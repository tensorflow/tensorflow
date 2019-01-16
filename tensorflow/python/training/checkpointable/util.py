"""Utilities for saving/loading Checkpointable objects."""
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

import abc
import collections
import os
import weakref

from tensorflow.core.protobuf import checkpointable_object_graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import optimizer as optimizer_v1
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.checkpointable import base
from tensorflow.python.training.checkpointable import data_structures
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


_ESCAPE_CHAR = "."  # For avoiding conflicts with user-specified names.

# Keyword for identifying that the next bit of a checkpoint variable name is a
# slot name. Checkpoint names for slot variables look like:
#
#   <path to variable>/<_OPTIMIZER_SLOTS_NAME>/<path to optimizer>/<slot name>
#
# Where <path to variable> is a full path from the checkpoint root to the
# variable being slotted for.
_OPTIMIZER_SLOTS_NAME = _ESCAPE_CHAR + "OPTIMIZER_SLOT"
# Keyword for separating the path to an object from the name of an
# attribute in checkpoint names. Used like:
#   <path to variable>/<_OBJECT_ATTRIBUTES_NAME>/<name of attribute>
_OBJECT_ATTRIBUTES_NAME = _ESCAPE_CHAR + "ATTRIBUTES"


class _CheckpointRestoreCoordinator(object):
  """Holds the status of an object-based checkpoint load."""

  def __init__(self, object_graph_proto, save_path, save_path_tensor,
               restore_op_cache, saveable_object_cache):
    """Specify the checkpoint being loaded.

    Args:
      object_graph_proto: The CheckpointableObjectGraph protocol buffer
        associated with this checkpoint.
      save_path: A string, the path to the checkpoint, as returned by
        `tf.train.latest_checkpoint`.
      save_path_tensor: A string `Tensor` which contains or will be fed the save
        path.
      restore_op_cache: A dictionary shared between
        `_CheckpointRestoreCoordinator`s for the same Python objects, used to
        look up restore ops by name to avoid re-creating them across multiple
        `restore()` calls.
      saveable_object_cache: A mapping of checkpointable objects -> attribute
        names -> list(`SaveableObject`s), used when `SaveableObjects` must be
        referenced every restore (e.g. for Python state); otherwise they would
        create their own ops every restore.
    """
    self.object_graph_proto = object_graph_proto
    self.restore_uid = ops.uid()
    # Maps from objects to lists of attributes which were in the checkpoint but
    # not loaded into any object, for error checking.
    self.unused_attributes = weakref.WeakKeyDictionary()
    # Dictionary mapping from an id in the protocol buffer flat array to
    # Checkpointable Python objects. This mapping may be deferred if a
    # checkpoint is restored before all dependencies have been tracked. Uses
    # weak references so that partial restorations don't create reference cycles
    # (as objects with deferred dependencies will generally have references to
    # this object).
    self.object_by_proto_id = weakref.WeakValueDictionary()
    # A set of all Python objects we've seen as dependencies, even if we didn't
    # use them (for example because of inconsistent references when
    # loading). Used to make status assertions fail when loading checkpoints
    # that don't quite match.
    self.all_python_objects = _ObjectIdentityWeakSet()
    self.save_path_tensor = save_path_tensor
    self.save_path_string = save_path
    self.dtype_map = pywrap_tensorflow.NewCheckpointReader(
        save_path).get_variable_to_dtype_map()
    # A NewCheckpointReader for the most recent checkpoint, for streaming Python
    # state restoration.
    # When graph building, contains a list of ops to run to restore objects from
    # this checkpoint.
    self.restore_ops = []
    self.restore_ops_by_name = restore_op_cache
    self.saveable_object_cache = saveable_object_cache
    self.new_restore_ops_callback = None
    # A mapping from optimizer proto ids to lists of slot variables to be
    # restored when the optimizer is tracked. Only includes slot variables whose
    # regular variables have already been created, and only for optimizer
    # objects which have not yet been created/tracked.
    self.deferred_slot_restorations = {}
    # A mapping from variable proto ids to lists of slot variables to be
    # restored when the variable is created/tracked. These get shifted over to
    # deferred_slot_restorations if the optimizer hasn't been created when that
    # happens.
    self.slot_restorations = {}
    for node_index, node in enumerate(self.object_graph_proto.nodes):
      for slot_reference in node.slot_variables:
        # `node` refers to an `Optimizer`, since only these have slot variables.
        self.slot_restorations.setdefault(
            slot_reference.original_variable_node_id, []).append(
                base._SlotVariableRestoration(  # pylint: disable=protected-access
                    optimizer_id=node_index,
                    slot_variable_id=slot_reference.slot_variable_node_id,
                    slot_name=slot_reference.slot_name))

  def new_restore_ops(self, new_ops):
    self.restore_ops.extend(new_ops)
    if self.new_restore_ops_callback:
      self.new_restore_ops_callback(new_ops)  # pylint: disable=not-callable

  def restore_saveables(self, tensor_saveables, python_saveables):
    """Run or build restore operations for SaveableObjects.

    Args:
      tensor_saveables: `SaveableObject`s which correspond to Tensors.
      python_saveables: `PythonStateSaveable`s which correspond to Python
        values.

    Returns:
      When graph building, a list of restore operations, either cached or newly
      created, to restore `tensor_saveables`.
    """
    restore_ops = []
    # Eagerly run restorations for Python state.
    reader = pywrap_tensorflow.NewCheckpointReader(
        self.save_path_string)
    for saveable in python_saveables:
      spec_names = [spec.name for spec in saveable.specs]
      saveable.python_restore(
          [reader.get_tensor(name) for name in spec_names])

    # If we have new SaveableObjects, extract and cache restore ops.
    if tensor_saveables:
      validated_saveables = saveable_object_util.validate_and_slice_inputs(
          tensor_saveables)
      validated_names = set(saveable.name for saveable in validated_saveables)
      if set(tensor_saveables.keys()) != validated_names:
        raise AssertionError(
            ("Saveable keys changed when validating. Got back %s, was "
             "expecting %s") % (tensor_saveables.keys(), validated_names))
      new_restore_ops = functional_saver.restore_from_saveable_objects(
          self.save_path_tensor, validated_saveables)
      if not context.executing_eagerly():
        restore_ops.extend(new_restore_ops)
        for saveable, restore_op in zip(validated_saveables, new_restore_ops):
          assert saveable.name not in self.restore_ops_by_name
          self.restore_ops_by_name[saveable.name] = restore_op
    return restore_ops


class _NameBasedRestoreCoordinator(object):
  """Keeps the status of a name-based checkpoint restore."""

  def __init__(self, save_path, dtype_map=None):
    self.save_path = save_path
    self.dtype_map = dtype_map
    self.unused_attributes = weakref.WeakKeyDictionary()
    self.restore_uid = ops.uid()

  def globally_named_object_attributes(self, checkpointable):
    """Create globally named SaveableObjects from attributes.

    If an object's attribute has no global name specified (default construction
    for the SaveableObject factory), records the failure in
    `self.unused_attributes` (which can then be used to make status assertions
    fail; see `NameBasedSaverStatus`).

    Args:
      checkpointable: An object to save.

    Yields:
      SaveableObjects for `checkpointable`'s attributes.
    """
    for attribute_name, saveable_factory in (
        checkpointable._gather_saveables_for_checkpoint().items()):  # pylint: disable=protected-access
      if callable(saveable_factory):
        try:
          # This saveable object factory does not have a default name= argument,
          # which means there's no way to save/restore it using a name-based
          # checkpoint. Ignore the error now and make sure assert_consumed()
          # fails.
          saveable = saveable_factory()
        except TypeError:
          self.unused_attributes.setdefault(checkpointable, []).append(
              attribute_name)
          continue
      else:
        saveable = saveable_factory
      names_to_saveables = saveable_object_util.op_list_to_dict(
          [saveable],
          convert_variable_to_tensor=False)
      for name, op in names_to_saveables.items():
        for saveable_object in saveable_object_util.saveable_objects_for_op(
            op=op, name=name):
          yield saveable_object

  def eager_restore(self, checkpointable):
    """Runs restore ops for `checkpointable`'s attributes."""
    # When graph building, we don't add any restore ops to the graph until
    # run_restore_ops/initialize_or_restore on the status object for name-based
    # checkpoints.
    assert context.executing_eagerly()
    for saveable in self.globally_named_object_attributes(
        checkpointable):
      restored_tensors = []
      tensor_missing = False
      for spec in saveable.specs:
        if spec.name in self.dtype_map:
          with ops.device("cpu:0"):
            restored, = io_ops.restore_v2(
                prefix=self.save_path,
                tensor_names=[spec.name],
                shape_and_slices=[""],
                dtypes=[self.dtype_map[spec.name]],
                name="%s_checkpoint_read" % (spec.name,))
          restored_tensors.append(array_ops.identity(restored))
        else:
          tensor_missing = True

      if not tensor_missing:
        # Ignores values missing from the checkpoint, as with object-based
        # restore. Status assertions can be used to check exact matches,
        # although it's unlikely to ever happen for name-based checkpoints.
        saveable.restore(restored_tensors=restored_tensors,
                         restored_shapes=None)


# TODO(allenl): If this ends up in a public API, consider adding LINT.IfChange
# or consolidating the implementation with get_variable.
def _default_getter(name, shape, dtype, initializer=None,
                    partition_info=None, **kwargs):
  """A pared-down version of get_variable which does not reuse variables."""
  dtype = dtypes.as_dtype(dtype)
  shape_object = tensor_shape.as_shape(shape)
  with ops.init_scope():
    if initializer is None:
      initializer, initializing_from_value = (
          variable_scope._get_default_variable_store()._get_default_initializer(  # pylint: disable=protected-access
              name=name, shape=shape_object, dtype=dtype))
    else:
      initializing_from_value = not callable(initializer)
    # Same logic as get_variable
    variable_dtype = dtype.base_dtype
    if initializing_from_value:
      if shape is not None:
        raise ValueError("If initializer is a constant, do not specify shape.")
      initial_value = initializer
    else:
      # Instantiate initializer if provided initializer is a type object.
      if isinstance(initializer, type(init_ops.Initializer)):
        initializer = initializer(dtype=dtype)
      def initial_value():
        return initializer(
            shape_object.as_list(), dtype=dtype, partition_info=partition_info)
    return variables.VariableV1(
        initial_value=initial_value,
        name=name,
        dtype=variable_dtype,
        use_resource=True,
        **kwargs
    )


def add_variable(checkpointable, name, shape=None, dtype=dtypes.float32,
                 initializer=None):
  """Add a variable to a Checkpointable with no scope influence."""
  return checkpointable._add_variable_with_custom_getter(  # pylint: disable=protected-access
      name=name, shape=shape, dtype=dtype,
      initializer=initializer, getter=_default_getter)


def object_metadata(save_path):
  """Retrieves information about the objects in a checkpoint.

  Example usage:

  ```python
  object_graph = tf.contrib.checkpoint.object_metadata(
      tf.train.latest_checkpoint(checkpoint_directory))
  ckpt_variable_names = set()
  for node in object_graph.nodes:
    for attribute in node.attributes:
      ckpt_variable_names.add(attribute.full_name)
  ```

  Args:
    save_path: The path to the checkpoint, as returned by `save` or
      `tf.train.latest_checkpoint`.
  Returns:
    A parsed `tf.contrib.checkpoint.CheckpointableObjectGraph` protocol buffer.
  Raises:
    ValueError: If an object graph was not found in the checkpoint.
  """
  reader = pywrap_tensorflow.NewCheckpointReader(save_path)
  try:
    object_graph_string = reader.get_tensor(
        base.OBJECT_GRAPH_PROTO_KEY)
  except errors_impl.NotFoundError:
    raise ValueError(
        ('The specified checkpoint "%s" does not appear to be object-based (it '
         'is missing the key "%s"). Likely it was created with a name-based '
         'saver and does not contain an object dependency graph.') % (
             save_path, base.OBJECT_GRAPH_PROTO_KEY))
  object_graph_proto = (
      checkpointable_object_graph_pb2.CheckpointableObjectGraph())
  object_graph_proto.ParseFromString(object_graph_string)
  return object_graph_proto


class _ObjectIdentityWrapper(object):
  """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like checkpointable
  _ListWrapper objects to object-identity collections.
  """

  def __init__(self, wrapped):
    self._wrapped = wrapped

  @property
  def unwrapped(self):
    return self._wrapped

  def __eq__(self, other):
    if isinstance(other, _ObjectIdentityWrapper):
      return self._wrapped is other._wrapped  # pylint: disable=protected-access
    return self._wrapped is other

  def __hash__(self):
    # Wrapper id() is also fine for weakrefs. In fact, we rely on
    # id(weakref.ref(a)) == id(weakref.ref(a)) and weakref.ref(a) is
    # weakref.ref(a) in _WeakObjectIdentityWrapper.
    return id(self._wrapped)


class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):

  def __init__(self, wrapped):
    super(_WeakObjectIdentityWrapper, self).__init__(weakref.ref(wrapped))

  @property
  def unwrapped(self):
    return self._wrapped()


class ObjectIdentityDictionary(collections.MutableMapping):
  """A mutable mapping data structure which compares using "is".

  This is necessary because we have checkpointable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """

  def __init__(self):
    self._storage = {}

  def _wrap_key(self, key):
    return _ObjectIdentityWrapper(key)

  def __getitem__(self, key):
    return self._storage[self._wrap_key(key)]

  def __setitem__(self, key, value):
    self._storage[self._wrap_key(key)] = value

  def __delitem__(self, key):
    del self._storage[self._wrap_key(key)]

  def __len__(self):
    return len(self._storage)

  def __iter__(self):
    for key in self._storage:
      yield key.unwrapped


class _ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
  """Like weakref.WeakKeyDictionary, but compares objects with "is"."""

  def _wrap_key(self, key):
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # Iterate, discarding old weak refs
    return len(list(self._storage))

  def __iter__(self):
    keys = self._storage.keys()
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        del self[key]
      else:
        yield unwrapped


class _ObjectIdentitySet(collections.MutableSet):
  """Like the built-in set, but compares objects with "is"."""

  def __init__(self, *args):
    self._storage = set([self._wrap_key(obj) for obj in list(*args)])

  def _wrap_key(self, key):
    return _ObjectIdentityWrapper(key)

  def __contains__(self, key):
    return self._wrap_key(key) in self._storage

  def discard(self, key):
    self._storage.discard(self._wrap_key(key))

  def add(self, key):
    self._storage.add(self._wrap_key(key))

  def __len__(self):
    return len(self._storage)

  def __iter__(self):
    keys = list(self._storage)
    for key in keys:
      yield key.unwrapped


class _ObjectIdentityWeakSet(_ObjectIdentitySet):
  """Like weakref.WeakSet, but compares objects with "is"."""

  def _wrap_key(self, key):
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # Iterate, discarding old weak refs
    return len([_ for _ in self])

  def __iter__(self):
    keys = list(self._storage)
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        self.discard(key)
      else:
        yield unwrapped


def _breadth_first_checkpointable_traversal(root_checkpointable):
  """Find shortest paths to all variables owned by dependencies of root."""
  bfs_sorted = []
  to_visit = collections.deque([root_checkpointable])
  path_to_root = ObjectIdentityDictionary()
  path_to_root[root_checkpointable] = ()
  while to_visit:
    current_checkpointable = to_visit.popleft()
    if isinstance(current_checkpointable, tracking.NotCheckpointable):
      raise NotImplementedError(
          ("The object %s does not support object-based saving. File a feature "
           "request if this limitation bothers you. In the meantime, you can "
           "remove the dependency on this object and save everything else.")
          % (current_checkpointable,))
    current_checkpointable._maybe_initialize_checkpointable()  # pylint: disable=protected-access
    bfs_sorted.append(current_checkpointable)
    for child_checkpointable in (
        current_checkpointable._checkpoint_dependencies):  # pylint: disable=protected-access
      if child_checkpointable.ref not in path_to_root:
        path_to_root[child_checkpointable.ref] = (
            path_to_root[current_checkpointable] + (child_checkpointable,))
        to_visit.append(child_checkpointable.ref)
  return bfs_sorted, path_to_root


def _escape_local_name(name):
  # We need to support slashes in local names for compatibility, since this
  # naming scheme is being patched in to things like Layer.add_variable where
  # slashes were previously accepted. We also want to use slashes to indicate
  # edges traversed to reach the variable, so we escape forward slashes in
  # names.
  return (name.replace(_ESCAPE_CHAR, _ESCAPE_CHAR + _ESCAPE_CHAR)
          .replace(r"/", _ESCAPE_CHAR + "S"))


def _object_prefix_from_path(path_to_root):
  return "/".join(
      (_escape_local_name(checkpointable.name)
       for checkpointable in path_to_root))


def _slot_variable_naming_for_optimizer(optimizer_path):
  """Make a function for naming slot variables in an optimizer."""
  # Name slot variables:
  #
  #   <variable name>/<_OPTIMIZER_SLOTS_NAME>/<optimizer path>/<slot name>
  #
  # where <variable name> is exactly the checkpoint name used for the original
  # variable, including the path from the checkpoint root and the local name in
  # the object which owns it. Note that we only save slot variables if the
  # variable it's slotting for is also being saved.

  optimizer_identifier = "/%s/%s/" % (_OPTIMIZER_SLOTS_NAME, optimizer_path)

  def _name_slot_variable(variable_path, slot_name):
    """With an optimizer specified, name a slot variable."""
    return (variable_path
            + optimizer_identifier
            + _escape_local_name(slot_name))

  return _name_slot_variable


def _serialize_slot_variables(checkpointable_objects, node_ids, object_names):
  """Gather and name slot variables."""
  non_slot_objects = list(checkpointable_objects)
  slot_variables = ObjectIdentityDictionary()
  for checkpointable in non_slot_objects:
    if (isinstance(checkpointable, optimizer_v1.Optimizer)
        # TODO(b/110718070): Fix Keras imports.
        or hasattr(checkpointable, "_create_or_restore_slot_variable")):
      naming_scheme = _slot_variable_naming_for_optimizer(
          optimizer_path=object_names[checkpointable])
      slot_names = checkpointable.get_slot_names()
      for slot_name in slot_names:
        for original_variable_node_id, original_variable in enumerate(
            non_slot_objects):
          try:
            slot_variable = checkpointable.get_slot(
                original_variable, slot_name)
          except (AttributeError, KeyError):
            slot_variable = None
          if slot_variable is None:
            continue
          slot_variable._maybe_initialize_checkpointable()  # pylint: disable=protected-access
          if slot_variable._checkpoint_dependencies:  # pylint: disable=protected-access
            # TODO(allenl): Gather dependencies of slot variables.
            raise NotImplementedError(
                "Currently only variables with no dependencies can be saved as "
                "slot variables. File a feature request if this limitation "
                "bothers you.")
          if slot_variable in node_ids:
            raise NotImplementedError(
                "A slot variable was re-used as a dependency of a "
                "Checkpointable object. This is not currently allowed. File a "
                "feature request if this limitation bothers you.")
          checkpoint_name = naming_scheme(
              variable_path=object_names[original_variable],
              slot_name=slot_name)
          object_names[slot_variable] = checkpoint_name
          slot_variable_node_id = len(checkpointable_objects)
          node_ids[slot_variable] = slot_variable_node_id
          checkpointable_objects.append(slot_variable)
          slot_variable_proto = (
              checkpointable_object_graph_pb2.CheckpointableObjectGraph
              .CheckpointableObject.SlotVariableReference(
                  slot_name=slot_name,
                  original_variable_node_id=original_variable_node_id,
                  slot_variable_node_id=slot_variable_node_id))
          slot_variables.setdefault(checkpointable, []).append(
              slot_variable_proto)
  return slot_variables


def _add_attributes_to_object_graph(
    checkpointable_objects, object_graph_proto, node_ids, object_names,
    saveables_cache, object_map):
  """Create SaveableObjects and corresponding SerializedTensor protos."""
  named_saveable_objects = []
  if saveables_cache is None:
    # No SaveableObject caching. Either we're executing eagerly, or building a
    # static save which is specialized to the current Python state.
    feed_additions = None
  else:
    # If we are caching SaveableObjects, we need to build up a feed_dict with
    # functions computing volatile Python state to be saved with the checkpoint.
    feed_additions = {}
  for checkpoint_id, (checkpointable, object_proto) in enumerate(
      zip(checkpointable_objects, object_graph_proto.nodes)):
    assert node_ids[checkpointable] == checkpoint_id
    object_name = object_names[checkpointable]
    if object_map:
      object_to_save = object_map.get(checkpointable, checkpointable)
    else:
      object_to_save = checkpointable
    if saveables_cache is not None:
      cached_attributes = saveables_cache.setdefault(object_to_save, {})
    else:
      cached_attributes = None

    for name, saveable_factory in (
        object_to_save._gather_saveables_for_checkpoint().items()):  # pylint: disable=protected-access
      attribute = object_proto.attributes.add()
      attribute.name = name
      attribute.checkpoint_key = "%s/%s/%s" % (
          object_name, _OBJECT_ATTRIBUTES_NAME, _escape_local_name(name))
      if cached_attributes is None:
        saveables = None
      else:
        saveables = cached_attributes.get(name, None)
        if saveables is not None:
          for saveable in saveables:
            if attribute.checkpoint_key not in saveable.name:
              # The checkpoint key for this SaveableObject is different. We need
              # to re-create it.
              saveables = None
              del cached_attributes[name]
              break
      if saveables is None:
        if callable(saveable_factory):
          maybe_saveable = saveable_factory(name=attribute.checkpoint_key)
        else:
          maybe_saveable = saveable_factory
        if isinstance(maybe_saveable, saveable_object_lib.SaveableObject):
          saveables = (maybe_saveable,)
        else:
          # Figure out the name-based Saver's name for this variable. If it's
          # already a SaveableObject we'd just get the checkpoint key back, so
          # we leave full_name blank.
          saver_dict = saveable_object_util.op_list_to_dict(
              [maybe_saveable], convert_variable_to_tensor=False)
          full_name, = saver_dict.keys()
          saveables = tuple(saveable_object_util.saveable_objects_for_op(
              op=maybe_saveable, name=attribute.checkpoint_key))
          for saveable in saveables:
            saveable.full_name = full_name
        for saveable in saveables:
          if attribute.checkpoint_key not in saveable.name:
            raise AssertionError(
                ("The object %s produced a SaveableObject with name '%s' for "
                 "attribute '%s'. Expected a name containing '%s'.")
                % (checkpointable, name, saveable.name,
                   attribute.checkpoint_key))
        if cached_attributes is not None:
          cached_attributes[name] = saveables

      optional_restore = None
      for saveable in saveables:
        if optional_restore is None:
          optional_restore = saveable.optional_restore
        else:
          optional_restore = optional_restore and saveable.optional_restore

        if hasattr(saveable, "full_name"):
          attribute.full_name = saveable.full_name
        if isinstance(saveable, base.PythonStateSaveable):
          if feed_additions is None:
            assert saveables_cache is None
            # If we're not caching saveables, then we're either executing
            # eagerly or building a static save/restore (e.g. for a
            # SavedModel). In either case, we should embed the current Python
            # state in the graph rather than relying on a feed dict.
            saveable = saveable.freeze()
          else:
            saveable_feed_dict = saveable.feed_dict_additions()
            for new_feed_key in saveable_feed_dict.keys():
              if new_feed_key in feed_additions:
                raise AssertionError(
                    ("The object %s tried to feed a value for the Tensor %s "
                     "when saving, but another object is already feeding a "
                     "value.")
                    % (checkpointable, new_feed_key))
            feed_additions.update(saveable_feed_dict)
        named_saveable_objects.append(saveable)
      if optional_restore is None:
        optional_restore = False
      attribute.optional_restore = optional_restore

  return named_saveable_objects, feed_additions


def fill_object_graph_proto(checkpointable_objects,
                            node_ids,
                            slot_variables,
                            object_graph_proto=None):
  """Name non-slot `Checkpointable`s and add them to `object_graph_proto`."""
  if object_graph_proto is None:
    object_graph_proto = (
        checkpointable_object_graph_pb2.CheckpointableObjectGraph())
  for checkpoint_id, checkpointable in enumerate(checkpointable_objects):
    assert node_ids[checkpointable] == checkpoint_id
    object_proto = object_graph_proto.nodes.add()
    object_proto.slot_variables.extend(slot_variables.get(checkpointable, ()))
    for child in checkpointable._checkpoint_dependencies:  # pylint: disable=protected-access
      child_proto = object_proto.children.add()
      child_proto.node_id = node_ids[child.ref]
      child_proto.local_name = child.name
  return object_graph_proto


def _serialize_gathered_objects(
    checkpointable_objects, path_to_root, saveables_cache, object_map):
  """Create SaveableObjects and protos for gathered objects."""
  object_names = ObjectIdentityDictionary()
  for obj, path in path_to_root.items():
    object_names[obj] = _object_prefix_from_path(path)
  node_ids = ObjectIdentityDictionary()
  for node_id, node in enumerate(checkpointable_objects):
    node_ids[node] = node_id
  slot_variables = _serialize_slot_variables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names)
  object_graph_proto = fill_object_graph_proto(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      slot_variables=slot_variables)
  named_saveable_objects, feed_additions = _add_attributes_to_object_graph(
      checkpointable_objects=checkpointable_objects,
      object_graph_proto=object_graph_proto,
      node_ids=node_ids,
      object_names=object_names,
      saveables_cache=saveables_cache,
      object_map=object_map)
  return named_saveable_objects, object_graph_proto, feed_additions


def _serialize_object_graph(root_checkpointable, saveables_cache):
  """Determine checkpoint keys for variables and build a serialized graph.

  Non-slot variables are keyed based on a shortest path from the root saveable
  to the object which owns the variable (i.e. the one which called
  `Checkpointable._add_variable` to create it).

  Slot variables are keyed based on a shortest path to the variable being
  slotted for, a shortest path to their optimizer, and the slot name.

  Args:
    root_checkpointable: A `Checkpointable` object whose variables (including
      the variables of dependencies, recursively) should be saved.
    saveables_cache: A dictionary mapping `Checkpointable` objects -> attribute
      names -> SaveableObjects, used to avoid re-creating SaveableObjects when
      graph building.

  Returns:
    A tuple of (named_variables, object_graph_proto, feed_additions):
      named_variables: A dictionary mapping names to variable objects.
      object_graph_proto: A CheckpointableObjectGraph protocol buffer containing
        the serialized object graph and variable references.
      feed_additions: A dictionary mapping from Tensors to values which should
        be fed when saving.

  Raises:
    ValueError: If there are invalid characters in an optimizer's slot names.
  """
  checkpointable_objects, path_to_root = (
      _breadth_first_checkpointable_traversal(root_checkpointable))
  return _serialize_gathered_objects(
      checkpointable_objects, path_to_root, saveables_cache, object_map=None)


def named_saveables(root_checkpointable):
  """Gather list of all SaveableObjects in the Checkpointable object."""
  return _serialize_object_graph(root_checkpointable, None)[0]


def find_objects(root_checkpointable):
  """Find and number objects which are dependencies of `root_checkpointable`."""
  checkpointable_objects, path_to_root = (
      _breadth_first_checkpointable_traversal(root_checkpointable))
  object_names = ObjectIdentityDictionary()
  for obj, path in path_to_root.items():
    object_names[obj] = _object_prefix_from_path(path)
  node_ids = ObjectIdentityDictionary()
  for node_id, node in enumerate(checkpointable_objects):
    node_ids[node] = node_id
  slot_variables = _serialize_slot_variables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names)
  return checkpointable_objects, node_ids, slot_variables


def list_objects(root_checkpointable):
  """Traverse the object graph and list all accessible objects.

  Looks for `Checkpointable` objects which are dependencies of
  `root_checkpointable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_checkpointable`
  (i.e. if they would be saved with a checkpoint).

  Args:
    root_checkpointable: A `Checkpointable` object whose dependencies should be
      flattened.
  Returns:
    A flat list of objects.
  """
  checkpointable_objects, _, _ = find_objects(root_checkpointable)
  return checkpointable_objects


def gather_initializers(root_checkpointable):
  """Traverse the object graph and find initialization ops.

  Looks for `Checkpointable` objects which are dependencies of
  `root_checkpointable` and which have an `initializer` property. Includes
  initializers for slot variables only if the variable they are slotting for and
  the optimizer are dependencies of `root_checkpointable` (i.e. if they would be
  saved with a checkpoint).

  Args:
    root_checkpointable: A `Checkpointable` object to gather initializers for.
  Returns:
    A list of initialization ops.
  """
  checkpointable_objects = list_objects(root_checkpointable)
  return [c.initializer for c in checkpointable_objects
          if hasattr(c, "initializer") and c.initializer is not None]


@tf_contextlib.contextmanager
def capture_dependencies(template):
  """Capture variables created within this scope as `Template` dependencies.

  Requires that `template.variable_scope` is active.

  This scope is intended as a compatibility measure, allowing a checkpointable
  object to add dependencies on variables created in a block of code which is
  not aware of object-based saving (and instead uses variable names
  heavily). This is how `Template` objects add dependencies on variables and
  sub-`Template`s. Where possible, use `tf.make_template` directly.

  Args:
    template: The `Template` object to register dependencies with.

  Yields:
    None (when used as a context manager).
  """
  name_prefix = template.variable_scope.name

  def _checkpointable_custom_creator(next_creator, name, initial_value,
                                     checkpointable_parent=None, **kwargs):
    """A variable creation hook which adds Checkpointable dependencies.

    Set for example during a `Template`'s first wrapped function
    execution. Ensures that (a) `template` depends on any checkpointable
    objects using their own `capture_dependencies` scope inside this scope which
    create variables, and (b) that any variables not in a more deeply nested
    scope are added as dependencies directly.

    The `checkpointable_parent` argument is passed between custom creators but
    ignored when the variable object itself is created. This argument indicates
    (if not `None`) that a more deeply nested scope has already added the
    variable as a dependency, and that parent scopes should add a dependency on
    that object rather than on the variable directly.

    Args:
      next_creator: See `variable_scope.variable_creator_scope`; the next
        creator in the chain.
      name: The (full, scope-influenced) name of the variable. The `name_prefix`
        itself is stripped for the purposes of object-based dependency tracking,
        but scopes opened within this scope are respected.
      initial_value: See `variable_scope.variable_creator_scope`. Taken
        explicitly so the argument can be re-named and used with
        `Checkpointable._add_variable_with_custom_getter`.
      checkpointable_parent: If not None, a more deeply nested checkpointable
        object and its name prefix which were passed to `capture_dependencies`
        to add a dependency on (rather than depending on the variable directly).
      **kwargs: Passed through to the next creator.

    Returns:
      The output of `next_creator`: the fetched/created variable object.
    """
    def _call_next_creator_renaming_initializer(initializer, **inner_kwargs):
      inner_kwargs.pop("name")  # Ignored; this is the scope-stripped name which
      # we don't want to propagate.
      return next_creator(
          initial_value=initializer,
          name=name,
          **inner_kwargs)
    if name is not None and name.startswith(name_prefix):
      scope_stripped_name = name[len(name_prefix) + 1:]
      if not checkpointable_parent:
        return template._add_variable_with_custom_getter(  # pylint: disable=protected-access
            initializer=initial_value,
            name=scope_stripped_name,
            getter=_call_next_creator_renaming_initializer,
            # Disable error checking for Checkpointable. Exceptions are instead
            # raised if necessary when the object-based saver tries to
            # save/restore the object.
            overwrite=True,
            checkpointable_parent=(template, name_prefix),
            **kwargs)
      else:
        parent_object, parent_name_prefix = checkpointable_parent
        template._track_checkpointable(  # pylint: disable=protected-access
            parent_object,
            name=parent_name_prefix[len(name_prefix) + 1:],
            overwrite=True)
    return next_creator(
        name=name, initial_value=initial_value,
        checkpointable_parent=(template, name_prefix), **kwargs)

  with variable_scope.variable_creator_scope(_checkpointable_custom_creator):
    yield


class _LoadStatus(object):
  """Abstract base for load status callbacks."""

  @abc.abstractmethod
  def assert_consumed(self):
    """Raises an exception unless a non-trivial restoration has completed."""
    pass

  @abc.abstractmethod
  def assert_existing_objects_matched(self):
    """Raises an exception unless existing Python objects have been matched."""
    pass

  @abc.abstractmethod
  def assert_nontrivial_match(self):
    """Raises an exception if only the root object matched."""
    pass

  @abc.abstractmethod
  def run_restore_ops(self, session=None):
    """Runs restore ops from the checkpoint. Requires a valid checkpoint."""
    pass

  @abc.abstractmethod
  def initialize_or_restore(self, session=None):
    """Runs restore ops from the checkpoint, or initializes variables."""
    pass


def streaming_restore(status, session=None):
  """When graph building, runs restore ops as soon as they come in.

  Args:
    status: A _LoadStatus objects from an object-based saver's
      restore(). Streaming restore from name-based checkpoints is not currently
      supported.
    session: A session to run new restore ops in.
  """
  if context.executing_eagerly():
    # Streaming restore is the default/only behavior when executing eagerly.
    return
  if session is None:
    session = ops.get_default_session()
  if isinstance(status, NameBasedSaverStatus):
    raise NotImplementedError(
        "Streaming restore not supported from name-based checkpoints. File a "
        "feature request if this limitation bothers you.")
  status.run_restore_ops(session=session)
  # pylint: disable=protected-access
  status._checkpoint.new_restore_ops_callback = (
      lambda ops: session.run(ops, feed_dict=status._feed_dict))
  # pylint: enable=protected-access


class CheckpointLoadStatus(_LoadStatus):
  """Checks the status of checkpoint loading and manages restore ops.

  Returned from `Saver.restore`. Since `restore` may defer the loading of values
  in the checkpoint which don't yet have corresponding Python objects,
  `CheckpointLoadStatus` provides a callback to verify that checkpoint loading
  is complete (`assert_consumed`).

  When graph building, `restore` does not run restore ops itself since their
  creation may be deferred. The `run_restore_ops` method must be called once all
  Python objects with values to restore have been created and added to the
  dependency graph (this does not necessarily have to be the whole checkpoint;
  calling `run_restore_ops` while `assert_consumed` fails is supported and will
  partially restore the checkpoint).

  See `Saver.restore` for usage examples.
  """

  def __init__(self, checkpoint, feed_dict, root_checkpointable):
    self._checkpoint = checkpoint
    self._feed_dict = feed_dict
    self._root_checkpointable = root_checkpointable

  def assert_consumed(self):
    """Asserts that all objects in the checkpoint have been created/matched.

    Returns:
      `self` for chaining.
    Raises:
      AssertionError: If there are any Python objects in the dependency graph
        which have not been restored from this checkpoint or a later `restore`,
        or if there are any checkpointed values which have not been matched to
        Python objects.
    """
    self.assert_existing_objects_matched()
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      checkpointable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if checkpointable is None:
        raise AssertionError("Unresolved object in checkpoint: %s" % (node,))
    if self._checkpoint.slot_restorations:
      # Sanity check; this collection should be clear if everything has been
      # restored.
      raise AssertionError("Unresolved slot restorations: %s" % (
          self._checkpoint.slot_restorations,))
    if self._checkpoint.unused_attributes:
      raise AssertionError(
          ("Unused attributes in these objects (the attributes exist in the "
           "checkpoint but not in the objects): %s") % (
               list(self._checkpoint.unused_attributes.items()),))
    return self

  def assert_existing_objects_matched(self):
    """Asserts that checkpointable Python objects have been matched.

    Note that this is a weaker assertion than `assert_consumed`. It will only
    fail for existing Python objects which are (transitive) dependencies of the
    root object and which do not have an entry in the checkpoint.

    It will not fail, for example, if a `tf.keras.Layer` object has not yet been
    built and so has not created any `tf.Variable` objects.

    Returns:
      `self` for chaining.

    Raises:
      AssertionError: If a Python object exists in the transitive dependencies
        of the root object but does not have a value in the checkpoint.
    """
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      checkpointable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if (checkpointable is not None
          and checkpointable._update_uid < self._checkpoint.restore_uid):  # pylint: disable=protected-access
        raise AssertionError(
            "Object not assigned a value from checkpoint: %s" % (node,))
    for checkpointable_object in list_objects(self._root_checkpointable):
      # Remove data structures that do not contain any variables from
      # restoration checks.
      if (isinstance(checkpointable_object,
                     data_structures.CheckpointableDataStructure) and
          not checkpointable_object._checkpoint_dependencies):
        continue
      self._checkpoint.all_python_objects.add(checkpointable_object)
    unused_python_objects = (
        _ObjectIdentitySet(self._checkpoint.all_python_objects)
        - _ObjectIdentitySet(self._checkpoint.object_by_proto_id.values()))
    if unused_python_objects:
      raise AssertionError(
          ("Some Python objects were not bound to checkpointed values, likely "
           "due to changes in the Python program: %s")
          % (list(unused_python_objects),))
    return self

  def assert_nontrivial_match(self):
    """Raises an exception if only the root object matched."""
    for checkpointable_object in list_objects(self._root_checkpointable):
      self._checkpoint.all_python_objects.add(checkpointable_object)
    if len(self._checkpoint.object_by_proto_id) <= 1:
      unused_python_objects = (
          _ObjectIdentitySet(self._checkpoint.all_python_objects)
          - _ObjectIdentitySet(self._checkpoint.object_by_proto_id.values()))
      if unused_python_objects:
        raise AssertionError(
            ("Nothing except the root object matched a checkpointed value. "
             "Typically this means that the checkpoint does not match the "
             "Python program. The following objects have no matching "
             "checkpointed value: %s") % (list(unused_python_objects),))
      else:
        raise AssertionError(
            "Nothing to load. No dependencies have been added to %s yet." % (
                self._root_checkpointable,))
    return self

  def run_restore_ops(self, session=None):
    """Run operations to restore objects in the dependency graph."""
    if context.executing_eagerly():
      return  # Run eagerly
    if session is None:
      session = ops.get_default_session()
    session.run(self._checkpoint.restore_ops, feed_dict=self._feed_dict)

  def initialize_or_restore(self, session=None):
    """Run operations to initialize or restore objects in the dependency graph.

    Any objects in the dependency graph which have initializers but are not in
    the checkpoint will have those initializers run, unless those variables are
    being restored by a later call to `tf.train.Checkpoint.restore()`.

    This method has a sibling in `InitializationOnlyStatus` which instead
    initializes variables. That type is returned if no checkpoint is specified
    in `Saver.restore`.

    Args:
      session: The session to run init/restore ops in. If `None`, uses the
        default session.
    """
    if context.executing_eagerly():
      return  # Initialization and restoration ops are run eagerly
    if session is None:
      session = ops.get_default_session()
    all_objects = list_objects(self._root_checkpointable)
    already_initialized_objects = _ObjectIdentitySet(
        self._checkpoint.object_by_proto_id.values())
    initializers_for_non_restored_variables = [
        c.initializer for c in all_objects
        if hasattr(c, "initializer")
        and c not in already_initialized_objects
        and (getattr(c, "_update_uid", self._checkpoint.restore_uid - 1)
             < self._checkpoint.restore_uid)]
    self.run_restore_ops(session=session)
    session.run(initializers_for_non_restored_variables)


class InitializationOnlyStatus(_LoadStatus):
  """Returned from `Saver.restore` when no checkpoint has been specified.

  Objects of this type have the same `assert_consumed` method as
  `CheckpointLoadStatus`, but it always fails. However,
  `initialize_or_restore` works on objects of both types, and will
  initialize variables in `InitializationOnlyStatus` objects or restore them
  otherwise.
  """

  def __init__(self, root_checkpointable, restore_uid):
    self._restore_uid = restore_uid
    self._root_checkpointable = root_checkpointable

  def assert_consumed(self):
    """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")

  def assert_existing_objects_matched(self):
    """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")

  def assert_nontrivial_match(self):
    """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
    raise AssertionError(
        "No checkpoint specified (save_path=None); nothing is being restored.")

  def run_restore_ops(self, session=None):
    """For consistency with `CheckpointLoadStatus`.

    Use `initialize_or_restore` for initializing if no checkpoint was passed
    to `Saver.restore` and restoring otherwise.

    Args:
      session: Not used.
    """
    raise AssertionError(
        "No checkpoint specified, so no restore ops are available "
        "(save_path=None to Saver.restore).")

  def initialize_or_restore(self, session=None):
    """Runs initialization ops for variables.

    Objects which would be saved by `Saver.save` will be initialized, unless
    those variables are being restored by a later call to
    `tf.train.Checkpoint.restore()`.

    This method does nothing when executing eagerly (initializers get run
    eagerly).

    Args:
      session: The session to run initialization ops in. If `None`, uses the
        default session.
    """
    if context.executing_eagerly():
      return  # run eagerly
    if session is None:
      session = ops.get_default_session()
    checkpointable_objects = list_objects(self._root_checkpointable)
    initializers = [
        c.initializer for c in checkpointable_objects
        if hasattr(c, "initializer") and c.initializer is not None
        and (getattr(c, "_update_uid", self._restore_uid - 1)
             < self._restore_uid)]
    session.run(initializers)


_DEPRECATED_RESTORE_INSTRUCTIONS = (
    "Restoring a name-based tf.train.Saver checkpoint using the object-based "
    "restore API. This mode uses global names to match variables, and so is "
    "somewhat fragile. It also adds new restore ops to the graph each time it "
    "is called when graph building. Prefer re-encoding training checkpoints in "
    "the object-based format: run save() on the object-based saver (the same "
    "one this message is coming from) and use that checkpoint in the future.")


class NameBasedSaverStatus(_LoadStatus):
  """Status for loading a name-based training checkpoint."""

  # Ideally this deprecation decorator would be on the class, but that
  # interferes with isinstance checks.
  @deprecation.deprecated(
      date=None, instructions=_DEPRECATED_RESTORE_INSTRUCTIONS)
  def __init__(self, checkpoint, root_checkpointable):
    self._checkpoint = checkpoint
    self._root_checkpointable = root_checkpointable

  def assert_consumed(self):
    """Raises an exception if any variables/objects are unmatched."""
    unused_attributes = dict(self._checkpoint.unused_attributes)
    if unused_attributes:
      raise AssertionError(
          "Some objects had attributes which were not restored: %s"
          % (unused_attributes,))
    for checkpointable in list_objects(self._root_checkpointable):
      # pylint: disable=protected-access
      checkpointable._maybe_initialize_checkpointable()
      if checkpointable._update_uid < self._checkpoint.restore_uid:
        raise AssertionError("Object not restored: %s" % (checkpointable,))
      # pylint: enable=protected-access
    return self

  def assert_existing_objects_matched(self):
    """Raises an exception if currently created objects are unmatched."""
    # For name-based checkpoints there's no object information in the
    # checkpoint, so there's no distinction between
    # assert_existing_objects_matched and assert_consumed (and both are less
    # useful since we don't touch Python objects or Python state).
    return self.assert_consumed()

  def assert_nontrivial_match(self):
    """Raises an exception if currently created objects are unmatched."""
    # For name-based checkpoints there's no object information in the
    # checkpoint, so there's no distinction between
    # assert_nontrivial_match and assert_consumed (and both are less
    # useful since we don't touch Python objects or Python state).
    return self.assert_consumed()

  def _gather_saveable_objects(self):
    """Walk the object graph, using global names for SaveableObjects."""
    objects = list_objects(self._root_checkpointable)
    saveable_objects = []
    for checkpointable in objects:
      # pylint: disable=protected-access
      checkpointable._maybe_initialize_checkpointable()
      if checkpointable._update_uid < self._checkpoint.restore_uid:
        checkpointable._update_uid = self._checkpoint.restore_uid
      else:
        continue
      # pylint: enable=protected-access
      saveable_objects.extend(
          self._checkpoint.globally_named_object_attributes(
              checkpointable))
    return saveable_objects

  def run_restore_ops(self, session=None):
    """Load the name-based training checkpoint using a new `tf.train.Saver`."""
    if context.executing_eagerly():
      return  # Nothing to do, variables are restored on creation.
    if session is None:
      session = ops.get_default_session()
    with ops.device("/cpu:0"):
      saveables = self._gather_saveable_objects()
      v1_saver_lib.Saver(saveables).restore(
          sess=session, save_path=self._checkpoint.save_path)

  def initialize_or_restore(self, session=None):
    """Alias for `run_restore_ops`."""
    self.run_restore_ops(session=session)


class _SessionWithFeedDictAdditions(session_lib.SessionInterface):
  """Pretends to be a session, inserts extra feeds on run()."""

  def __init__(self, session, feed_additions):
    self._wrapped_session = session
    self._feed_additions = feed_additions

  def run(self, fetches, feed_dict=None, **kwargs):
    if feed_dict is None:
      feed_dict = {}
    else:
      feed_dict = feed_dict.copy()
    feed_dict.update(self._feed_additions)
    return self._wrapped_session.run(
        fetches=fetches, feed_dict=feed_dict, **kwargs)


class CheckpointableSaver(object):
  """Saves and restores a `Checkpointable` object and its dependencies.

  See `Checkpointable` for details of dependency management. `Saver` wraps
  `tf.train.Saver` for saving, including extra information about the graph of
  dependencies between Python objects. When restoring, it uses this information
  about the save-time dependency graph to more robustly match objects with their
  checkpointed values. When executing eagerly, it supports restoring variables
  on object creation (see `Saver.restore`).

  Values in a checkpoint are mapped to `Checkpointable` Python objects
  (`Variable`s, `Optimizer`s, `Layer`s) based on the names provided when the
  checkpoint was written. To avoid breaking existing checkpoints when modifying
  a class, dependency names (the names of attributes to which `Checkpointable`
  objects are assigned) may not change. These names are local to objects, in
  contrast to the `Variable.name`-based save/restore from `tf.train.Saver`, and
  so allow additional program transformations.
  """

  def __init__(self, root_checkpointable):
    """Configure saving.

    Args:
      root_checkpointable: The root of the object graph to save/restore. This
        object and all of its dependencies are saved in the checkpoint. When
        restoring, objects are matched and restored starting from this root.
    """
    # Allow passing in a weak reference to avoid reference cycles when
    # `Checkpointable` objects save themselves.
    self._root_checkpointable_ref = root_checkpointable
    # The file prefix placeholder is created lazily when graph building (and not
    # at all when executing eagerly) to avoid creating ops in the constructor
    # (when they may never be necessary).
    self._file_prefix_placeholder = None

    # Op caching for save
    self._object_graph_feed_tensor = None
    self._last_save_object_graph = None
    self._file_prefix_feed_tensor = None
    self._cached_save_operation = None

    # Op caching for restore, shared between _CheckpointRestoreCoordinators
    self._restore_op_cache = {}

    if context.executing_eagerly():
      # SaveableObjects are always recreated when executing eagerly.
      self._saveable_object_cache = None
    else:
      # Maps Checkpointable objects -> attribute names -> list(SaveableObjects),
      # to avoid re-creating SaveableObjects when graph building.
      self._saveable_object_cache = _ObjectIdentityWeakKeyDictionary()

  @property
  def _root_checkpointable(self):
    if isinstance(self._root_checkpointable_ref, weakref.ref):
      derefed = self._root_checkpointable_ref()
      assert derefed is not None
      return derefed
    else:
      return self._root_checkpointable_ref

  def _gather_saveables(
      self, object_graph_tensor=None, saveable_object_cache=None):
    """Wraps _serialize_object_graph to include the object graph proto."""
    assert ((object_graph_tensor is None and saveable_object_cache is None)
            or (object_graph_tensor is not None
                and saveable_object_cache is not None))
    (named_saveable_objects, graph_proto,
     feed_additions) = _serialize_object_graph(
         self._root_checkpointable,
         saveables_cache=saveable_object_cache)
    if object_graph_tensor is None:
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
    else:
      feed_additions.update(
          {object_graph_tensor: graph_proto.SerializeToString()})
    assert base.OBJECT_GRAPH_PROTO_KEY not in named_saveable_objects
    named_saveable_objects.append(
        base.NoRestoreSaveable(
            tensor=object_graph_tensor,
            name=base.OBJECT_GRAPH_PROTO_KEY))
    return named_saveable_objects, graph_proto, feed_additions

  def gather_objects(self, object_map=None, to_graph=None):
    """Creates SaveableObjects with the current object graph frozen."""
    checkpointable_objects, path_to_root = (
        _breadth_first_checkpointable_traversal(self._root_checkpointable))
    if to_graph:
      target_context = to_graph.as_default
    else:
      target_context = ops.NullContextmanager
    with target_context():
      named_saveable_objects, graph_proto, _ = _serialize_gathered_objects(
          checkpointable_objects,
          path_to_root,
          saveables_cache=None,
          object_map=object_map)
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
      named_saveable_objects.append(
          base.NoRestoreSaveable(
              tensor=object_graph_tensor,
              name=base.OBJECT_GRAPH_PROTO_KEY))
    return named_saveable_objects

  def freeze(self, object_map=None, to_graph=None):
    named_saveable_objects = self.gather_objects(
        object_map=object_map, to_graph=to_graph)
    return functional_saver.Saver(named_saveable_objects)

  def _save_cached_when_graph_building(
      self,
      file_prefix,
      object_graph_tensor=None,
      saveable_object_cache=None):
    """Create or retrieve save ops.

    When graph building, `saveable_object_cache` will typically be non-`None`,
    meaning that existing `SaveableObject`s are re-used across calls to
    `_prepare_save` even if the object graph has grown. This avoids
    unnecessarily re-creating save ops.

    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      saveable_object_cache: A dictionary; if specified, used to cache
        `SaveableObject`s.

    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
    (named_saveable_objects, graph_proto,
     feed_additions) = self._gather_saveables(
         object_graph_tensor=object_graph_tensor,
         saveable_object_cache=saveable_object_cache)
    if (self._last_save_object_graph != graph_proto
        # When executing eagerly, we need to re-create SaveableObjects each time
        # save() is called so they pick up new Tensors passed to their
        # constructors. That means the Saver needs to be copied with a new
        # var_list.
        or context.executing_eagerly()):
      saver = functional_saver.Saver(named_saveable_objects)
      with ops.device("/cpu:0"):
        self._cached_save_operation = saver.save(file_prefix)
      self._last_save_object_graph = graph_proto
    return self._cached_save_operation, feed_additions

  def save(self, file_prefix, checkpoint_number=None, session=None):
    """Save a training checkpoint.

    The saved checkpoint includes variables created by this object and any
    Checkpointable objects it depends on at the time `Saver.save()` is called.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `checkpoint_number`, if provided.
      checkpoint_number: An integer variable or Tensor, used to number
        checkpoints. Typically this value is saved along with other variables in
        training checkpoints, which will happen automatically if it was created
        by `root_checkpointable` or one of its dependencies (via
        `Checkpointable._add_variable`).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      The full path to the checkpoint.
    """
    feed_dict = {}
    graph_building = not context.executing_eagerly()
    if checkpoint_number:
      file_prefix = "%s-%d" % (file_prefix, checkpoint_number)
    if graph_building:
      if self._object_graph_feed_tensor is None:
        with ops.device("/cpu:0"):
          self._object_graph_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
          self._file_prefix_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
      object_graph_tensor = self._object_graph_feed_tensor
      file_prefix_tensor = self._file_prefix_feed_tensor
      feed_dict[file_prefix_tensor] = file_prefix
    else:
      with ops.device("/cpu:0"):
        file_prefix_tensor = constant_op.constant(
            file_prefix, dtype=dtypes.string)
      object_graph_tensor = None

    file_io.recursive_create_dir(os.path.dirname(file_prefix))
    save_path, new_feed_additions = self._save_cached_when_graph_building(
        file_prefix=file_prefix_tensor,
        object_graph_tensor=object_graph_tensor,
        saveable_object_cache=self._saveable_object_cache)
    if new_feed_additions:
      feed_dict.update(new_feed_additions)
    if not graph_building:
      session = None
    elif session is None:
      session = ops.get_default_session()

    if session:
      save_path = session.run(save_path, feed_dict=feed_dict)
    else:
      save_path = save_path.numpy()
    return save_path

  def restore(self, save_path):
    """Restore a training checkpoint.

    Restores `root_checkpointable` and any objects that it tracks
    (transitive). Either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added to the `root_checkpointable` passed to the
    constructor after this call will be matched if they have a corresponding
    object in the checkpoint.

    When building a graph, restorations are added to the graph but not run.

    To disallow deferred loading, assert immediately that all checkpointed
    variables have been matched to variable objects:

    ```python
    saver = Saver(root)
    saver.restore(path).assert_consumed()
    ```

    An exception will be raised unless every object was matched and its
    variables already exist.

    When graph building, `assert_consumed()` indicates that all of the restore
    ops which will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` function of the status object:

    ```python
    saver.restore(path).assert_consumed().run_restore_ops()
    ```

    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.

    Name-based `tf.train.Saver` checkpoints can be loaded using this
    method. There is no deferred loading, and names are used to match
    variables. No restore ops are created/run until `run_restore_ops()` or
    `initialize_or_restore()` are called on the returned status object, even
    when executing eagerly. Re-encode name-based checkpoints using this
    object-based `Saver.save` as soon as possible.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency
        graph. If the checkpoint was written by the name-based `tf.train.Saver`,
        names are used to match variables.

    Returns:
      A load status object, which can be used to make assertions about the
      status of checkpoint restoration and run initialization/restore ops
      (of type `CheckpointLoadStatus`, or `InitializationOnlyStatus` if
      `save_path` is `None`).

      If `save_path` points to a name-based checkpoint, a `NameBasedSaverStatus`
      object is returned which runs restore ops from a name-based saver.
    """
    if save_path is None:
      return InitializationOnlyStatus(self._root_checkpointable, ops.uid())
    reader = pywrap_tensorflow.NewCheckpointReader(save_path)
    graph_building = not context.executing_eagerly()
    if graph_building:
      dtype_map = None
    else:
      dtype_map = reader.get_variable_to_dtype_map()
    try:
      object_graph_string = reader.get_tensor(
          base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError:
      # The object graph proto does not exist in this checkpoint. Try the
      # name-based compatibility mode.
      restore_coordinator = _NameBasedRestoreCoordinator(
          save_path=save_path, dtype_map=dtype_map)
      if not graph_building:
        for existing_checkpointable in list_objects(self._root_checkpointable):
          # pylint: disable=protected-access
          existing_checkpointable._maybe_initialize_checkpointable()
          existing_checkpointable._name_based_restores.add(restore_coordinator)
          existing_checkpointable._name_based_attribute_restore(
              restore_coordinator)
          # pylint: enable=protected-access
      return NameBasedSaverStatus(
          restore_coordinator, root_checkpointable=self._root_checkpointable)

    if graph_building:
      if self._file_prefix_placeholder is None:
        with ops.device("/cpu:0"):
          self._file_prefix_placeholder = constant_op.constant("model")
      file_prefix_tensor = self._file_prefix_placeholder
      file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
    else:
      with ops.device("/cpu:0"):
        file_prefix_tensor = constant_op.constant(save_path)
      file_prefix_feed_dict = None
    object_graph_proto = (
        checkpointable_object_graph_pb2.CheckpointableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    checkpoint = _CheckpointRestoreCoordinator(
        object_graph_proto=object_graph_proto,
        save_path=save_path,
        save_path_tensor=file_prefix_tensor,
        restore_op_cache=self._restore_op_cache,
        saveable_object_cache=self._saveable_object_cache)
    base._CheckpointPosition(  # pylint: disable=protected-access
        checkpoint=checkpoint, proto_id=0).restore(self._root_checkpointable)
    load_status = CheckpointLoadStatus(
        checkpoint,
        root_checkpointable=self._root_checkpointable,
        feed_dict=file_prefix_feed_dict)
    return load_status


def frozen_saver(root_checkpointable):
  """Creates a static `tf.train.Saver` from a checkpointable object.

  The returned `Saver` saves object-based checkpoints, but these checkpoints
  will no longer reflect structural changes to the object graph, only changes to
  the values of `Variable`s added as dependencies of the root object before
  `freeze` was called.

  `restore` works on the returned `Saver`, but requires that the object graph of
  the checkpoint being loaded exactly matches the object graph when `freeze` was
  called. This is in contrast the object-based restore performed by
  `tf.train.Checkpoint` which attempts a fuzzy matching between a checkpoint's
  object graph and the current Python object graph.

  Args:
    root_checkpointable: A checkpointable object to save.

  Returns:
    A `tf.train.Saver` which saves object-based checkpoints for the object graph
    frozen at the time `frozen_saver` was called.
  """
  return CheckpointableSaver(root_checkpointable).freeze()


@tf_export("train.Checkpoint")
class Checkpoint(tracking.Checkpointable):
  """Groups checkpointable objects, saving and restoring them.

  `Checkpoint`'s constructor accepts keyword arguments whose values are types
  that contain checkpointable state, such as `tf.train.Optimizer`
  implementations, `tf.Variable`, `tf.keras.Layer` implementations, or
  `tf.keras.Model` implementations. It saves these values with a checkpoint, and
  maintains a `save_counter` for numbering checkpoints.

  Example usage when graph building:

  ```python
  import tensorflow as tf
  import os

  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  train_op = optimizer.minimize( ... )
  status.assert_consumed()  # Optional sanity checks.
  with tf.Session() as session:
    # Use the Session to restore variables, or initialize them if
    # tf.train.latest_checkpoint returned None.
    status.initialize_or_restore(session)
    for _ in range(num_training_steps):
      session.run(train_op)
    checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  Example usage with eager execution enabled:

  ```python
  import tensorflow as tf
  import os

  tf.enable_eager_execution()

  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )  # Variables will be restored on creation.
  status.assert_consumed()  # Optional sanity checks.
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  `Checkpoint.save` and `Checkpoint.restore` write and read object-based
  checkpoints, in contrast to `tf.train.Saver` which writes and reads
  `variable.name` based checkpoints. Object-based checkpointing saves a graph of
  dependencies between Python objects (`Layer`s, `Optimizer`s, `Variable`s,
  etc.) with named edges, and this graph is used to match variables when
  restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables when executing
  eagerly. Prefer `tf.train.Checkpoint` over `tf.train.Saver` for new code.

  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their variables (e.g. "kernel" and "bias" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:

  ```python
  class Regress(tf.keras.Model):

    def __init__(self):
      super(Regress, self).__init__()
      self.input_transform = tf.keras.layers.Dense(10)
      # ...

    def call(self, inputs):
      x = self.input_transform(inputs)
      # ...
  ```

  This `Model` has a dependency named "input_transform" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.

  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """

  def __init__(self, **kwargs):
    """Group objects into a training checkpoint.

    Args:
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. Values must be checkpointable objects.
    Raises:
      ValueError: If objects in `kwargs` are not checkpointable.
    """
    super(Checkpoint, self).__init__()
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      if not isinstance(v, (base.CheckpointableBase, def_function.Function)):
        raise ValueError(
            ("`Checkpoint` was expecting a checkpointable object (an object "
             "derived from `CheckpointableBase`), got %s. If you believe this "
             "object should be checkpointable (i.e. it is part of the "
             "TensorFlow Python API and manages state), please open an issue.")
            % (v,))
      setattr(self, k, v)
    self._save_counter = None  # Created lazily for restore-on-create.
    self._save_assign_op = None
    self._saver = CheckpointableSaver(weakref.ref(self))

  def _maybe_create_save_counter(self):
    """Create a save counter if it does not yet exist."""
    if self._save_counter is None:
      # Initialized to 0 and incremented before saving.
      with ops.device("/cpu:0"):
        # add_variable creates a dependency named "save_counter"; NoDependency
        # prevents creating a second dependency named "_save_counter".
        self._save_counter = data_structures.NoDependency(
            add_variable(self, name="save_counter", initializer=0,
                         dtype=dtypes.int64))

  def write(self, file_prefix, session=None):
    """Writes a training checkpoint.

    The checkpoint includes variables created by this object and any
    checkpointable objects it depends on at the time `Checkpoint.write()` is
    called.

    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
    return compat.as_str(self._saver.save(
        file_prefix=file_prefix,
        session=session))

  @property
  def save_counter(self):
    """An integer variable which starts at zero and is incremented on save.

    Used to number checkpoints.

    Returns:
      The save counter variable.
    """
    self._maybe_create_save_counter()
    return self._save_counter

  def save(self, file_prefix, session=None):
    """Saves a training checkpoint and provides basic checkpoint management.

    The saved checkpoint includes variables created by this object and any
    checkpointable objects it depends on at the time `Checkpoint.save()` is
    called.

    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write`
    (`tf.contrib.checkpoint.CheckpointManager` for example).

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      The full path to the checkpoint.
    """
    graph_building = not context.executing_eagerly()
    if graph_building:
      if session is None:
        session = ops.get_default_session()
      if self._save_counter is None:
        # When graph building, if this is a new save counter variable then it
        # needs to be initialized before assign_add. This is only an issue if
        # restore() has not been called first.
        session.run(self.save_counter.initializer)
    if not graph_building or self._save_assign_op is None:
      with ops.colocate_with(self.save_counter):
        assign_op = self.save_counter.assign_add(1, read_value=True)
      if graph_building:
        self._save_assign_op = data_structures.NoDependency(assign_op)
    if graph_building:
      checkpoint_number = session.run(self._save_assign_op)
    else:
      checkpoint_number = assign_op.numpy()
    file_path = self.write("%s-%d" % (file_prefix, checkpoint_number),
                           session=session)
    checkpoint_management.update_checkpoint_state_internal(
        save_dir=os.path.dirname(file_prefix),
        model_checkpoint_path=file_path,
        all_model_checkpoint_paths=[file_path])
    return file_path

  def restore(self, save_path):
    """Restore a training checkpoint.

    Restores this `Checkpoint` and any objects it depends on.

    When executing eagerly, either assigns values immediately if variables to
    restore have been created already, or defers restoration until the variables
    are created. Dependencies added after this call will be matched if they have
    a corresponding object in the checkpoint (the restore request will queue in
    any checkpointable object waiting for the expected dependency to be added).

    When graph building, restoration ops are added to the graph but not run
    immediately.

    To ensure that loading is complete and no more assignments will take place,
    use the `assert_consumed()` method of the status object returned by
    `restore`:

    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path).assert_consumed()
    ```

    An exception will be raised if any Python objects in the dependency graph
    were not found in the checkpoint, or if any checkpointed values do not have
    a matching Python object.

    When graph building, `assert_consumed()` indicates that all of the restore
    ops that will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` method of the status object:

    ```python
    checkpoint.restore(path).assert_consumed().run_restore_ops()
    ```

    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.

    Name-based `tf.train.Saver` checkpoints can be loaded using this
    method. Names are used to match variables. No restore ops are created/run
    until `run_restore_ops()` or `initialize_or_restore()` are called on the
    returned status object when graph building, but there is restore-on-creation
    when executing eagerly. Re-encode name-based checkpoints using
    `tf.train.Checkpoint.save` as soon as possible.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency
        graph. If the checkpoint was written by the name-based `tf.train.Saver`,
        names are used to match variables.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration and run initialization/restore ops.

      The returned status object has the following methods:

      * `assert_consumed()`:
          Raises an exception if any variables/objects are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with `initialize_or_restore` or `run_restore_ops`.

      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.train.Optimizer` was saved but only the state required for
          inference is being loaded. This method returns the status object, and
          so may be chained with `initialize_or_restore` or `run_restore_ops`.

      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.

      * `initialize_or_restore(session=None)`:
          When graph building, runs variable initializers if `save_path` is
          `None`, but otherwise runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (variables are initialized or restored eagerly).

      * `run_restore_ops(session=None)`:
          When graph building, runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (restore operations are run eagerly). May only be
          called when `save_path` is not `None`.
    """
    status = self._saver.restore(save_path=save_path)
    # Create the save counter now so it gets initialized with other variables
    # when graph building. Creating it earlier would lead to double
    # initialization when executing eagerly.
    self._maybe_create_save_counter()
    return status
