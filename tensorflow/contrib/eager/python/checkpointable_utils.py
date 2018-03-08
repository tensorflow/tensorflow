"""Utilities for working with Checkpointable objects."""
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
import weakref

from tensorflow.contrib.eager.proto import checkpointable_object_graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import checkpointable as core_checkpointable
from tensorflow.python.training import checkpointable_utils as core_checkpointable_utils
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import deprecation


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
# Key where the object graph proto is saved in a TensorBundle
_OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"


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
    return resource_variable_ops.ResourceVariable(
        initial_value=initial_value,
        name=name,
        dtype=variable_dtype,
        **kwargs
    )


def add_variable(checkpointable, name, shape=None, dtype=dtypes.float32,
                 initializer=None):
  """Add a variable to a Checkpointable with no scope influence."""
  return checkpointable._add_variable_with_custom_getter(  # pylint: disable=protected-access
      name=name, shape=shape, dtype=dtype,
      initializer=initializer, getter=_default_getter)


def _breadth_first_checkpointable_traversal(root_checkpointable):
  """Find shortest paths to all variables owned by dependencies of root."""
  bfs_sorted = []
  to_visit = collections.deque([root_checkpointable])
  path_to_root = {root_checkpointable: ()}
  while to_visit:
    current_checkpointable = to_visit.popleft()
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
  slot_variables = {}
  for checkpointable in non_slot_objects:
    if isinstance(checkpointable, optimizer_lib.Optimizer):
      naming_scheme = _slot_variable_naming_for_optimizer(
          optimizer_path=object_names[checkpointable])
      slot_names = checkpointable.get_slot_names()
      for slot_name in slot_names:
        for original_variable_node_id, original_variable in enumerate(
            non_slot_objects):
          try:
            slot_variable = checkpointable.get_slot(
                original_variable, slot_name)
          except AttributeError:
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
              .Object.SlotVariableReference(
                  slot_name=slot_name,
                  original_variable_node_id=original_variable_node_id,
                  slot_variable_node_id=slot_variable_node_id))
          slot_variables.setdefault(checkpointable, []).append(
              slot_variable_proto)
  return slot_variables


def _serialize_checkpointables(
    checkpointable_objects, node_ids, object_names, slot_variables):
  """Name non-slot `Checkpointable`s and add them to `object_graph_proto`."""
  object_graph_proto = (
      checkpointable_object_graph_pb2.CheckpointableObjectGraph())
  named_saveables = {}

  for checkpoint_id, checkpointable in enumerate(checkpointable_objects):
    assert node_ids[checkpointable] == checkpoint_id
    object_proto = object_graph_proto.nodes.add()
    object_proto.slot_variables.extend(slot_variables.get(checkpointable, ()))
    object_name = object_names[checkpointable]
    for name, saveable in (
        checkpointable._gather_saveables_for_checkpoint().items()):  # pylint: disable=protected-access
      attribute = object_proto.attributes.add()
      attribute.name = name
      attribute.checkpoint_key = "%s/%s/%s" % (
          object_name, _OBJECT_ATTRIBUTES_NAME, _escape_local_name(name))
      # Figure out the name-based Saver's name for this variable.
      saver_dict = saver_lib.BaseSaverBuilder.OpListToDict(
          [saveable], convert_variable_to_tensor=False)
      attribute.full_name, = saver_dict.keys()
      named_saveables[attribute.checkpoint_key] = saveable

    for child in checkpointable._checkpoint_dependencies:  # pylint: disable=protected-access
      child_proto = object_proto.children.add()
      child_proto.node_id = node_ids[child.ref]
      child_proto.local_name = child.name

  return named_saveables, object_graph_proto


def _serialize_object_graph(root_checkpointable):
  """Determine checkpoint keys for variables and build a serialized graph.

  Non-slot variables are keyed based on a shortest path from the root saveable
  to the object which owns the variable (i.e. the one which called
  `Checkpointable._add_variable` to create it).

  Slot variables are keyed based on a shortest path to the variable being
  slotted for, a shortest path to their optimizer, and the slot name.

  Args:
    root_checkpointable: A `Checkpointable` object whose variables (including
      the variables of dependencies, recursively) should be saved.

  Returns:
    A tuple of (named_variables, object_graph_proto):
      named_variables: A dictionary mapping names to variable objects.
      object_graph_proto: A CheckpointableObjectGraph protocol buffer containing
        the serialized object graph and variable references.

  Raises:
    ValueError: If there are invalid characters in an optimizer's slot names.
  """
  checkpointable_objects, path_to_root = (
      _breadth_first_checkpointable_traversal(root_checkpointable))
  object_names = {
      obj: _object_prefix_from_path(path)
      for obj, path in path_to_root.items()}
  node_ids = {node: node_id for node_id, node
              in enumerate(checkpointable_objects)}
  slot_variables = _serialize_slot_variables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names)
  return _serialize_checkpointables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names,
      slot_variables=slot_variables)


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
  # TODO(allenl): Extract out gathering logic so the naming logic doesn't have
  # to run.
  checkpointable_objects, path_to_root = (
      _breadth_first_checkpointable_traversal(root_checkpointable))
  object_names = {
      obj: _object_prefix_from_path(path)
      for obj, path in path_to_root.items()}
  node_ids = {node: node_id for node_id, node
              in enumerate(checkpointable_objects)}
  _serialize_slot_variables(
      checkpointable_objects=checkpointable_objects,
      node_ids=node_ids,
      object_names=object_names)
  return [c.initializer for c in checkpointable_objects
          if hasattr(c, "initializer") and c.initializer is not None]


class _NoRestoreSaveable(saver_lib.BaseSaverBuilder.SaveableObject):

  def __init__(self, tensor, name):
    spec = saver_lib.BaseSaverBuilder.SaveSpec(tensor, "", name)
    super(_NoRestoreSaveable, self).__init__(tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    return control_flow_ops.no_op()


class _LoadStatus(object):
  """Abstract base for load status callbacks."""

  @abc.abstractmethod
  def assert_consumed(self):
    """Raises an exception unless a non-trivial restoration has completed."""
    pass

  @abc.abstractmethod
  def run_restore_ops(self, session=None):
    """Runs restore ops from the checkpoint. Requires a valid checkpoint."""
    pass

  @abc.abstractmethod
  def initialize_or_restore(self, session=None):
    """Runs restore ops from the checkpoint, or initializes variables."""
    pass


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

  def __init__(self, checkpoint, feed_dict):
    self._checkpoint = checkpoint
    self._feed_dict = feed_dict

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
    for node_id, node in enumerate(self._checkpoint.object_graph_proto.nodes):
      checkpointable = self._checkpoint.object_by_proto_id.get(node_id, None)
      if checkpointable is None:
        raise AssertionError("Unresolved object in checkpoint: %s" % (node,))
      if checkpointable._update_uid < self._checkpoint.restore_uid:  # pylint: disable=protected-access
        raise AssertionError(
            "Object not assigned a value from checkpoint: %s" % (node,))
    if self._checkpoint.slot_restorations:
      # Sanity check; this collection should be clear if everything has been
      # restored.
      raise AssertionError("Unresolved slot restorations: %s" % (
          self._checkpoint.slot_restorations,))
    if self._checkpoint.unused_attributes:
      raise AssertionError(
          ("Unused attributes in these objects (the attributes exist in the "
           "checkpoint but not in the objects): %s") % (
               self._checkpoint.unused_attributes.items(),))
    return self

  def run_restore_ops(self, session=None):
    """Run operations to restore objects in the dependency graph."""
    if context.executing_eagerly():
      return  # Run eagerly
    if session is None:
      session = ops.get_default_session()
    session.run(self._checkpoint.restore_ops, feed_dict=self._feed_dict)

  def initialize_or_restore(self, session=None):
    """Alias for `run_restore_ops`.

    This method has a sibling in `InitializationOnlyStatus` which instead
    initializes variables. That type is returned if no checkpoint is specified
    in `Saver.restore`.

    Args:
      session: The session to run restore ops in. If `None`, uses the default
        session.
    """
    self.run_restore_ops(session=session)


class InitializationOnlyStatus(_LoadStatus):
  """Returned from `Saver.restore` when no checkpoint has been specified.

  Objects of this type have the same `assert_consumed` method as
  `CheckpointLoadStatus`, but it always fails. However,
  `initialize_or_restore` works on objects of both types, and will
  initialize variables in `InitializationOnlyStatus` objects or restore them
  otherwise.
  """

  def __init__(self, root_checkpointable):
    self._root_checkpointable = root_checkpointable

  def assert_consumed(self):
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

    Only objects which would be saved by `Saver.save` will be initialized. See
    `gather_initializers` for details.

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
    session.run(gather_initializers(self._root_checkpointable))


_DEPRECATED_RESTORE_INSTRUCTIONS = (
    "Restoring a name-based tf.train.Saver checkpoint using the object-based "
    "restore API. This mode uses global names to match variables, and so is "
    "somewhat fragile. It also adds new restore ops to the graph each time it "
    "is called. Prefer re-encoding training checkpoints in the object-based "
    "format: run save() on the object-based saver (the same one this message "
    "is coming from) and use that checkpoint in the future.")


class NameBasedSaverStatus(_LoadStatus):
  """Status for loading a name-based training checkpoint."""

  def __init__(self, object_saver, save_path):
    self._object_saver = object_saver
    self._save_path = save_path

  def assert_consumed(self):
    """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
    raise AssertionError(
        "Restoring a name-based checkpoint. No load status is available.")

  @deprecation.deprecated(
      date=None, instructions=_DEPRECATED_RESTORE_INSTRUCTIONS)
  def run_restore_ops(self, session=None):
    """Load the name-based training checkpoint using a new `tf.train.Saver`."""
    if session is None and not context.executing_eagerly():
      session = ops.get_default_session()
    saver_lib.Saver(self._object_saver._global_variable_names()).restore(  # pylint: disable=protected-access
        sess=session, save_path=self._save_path)

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
    if not context.executing_eagerly():
      with ops.device("/cpu:0"):
        self._file_prefix_placeholder = constant_op.constant("model")
    else:
      self._file_prefix_placeholder = None

    # Op caching for save
    self._object_graph_feed_tensor = None
    self._last_save_object_graph = None
    self._last_save_saver = None

    # Op caching for restore
    self._object_graph_restore_tensor = None
    self._last_restore_object_graph = None
    self._last_restore_checkpoint = None

  @property
  def _root_checkpointable(self):
    if isinstance(self._root_checkpointable_ref, weakref.ref):
      derefed = self._root_checkpointable_ref()
      assert derefed is not None
      return derefed
    else:
      return self._root_checkpointable_ref

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
    named_variables, graph_proto = _serialize_object_graph(
        self._root_checkpointable)
    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
      if session is None:
        session = ops.get_default_session()
      if self._object_graph_feed_tensor is None:
        with ops.device("/cpu:0"):
          self._object_graph_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
      object_graph_tensor = self._object_graph_feed_tensor
      feed_additions = {object_graph_tensor: graph_proto.SerializeToString()}
    else:
      session = None
      with ops.device("/cpu:0"):
        object_graph_tensor = constant_op.constant(
            graph_proto.SerializeToString(), dtype=dtypes.string)
      feed_additions = None
    assert _OBJECT_GRAPH_PROTO_KEY not in named_variables
    named_variables[_OBJECT_GRAPH_PROTO_KEY] = _NoRestoreSaveable(
        tensor=object_graph_tensor,
        name=_OBJECT_GRAPH_PROTO_KEY)
    if not in_graph_mode or self._last_save_object_graph != graph_proto:
      if self._last_save_object_graph is not None and in_graph_mode:
        raise NotImplementedError(
            "Using a single Saver to save a mutated object graph is not "
            "currently supported when graph building. Use a different Saver "
            "when the object graph changes (save ops will be duplicated), or "
            "file a feature request if this limitation bothers you.")
      saver = saver_lib.Saver(var_list=named_variables)
      if in_graph_mode:
        self._last_save_saver = saver
        self._last_save_object_graph = graph_proto
    else:
      saver = self._last_save_saver
    with ops.device("/cpu:0"):
      save_path = saver.save(
          sess=_SessionWithFeedDictAdditions(
              session=session, feed_additions=feed_additions),
          save_path=file_prefix,
          write_meta_graph=False,
          global_step=checkpoint_number)
    return save_path

  def _global_variable_names(self):
    """Generate a `tf.train.Saver`-style `var_list` using `variable.name`s."""
    named_saveables, graph_proto = _serialize_object_graph(
        self._root_checkpointable)
    saver_names = {}
    for object_proto in graph_proto.nodes:
      for attribute_proto in object_proto.attributes:
        saver_names[attribute_proto.full_name] = named_saveables[
            attribute_proto.checkpoint_key]
    return saver_names

  def restore(self, save_path, session=None):
    """Restore a training checkpoint.

    Restores `root_checkpointable` and any objects that it tracks
    (transitive). Either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added to the `root_checkpointable` passed to the
    constructor after this call will be matched if they have a corresponding
    object in the checkpoint.

    When building a graph, restorations are added to the graph but not run. A
    session is required to retrieve checkpoint metadata.

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
      session: The session to retrieve metadata with. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      A load status object, which can be used to make assertions about the
      status of checkpoint restoration and run initialization/restore ops
      (of type `CheckpointLoadStatus`, or `InitializationOnlyStatus` if
      `save_path` is `None`).

      If `save_path` points to a name-based checkpoint, a `NameBasedSaverStatus`
      object is returned which runs restore ops from a name-based saver.
    """
    if save_path is None:
      return InitializationOnlyStatus(self._root_checkpointable)
    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
      if session is None:
        session = ops.get_default_session()
      file_prefix_tensor = self._file_prefix_placeholder
      file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
    else:
      session = None
      with ops.device("/cpu:0"):
        file_prefix_tensor = constant_op.constant(save_path)
      file_prefix_feed_dict = None
    try:
      if not in_graph_mode or self._object_graph_restore_tensor is None:
        with ops.device("/cpu:0"):
          object_graph_string, = io_ops.restore_v2(
              prefix=file_prefix_tensor,
              tensor_names=[_OBJECT_GRAPH_PROTO_KEY],
              shape_and_slices=[""],
              dtypes=[dtypes.string],
              name="object_graph_proto_read")
        if in_graph_mode:
          self._object_graph_restore_tensor = object_graph_string
      if in_graph_mode:
        object_graph_string = session.run(
            self._object_graph_restore_tensor,
            feed_dict=file_prefix_feed_dict)
      else:
        object_graph_string = object_graph_string.numpy()
    except errors_impl.NotFoundError:
      # The object graph proto does not exist in this checkpoint. Try again with
      # name-based saving.
      return NameBasedSaverStatus(self, save_path)

    object_graph_proto = (
        checkpointable_object_graph_pb2.CheckpointableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    if in_graph_mode and object_graph_proto == self._last_restore_object_graph:
      checkpoint = self._last_restore_checkpoint
    else:
      if in_graph_mode:
        dtype_map = None
      else:
        reader = pywrap_tensorflow.NewCheckpointReader(save_path)
        dtype_map = reader.get_variable_to_dtype_map()
      checkpoint = core_checkpointable_utils._Checkpoint(  # pylint: disable=protected-access
          object_graph_proto=object_graph_proto,
          save_path=file_prefix_tensor,
          dtype_map=dtype_map)
      if in_graph_mode:
        if self._last_restore_object_graph is not None:
          raise NotImplementedError(
              "Using a single Saver to restore different object graphs is not "
              "currently supported when graph building. Use a different Saver "
              "for each object graph (restore ops will be duplicated), or "
              "file a feature request if this limitation bothers you.")
        self._last_restore_checkpoint = checkpoint
        self._last_restore_object_graph = object_graph_proto
    core_checkpointable._CheckpointPosition(  # pylint: disable=protected-access
        checkpoint=checkpoint, proto_id=0).restore(self._root_checkpointable)
    load_status = CheckpointLoadStatus(
        checkpoint, feed_dict=file_prefix_feed_dict)
    return load_status


class Checkpoint(core_checkpointable.Checkpointable):
  """A utility class which groups `Checkpointable` objects.

  Accepts arbitrary keyword arguments to its constructor and saves those values
  with a checkpoint. Maintains a `save_counter` for numbering checkpoints.

  Example usage:

  ```python
  import tensorflow as tf
  import tensorflow.contrib.eager as tfe
  import os

  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

  root = tfe.Checkpoint(optimizer=optimizer, model=model)
  root.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )
  root.save(file_prefix=checkpoint_prefix)
  ```

  For more manual control over saving, use `tfe.CheckpointableSaver` directly.

  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """

  def __init__(self, **kwargs):
    """Group objects into a training checkpoint.

    Args:
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. Attribute values must derive from
        `CheckpointableBase`.
    Raises:
      ValueError: If objects in `kwargs` are not Checkpointable.
    """
    super(Checkpoint, self).__init__()
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      if not isinstance(v, core_checkpointable.CheckpointableBase):
        raise ValueError(
            ("`Checkpoint` was expecting an object derived from "
             "`CheckpointableBase`, got %s.") % (v,))
      setattr(self, k, v)
    self._save_counter = None  # Created lazily for restore-on-create.
    self._saver = CheckpointableSaver(weakref.ref(self))

  def _maybe_create_save_counter(self):
    """Create a save counter if it does not yet exist."""
    if self._save_counter is None:
      # Initialized to 0 and incremented before saving.
      with ops.device("/cpu:0"):
        self._save_counter = add_variable(
            self, name="save_counter", initializer=0, dtype=dtypes.int64)

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
    """Save a checkpoint. Wraps `tfe.CheckpointableSaver.save`."""
    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
      if session is None:
        session = ops.get_default_session()
      if self._save_counter is None:
        # When graph building, if this is a new save counter variable then it
        # needs to be initialized before assign_add. This is only an issue if
        # restore() has not been called first.
        session.run(self.save_counter.initializer)
    with ops.colocate_with(self.save_counter):
      assign_op = self.save_counter.assign_add(1)
    if in_graph_mode:
      session.run(assign_op)
    return self._saver.save(
        file_prefix=file_prefix,
        checkpoint_number=self.save_counter,
        session=session)

  def restore(self, save_path):
    """Restore a checkpoint. Wraps `tfe.CheckpointableSaver.restore`."""
    status = self._saver.restore(save_path=save_path)
    # Create the save counter now so it gets initialized with other variables
    # when graph building. Creating it earlier would lead to double
    # initialization when executing eagerly.
    self._maybe_create_save_counter()
    return status
