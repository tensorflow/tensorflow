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
"""A Network is a composition of Layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import weakref

from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util

# pylint: disable=protected-access
# Explanation for protected-access disable: Network has lots of same-class and
# parent-class references across different objects, and some to private
# functions in base.py which should be reused.


_DeferredRestoration = collections.namedtuple(

    "_DeferredRestoration",
    [
        # The map_func to use (either user-specified or the default).
        "map_func",
        # Boolean, True if the user specified an explicit map_func, for error
        # messages.
        "map_func_is_user",
        # A mapping from checkpoint names to initial values of not-yet-created
        # variables which should be restored. These values come from parsing a
        # checkpoint.
        "checkpointed_variables_to_restore",
        # A mapping from checkpoint name to variable objects of variables which
        # have already been restored, for error checking.
        "restored_variables",
        # The session to restore with (if in graph mode).
        "session",
        # Names of the Network where the restore was requested, for error
        # messages.
        "network_name",
        "network_scope_name"
    ])


def _default_naming_conflict_error_message(
    mapped_name, first_variable, second_variable,
    network_name, network_scope_name):
  return (
      ("The default checkpoint variable name mapping strategy for Network "
       "'%s' resulted in a naming conflict. We attempted to strip off the "
       "variable prefix for the Network ('%s'), but this resulted in two "
       "variables named '%s' (originally '%s' and '%s'). This should only "
       "happen when using variable sharing (i.e. the Network contains Networks "
       "or Layers which were first added to another Network, and therefore "
       "have that Network's variable prefix). One solution is to pass "
       "`map_func=lambda n: n` to Network.save and Network.restore to use "
       "fully qualified variable names in the checkpoint, although this will "
       "require that the variable prefix of the Network being restored into "
       "is also '%s'. You may alternatively write an arbitrary mapping.")
      % (
          network_name, network_scope_name, mapped_name,
          first_variable._shared_name,
          second_variable._shared_name, network_scope_name
      ))


def _restore_custom_map_func_error_message(
    mapped_name, first_variable, second_variable,
    network_name, network_scope_name):
  return (
      ("The map_func passed to Network.restore for the Network '%s' "
       "resulted in two variables named '%s' (originally '%s' and '%s'). Since "
       "this is also an error on Network.save, this Network was "
       "probably not saved with this map_func. Note that map_func "
       "always maps from full variable names to checkpoint names; "
       "there is no need to specify an inverse mapping.\n\n"
       "Try stripping less from the variable names, or renaming parts "
       "of the Network. For reference, variables created by sub-Layers "
       "of this Network are prefixed with '%s', but if they are "
       "re-used after being added to another Network they will have "
       "that Network's full variable prefix instead.") % (
           network_name, mapped_name,
           first_variable._shared_name,
           second_variable._shared_name,
           network_scope_name))


def _make_custom_getter_for_deferred_restorations():
  """Returns a custom getter which searches `deferred_restorations`.

  Returns: A tuple of (_custom_getter, deferred_restorations)
    _custom_getter: The getter which should be added to variable_scopes where
      variables will be created.
    deferred_restorations: A list for _DeferredRestoration objects. Typically
      empty when the getter is set, and expanded as deferred restorations are
      requested. All new deferred restorations should be appended to the end of
      the list, where they will have priority over older deferred restorations.
  """
  deferred_restorations = []

  def _custom_getter(getter, name, shape=None, dtype=None,
                     initializer=None,
                     *args, **kwargs):
    """A custom getter which processes deferred restorations."""
    # Iterate over restorations, newest first (newer restorations will take
    # precedence over older restorations, just like with immediate restorations
    # into existing variables).
    delayed_restoration = None
    found_value = False
    value_to_restore = None
    for delayed_restoration in reversed(
        deferred_restorations):
      checkpoint_name = delayed_restoration.map_func(name)
      if (checkpoint_name
          in delayed_restoration.checkpointed_variables_to_restore):
        found_value = True
        value_to_restore = (
            delayed_restoration.checkpointed_variables_to_restore[
                checkpoint_name])
      if found_value:
        break
    # value_to_restore may be False because this variable is not in any
    # checkpoint we are restoring, or None because we have explicitly set it to
    # None when it was previously fetched. In either case, we don't need to
    # set an initializer.
    if found_value and value_to_restore is not None:
      initializer = value_to_restore
      shape = None
    variable = getter(name, shape=shape, dtype=dtype, initializer=initializer,
                      *args, **kwargs)
    if found_value and value_to_restore is not None:
      # Mark as already restored from this checkpoint.
      delayed_restoration.checkpointed_variables_to_restore[
          checkpoint_name] = None
      if context.in_graph_mode():
        delayed_restoration.session.run(variable.initializer)
    if found_value:
      # Error checking should run even if we've already restored a value.
      if delayed_restoration.restored_variables.setdefault(
          checkpoint_name, variable) is not variable:
        # Naming conflict. We've tried to initialize two variables with the
        # same value from the checkpoint.
        if delayed_restoration.map_func_is_user:
          raise ValueError(
              _restore_custom_map_func_error_message(
                  mapped_name=checkpoint_name,
                  first_variable=delayed_restoration.restored_variables[
                      checkpoint_name],
                  second_variable=variable,
                  network_name=delayed_restoration.network_name,
                  network_scope_name=delayed_restoration.network_scope_name))
        else:
          raise ValueError(
              _default_naming_conflict_error_message(
                  mapped_name=checkpoint_name,
                  first_variable=delayed_restoration.restored_variables[
                      checkpoint_name],
                  second_variable=variable,
                  network_name=delayed_restoration.network_name,
                  network_scope_name=delayed_restoration.network_scope_name))
    return variable
  return _custom_getter, deferred_restorations


class Network(base.Layer):
  """Represents the composition of a set of Layers.

  TODO(josh11b,ashankar):
  - Should "trainable" be changeable on the Network object?
  - Do we allow add_variable in Network?
  - Detect layers used in __call__ that weren't registered with track_layer.
  - Convert inputs to __call__ to tensors.
  - Prevent variables from being created after the first __call__?
    (Think about restoring from a checkpoint).
  """

  def __init__(self, name=None):
    if isinstance(name, variable_scope.VariableScope):
      raise ValueError("VariableScopes are not valid Network names.")
    if name is not None and "/" in name:
      raise ValueError(
          "Forward slashes ('/') are not allowed in Network names.")
    super(Network, self).__init__(name=name)
    self._layers = []
    self._sub_layer_name_uids = collections.defaultdict(int)
    # Initially None, but set to False for networks which are first built as
    # top-level.
    self._first_parent = None  # A weak reference to our first parent.
    self._non_network_sublayers = []
    self._owned_layers = {}
    # The scope to use if we end up without a parent.
    self._default_parent_variable_scope = variable_scope.get_variable_scope()
    self._custom_getter, self._deferred_restorations = (
        _make_custom_getter_for_deferred_restorations())

  def _init_set_name(self, name):
    # Anonymous Networks (name=None) defer setting a final name until they are
    # (1) added to another Network, or (2) built/called (where (2) is only used
    # for a "top level" network).
    #
    # However, if we were provided an explicit name (name is not None), that
    # will always be the final name of the Network; if it turns out not to be
    # unique or if variable names can't be prefixed by it we will throw an
    # error.
    self._name = name
    self._base_name = None

  def _finalize_name(self, parent_network):
    if not self._name:
      if not parent_network:
        name_uid_map = base._get_default_graph_uid_map()
      else:
        name_uid_map = parent_network._sub_layer_name_uids
      # Were were not passed a name explicitly (or it was blank), so this is an
      # anonymous Network. We make up a unique name.
      if parent_network:
        avoid_names = parent_network._owned_layers
      else:
        avoid_names = None
      self._name, self._base_name = self._make_unique_name(
          name_uid_map=name_uid_map, avoid_names=avoid_names)
    if self._first_parent is None or (self._first_parent  # False = no parent
                                      and self._first_parent() is None):
      # Save a pointer to the parent Network so that we can later check that the
      # scope name we get is correct.
      if not parent_network:
        self._first_parent = parent_network
      else:
        self._first_parent = weakref.ref(parent_network)

  def _set_scope(self, scope=None):
    if self._scope is None:
      if not self._first_parent:
        first_parent = self._first_parent
      else:
        first_parent = self._first_parent()
      if first_parent is None:
        # If we were never added to another Network, or that Network has beed
        # garbage collected before being called, then we're a top-level Network.
        self._finalize_name(
            # Use False to make sure the value sticks and we don't inherit a
            # parent if we're added to a network later.
            parent_network=False)
      if scope is not None:
        raise ValueError("Networks may not be created with explicit scopes.")
      if first_parent:
        first_parent._set_scope()
        parent_scope = first_parent._scope
      else:
        parent_scope = self._default_parent_variable_scope
      with variable_scope.variable_scope(parent_scope):
        # Make sure variables with this prefix will be unique.
        with variable_scope.variable_scope(
            None, use_resource=True, default_name=self._name) as scope:
          self._scope = scope
          scope_name = scope.name
          suffix_start = scope_name.rfind("/") + 1
          # rfind is -1 if there is no slash in the string, in which case the
          # suffix starts at the beginning of the string (there is no prefix).
          scope_suffix = scope_name[suffix_start:]
          scope_prefix = scope_name[:suffix_start]
          if scope_suffix != self._name:
            raise ValueError(
                ("A Network named '%s' already exists (or a variable_scope was "
                 "created with this name). Names must be unique.") % (
                     self._name,))
          if (first_parent
              and scope_prefix[:-1] != first_parent._scope.name):
            raise ValueError(
                ("Network variable names must match a nesting of sub-Network "
                 "names. Expected prefix '%s' from parent network, but got "
                 "'%s' when attempting to create a variable_scope for Network "
                 "'%s'. Likely an explicit variable_scope was inserted into "
                 "the nesting.") % (
                     first_parent._scope.name,
                     scope_prefix[:-1],
                     self._name))
          elif not first_parent and scope_prefix:
            # For the case when this Network is not nested inside any other
            # Network, but is in a variable_scope. This is an error for now.
            raise ValueError(
                "Creating Networks inside named variable_scopes is currently "
                "not supported (to ensure that variable names match the names "
                "of Networks in which they were first created). To set "
                "options, try `with tf.variable_scope(''):`. If this "
                "limitation bothers you, please file a feature request.")
      for non_network_sublayer in self._non_network_sublayers:
        self._set_scope_for_nonnetwork_sublayer(non_network_sublayer)

  def _set_scope_for_nonnetwork_sublayer(self, sublayer):
    if sublayer._scope is None:
      if sublayer._first_parent is None:
        constituent_first_parent = None
      else:
        constituent_first_parent = sublayer._first_parent()
      if constituent_first_parent:
        constituent_first_parent._set_scope()
        parent_scope = constituent_first_parent._scope
      else:
        self._finalize_name(False)
        raise ValueError(
            ("The parent of a Layer added to Network %s was garbage collected "
             "before the Layer was built. If this limitation bothers you "
             "please, file a feature request.") % (self.name,))
      with variable_scope.variable_scope(parent_scope):
        # Horrid hack to make Layer variable names which are direct
        # sub-layers of Networks conform to the Network variable naming
        # conventions.
        with variable_scope.variable_scope(
            None, use_resource=True,
            default_name=sublayer.name) as sub_scope:
          sublayer._scope = sub_scope

  @base.Layer.name.getter
  def name(self):
    if self._name is None:
      raise ValueError(
          "The network does not yet have a final name, but a name was "
          "requested for it. Networks get a name when they are added to "
          "another Network via track_layer, or when they are first "
          "called/built.")
    return self._name

  def track_layer(self, layer):
    """Track a Layer in this Network.

    `Network` requires that all `Layer`s used in `call()` be tracked so that the
    `Network` can export a complete list of variables.

    Args:
      layer: A `tf.layers.Layer` object.

    Returns:
      The passed in `layer`.

    Raises:
      RuntimeError: If __init__ has not been called.
      TypeError: If `layer` is the wrong type.
      ValueError: If a `Layer` with the same name has already been added.
    """
    if not hasattr(self, "_layers"):
      raise RuntimeError("Need to call Network.__init__ before adding layers")
    if not isinstance(layer, base.Layer):
      raise TypeError(
          "Network.track_layer() passed type %s, not a tf.layers.Layer" %
          (type(layer),))
    if isinstance(layer, Network):
      layer._finalize_name(parent_network=self)
    else:
      # `layer` is a non-Network, so it hasn't been named to follow Network
      # conventions for contained Layers (i.e. the same conventions as for
      # sub-Networks). This renaming is necessary to isolate Network variable
      # naming from Layers constructed outside the Network and never added to it
      # (because Layers are named globally).
      if not layer.built:
        if not hasattr(layer, "_first_parent"):
          dereferenced_layer_first_parent = None
        else:
          dereferenced_layer_first_parent = layer._first_parent()
        if dereferenced_layer_first_parent is None:
          if layer._name != layer._base_name:
            # If name and base_name do not match, then this Layer used anonymous
            # naming and we have to rename it. Otherwise there's an explicit
            # name, and we should respect it (subject to error checking).
            layer._name, layer._base_name = layer._make_unique_name(
                name_uid_map=self._sub_layer_name_uids,
                avoid_names=self._owned_layers)
          layer._first_parent = weakref.ref(self)
        self._non_network_sublayers.append(layer)
    if (not layer.built
        and layer._first_parent
        and self is layer._first_parent()):
      if layer.name in self._owned_layers:
        if self._owned_layers[layer.name] is layer:
          return layer
        raise ValueError(
            "Attempt to add two Layers with the name '%s' to the same Network."
            % (layer.name))
      self._owned_layers[layer.name] = layer
    self._layers.append(layer)
    return layer

  def get_layer(self, name=None, index=None):
    """Get a contained `tf.layers.Layer` either by name or index.

    Args:
      name: String matching one of the names of a contained `Layer`. Note that
        the names of `Layer`s added to `Network`s may not be unique when doing
        layer sharing (i.e. adding a `Layer` to this `Network` which was already
        added to another `Network`). The lowest index `Layer` with a matching
        name will be returned.
      index: Integer in [0, number of layers). Layers are assigned an index
        by the order they are added.

    Returns:
      A `tf.layers.Layer` object.

    Raises:
      ValueError: If neither or both of 'index' or 'name' is specified, or the
        lookup failed.
    """
    if index is not None:
      if name is not None:
        raise ValueError("Exactly one of 'index' or 'name' must be provided")
      if len(self._layers) <= index:
        raise ValueError("Was asked to retrieve layer at index " + str(index) +
                         " but model only has " + str(len(self._layers)) +
                         " layers.")
      else:
        return self._layers[index]
    else:
      if not name:
        raise ValueError("Provide either a layer name or layer index.")
    for layer in self._layers:
      if layer.name == name:
        return layer
    raise ValueError("No such layer: " + name)

  # The following methods are for implementing the Layer interface.

  @property
  def weights(self):
    # TODO(josh11b): Should this return a set or perform de-duplication of
    # variables in the case of shared layers/variables that appear in
    # multiple places in the Network?
    weights = []
    for layer in self._layers:
      weights += layer.weights
    return weights

  @property
  def trainable_weights(self):
    weights = []
    for layer in self._layers:
      weights += layer.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for layer in self._layers:
      weights += layer.non_trainable_weights
    return weights

  @property
  def trainable(self):
    return True

  @trainable.setter
  def trainable(self, value):
    if not value:
      # We believe it better to decide which layers & networks are trainable
      # at the Trainer level than here. Otherwise you can run into trouble if a
      # layer/network is shared between two models, but is trainable in one
      # but not the other (like with adversarial networks).
      raise AttributeError("cannot mark Network as not trainable")

  @property
  def layers(self):
    return self._layers

  def add_variable(self, name, shape, dtype=None, initializer=None,
                   regularizer=None, trainable=True, constraint=None):
    raise RuntimeError(
        "add_variable not supported in Network class yet. Please file an issue "
        "at https://github.com/tensorflow/tensorflow/issues/new if this is "
        "important to you")

  def _strip_variable_prefix(self, original_variable_name):
    """The default map_func for saving or restoring variables.

    Strips the variable prefix for the Network on which save/restore was called,
    and leaves other variable names fully qualified in the checkpoint.

    Args:
      original_variable_name: The _shared_name of the variable (no :0
        suffix) to map.
    Returns:
      The checkpoint name of the variable.
    """
    scope_name_with_slash = self.scope_name + "/"
    if original_variable_name.startswith(scope_name_with_slash):
      return original_variable_name[len(scope_name_with_slash):]
    else:
      return original_variable_name

  def save(self, save_path, global_step=None, map_func=None):
    """Save variables from the Network to a checkpoint.

    Args:
      save_path: Either a checkpoint prefix or the name of a directory to save
        the checkpoint in (in which case the checkpoint will be named based on
        the Network name).
      global_step: The global step to use when naming the checkpoint. If None
        (default), we will first try to get the default global step. If that
        fails because no default global step exists, then the checkpoint is
        created without a global step suffix.
      map_func: A function mapping fully qualified variable names
        (e.g. 'my_network_1/dense_1/kernel') to names in the checkpoint. By
        default (if `map_func=None`), the variable prefix for the network being
        restored (`Network.scope_name + '/'`, e.g. 'my_network_1/') is stripped
        and all other variable names (shared with other Networks) are left
        unchanged.
    Returns:
      The checkpoint prefix for the saved checkpoint, which may be passed to
      `Network.restore`.
    Raises:
      ValueError: If the Network has not yet been called, or if map_func results
        in a name collision.
    """
    if not self.built:
      raise ValueError(
          "Attempt to save the Network before it was first called. This means "
          "variables have not yet been created, so there is nothing to save.")
    self._set_scope()  # scope_name should be available to map_funcs
    if global_step is None:
      global_step = training_util.get_global_step()
    if os.path.isdir(save_path):
      # If we were passed a directory, default to naming based on the Network
      # name.
      save_path = os.path.join(save_path, self.name)
    user_map_func = map_func
    if map_func is None:
      map_func = self._strip_variable_prefix
    variable_map = {}
    for variable in self.variables:
      mapped_name = map_func(variable._shared_name)
      if variable_map.setdefault(mapped_name, variable) is not variable:
        if user_map_func is None:
          # Instead of erroring out, we could just re-try and silently use the
          # full variable names in the checkpoint. This could be odd for deeply
          # nested sub-Networks (since the full prefix from the nesting would
          # get added), so for now we'll let the user deal with this case.
          raise ValueError(_default_naming_conflict_error_message(
              mapped_name=mapped_name,
              first_variable=variable_map[mapped_name],
              second_variable=variable,
              network_name=self.name,
              network_scope_name=self.scope_name))
        else:
          # The user passed their own problematic map_func.
          raise ValueError(
              ("The map_func passed to Network.save for the Network '%s' "
               "resulted in two variables named '%s' ('%s' and '%s'). Try "
               "stripping less from the variable names, or renaming parts of "
               "the Network. For reference, variables created by sub-Layers of "
               "this Network are prefixed with '%s', but if they are re-used "
               "after being added to another Network, they will have that "
               "Network's full variable prefix instead.") % (
                   self.name, mapped_name,
                   variable_map[mapped_name]._shared_name,
                   variable._shared_name,
                   self.scope_name))
    if context.in_eager_mode():
      sess = None
    else:
      sess = ops.get_default_session()
    return saver_lib.Saver(variable_map).save(
        sess=sess, save_path=save_path, write_meta_graph=False,
        global_step=global_step)

  def _restore_existing_variables(self, save_path, map_func, user_map_func):
    """Use a standard Saver to restore existing variables from a checkpoint.

    Args:
      save_path: The checkpoint prefix or directory to read from.
      map_func: The function to use when mapping from variable names to
        checkpoint names.
      user_map_func: The original map_func passed by the user, for error
        checking.
    Returns:
      A dictionary mapping from checkpoint names to variable objects which have
      been restored (for bookkeeping to avoid deferred restorations on these
      variables).
    Raises:
      ValueError: If there is a name collision.
    """
    existing_variables_by_checkpoint_name = {}
    for variable in self.variables:
      checkpoint_name = map_func(variable._shared_name)
      if existing_variables_by_checkpoint_name.setdefault(
          checkpoint_name, variable) is not variable:
        if user_map_func is None:
          raise ValueError(_default_naming_conflict_error_message(
              mapped_name=checkpoint_name,
              first_variable=existing_variables_by_checkpoint_name[
                  checkpoint_name],
              second_variable=variable,
              network_name=self.name,
              network_scope_name=self.scope_name))
        else:
          raise ValueError(_restore_custom_map_func_error_message(
              mapped_name=checkpoint_name,
              first_variable=existing_variables_by_checkpoint_name[
                  checkpoint_name],
              second_variable=variable,
              network_name=self.name,
              network_scope_name=self.scope_name))
    if existing_variables_by_checkpoint_name:
      if context.in_eager_mode():
        sess = None
      else:
        sess = ops.get_default_session()
      saver_lib.Saver(var_list=existing_variables_by_checkpoint_name).restore(
          sess=sess, save_path=save_path)
    return existing_variables_by_checkpoint_name

  def _set_restore_on_create(self, save_path, map_func, user_map_func,
                             existing_variables_by_checkpoint_name):
    """If necessary, request deferred restorations of variables."""
    checkpoint_reader = checkpoint_utils.load_checkpoint(save_path)
    checkpointed_variables_to_restore = {}
    for checkpoint_name, _ in checkpoint_utils.list_variables(save_path):
      if checkpoint_name in existing_variables_by_checkpoint_name:
        # This variable was already created and restored.
        continue
      # Save the variable for later restoration in a custom getter.
      checkpointed_variables_to_restore[checkpoint_name] = (
          checkpoint_reader.get_tensor(checkpoint_name))
    # Only set a deferred restoration if there are checkpoint variables which
    # have not been assigned to existing variables. Note that this loses out on
    # some opportunity for error checking, but avoids creating
    # _DeferredRestoration objects once a Network has been built (so that
    # restoring in a loop does not take increasing amounts of memory).
    if checkpointed_variables_to_restore:
      if context.in_eager_mode():
        sess = None
      else:
        sess = ops.get_default_session()
      # We need a name for error messages. If we haven't been added to another
      # Network yet, we're top-level.
      self._finalize_name(False)
      self._set_scope()
      # Save a record of this restoration for use in the custom getter.
      deferred_restoration = _DeferredRestoration(
          map_func=map_func,
          map_func_is_user=(user_map_func is not None),
          checkpointed_variables_to_restore=checkpointed_variables_to_restore,
          restored_variables={},
          session=sess,
          network_name=self.name,
          network_scope_name=self.scope_name)
      self._deferred_restorations.append(deferred_restoration)
      # Add the deferred registration to non-Network children, and request that
      # Networks propagate the request to their children.
      self._add_deferred_restoration(deferred_restoration)

  def _add_deferred_restoration(self, deferred_restoration):
    """Add a deferred restoration to this Network and all children.

    Restorations which are requested later have higher priority, and the highest
    priority matching restoration is applied to a variable when it is created.

    Args:
      deferred_restoration: A _DeferredRestoration object.
    """
    # Networks don't create variables at the moment, so this append isn't
    # strictly necessary. We could get by with only adding deferred restorations
    # to non-Network Layers.
    self._set_scope()
    # We use set_custom_getter because it avoids recursively calling up the
    # variable_scope tree. We've done the tree traversal ourselves and have
    # added the request to each Layer which needs it.
    self._scope.set_custom_getter(self._custom_getter)
    self._deferred_restorations.append(deferred_restoration)
    for layer in self.layers:
      if isinstance(layer, Network):
        # For Networks, request that they propagate this deferred restoration
        # to all of their children recursively.
        layer._add_deferred_restoration(deferred_restoration)
      else:
        # For non-Network Layers, make sure they have a deferred restoration
        # queue and a custom getter, then add our request to it.
        if not hasattr(layer, "_custom_getter"):
          assert not hasattr(layer, "_deferred_restorations")
          layer._custom_getter, layer._deferred_restorations = (
              _make_custom_getter_for_deferred_restorations())
          self._set_scope_for_nonnetwork_sublayer(layer)
          layer._scope.set_custom_getter(layer._custom_getter)
        layer._deferred_restorations.append(deferred_restoration)

  def restore(self, save_path, map_func=None):
    """Restore the Network from a checkpoint.

    If variables have already been created (typically when some or all of the
    `Network` is built), they are assigned values from the checkpoint
    immediately, overwriting any existing values (in graph mode the default
    session is used for the assignments).

    If there are checkpoint entries which do not correspond to any existing
    variables in the `Network`, these values are saved for deferred restoration;
    their initial values will be the checkpointed values once they are
    created. Requests for multiple deferred restorations behave the same way as
    immediate restorations, in that later requests will take priority over
    earlier requests relevant to the same variable.

    If this `Network` shares `Layer`s with another network, those `Layer`s will
    also have their variables restored from the checkpoint.

    Args:
      save_path: The return value of `Network.save`, or a directory to search
        for a checkpoint.
      map_func: A function mapping fully qualified variable names
        (e.g. 'my_network_1/dense_1/kernel') to names in the checkpoint. By
        default (if `map_func=None`), the variable prefix for the network being
        restored (`Network.scope_name + '/'`, e.g. 'my_network_1/') is stripped
        and all other variable names (shared with other Networks) are left
        unchanged. Note that this is the _same_ map_func as `Network.save`, not
        an inverse mapping.
    """
    self._finalize_name(parent_network=False)
    self._set_scope()  # scope_name should be available to map_funcs
    if os.path.isdir(save_path):
      # If we don't have a name yet, set no parent.
      save_path = os.path.join(save_path, self.name)
    user_map_func = map_func
    if map_func is None:
      map_func = self._strip_variable_prefix
    # Step one is to restore any existing variables from the checkpoint.
    existing_variables_by_checkpoint_name = self._restore_existing_variables(
        save_path=save_path,
        map_func=map_func,
        user_map_func=user_map_func)
    # Step two is to set a custom getter which restores variables on creation,
    # for those variables which have not been added to sub-Layers yet.
    self._set_restore_on_create(
        save_path=save_path,
        map_func=map_func,
        user_map_func=user_map_func,
        existing_variables_by_checkpoint_name=(
            existing_variables_by_checkpoint_name))

  # TODO(josh11b): Support other Layer methods needed for graph mode, such as for
  # losses and updates


class Sequential(Network):
  """Represents a linear sequence of Layers or functions.

  The output of each layer/function is provided as the input to the next.
  The inputs passed to `__call__` are passed to the inputs of the first
  Layer, and it returns the outputs of the last Layer.

  Args:
    layers_funcs: An optional sequence where each element is either a
      tf.layers.Layer object or a callable.
    name: An optional string name to use for this Network.
  """

  def __init__(self, layers_funcs=None, name=None):
    super(Sequential, self).__init__(name=name)
    self._layers_funcs = []
    if layers_funcs:
      for l in layers_funcs:
        self.add(l)

  def add(self, layer_func):
    if isinstance(layer_func, base.Layer):
      args = estimator_util.fn_args(layer_func.call)
      self.track_layer(layer_func)
    elif callable(layer_func):
      args = estimator_util.fn_args(layer_func)
    else:
      raise TypeError(
          "Sequential.add() takes only tf.layers.Layer objects or callables; "
          "not '%s' of type '%s'." % (layer_func, type(layer_func)))
    self._layers_funcs.append((("training" in args), layer_func))

  def call(self, inputs, training=None):
    """Call each Layer in the order they were added."""
    # TODO(josh11b): Support "mode" and maybe other arguments
    if training is None:
      for _, l in self._layers_funcs:
        inputs = l(inputs)
    else:
      for has_training_arg, l in self._layers_funcs:
        if has_training_arg:
          inputs = l(inputs, training)
        else:
          inputs = l(inputs)
    return inputs
