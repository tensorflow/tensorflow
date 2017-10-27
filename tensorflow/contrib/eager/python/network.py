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
import weakref

from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope

# pylint: disable=protected-access
# Explanation for protected-access disable: Network has lots of same-class and
# parent-class references across different objects, and some to private
# functions in base.py which should be reused.


class Network(base.Layer):
  """Represents the composition of a set of Layers.

  TODO(josh11b,ashankar):
  - Should "trainable" be changeable on the Network object?
  - Do we allow add_variable in Network?
  - Detect layers used in __call__ that weren't registered with track_layer.
  - Convert inputs to __call__ to tensors.
  - Prevent variables from being created after the first __call__?
    (Think about restoring from a checkpoint).
  - Save & restore
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
    if self._first_parent is None or self._first_parent() is None:
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
      for non_network_constituent in self._non_network_sublayers:
        if non_network_constituent._scope is None:
          if non_network_constituent._first_parent is None:
            constituent_first_parent = None
          else:
            constituent_first_parent = non_network_constituent._first_parent()
          if constituent_first_parent:
            constituent_first_parent._set_scope()
            parent_scope = constituent_first_parent._scope
          else:
            parent_scope = (
                non_network_constituent._default_parent_variable_scope)
          with variable_scope.variable_scope(parent_scope):
            # Horrid hack to make Layer variable names which are direct
            # sub-layers of Networks conform to the Network variable naming
            # conventions.
            with variable_scope.variable_scope(
                None, use_resource=True,
                default_name=non_network_constituent.name) as sub_scope:
              non_network_constituent._scope = sub_scope

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
