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
import uuid

import six

from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope


class Network(base.Layer):
  """Represents the composition of a set of Layers.

  TODO(josh11b,ashankar):
  - Should "trainable" be changeable on the Network object?
  - Do we allow add_variable in Network?
  - Layer.name and Layer.variables.names are not in sync today
    d = tf.layers.Dense(1)
    d(tf.constant([[1.]]))
    print(d.name)
    print(d.variables)
  - Note that name provided to __init__ is only for error messages?
  - Detect layers used in __call__ that weren't registered with track_layer.
  - Convert inputs to __call__ to tensors.
  - Prevent variables from being created after the first __call__?
    (Think about restoring from a checkpoint).
  - Save & restore
  """

  def __init__(self, name=None):
    super(Network, self).__init__(name=name)
    self._container = uuid.uuid4().hex
    self._layers = collections.OrderedDict()

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
    if layer.name in self._layers:
      if self._layers[layer.name] is layer:
        return layer
      raise ValueError(
          "Attempt to add two Layers with the name '%s' to the same Network "
          "'%s'" % (layer.name, self.name))
    self._layers[layer.name] = layer
    return layer

  def get_layer(self, name=None, index=None):
    """Get a contained `tf.layers.Layer` either by name or index.

    Args:
      name: String matching one of the names of a contained `Layer`.
      index: Integer in [0, number of layers). Layers are assigned an index
        by the order they are added.

    Returns:
      A `tf.layers.Layer` object.

    Raises:
      ValueError: If neither or both of 'index' or 'name' is specified.
    """
    if index is not None:
      if name is not None:
        raise ValueError("Exactly one of 'index' or 'name' must be provided")
      if len(self._layers) <= index:
        raise ValueError("Was asked to retrieve layer at index " +
                         str(index) + " but model only has " + str(
                             len(self._layers)) + " layers.")
      return list(self._layers.values())[index]
    if name is None:
      raise ValueError("Exactly one of 'index' or 'name' must be provided")
    return self._layers[index]

  # The following methods are for implementing the Layer interface.

  @property
  def weights(self):
    # TODO(josh11b): Should this return a set or perform de-duplication of
    # variables in the case of shared layers/variables that appear in
    # multiple places in the Network?
    weights = []
    for layer in six.itervalues(self._layers):
      weights += layer.weights
    return weights

  @property
  def trainable_weights(self):
    weights = []
    for layer in six.itervalues(self._layers):
      weights += layer.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for layer in six.itervalues(self._layers):
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
    return self._layers.values()

  def add_variable(self, name, shape, dtype=None, initializer=None,
                   regularizer=None, trainable=True, constraint=None):
    raise RuntimeError(
        "add_variable not supported in Network class yet. Please file an issue "
        "at https://github.com/tensorflow/tensorflow/issues/new if this is "
        "important to you")

  def __call__(self, inputs, *args, **kwargs):
    # TODO(josh11b,ashankar,agarwal): Can we reduce the number of context
    # managers here and/or move some of the work into the constructor
    # for performance reasons?
    with ops.container(self._container):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         use_resource=True):
        return super(Network, self).__call__(inputs, *args, **kwargs)

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
