# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Contains the base Layer class, from which all layers inherit."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util import function_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


InputSpec = base_layer.InputSpec  # pylint: disable=invalid-name


@tf_export('layers.Layer')
class Layer(base_layer.Layer):
  """Base layer class.

  It is considered legacy, and we recommend the use of `tf.keras.layers.Layer`
  instead.

  Arguments:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: Default dtype of the layer's weights (default of `None` means use the
      type of the first input).

  Read-only properties:
    name: The name of the layer (string).
    dtype: Default dtype of the layer's weights (default of `None` means use the
      type of the first input).
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and
      non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.
    trainable_weights: List of variables to be included in backprop.
    non_trainable_weights: List of variables that should not be
      included in backprop.
    weights: The concatenation of the lists trainable_weights and
      non_trainable_weights (in this order).

  Mutable properties:
    trainable: Whether the layer should be trained (boolean).
    input_spec: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.
  """

  def __init__(self, trainable=True, name=None, dtype=None,
               **kwargs):
    # For backwards compatibility, legacy layers do not use `ResourceVariable`
    # by default.
    self._use_resource_variables = False
    scope = kwargs.pop('_scope', None)
    self._reuse = kwargs.pop('_reuse', None)

    # Avoid an incorrect lint error
    self._trainable_weights = []
    self.built = False

    super(Layer, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                **kwargs)

    self._graph = None
    self._call_has_scope_arg = 'scope' in self._call_fn_args
    if scope:
      with vs.variable_scope(scope) as captured_scope:
        self._scope = captured_scope
    else:
      self._scope = None
    self._current_scope = None

  @property
  def graph(self):
    if context.executing_eagerly():
      raise RuntimeError('Layer.graph not supported when executing eagerly.')
    return self._graph

  def _init_set_name(self, name):
    # Determine layer name (non-unique).
    if isinstance(name, vs.VariableScope):
      base_name = name.name
    else:
      base_name = name
      self._name = name
    if not name:
      self._name, base_name = self._make_unique_name()
    self._base_name = base_name

  def _make_unique_name(self, name_uid_map=None, avoid_names=None,
                        namespace='', zero_based=False):
    base_name = base_layer.to_snake_case(self.__class__.__name__)
    name = base_layer.unique_layer_name(base_name,
                                        name_uid_map=name_uid_map,
                                        avoid_names=avoid_names,
                                        namespace=namespace,
                                        zero_based=zero_based)
    return (name, base_name)

  @property
  def scope_name(self):
    if not self._scope:
      raise ValueError('No name available for layer scope because the layer "' +
                       self._name + '" has not been used yet. The scope name ' +
                       ' is determined the first time the layer instance is ' +
                       'called. You must therefore call the layer before ' +
                       'querying `scope_name`.')
    return self._scope.name

  def add_loss(self, losses, inputs=None):
    previous_losses_length = len(self._losses)
    super(Layer, self).add_loss(losses, inputs=inputs)
    # TODO(fchollet): deprecate collection below.
    new_losses = self._losses[previous_losses_length:]
    _add_elements_to_collection(new_losses, ops.GraphKeys.REGULARIZATION_LOSSES)

  def _name_scope(self):
    """Determines op naming for the Layer."""
    return self._current_scope.original_name_scope

  def _set_scope(self, scope=None):
    if self._scope is None:
      # If constructed with _scope=None, lazy setting of scope.
      if self._reuse:
        with vs.variable_scope(
            scope if scope is not None else self._base_name) as captured_scope:
          self._scope = captured_scope
      else:
        with vs.variable_scope(
            scope, default_name=self._base_name) as captured_scope:
          self._scope = captured_scope

  def add_weight(self,
                 name,
                 shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=vs.VariableSynchronization.AUTO,
                 aggregation=vs.VariableAggregation.NONE,
                 partitioner=None):
    """Adds a new variable to the layer, or gets an existing one; returns it.

    Arguments:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
        Note, if the current variable scope is marked as non-trainable
        then this parameter is ignored and any added variables are also
        marked as non-trainable. `trainable` defaults to `True` unless
        `synchronization` is set to `ON_READ`.
      constraint: constraint instance (callable).
      use_resource: Whether to use `ResourceVariable`.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize. If `synchronization` is set to `ON_READ`,
        `trainable` must not be set to `True`.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      partitioner: (optional) partitioner instance (callable).  If
        provided, when the requested variable is created it will be split
        into multiple partitions according to `partitioner`.  In this case,
        an instance of `PartitionedVariable` is returned.  Available
        partitioners include `tf.fixed_size_partitioner` and
        `tf.variable_axis_size_partitioner`.  For more details, see the
        documentation of `tf.get_variable` and the  "Variable Partitioners
        and Sharding" section of the API guide.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partioned variable regularization and
        eager execution is enabled.
      ValueError: When trainable has been set to True with synchronization
        set as `ON_READ`.
    """
    if synchronization == vs.VariableSynchronization.ON_READ:
      if trainable:
        raise ValueError(
            'Synchronization value can be set to '
            'VariableSynchronization.ON_READ only for non-trainable variables. '
            'You have specified trainable=True and '
            'synchronization=VariableSynchronization.ON_READ.')
      else:
        # Set trainable to be false when variable is to be synced on read.
        trainable = False
    elif trainable is None:
      trainable = True

    def _should_add_regularizer(variable, existing_variable_set):
      if isinstance(variable, tf_variables.PartitionedVariable):
        for var in variable:
          if var in existing_variable_set:
            return False
        return True
      else:
        return variable not in existing_variable_set

    init_graph = None
    if not context.executing_eagerly():
      default_graph = ops.get_default_graph()
      if default_graph.building_function:
        with ops.init_scope():
          # Retrieve the variables from the graph into which variables
          # will be lifted; if initialization ops will be lifted into
          # the eager context, then there is nothing to retrieve, since variable
          # collections are not supported when eager execution is enabled.
          if not context.executing_eagerly():
            init_graph = ops.get_default_graph()
            existing_variables = set(tf_variables.global_variables())
      else:
        # Initialization ops will not be lifted out of the default graph.
        init_graph = default_graph
        existing_variables = set(tf_variables.global_variables())

    if dtype is None:
      dtype = self.dtype or dtypes.float32

    self._set_scope(None)
    reuse = self.built or self._reuse
    prev_len_trainable = len(self._trainable_weights)
    with vs.variable_scope(
        self._scope, reuse=reuse, auxiliary_name_scope=False) as scope:
      self._current_scope = scope
      with ops.name_scope(self._name_scope()):
        use_resource = (use_resource or
                        self._use_resource_variables or
                        scope.use_resource)
        if initializer is None:
          initializer = scope.initializer
        variable = super(Layer, self).add_weight(
            name,
            shape,
            dtype=dtypes.as_dtype(dtype),
            initializer=initializer,
            trainable=trainable,
            constraint=constraint,
            partitioner=partitioner,
            use_resource=use_resource,
            synchronization=synchronization,
            aggregation=aggregation,
            getter=vs.get_variable)

        if regularizer:
          if context.executing_eagerly() or _should_add_regularizer(
              variable, existing_variables):
            self._handle_weight_regularization(name, variable, regularizer)

        if init_graph is not None:
          # Handle edge case where a custom getter has overridden `trainable`.
          # There is one known occurrence of this, in unit test
          # testBasicRNNCellNotTrainable in
          # contrib.rnn.python.kernel_tests.core_rnn_cell_test
          with init_graph.as_default():
            trainable_variables = tf_variables.trainable_variables()
          if (trainable and self.trainable and
              variable not in trainable_variables):
            # A custom getter / variable scope overrode the trainable flag.
            extra_trainable_vars = self._trainable_weights[prev_len_trainable:]
            self._trainable_weights = self._trainable_weights[
                :prev_len_trainable]
            self._non_trainable_weights += extra_trainable_vars
    return variable

  def __call__(self, inputs, *args, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
        **Note**: kwarg `scope` is reserved for use by the layer.

    Returns:
      Output tensor(s).

    Note:
      - If the layer's `call` method takes a `scope` keyword argument,
        this argument will be automatically set to the current variable scope.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
    """
    self._set_scope(kwargs.pop('scope', None))

    if not context.executing_eagerly():
      try:
        # Set layer's "graph" at build time
        self._graph = ops._get_graph_from_inputs(nest.flatten(inputs),  # pylint: disable=protected-access
                                                 graph=self._graph)
      except ValueError as e:
        raise ValueError('Input graph and Layer graph are not the same: %s' % e)

    if self.built:
      try:
        # Some classes which inherit from Layer do not use its constructor, so
        # rather than initializing to None we check for an AttributeError.
        scope_context_manager = self._always_reuse_variable_scope
      except AttributeError:
        # From this point we will always set reuse=True, so create a "final"
        # variable scope with this setting. We avoid re-creating variable scopes
        # after this point as an optimization.
        self._always_reuse_variable_scope = vs.variable_scope(
            self._scope, reuse=True, auxiliary_name_scope=False)
        scope_context_manager = self._always_reuse_variable_scope
    else:
      scope_context_manager = vs.variable_scope(
          self._scope, reuse=self._reuse, auxiliary_name_scope=False)

    with scope_context_manager as scope:
      self._current_scope = scope

      try:
        call_has_scope_arg = self._call_has_scope_arg
      except AttributeError:
        self._call_fn_args = function_utils.fn_args(self.call)
        self._call_has_scope_arg = 'scope' in self._call_fn_args
        call_has_scope_arg = self._call_has_scope_arg
      if call_has_scope_arg:
        kwargs['scope'] = scope

      # Actually call layer
      outputs = super(Layer, self).__call__(inputs, *args, **kwargs)

    if not context.executing_eagerly():
      # Update global default collections.
      _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
    return outputs

  def __deepcopy__(self, memo):
    no_copy = set(['_graph'])
    shallow_copy = set(['_scope', '_always_reuse_variable_scope'])
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k in no_copy:
        setattr(result, k, v)
      elif k in shallow_copy:
        setattr(result, k, copy.copy(v))
      elif base_layer.is_tensor_or_tensor_list(v):
        setattr(result, k, v)
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result


def _add_elements_to_collection(elements, collection_list):
  if context.executing_eagerly():
    raise RuntimeError('Using collections from Layers not supported in Eager '
                       'mode. Tried to add %s to %s' % (elements,
                                                        collection_list))
  elements = nest.flatten(elements)
  collection_list = nest.flatten(collection_list)
  for name in collection_list:
    collection = ops.get_collection_ref(name)
    collection_set = set(collection)
    for element in elements:
      if element not in collection_set:
        collection.append(element)
