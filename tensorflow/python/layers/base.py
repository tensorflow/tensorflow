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

# pylint: disable=unused-import,g-bad-import-order
"""Contains the base Layer class, from which all layers inherit."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import re
import weakref

import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils as layers_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export('layers.Layer')
class Layer(checkpointable.CheckpointableBase):
  """Base layer class.

  This is the class from which all layers inherit, implementing common
  infrastructure functionality.

  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing variables,
  losses, and updates, as well as applying TensorFlow ops to input tensors.

  Users will just instantiate it and then treat it as a callable.

  We recommend that descendants of Layer implement the following methods:
  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_variable()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).

  Read-only properties:
    `name`: The name of the layer (string).
    `dtype`: Default dtype of the layer (default of `None` means use the
      type of the first input).
    `trainable_variables`: List of trainable variables.
    `non_trainable_variables`: List of non-trainable variables.
    `variables`: List of all variables of this layer, trainable and
      non-trainable.
    `updates`: List of update ops of this layer.
    `losses`: List of losses added by this layer.

  Mutable properties:
    `trainable`: Whether the layer should be trained (boolean).
    `input_spec`: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.
  """

  def __init__(self, trainable=True, name=None, dtype=None,
               activity_regularizer=None, **kwargs):
    # We use a kwargs dict here because these kwargs only exist
    # for compatibility reasons.
    # The list of kwargs is subject to changes in the future.
    # We do not want to commit to it or to expose the list to users at all.
    # Note this is exactly as safe as defining kwargs in the function signature,
    # the only difference being that the list of valid kwargs is defined
    # below rather rather in the signature, and default values are defined
    # in calls to kwargs.get().
    allowed_kwargs = {
        '_scope',
        '_reuse',
        'input_shape',  # For compatibility with Keras `Sequential` model.
        'batch_size',  # For compatibility with Keras `Sequential` model.
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    # Mutable properties
    # Indicates whether the layer's weights are updated during training
    # and whether the layer's updates are run during training
    self.trainable = trainable
    # A stateful layer is a layer whose updates are run during inference too,
    # for instance stateful RNNs.
    self.stateful = False
    # Indicates whether `build` needs to be called upon layer call, to create
    # the layer's weights.
    self.built = False
    # Provides information about which inputs are compatible with the layer.
    self.input_spec = None

    if activity_regularizer and context.in_eager_mode():
      raise ValueError(
          ('Activity regularization is not supported when executing eagerly. '
           'Got activity_regularizer=%s') % (activity_regularizer,))
    self._activity_regularizer = activity_regularizer
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    # When executing eagerly, _losses is a list of zero-argument lambdas which
    # return tensors. When using graph execution, _losses is a list of ops.
    self._losses = []
    self._reuse = kwargs.get('_reuse')
    self._graph = None  # Will be set at build time.
    self._dtype = None if dtype is None else dtypes.as_dtype(dtype).name
    self._call_fn_args = estimator_util.fn_args(self.call)
    self._compute_previous_mask = ('mask' in self._call_fn_args or
                                   hasattr(self, 'compute_mask'))
    self._call_has_scope_arg = 'scope' in self._call_fn_args

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    self._inbound_nodes = []
    self._outbound_nodes = []

    self._init_set_name(name)

    # Determine variable scope.
    scope = kwargs.get('_scope')
    if scope:
      with vs.variable_scope(scope) as captured_scope:
        self._scope = captured_scope
    else:
      self._scope = None

    # Set `_batch_input_shape` attribute
    # for compatibility with Keras `Sequential` model.
    if 'input_shape' in kwargs:
      batch_size = kwargs.get('batch_size')
      self._batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])

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

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  @property
  def activity_regularizer(self):
    """Optional regularizer function for the output of this layer."""
    return self._activity_regularizer

  @property
  def scope_name(self):
    if not self._scope:
      raise ValueError('No name available for layer scope because the layer "' +
                       self._name + '" has not been used yet. The scope name ' +
                       ' is determined the first time the layer instance is ' +
                       'called. You must therefore call the layer before ' +
                       'querying `scope_name`.')
    return self._scope.name

  @property
  def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

  @property
  def non_trainable_weights(self):
    if self.trainable:
      return self._non_trainable_weights
    else:
      return self._trainable_weights + self._non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.trainable_weights + self.non_trainable_weights

  @property
  def variables(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.weights

  @property
  def updates(self):
    if context.in_eager_mode():
      raise RuntimeError('Layer.updates not supported in Eager mode.')
    if not self.trainable and not self.stateful:
      return []
    return self._updates

  def add_update(self, updates, inputs=None):
    """Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_updates_for` method allows to retrieve the updates relevant to a
    specific set of inputs.

    This call is ignored in Eager mode.

    Arguments:
      updates: Update op, or list/tuple of update ops.
      inputs: If anything other than None is passed, it signals the updates
        are conditional on some of the layer's inputs,
        and thus they should only be run where these inputs are available.
        This is the case for BatchNormalization updates, for instance.
        If None, the updates will be taken into account unconditionally,
        and you are responsible for making sure that any dependency they might
        have is available at runtime.
        A step counter might fall into this category.
    """
    if context.in_eager_mode():
      return  # Updates already applied when in eager mode.

    updates = _to_list(updates)
    updates = [x if isinstance(x, ops.Operation)
               else ops.convert_to_tensor(x) for x in updates]
    self._updates += updates
    if inputs is None:
      for u in updates:
        u._unconditional_update = True  # pylint: disable=protected-access
    else:
      for u in updates:
        u._unconditional_update = False  # pylint: disable=protected-access

  def get_updates_for(self, inputs):
    """Retrieves updates relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of update ops of the layer that depend on `inputs`.

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('`get_updates_for()` not supported in Eager mode.')

    # Updates disabled if layer is not trainable and not explicitly stateful.
    if not self.trainable and not self.stateful:
      return []

    if inputs is None:
      # Requesting unconditional updates.
      return [x for x in self.updates if x._unconditional_update]  # pylint: disable=protected-access

    # Requesting input-conditional updates.
    inputs = nest.flatten(inputs)
    reachable = layers_util.get_reachable_from_inputs(inputs, self.updates)
    updates = []
    for update in self.updates:
      if update in reachable:
        updates.append(update)
    return updates

  @property
  def losses(self):
    """Losses which are associated with this `Layer`.

    Note that when executing eagerly, getting this property evaluates
    regularizers. When using graph execution, variable regularization ops have
    already been created and are simply returned here.

    Returns:
      A list of tensors.
    """
    if context.in_eager_mode():
      # _losses may only contain variable regularization losses when executing
      # eagerly, and they have been saved as lambdas to be executed when
      # requested.
      return [regularizer() for regularizer in self._losses]
    else:
      return self._losses

  def add_loss(self, losses, inputs=None):
    """Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    The `get_losses_for` method allows to retrieve the losses relevant to a
    specific set of inputs.

    Note that `add_loss` is not supported when executing eagerly. Instead,
    variable regularizers may be added through `add_variable`. Activity
    regularization is not supported directly (but such losses may be returned
    from `Layer.call()`).

    Arguments:
      losses: Loss tensor, or list/tuple of tensors.
      inputs: If anything other than None is passed, it signals the losses
        are conditional on some of the layer's inputs,
        and thus they should only be run where these inputs are available.
        This is the case for activity regularization losses, for instance.
        If `None` is passed, the losses are assumed
        to be unconditional, and will apply across all dataflows of the layer
        (e.g. weight regularization losses).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      # TODO(fchollet): it should be possible (and highly desirable) to support
      # `add_loss` in eager mode. This allows great convenience and flexibility
      # in defining custom losses on the fly (e.g. in VAEs).
      # Simply appending the loss value to `self._losses`
      # is the correct behavior.
      # The only caveat is that we need to force the user to only call
      # `add_loss` from inside a model or Layer's `call` method
      # (otherwise the loss computation cannot be backproped through).
      raise RuntimeError('Layer.add_loss not supported in Eager mode.')

    losses = _to_list(losses)
    self._losses += losses
    if inputs is None:
      for loss in losses:
        loss._unconditional_loss = True  # pylint: disable=protected-access
    else:
      for loss in losses:
        loss._unconditional_loss = False  # pylint: disable=protected-access
    # TODO(fchollet): deprecate collection below.
    _add_elements_to_collection(losses, ops.GraphKeys.REGULARIZATION_LOSSES)

  def get_losses_for(self, inputs):
    """Retrieves losses relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_losses_for not supported in Eager mode.')

    if inputs is None:
      # Requesting unconditional losses.
      return [x for x in self.losses if x._unconditional_loss]  # pylint: disable=protected-access

    # Requesting input-conditional losses.
    inputs = nest.flatten(inputs)
    # Retrieve the set of tensors in the TF graph that depend on `inputs`.
    # The losses we want to return will be part of this set.
    # To avoid unnecessary work, we stop the search in case all of
    # `self.losses` have been retrieved.
    reachable = layers_util.get_reachable_from_inputs(inputs, self.losses)
    losses = []
    for loss in self.losses:
      if loss in reachable:
        losses.append(loss)
    return losses

  def build(self, _):
    """Creates the variables of the layer."""
    self.built = True

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """The logic of the layer lives here.

    Arguments:
      inputs: input tensor(s).
      **kwargs: additional keyword arguments.

    Returns:
      Output tensor(s).
    """
    return inputs

  def _name_scope_name(self, current_variable_scope):
    """Determines op naming for the Layer."""
    return current_variable_scope.original_name_scope

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer given the input shape.

    Args:
      input_shape: A (possibly nested tuple of) `TensorShape`.  It need not
        be fully defined (e.g. the batch size may be unknown).

    Returns:
      A (possibly nested tuple of) `TensorShape`.

    Raises:
      TypeError: if `input_shape` is not a (possibly nested tuple of)
        `TensorShape`.
      ValueError: if `input_shape` is incomplete or is incompatible with the
        the layer.
    """
    raise NotImplementedError

  def _make_unique_name(self, name_uid_map=None, avoid_names=None,
                        namespace='', zero_based=False):
    base_name = _to_snake_case(self.__class__.__name__)
    name = _unique_layer_name(base_name, name_uid_map=name_uid_map,
                              avoid_names=avoid_names, namespace=namespace,
                              zero_based=zero_based)
    return (name, base_name)

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

  def add_variable(self, name, shape, dtype=None,
                   initializer=None, regularizer=None,
                   trainable=True, constraint=None,
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
        marked as non-trainable.
      constraint: constraint instance (callable).
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
    """

    # `init_graph` should point to the graph in which variable initialization
    # will occur; it should be None if and only if initialization will take
    # place in the eager context.
    init_graph = None
    if context.in_graph_mode():
      default_graph = ops.get_default_graph()
      if default_graph.building_function:
        with ops.init_scope():
          # Retrieve the variables from the graph into which variables
          # will be lifted; if initialization ops will be lifted into
          # the eager context, then there is nothing to retrieve, since variable
          # collections are not supported when eager execution is enabled.
          if context.in_graph_mode():
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
    with vs.variable_scope(
        self._scope, reuse=reuse, auxiliary_name_scope=False) as scope:
      with ops.name_scope(self._name_scope_name(scope)):
        variable = self._add_variable_with_custom_getter(
            name=name,
            shape=shape,
            getter=vs.get_variable,
            # Manage errors in Layer rather than Checkpointable.
            overwrite=True,
            initializer=initializer,
            dtype=dtypes.as_dtype(dtype),
            constraint=constraint,
            trainable=trainable and self.trainable,
            partitioner=partitioner)

        if init_graph is not None:  # pylint: disable=protected-access
          # The variable was created and initialized in a graph.

          if variable in existing_variables:
            # To match the behavior of tf.get_variable(), we only apply
            # regularization if the variable is newly created.
            return variable

          with init_graph.as_default():
            trainable_variables = tf_variables.trainable_variables()
          if (trainable and self.trainable and
              variable not in trainable_variables):
            # A custom getter / variable scope overrode the trainable flag.
            trainable = False

          if regularizer:
            if isinstance(variable, tf_variables.PartitionedVariable):
              for v in variable:
                with ops.colocate_with(v.op):
                  with ops.name_scope(name + '/Regularizer'):
                    regularization = regularizer(v)
                if regularization is not None:
                  self.add_loss(regularization)
            else:
              with ops.colocate_with(variable.op):
                with ops.name_scope(name + '/Regularizer'):
                  regularization = regularizer(variable)
              if regularization is not None:
                self.add_loss(regularization)
        elif regularizer:  # and initialization took place in an eager context
          if isinstance(variable, tf_variables.PartitionedVariable):
            raise RuntimeError(
                'Partitioned variable regularization is not yet '
                'supported when executing eagerly. File a feature request'
                'if this is important to you.')
          # Save a zero-argument lambda which runs the regularizer on the
          # variable, to be executed when `Layer.losses` is requested.
          # This makes losses responsive to variable updates when executing
          # eagerly.
          #
          # TODO(akshayka): Do the same for graphs as well, so that losses
          # collected in a while_loop can be run outside its control flow
          # context and so that losses won't be swallowed up by graph functions
          # (i.e., `.losses()` should always create regularizers).
          self._losses.append(lambda: regularizer(variable))

    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
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
    input_list = nest.flatten(inputs)

    in_graph_mode = context.in_graph_mode()
    in_deferred_mode = isinstance(input_list[0], _DeferredTensor)
    # Ensure the Layer, if being reused, is working with inputs from
    # the same graph as where it was created.
    if in_graph_mode:
      try:
        # Set layer's "graph" at build time
        self._graph = ops._get_graph_from_inputs(input_list, graph=self._graph)  # pylint: disable=protected-access
      except ValueError as e:
        raise ValueError('Input graph and Layer graph are not the same: %s' % e)
    if in_graph_mode or in_deferred_mode:
      user_kwargs = copy.copy(kwargs)

    # Handle Keras mask propagation from previous layer to current layer.
    previous_mask = None
    if (not hasattr(self, '_compute_previous_mask') or
        self._compute_previous_mask):
      previous_mask = _collect_previous_mask(inputs)
      if not hasattr(self, '_call_fn_args'):
        self._call_fn_args = estimator_util.fn_args(self.call)
      if ('mask' in self._call_fn_args and 'mask' not in kwargs and
          not _is_all_none(previous_mask)):
        # The previous layer generated a mask, and mask was not explicitly pass
        # to __call__, hence we set previous_mask as the default value.
        kwargs['mask'] = previous_mask

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
    input_shapes = None
    with scope_context_manager as scope:
      with ops.name_scope(self._name_scope_name(scope)):
        if not self.built:
          if not in_graph_mode:
            # Activity regularization is currently unsupported in Eager mode.
            if self._activity_regularizer:
              raise ValueError('activity_regularizer currently unsupported in '
                               'Eager mode. Found an activity_regularizer in '
                               '%s(%s).' % (self.__class__.__name__, self))
          if not in_graph_mode and not in_deferred_mode:
            # TODO(agarwal): support _keras_history in Eager mode.
            for x in input_list:
              if hasattr(x, '_keras_history'):
                raise ValueError('_keras_history currently unsupported in '
                                 'Eager mode. Found _keras_history in %s while '
                                 'executing __call__ for %s(%s)' %
                                 (x, self.__class_.__name__, self))

          # Check input assumptions set before layer building, e.g. input rank.
          self._assert_input_compatibility(inputs)
          if input_list and self._dtype is None:
            try:
              self._dtype = input_list[0].dtype.base_dtype.name
            except AttributeError:
              pass
          input_shapes = nest.map_structure(lambda x: x.get_shape(), inputs)
          self.build(input_shapes)
        try:
          # Note: not all sub-classes of Layer call Layer.__init__ (especially
          # the ones under tensorflow/python/keras). Hence we recompute this
          # attribute here if it is not set.
          # TODO(agarwal): Fix the sub-classes and avoid this complexity.
          call_has_scope_arg = self._call_has_scope_arg
        except AttributeError:
          self._call_fn_args = estimator_util.fn_args(self.call)
          self._call_has_scope_arg = 'scope' in self._call_fn_args
          call_has_scope_arg = self._call_has_scope_arg
        if call_has_scope_arg:
          kwargs['scope'] = scope
        # Check input assumptions set after layer building, e.g. input shape.
        if in_graph_mode or in_deferred_mode:
          self._assert_input_compatibility(inputs)

        if not in_deferred_mode:
          outputs = self.call(inputs, *args, **kwargs)
          if outputs is None:
            raise ValueError('A layer\'s `call` method should return a Tensor '
                             'or a list of Tensors, not None.')
        else:
          # Deferred mode behavior: use `compute_output_shape` to
          # infer the number of outputs of the layer and their shapes.
          if input_shapes is None:
            input_shapes = nest.map_structure(lambda x: x.get_shape(), inputs)

          output_shapes = self.compute_output_shape(input_shapes)
          output_shapes = nest.flatten(output_shapes)
          outputs = [
              # TODO(fchollet): name the deferred tensors?
              _DeferredTensor(shape=shape, dtype=self._dtype)
              for shape in output_shapes
          ]
          if len(outputs) == 1:
            outputs = outputs[0]

        if in_graph_mode:
          # Apply activity regularization.
          # Note that it should be applied every time the layer creates a new
          # output, since it is output-specific.
          if self._activity_regularizer:
            output_list = nest.flatten(outputs)
            for output in output_list:
              with ops.name_scope('ActivityRegularizer'):
                activity_regularization = self._activity_regularizer(output)
              self.add_loss(activity_regularization, inputs=inputs)

          # TODO(fchollet): consider enabling masking for Eager mode.
          if hasattr(self, 'compute_mask'):
            output_mask = self.compute_mask(inputs, previous_mask)
            if isinstance(outputs, (list, tuple)):
              if output_mask is None:
                output_mask = [None for _ in range(len(outputs))]
              for x, m in zip(outputs, output_mask):
                x._keras_mask = m  # pylint: disable=protected-access
            else:
              outputs._keras_mask = output_mask  # pylint: disable=protected-access

    if in_graph_mode:
      # If all input tensors have history metadata,
      # we update the output tensors
      # with corresponding history metadata, thus eventually allowing to use
      # these tensors to instantiate a Network.
      if _have_all_keras_metadata(inputs):
        # If the layer returns tensors from its inputs, unmodified,
        # we copy them to avoid loss of tensor metadata.
        output_ls = nest.flatten(outputs)
        output_ls_copy = []
        for x in output_ls:
          if x in input_list:
            with ops.name_scope(scope.original_name_scope):
              x = array_ops.identity(x)
          output_ls_copy.append(x)
        if len(output_ls_copy) == 1:
          outputs = output_ls_copy[0]
        else:
          outputs = output_ls_copy

      # Update global default collections.
      _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)

    if in_deferred_mode or in_graph_mode:
      if _have_all_keras_metadata(inputs):
        # Add an inbound node to the layer, so it can keep track of this call.
        # This updates the layer history of the output tensor(s).
        self._add_inbound_node(
            input_tensors=inputs, output_tensors=outputs, arguments=user_kwargs)

    self.built = True
    return outputs

  @property
  def graph(self):
    if context.in_eager_mode():
      raise RuntimeError('Layer.graph not supported in Eager mode.')
    return self._graph

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
      elif _is_tensor_or_tensor_list(v):
        setattr(result, k, v)
      else:
        setattr(result, k, copy.deepcopy(v, memo))
    return result

  def apply(self, inputs, *args, **kwargs):
    """Apply the layer on a input.

    This simply wraps `self.__call__`.

    Arguments:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    return self.__call__(inputs, *args, **kwargs)

  def _add_inbound_node(self,
                        input_tensors,
                        output_tensors,
                        arguments=None):
    """Internal method to create an inbound node for the layer.

    Arguments:
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.
    """
    input_tensors = nest.flatten(input_tensors)
    output_tensors = nest.flatten(output_tensors)

    # Collect input tensor(s) coordinates.
    inbound_layers = []
    node_indices = []
    tensor_indices = []
    for x in input_tensors:
      assert hasattr(x, '_keras_history')
      inbound_layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      inbound_layers.append(inbound_layer)
      node_indices.append(node_index)
      tensor_indices.append(tensor_index)

    # Create node, add it to inbound nodes.
    Node(
        self,
        inbound_layers=inbound_layers,
        node_indices=node_indices,
        tensor_indices=tensor_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        arguments=arguments)

    # Update tensor history metadata.
    for i in range(len(output_tensors)):
      # The metadata attribute consists of 1) a layer instance
      # 2) a node index for the layer, 3) a tensor index for the node.
      # The allows layer reuse (multiple nodes per layer) and multi-output
      # or multi-input layers (e.g. a layer can return multiple tensors,
      # and each can be sent to a different layer).
      output_tensors[i]._keras_history = (self, len(self._inbound_nodes) - 1, i)  # pylint: disable=protected-access

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    """Private utility to retrieves an attribute (e.g. inputs) from a node.

    This is used to implement the methods:
        - get_input_shape_at
        - get_output_shape_at
        - get_input_at
        etc...

    Arguments:
        node_index: Integer index of the node from which
            to retrieve the attribute.
        attr: Exact node attribute name.
        attr_name: Human-readable attribute name, for error messages.

    Returns:
        The layer's attribute `attr` at the node of index `node_index`.

    Raises:
        RuntimeError: If the layer has no inbound nodes, or if called in Eager
        mode.
        ValueError: If the index provided does not match any node.
    """
    assert context.in_graph_mode()
    if not self._inbound_nodes:
      raise RuntimeError('The layer has never been called '
                         'and thus has no defined ' + attr_name + '.')
    if not len(self._inbound_nodes) > node_index:
      raise ValueError('Asked to get ' + attr_name + ' at node ' +
                       str(node_index) + ', but the layer has only ' +
                       str(len(self._inbound_nodes)) + ' inbound nodes.')
    values = getattr(self._inbound_nodes[node_index], attr)
    if len(values) == 1:
      return values[0]
    else:
      return values

  def get_input_shape_at(self, node_index):
    """Retrieves the input shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          'Layer.get_input_shape_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'input_shapes',
                                             'input shape')

  def get_output_shape_at(self, node_index):
    """Retrieves the output shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError(
          'Layer.get_output_shape_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'output_shapes',
                                             'output shape')

  def get_input_at(self, node_index):
    """Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_input_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  def get_output_at(self, node_index):
    """Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.get_output_at not supported in Eager mode.')
    return self._get_node_attribute_at_index(node_index, 'output_tensors',
                                             'output')

  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.input not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name +
                           ' is not connected, no input to return.')
    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.output not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

  @property
  def input_shape(self):
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.input_shape not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined input shape.')
    all_input_shapes = set(
        [str(node.input_shapes) for node in self._inbound_nodes])
    if len(all_input_shapes) == 1:
      input_shapes = self._inbound_nodes[0].input_shapes
      if len(input_shapes) == 1:
        return tuple(tensor_shape.TensorShape(input_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in input_shapes
        ]
    else:
      raise AttributeError('The layer "' + str(self.name) +
                           ' has multiple inbound nodes, '
                           'with different input shapes. Hence '
                           'the notion of "input shape" is '
                           'ill-defined for the layer. '
                           'Use `get_input_shape_at(node_index)` '
                           'instead.')

  def count_params(self):
    """Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    """
    if not self.built:
      if self.__class__.__name__ == 'Sequential':
        self.build()  # pylint: disable=no-value-for-parameter
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    weight_shapes = [w.get_shape().as_list() for w in self.weights]
    return int(sum([np.prod(w) for w in weight_shapes]))

  @property
  def output_shape(self):
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    if context.in_eager_mode():
      raise RuntimeError('Layer.output_shape not supported in Eager mode.')
    if not self._inbound_nodes:
      raise AttributeError('The layer has never been called '
                           'and thus has no defined output shape.')
    all_output_shapes = set(
        [str(node.output_shapes) for node in self._inbound_nodes])
    if len(all_output_shapes) == 1:
      output_shapes = self._inbound_nodes[0].output_shapes
      if len(output_shapes) == 1:
        return tuple(tensor_shape.TensorShape(output_shapes[0]).as_list())
      else:
        return [
            tuple(tensor_shape.TensorShape(shape).as_list())
            for shape in output_shapes
        ]
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  @property
  def inbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._inbound_nodes

  @property
  def outbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._outbound_nodes

  def _assert_input_compatibility(self, inputs):
    """Checks compatibility between the layer and provided inputs.

    This checks that the tensor(s) `inputs` verify the input assumptions
    of the layer (if any). If not, a clear and actional exception gets raised.

    Arguments:
        inputs: input tensor or list of input tensors.

    Raises:
        ValueError: in case of mismatch between
            the provided inputs and the expectations of the layer.
    """
    if not self.input_spec:
      return
    if not isinstance(self.input_spec, (list, tuple)):
      input_spec = nest.flatten(self.input_spec)
    else:
      input_spec = self.input_spec
    inputs = nest.flatten(inputs)
    if len(inputs) != len(input_spec):
      raise ValueError('Layer ' + self.name + ' expects ' +
                       str(len(input_spec)) + ' inputs, '
                       'but it received ' + str(len(inputs)) +
                       ' input tensors. Inputs received: ' + str(inputs))
    for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
      if spec is None:
        continue

      if (spec.ndim is not None or
          spec.min_ndim is not None or
          spec.max_ndim is not None):
        if x.get_shape().ndims is None:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'its rank is undefined, but the layer requires a '
                           'defined rank.')

      # Check ndim.
      if spec.ndim is not None:
        ndim = x.get_shape().ndims
        if ndim != spec.ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected ndim=' + str(spec.ndim) + ', found ndim=' +
                           str(ndim) + '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      if spec.max_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim > spec.max_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected max_ndim=' + str(spec.max_ndim) +
                           ', found ndim=' + str(ndim))
      if spec.min_ndim is not None:
        ndim = x.get_shape().ndims
        if ndim is not None and ndim < spec.min_ndim:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           ': expected min_ndim=' + str(spec.min_ndim) +
                           ', found ndim=' + str(ndim) +
                           '. Full shape received: ' +
                           str(x.get_shape().as_list()))
      # Check dtype.
      if spec.dtype is not None:
        if x.dtype != spec.dtype:
          raise ValueError('Input ' + str(input_index) + ' of layer ' +
                           self.name + ' is incompatible with the layer: '
                           'expected dtype=' + str(spec.dtype) +
                           ', found dtype=' + str(x.dtype))
      # Check specific shape axes.
      if spec.axes:
        shape = x.get_shape().as_list()
        if shape is not None:
          for axis, value in spec.axes.items():
            if hasattr(value, 'value'):
              value = value.value
            if value is not None and shape[int(axis)] not in {value, None}:
              raise ValueError(
                  'Input ' + str(input_index) + ' of layer ' + self.name + ' is'
                  ' incompatible with the layer: expected axis ' + str(axis) +
                  ' of input shape to have value ' + str(value) +
                  ' but received input with shape ' + str(shape))
      # Check shape.
      if spec.shape is not None:
        shape = x.get_shape().as_list()
        if shape is not None:
          for spec_dim, dim in zip(spec.shape, shape):
            if spec_dim is not None and dim is not None:
              if spec_dim != dim:
                raise ValueError('Input ' + str(input_index) +
                                 ' is incompatible with layer ' + self.name +
                                 ': expected shape=' + str(spec.shape) +
                                 ', found shape=' + str(shape))


@tf_export('keras.layers.InputSpec', 'layers.InputSpec')
class InputSpec(object):
  """Specifies the ndim, dtype and shape of every input to a layer.

  Every layer should expose (if appropriate) an `input_spec` attribute:
  a list of instances of InputSpec (one per input tensor).

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.

  Arguments:
      dtype: Expected DataType of the input.
      shape: Shape tuple, expected shape of the input
          (may include None for unchecked axes).
      ndim: Integer, expected rank of the input.
      max_ndim: Integer, maximum rank of the input.
      min_ndim: Integer, minimum rank of the input.
      axes: Dictionary mapping integer axes to
          a specific dimension value.
  """

  def __init__(self,
               dtype=None,
               shape=None,
               ndim=None,
               max_ndim=None,
               min_ndim=None,
               axes=None):
    self.dtype = dtype
    self.shape = shape
    if shape is not None:
      self.ndim = len(shape)
    else:
      self.ndim = ndim
    self.max_ndim = max_ndim
    self.min_ndim = min_ndim
    self.axes = axes or {}

  def __repr__(self):
    spec = [('dtype=' + str(self.dtype)) if self.dtype else '',
            ('shape=' + str(self.shape)) if self.shape else '',
            ('ndim=' + str(self.ndim)) if self.ndim else '',
            ('max_ndim=' + str(self.max_ndim)) if self.max_ndim else '',
            ('min_ndim=' + str(self.min_ndim)) if self.min_ndim else '',
            ('axes=' + str(self.axes)) if self.axes else '']
    return 'InputSpec(%s)' % ', '.join(x for x in spec if x)


class Node(object):
  """A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer._inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer._outbound_nodes`.

  Arguments:
      outbound_layer: the layer that takes
          `input_tensors` and turns them into `output_tensors`
          (the node gets created when the `call`
          method of the layer was called).
      inbound_layers: a list of layers, the same length as `input_tensors`,
          the layers from where `input_tensors` originate.
      node_indices: a list of integers, the same length as `inbound_layers`.
          `node_indices[i]` is the origin node of `input_tensors[i]`
          (necessary since each inbound layer might have several nodes,
          e.g. if the layer is being shared with a different data stream).
      tensor_indices: a list of integers,
          the same length as `inbound_layers`.
          `tensor_indices[i]` is the index of `input_tensors[i]` within the
          output of the inbound layer
          (necessary since each inbound layer might
          have multiple tensor outputs, with each one being
          independently manipulable).
      input_tensors: list of input tensors.
      output_tensors: list of output tensors.
      arguments: dictionary of keyword arguments that were passed to the
          `call` method of the layer at the call that created the node.

  `node_indices` and `tensor_indices` are basically fine-grained coordinates
  describing the origin of the `input_tensors`.

  A node from layer A to layer B is added to:
    - A._outbound_nodes
    - B._inbound_nodes
  """

  def __init__(self,
               outbound_layer,
               inbound_layers,
               node_indices,
               tensor_indices,
               input_tensors,
               output_tensors,
               arguments=None):
    # Layer instance (NOT a list).
    if isinstance(outbound_layer, list):
      raise ValueError(
          '`outbound_layer` should be a layer instance, not a list.')
    # this is the layer that takes a list of input tensors
    # and turns them into a list of output tensors.
    # the current node will be added to
    # the inbound_nodes of outbound_layer.
    self.outbound_layer = outbound_layer

    # The following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.

    # List of layer instances.
    self.inbound_layers = inbound_layers
    # List of integers, 1:1 mapping with inbound_layers.
    self.node_indices = node_indices
    # List of integers, 1:1 mapping with inbound_layers.
    self.tensor_indices = tensor_indices

    # Following 2 properties:
    # tensor inputs and outputs of outbound_layer.

    # List of tensors. 1:1 mapping with inbound_layers.
    self.input_tensors = input_tensors
    # List of tensors, created by outbound_layer.call().
    self.output_tensors = output_tensors

    # Following 2 properties: input and output shapes.

    # List of shape tuples, shapes of input_tensors.
    self.input_shapes = [layers_util.static_shape(x) for x in input_tensors]
    # List of shape tuples, shapes of output_tensors.
    self.output_shapes = [layers_util.static_shape(x) for x in output_tensors]

    # Optional keyword arguments to layer's `call`.
    self.arguments = arguments

    # Add nodes to all layers involved.
    for layer in inbound_layers:
      if layer is not None:
        # For compatibility with external Keras, we use the deprecated
        # accessor here.
        layer.outbound_nodes.append(self)
    # For compatibility with external Keras, we use the deprecated
    # accessor here.
    outbound_layer.inbound_nodes.append(self)

  def get_config(self):
    inbound_names = []
    for layer in self.inbound_layers:
      if layer:
        inbound_names.append(layer.name)
      else:
        inbound_names.append(None)
    return {
        'outbound_layer': self.outbound_layer.name,
        'inbound_layers': inbound_names,
        'node_indices': self.node_indices,
        'tensor_indices': self.tensor_indices
    }


class _DeferredTensor(object):
  """Tensor-like object used to build graphs of layers in Eager mode.

  When calling a layer on a DeferredTensor, the layer will not perform any
  computation and will simply perfom shape inference to return new
  DeferredTensors with appropriate shape information. Thus DeferredTensor
  behaves like a graph-mode Tensor when manipulated by layers.
  """

  def __init__(self, shape, dtype, name=None):
    self.shape = tensor_shape.TensorShape(shape)
    if dtype is None:
      self.dtype = dtypes.as_dtype(np.float32)
    else:
      self.dtype = dtypes.as_dtype(dtype)
    self.name = name

  def get_shape(self):
    return self.shape

  def __str__(self):
    return "DeferredTensor('%s', shape=%s, dtype=%s)" % (self.name,
                                                         self.get_shape(),
                                                         self.dtype.name)

  def __repr__(self):
    return "<_DeferredTensor '%s' shape=%s dtype=%s>" % (self.name,
                                                         self.get_shape(),
                                                         self.dtype.name)


def _is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False


def _to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def _to_list(x):
  """This normalizes a list/tuple or single element into a list.

  If a single element is passed, we return
  a list of size 1 containing the element.

  Arguments:
    x: list or tuple or single element.

  Returns:
    A list.
  """
  if isinstance(x, (list, tuple)):
    return list(x)
  return [x]


def _add_elements_to_collection(elements, collection_list):
  if context.in_eager_mode():
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


def _is_all_none(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  # We cannot use Python's `any` because the iterable may return Tensors.
  for element in iterable:
    if element is not None:
      return False
  return True


def _have_all_keras_metadata(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  return all([hasattr(x, '_keras_history') for x in iterable])


def _collect_previous_mask(input_tensors):
  """Retrieves the output mask(s) of the previous node.

  Arguments:
      input_tensors: A tensor or list of tensors.

  Returns:
      A mask tensor or list of mask tensors.
  """
  input_tensors = nest.flatten(input_tensors)
  masks = []
  for x in input_tensors:
    if hasattr(x, '_keras_mask'):
      mask = x._keras_mask  # pylint: disable=protected-access
      masks.append(mask)
    else:
      masks.append(None)
  if len(masks) == 1:
    return masks[0]
  return masks


# A global dictionary mapping graph objects to an index of counters used
# for various layer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_LAYER_NAME_UIDS = weakref.WeakKeyDictionary()


def _get_default_graph_uid_map():
  graph = ops.get_default_graph()
  name_uid_map = PER_GRAPH_LAYER_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections.defaultdict(int)
    PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map
  return name_uid_map


def _unique_layer_name(name, name_uid_map=None, avoid_names=None, namespace='',
                       zero_based=False):
  """Makes a layer name (or arbitrary string) unique within a TensorFlow graph.

  Arguments:
    name: String name to make unique.
    name_uid_map: An optional defaultdict(int) to use when creating unique
      names. If None (default), uses a per-Graph dictionary.
    avoid_names: An optional set or dict with names which should not be used. If
      None (default) does not avoid any names.
    namespace: Gets a name which is unique within the (graph, namespace). Layers
      which are not Networks use a blank namespace and so get graph-global
      names.
    zero_based: If True, name sequences start with no suffix (e.g. "dense",
      "dense_1"). If False, naming is one-based ("dense_1", "dense_2").

  Returns:
    Unique string name.

  Example:

  ```python
  _unique_layer_name('dense')  # dense_1
  _unique_layer_name('dense')  # dense_2
  ```
  """
  if name_uid_map is None:
    name_uid_map = _get_default_graph_uid_map()
  if avoid_names is None:
    avoid_names = set()
  proposed_name = None
  while proposed_name is None or proposed_name in avoid_names:
    name_key = (namespace, name)
    if zero_based:
      number = name_uid_map[name_key]
      if number:
        proposed_name = name + '_' + str(number)
      else:
        proposed_name = name
      name_uid_map[name_key] += 1
    else:
      name_uid_map[name_key] += 1
      proposed_name = name + '_' + str(name_uid_map[name_key])
  return proposed_name
