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
# ==============================================================================
# pylint: disable=protected-access
"""Contains the base Layer class, from which all layers inherit."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect  # Necessary supplement to tf_inspect to deal with variadic args.
import re

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.Layer')
class Layer(checkpointable.CheckpointableBase):
  """Base layer class.

  This is the class from which all layers inherit.

  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing weights,
  losses, updates, and inter-layer connectivity.

  Users will just instantiate a layer and then treat it as a callable.

  We recommend that descendants of `Layer` implement the following methods:
  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_weight()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).

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

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    # These properties should be set by the user via keyword arguments.
    # note that 'dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
    }
    # Validate optional keyword arguments.
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

    self._init_set_name(name)

    activity_regularizer = kwargs.pop('activity_regularizer', None)
    if activity_regularizer and context.executing_eagerly():
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
    self._dtype = None if dtype is None else dtypes.as_dtype(dtype).name
    self._call_fn_args = estimator_util.fn_args(self.call)
    self._compute_previous_mask = ('mask' in self._call_fn_args or
                                   hasattr(self, 'compute_mask'))
    self._uses_inputs_arg = True

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    self._inbound_nodes = []
    self._outbound_nodes = []

    self.supports_masking = False

    # Manage input shape information if passed.
    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
      # In this case we will later create an input layer
      # to insert before the current layer
      if 'batch_input_shape' in kwargs:
        batch_input_shape = tuple(kwargs['batch_input_shape'])
      elif 'input_shape' in kwargs:
        if 'batch_size' in kwargs:
          batch_size = kwargs['batch_size']
        else:
          batch_size = None
        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
      self._batch_input_shape = batch_input_shape

    # Manage initial weight values if passed.
    if 'weights' in kwargs:
      self._initial_weights = kwargs['weights']
    else:
      self._initial_weights = None

  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = unique_layer_name(
          to_snake_case(self.__class__.__name__), zero_based=zero_based)
    else:
      self._name = name

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

  @activity_regularizer.setter
  def activity_regularizer(self, regularizer):
    """Optional regularizer function for the output of this layer."""
    self._activity_regularizer = regularizer

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
    if context.executing_eagerly():
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

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

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
    if context.executing_eagerly():
      return  # Updates already applied when in eager mode.

    def process_update(x):
      if isinstance(x, ops.Operation):
        return x
      elif hasattr(x, 'op'):
        return x.op
      else:
        return ops.convert_to_tensor(x)

    updates = generic_utils.to_list(updates)
    updates = [process_update(x) for x in updates]
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
    if context.executing_eagerly():
      raise RuntimeError('`get_updates_for()` not supported in Eager mode.')

    # Updates disabled if layer is not trainable and not explicitly stateful.
    if not self.trainable and not self.stateful:
      return []

    if inputs is None:
      # Requesting unconditional updates.
      return [x for x in self.updates if x._unconditional_update]  # pylint: disable=protected-access

    # Requesting input-conditional updates.
    inputs = nest.flatten(inputs)
    reachable = get_reachable_from_inputs(inputs, self.updates)
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
    if context.executing_eagerly():
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
    if context.executing_eagerly():
      # TODO(fchollet): it should be possible (and highly desirable) to support
      # `add_loss` in eager mode. This allows great convenience and flexibility
      # in defining custom losses on the fly (e.g. in VAEs).
      # Simply appending the loss value to `self._losses`
      # is the correct behavior.
      # The only caveat is that we need to force the user to only call
      # `add_loss` from inside a model or Layer's `call` method
      # (otherwise the loss computation cannot be backproped through).
      raise RuntimeError('Layer.add_loss not supported in Eager mode.')

    losses = generic_utils.to_list(losses)
    self._losses += losses
    if inputs is None:
      for loss in losses:
        loss._unconditional_loss = True  # pylint: disable=protected-access
    else:
      for loss in losses:
        loss._unconditional_loss = False  # pylint: disable=protected-access

  def get_losses_for(self, inputs):
    """Retrieves losses relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.

    Raises:
      RuntimeError: If called in Eager mode.
    """
    if context.executing_eagerly():
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
    reachable = get_reachable_from_inputs(inputs, self.losses)
    losses = []
    for loss in self.losses:
      if loss in reachable:
        losses.append(loss)
    return losses

  def _name_scope(self):
    return self.name

  def build(self, _):
    """Creates the variables of the layer."""
    self.built = True

  def add_variable(self, *args, **kwargs):
    """Alias for `add_weight`."""
    return self.add_weight(*args, **kwargs)

  def add_weight(self, name, shape,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 constraint=None,
                 partitioner=None,
                 use_resource=None,
                 getter=None):
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
      partitioner: Partitioner to be passed to the `Checkpointable` API.
      use_resource: Whether to use `ResourceVariable`.
      getter: Variable getter argument to be passed to the `Checkpointable` API.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partioned variable regularization and
        eager execution is enabled.
      ValueError: When giving unsupported dtype and no initializer.
    """
    if dtype is None:
      dtype = self.dtype or backend.floatx()
    else:
      dtype = dtypes.as_dtype(dtype)
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    constraint = constraints.get(constraint)

    # Initialize variable when no initializer provided
    if initializer is None:
      # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
      if dtype.is_floating:
        initializer = initializers.glorot_uniform()
      # If dtype is DT_INT/DT_UINT, provide a default value `zero`
      # If dtype is DT_BOOL, provide a default value `FALSE`
      elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        initializer = initializers.zeros()
      # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
      else:
        raise ValueError('An initializer for variable %s of type %s is required'
                         ' for layer %s' % (name, dtype.base_dtype, self.name))

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        # TODO(allenl): a `make_variable` equivalent should be added as a
        # `Checkpointable` method.
        getter=getter or make_variable,
        # Manage errors in Layer rather than Checkpointable.
        overwrite=True,
        initializer=initializer,
        dtype=dtypes.as_dtype(dtype),
        constraint=constraint,
        trainable=trainable and self.trainable,
        partitioner=partitioner,
        use_resource=use_resource)

    if regularizer is not None:
      # TODO(fchollet): in the future, this should be handled at the
      # level of variable creation, and weight regularization losses
      # should be variable attributes.
      self._handle_weight_regularization(name, variable, regularizer)

    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    return variable

  def _handle_weight_regularization(self, name, variable, regularizer):
    # `init_graph` should point to the graph in which variable initialization
    # will occur; it should be None if and only if initialization will take
    # place in the eager context.
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
      else:
        # Initialization ops will not be lifted out of the default graph.
        init_graph = default_graph

    if init_graph is not None:  # pylint: disable=protected-access
      # The variable was created and initialized in a graph.
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
    elif regularizer:  # initialization took place in an eager context
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

  def _handle_activity_regularization(self, inputs, outputs):
    # Apply activity regularization.
    # Note that it should be applied every time the layer creates a new
    # output, since it is output-specific.
    if self._activity_regularizer:
      output_list = nest.flatten(outputs)
      for output in output_list:
        with ops.name_scope('ActivityRegularizer'):
          activity_regularization = self._activity_regularizer(output)
        self.add_loss(activity_regularization, inputs=inputs)

  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
    """This is where the layer's logic lives.

    Arguments:
        inputs: Input tensor, or list/tuple of input tensors.
        **kwargs: Additional keyword arguments.

    Returns:
        A tensor or list/tuple of tensors.
    """
    return inputs

  def __call__(self, inputs, *args, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).

    Note:
      - The following optional keyword arguments are reserved for specific uses:
        * `training`: Boolean scalar tensor of Python boolean indicating
          whether the `call` is meant for training or inference.
        * `mask`: Boolean input mask.
      - If the layer's `call` method takes a `mask` argument (as some Keras
        layers do), its default value will be set to the mask generated
        for `inputs` by the previous layer (if `input` did come from
        a layer that generated a corresponding mask, i.e. if it came from
        a Keras layer with masking support.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
    """
    input_list = nest.flatten(inputs)

    build_graph = not context.executing_eagerly()
    # TODO(fchollet, allenl): Make deferred mode work with subclassed Models
    # which don't use an "inputs" argument.
    in_deferred_mode = isinstance(input_list[0], DeferredTensor)

    # Handle Keras mask propagation from previous layer to current layer.
    previous_mask = None
    if (not hasattr(self, '_compute_previous_mask') or
        self._compute_previous_mask):
      previous_mask = collect_previous_mask(inputs)
      if not hasattr(self, '_call_fn_args'):
        self._call_fn_args = estimator_util.fn_args(self.call)
      if ('mask' in self._call_fn_args and 'mask' not in kwargs and
          not is_all_none(previous_mask)):
        # The previous layer generated a mask, and mask was not explicitly pass
        # to __call__, hence we set previous_mask as the default value.
        kwargs['mask'] = previous_mask

    input_shapes = None

    with ops.name_scope(self._name_scope()):
      if not self.built:
        if not build_graph:
          # Activity regularization is currently unsupported in Eager mode.
          if self._activity_regularizer:
            raise ValueError(
                'activity_regularizer currently unsupported with '
                'eager execution enabled. Found an activity_regularizer in '
                '%s(%s).' % (self.__class__.__name__, self))
        if not build_graph and not in_deferred_mode:
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
        if all(hasattr(x, 'get_shape') for x in input_list):
          input_shapes = nest.map_structure(lambda x: x.get_shape(), inputs)
        self.build(input_shapes)

      # Check input assumptions set after layer building, e.g. input shape.
      if build_graph or in_deferred_mode:
        self._assert_input_compatibility(inputs)

      if not in_deferred_mode:
        outputs = self.call(inputs, *args, **kwargs)
        if outputs is None:
          raise ValueError('A layer\'s `call` method should return a Tensor '
                           'or a list of Tensors, not None (layer: ' +
                           self.name + ').')
      else:
        # Deferred mode behavior: use `compute_output_shape` to
        # infer the number of outputs of the layer and their shapes.
        if input_shapes is None:
          input_shapes = nest.map_structure(lambda x: x.get_shape(), inputs)

        output_shapes = self.compute_output_shape(input_shapes)
        output_shapes = nest.flatten(output_shapes)
        outputs = [
            # TODO(fchollet): name the deferred tensors?
            DeferredTensor(shape=shape, dtype=self._dtype)
            for shape in output_shapes
        ]
        if len(outputs) == 1:
          outputs = outputs[0]

      if build_graph:
        self._handle_activity_regularization(inputs, outputs)
        # TODO(fchollet): consider enabling masking for Eager mode.
        self._set_mask_metadata(inputs, outputs, previous_mask)

      if in_deferred_mode or build_graph and have_all_keras_metadata(inputs):
        inputs, outputs = self._set_connectivity_metadata_(
            inputs, outputs, args, kwargs)

      self.built = True
      if context.executing_eagerly():
        return outputs

      if hasattr(self, '_symbolic_set_inputs') and not self.inputs:
        # Subclassed network: explicitly set metadata normally set by a call to
        # self._set_inputs(). This is not relevant in eager execution.
        self._symbolic_set_inputs(inputs, outputs)

      if in_deferred_mode or build_graph:
        self._set_learning_phase_metadata(inputs, outputs)

    # Optionally load weight values that were specified at layer instantiation.
    # TODO(fchollet): consider enabling this with eager execution too.
    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
      self.set_weights(self._initial_weights)
      del self._initial_weights
    return outputs

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

  def _set_learning_phase_metadata(self, inputs, outputs):
    # Update learning phase info. To work with subclassed models,
    # this should be done even if Keras metadata is absent.
    output_tensors = generic_utils.to_list(outputs)
    uses_lp = any(
        [getattr(x, '_uses_learning_phase', False)
         for x in generic_utils.to_list(inputs)])
    uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
    for i in range(len(output_tensors)):
      try:
        output_tensors[i]._uses_learning_phase = getattr(
            output_tensors[i], '_uses_learning_phase', False) or uses_lp
      except AttributeError:
        # An output element happens to be a C type (such as tuple or dict).
        # We don't track learning phase info in such edge cases.
        pass

  def _set_mask_metadata(self, inputs, outputs, previous_mask):
    if hasattr(self, 'compute_mask'):
      output_mask = self.compute_mask(inputs, previous_mask)
      if isinstance(outputs, (list, tuple)):
        if output_mask is None:
          output_mask = [None for _ in range(len(outputs))]
        for x, m in zip(outputs, output_mask):
          try:
            x._keras_mask = m  # pylint: disable=protected-access
          except AttributeError:
            pass  # C type such as dict. Masking not supported in this case.
      else:
        try:
          outputs._keras_mask = output_mask  # pylint: disable=protected-access
        except AttributeError:
          pass  # C type such as dict. Masking not supported in this case.

  def _set_connectivity_metadata_(self, inputs, outputs, args, kwargs):
    if args and getattr(self, '_uses_inputs_arg', True):
      raise TypeError(
          'This Layer takes an `inputs` argument to call(), and only the '
          '`inputs` argument may be specified as a positional argument. '
          'Pass everything else as a keyword argument (those arguments will'
          ' not be tracked as inputs to the Layer).')

    # If the layer returns tensors from its inputs, unmodified,
    # we copy them to avoid loss of tensor metadata.
    output_ls = nest.flatten(outputs)
    output_ls_copy = []
    for x in output_ls:
      if x in nest.flatten(inputs):
        with ops.name_scope(self.name):
          x = array_ops.identity(x)
      output_ls_copy.append(x)
    if len(output_ls_copy) == 1:
      outputs = output_ls_copy[0]
    else:
      outputs = output_ls_copy

    inputs, kwargs = self._inputs_from_call_args(
        call_args=(inputs,) + args, call_kwargs=kwargs)
    # Add an inbound node to the layer, so it can keep track of this call.
    # This updates the layer history of the output tensor(s).
    kwargs.pop('mask', None)  # `mask` should not be serialized.
    self._add_inbound_node(
        input_tensors=inputs, output_tensors=outputs, arguments=kwargs)
    return inputs, outputs

  def _inputs_from_call_args(self, call_args, call_kwargs):
    """Get Layer inputs from __call__ *args and **kwargs.

    Args:
      call_args: The positional arguments passed to __call__.
      call_kwargs: The keyword argument dict passed to __call__.

    Returns:
      A tuple of (inputs, non_input_kwargs). These may be the same objects as
      were passed in (call_args and call_kwargs).
    """
    if getattr(self, '_uses_inputs_arg', True):
      assert len(call_args) == 1  # TypeError raised earlier in __call__.
      return call_args[0], call_kwargs
    else:
      call_arg_spec = tf_inspect.getargspec(self.call)
      # There is no explicit "inputs" argument expected or provided to
      # call(). Arguments which have default values are considered non-inputs,
      # and arguments without are considered inputs.
      if call_arg_spec.defaults:
        if call_arg_spec.varargs is not None:
          raise TypeError(
              'Layer.call() may not accept both *args and arguments with '
              'default values (unable to determine which are inputs to the '
              'Layer).')
        keyword_arg_names = set(
            call_arg_spec.args[-len(call_arg_spec.defaults):])
      else:
        keyword_arg_names = set()
        # Training is never an input argument name, to allow signatures like
        # call(x, training).
      keyword_arg_names.add('training')
      _, unwrapped_call = tf_decorator.unwrap(self.call)
      bound_args = inspect.getcallargs(
          unwrapped_call, *call_args, **call_kwargs)
      if call_arg_spec.keywords is not None:
        var_kwargs = bound_args.pop(call_arg_spec.keywords)
        bound_args.update(var_kwargs)
        keyword_arg_names = keyword_arg_names.union(var_kwargs.keys())
      all_args = call_arg_spec.args
      if all_args and bound_args[all_args[0]] is self:
        # Ignore the 'self' argument of methods
        bound_args.pop(call_arg_spec.args[0])
        all_args = all_args[1:]
      non_input_arg_values = {}
      input_arg_values = []
      remaining_args_are_keyword = False
      for argument_name in all_args:
        if argument_name in keyword_arg_names:
          remaining_args_are_keyword = True
        else:
          if remaining_args_are_keyword:
            raise TypeError(
                'Found a positional argument to call() after a non-input '
                'argument. All arguments after "training" must be keyword '
                'arguments, and are not tracked as inputs to the Layer.')
        if remaining_args_are_keyword:
          non_input_arg_values[argument_name] = bound_args[argument_name]
        else:
          input_arg_values.append(bound_args[argument_name])
      if call_arg_spec.varargs is not None:
        input_arg_values.extend(bound_args[call_arg_spec.varargs])
      return input_arg_values, non_input_arg_values

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    Assumes that the layer will be built
    to match that input shape provided.

    Arguments:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.

    Returns:
        An input shape tuple.
    """
    raise NotImplementedError

  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    """Computes an output mask tensor.

    Arguments:
        inputs: Tensor or list of tensors.
        mask: Tensor or list of tensors.

    Returns:
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    """
    if not self.supports_masking:
      if mask is not None:
        if isinstance(mask, list):
          if any(m is not None for m in mask):
            raise TypeError('Layer ' + self.name + ' does not support masking, '
                            'but was passed an input_mask: ' + str(mask))
        else:
          raise TypeError('Layer ' + self.name + ' does not support masking, '
                          'but was passed an input_mask: ' + str(mask))
      # masking not explicitly supported: return None as mask
      return None
    # if masking is explicitly supported, by default
    # carry over the input mask
    return mask

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

  def get_input_mask_at(self, node_index):
    """Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple inputs).
    """
    inputs = self.get_input_at(node_index)
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  def get_output_mask_at(self, node_index):
    """Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple outputs).
    """
    output = self.get_output_at(node_index)
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

  @property
  def input_mask(self):
    """Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input mask tensor (potentially None) or list of input
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    inputs = self.input
    if isinstance(inputs, list):
      return [getattr(x, '_keras_mask', None) for x in inputs]
    else:
      return getattr(inputs, '_keras_mask', None)

  @property
  def output_mask(self):
    """Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Output mask tensor (potentially None) or list of output
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    """
    output = self.output
    if isinstance(output, list):
      return [getattr(x, '_keras_mask', None) for x in output]
    else:
      return getattr(output, '_keras_mask', None)

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

  def set_weights(self, weights):
    """Sets the weights of the layer, from Numpy arrays.

    Arguments:
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).

    Raises:
        ValueError: If the provided weights list does not match the
            layer's specifications.
    """
    params = self.weights
    if len(params) != len(weights):
      raise ValueError('You called `set_weights(weights)` on layer "' +
                       self.name + '" with a  weight list of length ' +
                       str(len(weights)) + ', but the layer was expecting ' +
                       str(len(params)) + ' weights. Provided weights: ' +
                       str(weights)[:50] + '...')
    if not params:
      return
    weight_value_tuples = []
    param_values = backend.batch_get_value(params)
    for pv, p, w in zip(param_values, params, weights):
      if pv.shape != w.shape:
        raise ValueError('Layer weight shape ' + str(pv.shape) +
                         ' not compatible with '
                         'provided weight shape ' + str(w.shape))
      weight_value_tuples.append((p, w))
    backend.batch_set_value(weight_value_tuples)

  def get_weights(self):
    """Returns the current weights of the layer.

    Returns:
        Weights values as a list of numpy arrays.
    """
    params = self.weights
    return backend.batch_get_value(params)

  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).

    Returns:
        Python dictionary.
    """
    config = {'name': self.name, 'trainable': self.trainable}
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    if hasattr(self, 'dtype'):
      config['dtype'] = self.dtype
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
        config: A Python dictionary, typically the
            output of get_config.

    Returns:
        A layer instance.
    """
    return cls(**config)


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
    self.input_shapes = [static_shape(x) for x in input_tensors]
    # List of shape tuples, shapes of output_tensors.
    self.output_shapes = [static_shape(x) for x in output_tensors]

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


class DeferredTensor(object):
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
    return "<DeferredTensor '%s' shape=%s dtype=%s>" % (self.name,
                                                        self.get_shape(),
                                                        self.dtype.name)


def shape_type_conversion(fn):
  """Decorator that handles tuple/TensorShape conversion.

  Used in `compute_output_shape` and `build`.

  Arguments:
    fn: function to wrap.

  Returns:
    Wrapped function.
  """

  def wrapper(instance, input_shape):
    if input_shape is not None:
      if isinstance(input_shape, list):
        input_shape = [
            tuple(tensor_shape.TensorShape(x).as_list()) for x in input_shape]
      else:
        input_shape = tuple(tensor_shape.TensorShape(input_shape).as_list())
    output_shape = fn(instance, input_shape)
    if output_shape is not None:
      if isinstance(output_shape, list):
        return [tensor_shape.TensorShape(x) for x in output_shape]
      return tensor_shape.TensorShape(output_shape)

  return wrapper


def object_list_uid(object_list):
  """Creates a single string from object ids."""
  object_list = nest.flatten(object_list)
  return ', '.join([str(abs(id(x))) for x in object_list])


def static_shape(x):
  """Get the static shape of a Tensor, or None if it is unavailable."""
  if x is None:
    return None
  try:
    return tuple(x.get_shape().as_list())
  except ValueError:
    return None


def get_reachable_from_inputs(inputs, targets=None):
  """Returns the set of tensors/ops reachable from `inputs`.

  Stops if all targets have been found (target is optional).

  Only valid in Symbolic mode, not Eager mode.

  Args:
    inputs: List of tensors.
    targets: List of tensors.

  Returns:
    A set of tensors reachable from the inputs (includes the inputs themselves).
  """
  reachable = set(inputs)
  if targets:
    targets = set(targets)
  queue = inputs[:]

  while queue:
    x = queue.pop()
    if isinstance(x, ops.Operation):
      outputs = x.outputs[:] or []
      outputs += x._control_outputs
    elif isinstance(x, ops.Tensor):
      outputs = x.consumers()
    elif isinstance(x, tf_variables.Variable):
      outputs = [x.op]
    else:
      raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))

    for y in outputs:
      if y not in reachable:
        reachable.add(y)
        queue.insert(0, y)

    if targets and targets.issubset(reachable):
      return reachable
  return reachable


def unique_layer_name(name, name_uid_map=None, avoid_names=None, namespace='',
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
    name_uid_map = get_default_graph_uid_map()
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


def to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def is_all_none(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  # We cannot use Python's `any` because the iterable may return Tensors.
  for element in iterable:
    if element is not None:
      return False
  return True


def have_all_keras_metadata(iterable_or_element):
  if not isinstance(iterable_or_element, (list, tuple)):
    iterable = [iterable_or_element]
  else:
    iterable = iterable_or_element
  return all([hasattr(x, '_keras_history') for x in iterable])


def collect_previous_mask(input_tensors):
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


def is_tensor_or_tensor_list(v):
  v = nest.flatten(v)
  if v and isinstance(v[0], ops.Tensor):
    return True
  else:
    return False


def get_default_graph_uid_map():
  # TODO(fchollet): refactor this into backend.
  graph = ops.get_default_graph()
  name_uid_map = backend.PER_GRAPH_LAYER_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections.defaultdict(int)
    backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map
  return name_uid_map


def make_variable(name,
                  shape=None,
                  dtype=dtypes.float32,
                  initializer=None,
                  partition_info=None,
                  trainable=True,
                  caching_device=None,
                  validate_shape=True,
                  constraint=None,
                  use_resource=None,
                  partitioner=None):  # pylint: disable=unused-argument
  """Temporary util to create a variable (relies on `variable_scope.variable`).

  Some reuse-related technicalities prevent us from using
  `variable_scope.get_variable()` directly, so we use a subcomponent
  that has fewer constraints (`variable_scope.variable()`).

  In the longer term, it seems like a similar "default variable creator" method
  should exist in `CheckpointableBase` instead. When this happens, we can get
  rid of this temporary solution.

  TODO(fchollet): remove this method when no longer needed.
  TODO(fchollet): handle `partitioner` argument.

  Arguments:
    name: Variable name.
    shape: Variable shape.
    dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
    initializer: Initializer instance (callable).
    partition_info: Not handled at this time.
    trainable: Whether the variable should be part of the layer's
      "trainable_variables" (e.g. variables, biases)
      or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
      Note, if the current variable scope is marked as non-trainable
      then this parameter is ignored and any added variables are also
      marked as non-trainable.
    caching_device: Passed to `vs.variable`.
    validate_shape: Passed to `vs.variable`.
    constraint: Constraint instance (callable).
    use_resource: Whether to use a `ResourceVariable`.
    partitioner: Not handled at this time.

  Returns:
    Variable instance.
  """
  initializing_from_value = False
  if initializer is not None and not callable(initializer):
    initializing_from_value = True

  with ops.init_scope():
    if initializing_from_value:
      init_val = initializer
      variable_dtype = None
    else:
      # Instantiate initializer if provided initializer is a type object.
      if isinstance(initializer, type(init_ops.Initializer)):
        initializer = initializer(dtype=dtype)
      init_val = lambda: initializer(  # pylint: disable=g-long-lambda
          shape, dtype=dtype, partition_info=partition_info)
      variable_dtype = dtype.base_dtype
  if use_resource is None:
    use_resource = True

  v = vs.variable(
      initial_value=init_val,
      name=name,
      trainable=trainable,
      caching_device=caching_device,
      dtype=variable_dtype,
      validate_shape=validate_shape,
      constraint=constraint,
      use_resource=use_resource)
  return v
