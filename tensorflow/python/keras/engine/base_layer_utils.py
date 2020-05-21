# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Contains private utilities used mainly by the base Layer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import threading

from tensorflow.python import tf2
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.training.tracking import base as tracking
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import keras_export

_call_context = threading.local()


def create_mean_metric(value, name=None):
  # import keras will import base_layer and then this module, and metric relies
  # on base_layer, which result into a cyclic dependency.
  from tensorflow.python.keras import metrics as metrics_module  # pylint: disable=g-import-not-at-top
  metric_obj = metrics_module.Mean(name=name, dtype=value.dtype)
  return metric_obj, metric_obj(value)


def make_variable(name,
                  shape=None,
                  dtype=dtypes.float32,
                  initializer=None,
                  trainable=None,
                  caching_device=None,
                  validate_shape=True,
                  constraint=None,
                  use_resource=None,
                  collections=None,
                  synchronization=tf_variables.VariableSynchronization.AUTO,
                  aggregation=tf_variables.VariableAggregation.NONE,
                  partitioner=None):  # pylint: disable=unused-argument
  """Temporary util to create a variable (relies on `variable_scope.variable`).

  Some reuse-related technicalities prevent us from using
  `variable_scope.get_variable()` directly, so we use a subcomponent
  that has fewer constraints (`variable_scope.variable()`).

  In the longer term, it seems like a similar "default variable creator" method
  should exist in `Trackable` instead. When this happens, we can get
  rid of this temporary solution.

  TODO(fchollet): remove this method when no longer needed.

  Arguments:
    name: Variable name.
    shape: Variable shape.
    dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
    initializer: Initializer instance (callable).
    trainable: Whether the variable should be part of the layer's
      "trainable_variables" (e.g. variables, biases)
      or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
      Note, if the current variable scope is marked as non-trainable
      then this parameter is ignored and any added variables are also
      marked as non-trainable. `trainable` defaults to `True` unless
      `synchronization` is set to `ON_READ`.
    caching_device: Passed to `tf.Variable`.
    validate_shape: Passed to `tf.Variable`.
    constraint: Constraint instance (callable).
    use_resource: Whether to use a `ResourceVariable`.
    collections: List of graph collections keys. The new variable is added to
      these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
    synchronization: Indicates when a distributed a variable will be
      aggregated. Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses
      when to synchronize. If `synchronization` is set to `ON_READ`,
      `trainable` must not be set to `True`.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.
    partitioner: Not handled at this time.

  Returns:
    Variable instance.
  """
  initializing_from_value = False
  if initializer is not None and not callable(initializer):
    initializing_from_value = True

  if initializing_from_value:
    init_val = initializer
    variable_dtype = None
  else:
    # Instantiate initializer if provided initializer is a type object.
    if isinstance(
        initializer,
        (type(init_ops.Initializer), type(init_ops_v2.Initializer))):
      initializer = initializer()
    init_val = functools.partial(initializer, shape, dtype=dtype)
    variable_dtype = dtype.base_dtype
  if use_resource is None:
    use_resource = True

  # TODO(apassos,rohanj) figure out how to remove collections from here so we
  # can remove the V1.
  variable_shape = tensor_shape.TensorShape(shape)
  return tf_variables.VariableV1(
      initial_value=init_val,
      name=name,
      trainable=trainable,
      caching_device=caching_device,
      dtype=variable_dtype,
      validate_shape=validate_shape,
      constraint=constraint,
      use_resource=use_resource,
      collections=collections,
      synchronization=synchronization,
      aggregation=aggregation,
      shape=variable_shape if variable_shape else None)


def collect_previous_mask(input_tensors):
  """Retrieves the output mask(s) of the previous node.

  Arguments:
      input_tensors: An arbitrary structure of Tensors.

  Returns:
      A mask tensor or list of mask tensors.
  """

  def _collect_previous_mask(x):
    return getattr(x, '_keras_mask', None)

  return nest.map_structure(_collect_previous_mask, input_tensors)


def have_all_keras_metadata(tensors):
  return all(hasattr(x, '_keras_history') for x in nest.flatten(tensors))


def generate_placeholders_from_shape(shape):
  return array_ops.placeholder(shape=shape, dtype=backend.floatx())


def create_keras_history(tensors):
  """Wraps TensorFlow Operations for compatibility with the Functional API.

  This method checks to see if a Tensor in `tensors` is missing Keras metadata
  and has its origin in a Keras `Input` Layer. If so, this method will replace
  the raw TensorFlow Operations that created this tensor with
  `TensorFlowOpLayer` instances that create identical operations.

  Any Tensors not originating from a Keras `Input` Layer will be treated as
  constants when constructing `TensorFlowOpLayer` instances.

  Arguments:
    tensors: A structure of Tensors, some of which come from raw TensorFlow
      operations and need to have Keras metadata assigned to them.

  Returns:
    created_layers: List. The `TensorFlowOpLayer` instances created to wrap
      the raw Tensorflow operations.
  """
  _, created_layers = _create_keras_history_helper(tensors, set(), [])
  return created_layers


def _create_keras_history_helper(tensors, processed_ops, created_layers):
  """Helper method for `create_keras_history`.

  Arguments:
    tensors: A structure of Tensors for which to create Keras metadata.
    processed_ops: Set. TensorFlow operations that have already been wrapped in
      `TensorFlowOpLayer` instances.
    created_layers: List. The `TensorFlowOpLayer` instances created.

  Returns:
    Tuple. First element is the updated set of TensorFlow Operations that
    have been wrapped in `TensorFlowOpLayer` instances. Second element is
    a list of the `TensorFlowOpLayer` instances created.
  """
  # Import of `base_layer` needed in order to create `TensorFlowOpLayer`.
  # Cannot be imported at top because of circular dependencies.
  # TODO(omalleyt): Resolve circular dependency.
  from tensorflow.python.keras.engine import base_layer  # pylint: disable=g-import-not-at-top
  tensor_list = nest.flatten(tensors)
  for tensor in tensor_list:
    if getattr(tensor, '_keras_history', None) is not None:
      continue
    op = tensor.op  # The Op that created this Tensor.
    if op not in processed_ops:
      if op.type.startswith('Sparse'):
        lambda_example = """
        weights_mult = lambda x: tf.sparse.sparse_dense_matmul(x, weights)
        output = tf.keras.layers.Lambda(weights_mult)(input)
        """
        raise ValueError(
            'Sparse ops are not supported with functional models with built-in '
            'layer wrapping. Please wrap the sparse ops in a Lambda layer like'
            ': \n{lambda_example}\n'.format(lambda_example=lambda_example))

      # Recursively set `_keras_history`.
      op_inputs = list(op.inputs)
      constants = {}
      layer_inputs = []
      for i, op_input in enumerate(op_inputs):
        if uses_keras_history(op_input):
          layer_inputs.append(op_input)
        else:
          # Treat any value not originating from a `keras.Input` as
          # a constant. Variables cannot be supported.
          ds_with_session = (
              distribution_strategy_context.in_cross_replica_context() and
              not ops.executing_eagerly_outside_functions())
          using_xla = control_flow_util.GraphOrParentsInXlaContext(
              ops.get_default_graph())
          if ds_with_session or using_xla:
            # In Legacy Graph mode, evaluating here makes Session be
            # configured improperly. The downside of this is that saving
            # via `get_config` breaks, but SavedModel still works.
            constants[i] = op_input
          else:
            with ops.init_scope():
              constants[i] = backend.function([], op_input)([])
      layer_inputs = unnest_if_single_tensor(layer_inputs)
      processed_ops, created_layers = _create_keras_history_helper(
          layer_inputs, processed_ops, created_layers)
      name = op.name
      node_def = op.node_def.SerializeToString()
      op_layer = base_layer.TensorFlowOpLayer(
          node_def, constants=constants, name=name)
      created_layers.append(op_layer)
      op_layer._set_connectivity_metadata(  # pylint: disable=protected-access
          args=(layer_inputs,),
          kwargs={},
          outputs=op.outputs)
      processed_ops.update([op])
  return processed_ops, created_layers


def unnest_if_single_tensor(input_tensors):
  # Preserve compatibility with older configs
  flat_input_tensors = nest.flatten(input_tensors)
  # If this is a single element but not a dict, unwrap. If this is a dict,
  # assume the first layer expects a dict (as is the case with a
  # DenseFeatures layer); pass through.
  if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
    input_tensors = flat_input_tensors[0]
  return input_tensors


def needs_keras_history(tensors, ignore_call_context=False):
  """Check if any Tensors need to be wrapped in TensorFlowOpLayers.

  This will never return True inside a sublayer, because sublayers
  do not need to create Keras History. Otherwise, this returns True
  if one or more of `tensors` originates from a `keras.Input` and
  does not have `_keras_history` set.

  Arguments:
    tensors: An arbitrary nested structure of Tensors.
    ignore_call_context: Whether to ignore the check of if currently
      outside of a `call` context. This is `True` when creating
      KerasHistory inside `Node`, where we always know that Tensors
      are being used with the Functional API.

  Returns:
    Bool, whether at least one Tensor needs to be wrapped.
  """
  input_tensors = nest.flatten(tensors)
  if call_context().in_call and not ignore_call_context:
    return False
  if all(
      getattr(tensor, '_keras_history', None) is not None
      for tensor in input_tensors):
    # KerasHistory already set.
    return False
  return uses_keras_history(tensors)


def is_in_keras_graph():
  """Returns if currently executing inside of a Keras graph."""
  return call_context().in_keras_graph


def is_in_eager_or_tf_function():
  """Returns if in eager mode or inside of a tf.function."""
  return context.executing_eagerly() or is_in_tf_function()


def is_in_tf_function():
  """Returns if inside of a tf.function."""
  # Check if running in V1 graph mode.
  if not ops.executing_eagerly_outside_functions():
    return False
  if not ops.inside_function():
    return False
  # Check if inside Keras FuncGraph.
  if is_in_keras_graph():
    return False
  # Check for a v1 `wrap_function` FuncGraph.
  graph = ops.get_default_graph()
  if (getattr(graph, 'name', False) and
      graph.name.startswith('wrapped_function')):
    return False
  return True


def uses_keras_history(tensors):
  """Check if at least one Tensor originates from a `keras.Input`.

  This is `True` if at least one Tensor has its origin in a `keras.Input`.
  Any Tensor that originates from a `keras.Input` will have a dependency
  Tensor with a `_keras_history` attribute attached. Tensors that have
  already been checked to not originate from a `keras.Input`
  are marked as `_keras_history_checked`.

  Arguments:
    tensors: An arbitrary nested structure of Tensors.

  Returns:
    Bool, whether at least one Tensor originates from a `keras.Input`.
  """
  checked_tensors = set()
  tensors_to_check = nest.flatten(tensors)

  while tensors_to_check:
    new_tensors_to_check = []
    for tensor in tensors_to_check:
      if id(tensor) in checked_tensors:
        continue

      checked_tensors.add(id(tensor))

      if getattr(tensor, '_keras_history_checked', None) is not None:
        continue
      if getattr(tensor, '_keras_history', None) is not None:
        return True

      try:
        new_tensors_to_check.extend(tensor.op.inputs)
      except AttributeError:
        # In case `tensor` is a Variable created in an Eager context.
        pass

    tensors_to_check = new_tensors_to_check

  # Mark that these Tensors have been checked once for `_keras_history`,
  # and should not be checked again for performance reasons.
  mark_checked(tensors)
  return False


def mark_checked(tensors):
  """Marks that these Tensors should not be tracked.

  This prevents Layers from attempting to create TensorFlowOpLayers
  for these Tensors.

  Arguments:
    tensors: An arbitrary structure of Tensors.
  """

  def _mark_checked(tensor):
    tensor._keras_history_checked = True  # pylint: disable=protected-access

  nest.map_structure(_mark_checked, tensors)


def call_context():
  """Returns currently active `CallContext`."""
  if getattr(_call_context, 'call_context', None) is None:
    _call_context.call_context = CallContext()
  return _call_context.call_context


control_flow_util_v2._register_keras_layer_context_function(call_context)  # pylint: disable=protected-access


class CallContext(object):
  """Keeps track of properties currently inside a Layer/Model's `call`.

  Attributes:
    layer: The `Layer` whose `call` is currently active.
    inputs: The inputs to the currently active `Layer`.
    frozen: Whether currently executing inside a `Layer` with `trainable` set to
      `False`.
    in_call: Whether currently inside the `call` of a Layer.
    training: Whether currently executing in training or inference mode.
    in_keras_graph: Whether executing inside the Keras Graph.
    saving: Whether currently saving to SavedModel.
  """

  def __init__(self):
    self.layer = None
    self.inputs = None
    self.frozen = False
    self.in_call = False
    self.training = None
    self._in_keras_graph = False
    self.saving = False

  @tf_contextlib.contextmanager
  def enter(self, layer, inputs, build_graph, training, saving=None):
    """Push a Layer and its inputs and state onto the current call context."""
    prev_layer = self.layer
    prev_inputs = self.inputs
    prev_frozen = self.frozen
    prev_in_call = self.in_call
    prev_training = self.training
    prev_in_keras_graph = self._in_keras_graph
    prev_saving = self.saving

    self.layer = layer
    self.inputs = inputs
    self.frozen = self.frozen or not layer.trainable
    self.in_call = True
    self.training = training
    self._in_keras_graph = (
        self._in_keras_graph or
        (build_graph and
         getattr(backend.get_graph(), 'name', None) == 'keras_graph'))
    self.saving = prev_saving if saving is None else saving

    try:
      yield
    finally:
      self.layer = prev_layer
      self.inputs = prev_inputs
      self.frozen = prev_frozen
      self.in_call = prev_in_call
      self.training = prev_training
      self._in_keras_graph = prev_in_keras_graph
      self.saving = prev_saving

  @property
  def in_keras_graph(self):
    # Returns True even if in a subgraph of the Keras graph, such as those
    # created by control flow ops.
    if context.executing_eagerly():
      return False
    return (self._in_keras_graph or
            getattr(backend.get_graph(), 'name', None) == 'keras_graph')


def training_arg_passed_to_call(argspec, args, kwargs):
  """Returns whether a user passed the `training` argument in `__call__`."""
  # `argspec.args` starts with ['self', 'inputs']
  full_args = dict(zip(argspec.args[2:], args))
  full_args.update(kwargs)
  return 'training' in full_args and full_args['training'] is not None


def autocast_context_manager(dtype):
  """Returns a context manager to autocast AutoCastVariables.

  Under this context manager, AutoCastVariables will be casted to `dtype` if
  `dtype` is floating-point. Otherwise, AutoCastVariables will not be casted.

  Args:
    dtype: The dtype to cast AutoCastVariables to, or None.

  Returns:
    A context manager to automatically cast AutoCastVariables.
  """
  if dtype and not dtypes.as_dtype(dtype).is_floating:
    dtype = None
  return ops.get_default_graph()._enable_auto_casting_variables(dtype)  # pylint: disable=protected-access


def is_subclassed(layer):
  """Returns True if the object is a subclassed layer or subclassed model."""
  return (layer.__module__.find('keras.engine') == -1 and
          layer.__module__.find('keras.layers') == -1)


def from_saved_model(layer):
  """Returns whether the layer is loaded from a SavedModel."""
  return layer.__module__.find('keras.saving.saved_model') != -1


def check_graph_consistency(tensor=None, method='add_loss', force_raise=False):
  """Checks that tensors passed to `add_*` method match the Keras graph.

  When one of the `add_*` method is called inside a V2 conditional branch,
  the underlying tensor gets created in a FuncGraph managed by control_flow_v2.
  We need to raise clear error messages in such cases.

  Arguments:
    tensor: Tensor to check, or `False` if it is known that an error
      should be raised.
    method: Caller method, one of {'add_metric', 'add_loss', 'add_update'}.
    force_raise: If an error should be raised regardless of `tensor`.

  Raises:
    RuntimeError: In case of an out-of-graph tensor.
  """
  if (force_raise or
      (ops.executing_eagerly_outside_functions() and
       hasattr(tensor, 'graph') and
       isinstance(tensor.graph,
                  (control_flow_v2_func_graphs.CondBranchFuncGraph,
                   control_flow_v2_func_graphs.WhileCondFuncGraph,
                   control_flow_v2_func_graphs.WhileBodyFuncGraph)))):
    if method == 'activity_regularizer':
      bad_example = """
      class TestModel(tf.keras.Model):

        def __init__(self):
          super(TestModel, self).__init__(name='test_model')
          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')

        def call(self, x, training=None):
          if training:
            return self.dense(x)
          else:
            return self.dense(x)
      """
      correct_example = """
      class TestModel(tf.keras.Model):

        def __init__(self):
          super(TestModel, self).__init__(name='test_model')
          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')

        def call(self, x, training=None):
          return self.dense(x)
      """
      raise RuntimeError(
          'You are using a layer with `activity_regularizer` in a control flow '
          'branch, e.g.:\n{bad_example}\nThis is currently not supported. '
          'Please move your call to the layer with `activity_regularizer` out '
          'of the control flow branch, e.g.:\n{correct_example}\n'
          'You can also resolve this by marking your outer model/layer dynamic'
          ' (eager-only) by passing `dynamic=True` to the layer constructor. '
          'Any kind of control flow is supported with dynamic layers. '
          'Note that using `dynamic=True` requires you to implement static '
          'shape inference in the `compute_output_shape(input_shape)` '
          'method.'.format(
              bad_example=bad_example, correct_example=correct_example))

    if method == 'add_metric':
      bad_example = """
      def call(self, inputs, training=None):
        if training:
          metric = compute_metric(inputs)
          self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs
      """
      correct_example = """
      def call(self, inputs, training=None):
        if training:
          metric = compute_metric(inputs)
        else:
          metric = 0.
        self.add_metric(metric, name='my_metric', aggregation='mean')
        return inputs
      """
    elif method == 'add_loss':
      bad_example = """
      def call(self, inputs, training=None):
        if training:
          loss = compute_loss(inputs)
          self.add_loss(loss)
        return inputs
      """
      correct_example = """
      def call(self, inputs, training=None):
        if training:
          loss = compute_loss(inputs)
        else:
          loss = 0.
        self.add_loss(loss)
        return inputs
      """
    else:
      bad_example = """
      def call(self, inputs, training=None):
        if training:
          self.add_update(self.w.assign_add(1))
        return inputs
      """
      correct_example = """
      def call(self, inputs, training=None):
        if training:
          increment = 1
        else:
          increment = 0
        self.add_update(self.w.assign_add(increment))
        return inputs
      """
    raise RuntimeError(
        'You are using the method `{method}` in a control flow branch '
        'in your layer, e.g.:\n{bad_example}\n'
        'This is not currently supported. '
        'Please move your call to {method} out of the control flow branch, '
        'e.g.:\n{correct_example}\n'
        'You can also resolve this by marking your layer '
        'as dynamic (eager-only) by passing '
        '`dynamic=True` to the layer constructor. '
        'Any kind of control flow is supported with dynamic layers. '
        'Note that using `dynamic=True` requires you '
        'to implement static shape inference '
        'in the `compute_output_shape(input_shape)` method.'.format(
            method=method,
            bad_example=bad_example,
            correct_example=correct_example))


def mark_as_return(outputs, acd):
  """Marks `outputs` as the return values for automatic control deps."""

  def _mark_as_return(tensor):
    """Marks `tensor` as the return value for automatic control deps."""
    if not tensor_util.is_tensor(tensor):
      return tensor

    # pylint: disable=protected-access
    return_tensor = acd.mark_as_return(tensor)
    if getattr(tensor, '_keras_mask', None) is not None:
      return_tensor._keras_mask = acd.mark_as_return(tensor._keras_mask)
    else:
      return_tensor._keras_mask = None

    # Handle TensorFlow Probability attached metadata.
    # TODO(b/132076537): Remove this once TFP uses `CompositeTensor`.
    if getattr(tensor, '_tfp_distribution', None) is not None:
      return_tensor._tfp_distribution = tensor._tfp_distribution

    return return_tensor
    # pylint: enable=protected-access

  return nest.map_structure(_mark_as_return, outputs)


V2_DTYPE_BEHAVIOR = None


@keras_export(v1=['keras.layers.enable_v2_dtype_behavior'])
def enable_v2_dtype_behavior():
  """Enable the V2 dtype behavior for Keras layers.

  By default, the V2 dtype behavior is enabled in TensorFlow 2, so this function
  is only useful if `tf.compat.v1.disable_v2_behavior` has been called. Since
  mixed precision requires V2 dtype behavior to be enabled, this function allows
  you to use mixed precision in Keras layers if `disable_v2_behavior` has been
  called.

  When enabled, the dtype of Keras layers defaults to floatx (which is typically
  float32) instead of None. In addition, layers will automatically cast
  floating-point inputs to the layer's dtype.

  >>> x = tf.ones((4, 4, 4, 4), dtype='float64')
  >>> layer = tf.keras.layers.Conv2D(filters=4, kernel_size=2)
  >>> print(layer.dtype)  # float32 since V2 dtype behavior is enabled
  float32
  >>> y = layer(x)  # Layer casts inputs since V2 dtype behavior is enabled
  >>> print(y.dtype.name)
  float32

  A layer author can opt-out their layer from the automatic input casting by
  passing `autocast=False` to the base Layer's constructor. This disables the
  autocasting part of the V2 behavior for that layer, but not the defaulting to
  floatx part of the V2 behavior.

  When a global `tf.keras.mixed_precision.experimental.Policy` is set, a Keras
  layer's dtype will default to the global policy instead of floatx. Layers
  will automatically cast inputs to the policy's compute_dtype.
  """
  global V2_DTYPE_BEHAVIOR
  V2_DTYPE_BEHAVIOR = True


@keras_export(v1=['keras.layers.disable_v2_dtype_behavior'])
def disable_v2_dtype_behavior():
  """Disables the V2 dtype behavior for Keras layers.

  See `tf.compat.v1.keras.layers.enable_v2_dtype_behavior`.
  """
  global V2_DTYPE_BEHAVIOR
  V2_DTYPE_BEHAVIOR = False


def v2_dtype_behavior_enabled():
  """Returns True if the V2 dtype behavior is enabled."""
  if V2_DTYPE_BEHAVIOR is None:
    return tf2.enabled()
  return V2_DTYPE_BEHAVIOR


class TrackableWeightHandler(object):
  """Keras wrapper for handling tracking.Trackable object saving and restoring.

  This class handles Trackables in both V1 and V2 modes, ensuring that they can
  be saved and restored with the correct data and without adding additional ops
  on every save.

  Attributes:
    trackable: The trackable to wrap.
    num_tensors: The number of tensors that this trackable requires for saving.
  """

  def __init__(self, trackable):
    if not isinstance(trackable, tracking.Trackable):
      raise ValueError('%s is not a Trackable object.' % (trackable,))
    self._trackable = trackable

    # TODO(b/141682913): Figure out why this is private and fix it.
    saveables = trackable._gather_saveables_for_checkpoint().values()  # pylint: disable=protected-access
    if len(saveables) != 1:
      raise ValueError('Only Trackables with one Saveable are supported.')
    saveable = list(saveables)[0]

    if ops.executing_eagerly_outside_functions():
      # If we're in eager mode, we need to defer calling the Trackable's
      # saveable() callable until data export time.
      # However, it is safe to call the saveable as many times as we want, so
      # we will call it now to figure out how many tensors this Trackable will
      # produce.
      self._saveable = saveable
      self._num_tensors = len(self._saveable().specs)
      self._setter = lambda weights: self._saveable().restore(weights, None)
      self._getter = lambda: [spec.tensor for spec in self._saveable().specs]
    else:
      # If we're in Graph mode, we need to evaluate the Saveable only once and
      # cache the resulting restore graph. Failing to do this will result in
      # new assignment ops being added to the graph each time set_weights() is
      # called.
      self._placeholder_tensors = []
      self._saveable = saveable()
      self._num_tensors = len(self._saveable.specs)
      for spec in self._saveable.specs:
        tensor = spec.tensor
        self._placeholder_tensors.append(
            array_ops.placeholder(tensor.dtype, tensor.shape))
      self._assign_op = self._saveable.restore(self._placeholder_tensors, None)
      self._setter = self._set_weights_v1
      self._getter = lambda: [spec.tensor for spec in self._saveable.specs]

  @property
  def num_tensors(self):
    return self._num_tensors

  def set_weights(self, weights):
    if len(weights) != self._num_tensors:
      raise ValueError(
          ('Weight handler for trackable %s received the wrong number of ' +
           'weights: expected %s, got %s.') %
          (self._trackable, self._num_tensors, len(weights)))
    self._setter(weights)

  def get_tensors(self):
    return self._getter()

  def _set_weights_v1(self, weights):
    feed_dict = {}
    for idx, tensor in enumerate(weights):
      feed_dict[self._placeholder_tensors[idx]] = tensor
    backend.get_session().run(self._assign_op, feed_dict)


def no_ragged_support(inputs, layer_name):
  input_list = nest.flatten(inputs)
  if any(isinstance(x, ragged_tensor.RaggedTensor) for x in input_list):
    raise ValueError('Layer %s does not support RaggedTensors as input. '
                     'Inputs received: %s. You can try converting your '
                     'input to an uniform tensor.' % (layer_name, inputs))


# TODO(kathywu): This is a temporary hack. When a network of layers is revived
# from SavedModel, only the top-level layer will have losses. This causes issues
# in eager mode because the child layers may have graph losses
# (thus model.losses returns a mix of Eager and graph tensors). To fix this,
# whenever eager losses are added to one layer, add eager losses to all
# child layers. This causes `.losses` to only return eager losses.
REVIVED_LOSS_PLACEHOLDER = (
    'This layer\'s losses have been added to the parent layer.')
