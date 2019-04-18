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

import collections as collections_lib
import threading
import enum

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib

_call_context = threading.local()


class CallConvention(enum.Enum):
  """Calling conventions for passing `Layer` inputs to `Layer.call`."""
  # The Layer takes inputs as its first argument, named "inputs" for
  # compatibility with the signature of Layer.__call__. This is the mode assumed
  # for Layers which are not subclassed Models.
  EXPLICIT_INPUTS_ARGUMENT = 1
  # The Layer takes a single positional argument, not named "inputs". It's
  # treated like an "inputs" argument.
  SINGLE_POSITIONAL_ARGUMENT = 2
  # The Layer has multiple positional arguments to which its inputs should be
  # bound.
  POSITIONAL_ARGUMENTS_ARE_INPUTS = 3


def create_mean_metric(value, name=None):
  # TODO(psv): Remove this import when b/110718070 is fixed.
  from tensorflow.python.keras import metrics as metrics_module  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.distribute import distributed_training_utils  # pylint: disable=g-import-not-at-top
  metric_obj = metrics_module.Mean(name=name)
  return (metric_obj,
          distributed_training_utils.call_replica_local_fn(metric_obj, value))


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

  with ops.init_scope():
    if initializing_from_value:
      init_val = initializer
      variable_dtype = None
    else:
      # Instantiate initializer if provided initializer is a type object.
      if isinstance(
          initializer,
          (type(init_ops.Initializer), type(init_ops_v2.Initializer))):
        initializer = initializer()
      init_val = lambda: initializer(shape, dtype=dtype)
      variable_dtype = dtype.base_dtype
  if use_resource is None:
    use_resource = True

  # TODO(apassos,rohanj) figure out how to remove collections from here so we
  # can remove the V1.
  v = tf_variables.VariableV1(
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
      aggregation=aggregation)
  return v


def get_default_graph_uid_map():
  # TODO(fchollet): refactor this into backend.
  graph = ops.get_default_graph()
  name_uid_map = backend.PER_GRAPH_LAYER_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections_lib.defaultdict(int)
    backend.PER_GRAPH_LAYER_NAME_UIDS[graph] = name_uid_map
  return name_uid_map


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
    keras_tensors: The Tensors found that came from a Keras Layer.
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
          if (distribution_strategy_context.in_cross_replica_context() and
              not ops.executing_eagerly_outside_functions()):
            # In Legacy Graph mode, evaluating here makes Session be
            # configured improperly.
            constants[i] = op_input
          else:
            constants[i] = backend.function([], op_input)([])
      processed_ops, created_layers = _create_keras_history_helper(
          layer_inputs, processed_ops, created_layers)
      name = op.name
      node_def = op.node_def.SerializeToString()
      op_layer = base_layer.TensorFlowOpLayer(
          node_def, constants=constants, name=name)
      created_layers.append(op_layer)
      op_layer._add_inbound_node(  # pylint: disable=protected-access
          layer_inputs, op.outputs)
      processed_ops.update([op])
  return processed_ops, created_layers


def needs_keras_history(tensors):
  """Check if any Tensors need to be wrapped in TensorFlowOpLayers.

  This will never return True inside a sublayer, because sublayers
  do not need to create Keras History. Otherwise, this returns True
  if one or more of `tensors` originates from a `keras.Input` and
  does not have `_keras_history` set.

  Arguments:
    tensors: An arbitrary nested structure of Tensors.

  Returns:
    Bool, whether at least one Tensor needs to be wrapped.
  """
  input_tensors = nest.flatten(tensors)
  if is_in_call_context() or all(
      getattr(tensor, '_keras_history', None) is not None
      for tensor in input_tensors):
    # KerasHistory already set.
    return False
  return uses_keras_history(tensors)


def is_in_call_context():
  """Returns true if inside of a model/layer '__call__'."""
  return getattr(_call_context, 'in_call', False)


def is_in_frozen_context():
  """Returns if currently executing inside a `call` of a frozen Layer.

  A Layer is considered frozen if `layer.trainable=False`.

  Returns:
    Whether currently inside the `call` of a frozen Layer.
  """
  return getattr(_call_context, 'frozen', False)


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
    new_tensors_to_check = set()
    for tensor in tensors_to_check:
      if getattr(tensor, '_keras_history_checked', None) is not None:
        continue
      if getattr(tensor, '_keras_history', None) is not None:
        return True

      try:
        new_tensors_to_check.update(tensor.op.inputs)
      except AttributeError:
        # In case `tensor` is a Variable created in an Eager context.
        pass

    checked_tensors.update(tensors_to_check)
    tensors_to_check = list(new_tensors_to_check - checked_tensors)

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


@tf_contextlib.contextmanager
def call_context(layer):
  """Scope that marks when we are currently inside a Layer/Model's `call`."""
  was_in_call = is_in_call_context()
  was_frozen = is_in_frozen_context()
  _call_context.in_call = True
  if not layer.trainable:
    _call_context.frozen = True
  try:
    yield
  finally:
    _call_context.in_call = was_in_call
    _call_context.frozen = was_frozen


def training_arg_passed_to_call(argspec, args, kwargs):
  """Returns whether a user passed the `training` argument in `__call__`."""
  # `argspec.args` starts with ['self', 'inputs']
  full_args = dict(zip(argspec.args[2:], args))
  full_args.update(kwargs)
  return 'training' in full_args


class AutoAddUpdates(object):
  """Automatically track stateful ops with `add_update`.

  This context manager is used to automatically add stateful ops to a Layer
  or Model's `.updates`. This ensures that stateful ops are run in the Keras
  training loop. It also allows for these stateful ops to be disabled by
  setting `trainable=False`.

  Example:

  ```
  with AutoAddUpdates(layer, inputs) as auto_updates:
    outputs = layer.call(inputs)
    auto_updates.set_outputs(outputs)
  ```

  Attributes:
    layer: Layer or Model instance to add the updates to.
    inputs: The inputs to this Layer or Model, to be used for input-conditional
      updates.
    outputs: The outputs of this Layer or Model.
  """

  def __init__(self, layer, inputs):
    self.layer = layer
    self.inputs = inputs
    self.outputs = []

  def set_outputs(self, outputs):
    if self.outputs:
      raise RuntimeError('`set_outputs` should only be called once on an'
                         '`AutoAddUpdates` instance.')
    self.outputs = outputs

  def __enter__(self):
    # Only run in V2 Function mode.
    if (context.executing_eagerly() or
        not ops.executing_eagerly_outside_functions()):
      return self

    self._graph = ops.get_default_graph()
    self._num_operations = len(self._graph.get_operations())
    return self

  def __exit__(self, error_type, unused_value, unused_traceback):
    if error_type:
      # Allow errors that occurred inside this context manager to pass through
      # normally.
      return

    # Only run in V2 Function mode.
    if (context.executing_eagerly() or
        not ops.executing_eagerly_outside_functions()):
      return

    if (self._graph is not ops.get_default_graph() or
        self._graph.name != 'keras_graph'):
      # Only auto-track updates when the Keras Graph is the only one used.
      return

    new_operations = self._graph.get_operations()[self._num_operations:]
    new_stateful_ops = set()

    # pylint: disable=protected-access
    for op in new_operations:
      # While loop is not supported in general for automatic control
      # dependencies.
      if control_flow_util.IsInWhileLoop(op):
        continue

      # Track stateful ops via `add_update`.
      is_stateful_op = (
          op.type not in self._graph._registered_ops or
          auto_control_deps.op_is_stateful(
              self._graph._registered_ops[op.type]))

      # Ignore ReadVariableOps as they are not needed to be run separately.
      # This ensures existing Layers don't get extra updates.
      if is_stateful_op and op.type != 'ReadVariableOp':
        new_stateful_ops.add(op)

    explicit_updates = set([
        u for u in self.layer._get_unfiltered_updates(check_trainable=False)
        if not isinstance(u, tuple)
    ])
    # pylint: enable=protected-access

    # Don't add updates that will already be run by virtue of being consumed by
    # other stateful ops or by the Layer's outputs. This ensures that existing
    # Layers like `BatchNormalization` continue to return the same values for
    # `.update` calls.
    minimum_ops = set()
    targets = new_stateful_ops.union(
        set(nest.flatten(self.outputs)), explicit_updates)
    for op in new_stateful_ops:
      # Scrub any ops that are consumed by the outputs or other stateful ops.
      reachable = tf_utils.get_reachable_from_inputs(op)
      if not (targets - {op}).intersection(reachable):
        minimum_ops.add(op)
    new_stateful_ops = minimum_ops

    # Don't double-track updates added via explicitly calling `add_update`.
    # Also don't double-track updates already tracked in sublayers.
    new_stateful_ops = new_stateful_ops - explicit_updates

    # Decide whether to track as input-conditional or unconditional.
    input_reachable_ops = tf_utils.get_reachable_from_inputs(
        self.inputs, targets=new_stateful_ops)
    unconditional_updates = new_stateful_ops - input_reachable_ops
    conditional_updates = new_stateful_ops - unconditional_updates

    if unconditional_updates:
      self.layer.add_update(list(unconditional_updates))
    if conditional_updates:
      self.layer.add_update(list(conditional_updates), inputs=self.inputs)


def _get_var_read_dtype(input_list, should_cast):
  """Gets the dtype that AutoCastVariables should be read in."""
  if should_cast and input_list and input_list[0].dtype.is_floating:
    return input_list[0].dtype.base_dtype
  else:
    return None


def autocast_context_manager(input_list, should_cast):
  """Returns a context manager to autocast AutoCastVariables.

  Under this context manager, if `should_cast` is True, AutoCastVariables will
  be casted. If `should_cast` is False, AutoCastVariables will not be casted,
  which can be used to disable autocasting if nested under another
  call to `autocast_context_manager`.

  Args:
    input_list: The inputs to the layer with the AutoCastVariables.
    should_cast: Whether AutoCastVariables should be casted.

  Returns:
    A context manager to automatically cast AutoCastVariables.
  """
  var_read_dtype = _get_var_read_dtype(input_list, should_cast)
  return ops.get_default_graph()._enable_auto_casting_variables(  # pylint: disable=protected-access
      var_read_dtype)


def is_subclassed(layer):
  return (layer.__module__.find('keras.engine') == -1 and
          layer.__module__.find('keras.layers') == -1)
