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

import collections
import copy
import functools
import itertools
import threading
import warnings
import weakref

import numpy as np

from google.protobuf import json_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.tools.docs import doc_controls

# A module that only depends on `keras.layers` import these from here.

# pylint: disable=g-inconsistent-quotes
metrics_mod = generic_utils.LazyLoader(
    "metrics_mod", globals(),
    "tensorflow.python.keras.metrics")
# pylint: enable=g-inconsistent-quotes

# Prefix that is added to the TF op layer names.
_TF_OP_LAYER_NAME_PREFIX = 'tf_op_layer_'

# TODO(mdan): Should we have a single generic type for types that can be passed
# to tf.cast?
_AUTOCAST_TYPES = (tensor_lib.Tensor, sparse_tensor.SparseTensor,
                   ragged_tensor.RaggedTensor)


class Layer(module.Module, version_utils.LayerVersionSelector):
  """This is the class from which all layers inherit.

  A layer is a callable object that takes as input one or more tensors and
  that outputs one or more tensors. It involves *computation*, defined
  in the `call()` method, and a *state* (weight variables), defined
  either in the constructor `__init__()` or in the `build()` method.

  Users will just instantiate a layer and then treat it as a callable.

  Args:
    trainable: Boolean, whether the layer's variables should be trainable.
    name: String name of the layer.
    dtype: The dtype of the layer's computations and weights. Can also be a
      `tf.keras.mixed_precision.Policy`, which allows the computation and weight
      dtype to differ. Default of `None` means to use
      `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
      unless set to different value.
    dynamic: Set this to `True` if your layer should only be run eagerly, and
      should not be used to generate a static computation graph.
      This would be the case for a Tree-RNN or a recursive network,
      for example, or generally for any layer that manipulates tensors
      using Python control flow. If `False`, we assume that the layer can
      safely be used to generate a static computation graph.

  Attributes:
    name: The name of the layer (string).
    dtype: The dtype of the layer's weights.
    variable_dtype: Alias of `dtype`.
    compute_dtype: The dtype of the layer's computations. Layers automatically
      cast inputs to this dtype which causes the computations and output to also
      be in this dtype. When mixed precision is used with a
      `tf.keras.mixed_precision.Policy`, this will be different than
      `variable_dtype`.
    dtype_policy: The layer's dtype policy. See the
      `tf.keras.mixed_precision.Policy` documentation for details.
    trainable_weights: List of variables to be included in backprop.
    non_trainable_weights: List of variables that should not be
      included in backprop.
    weights: The concatenation of the lists trainable_weights and
      non_trainable_weights (in this order).
    trainable: Whether the layer should be trained (boolean), i.e. whether
      its potentially-trainable weights should be returned as part of
      `layer.trainable_weights`.
    input_spec: Optional (list of) `InputSpec` object(s) specifying the
      constraints on inputs that can be accepted by the layer.

  We recommend that descendants of `Layer` implement the following methods:

  * `__init__()`: Defines custom layer attributes, and creates layer state
    variables that do not depend on input shapes, using `add_weight()`.
  * `build(self, input_shape)`: This method can be used to create weights that
    depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
    will automatically build the layer (if it has not been built yet) by
    calling `build()`.
  * `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
    sure `build()` has been called. `call()` performs the logic of applying the
    layer to the input tensors (which should be passed in as argument).
    Two reserved keyword arguments you can optionally use in `call()` are:
      - `training` (boolean, whether the call is in inference mode or training
        mode). See more details in [the layer/model subclassing guide](
        https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
      - `mask` (boolean tensor encoding masked timesteps in the input, used
        in RNN layers). See more details in [the layer/model subclassing guide](
        https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
    A typical signature for this method is `call(self, inputs)`, and user could
    optionally add `training` and `mask` if the layer need them. `*args` and
    `**kwargs` is only useful for future extension when more input parameters
    are planned to be added.
  * `get_config(self)`: Returns a dictionary containing the configuration used
    to initialize this layer. If the keys differ from the arguments
    in `__init__`, then override `from_config(self)` as well.
    This method is used when saving
    the layer or a model that contains this layer.

  Examples:

  Here's a basic example: a layer with two variables, `w` and `b`,
  that returns `y = w . x + b`.
  It shows how to implement `build()` and `call()`.
  Variables set as attributes of a layer are tracked as weights
  of the layers (in `layer.weights`).

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
      w_init = tf.random_normal_initializer()
      self.w = tf.Variable(
          initial_value=w_init(shape=(input_shape[-1], self.units),
                               dtype='float32'),
          trainable=True)
      b_init = tf.zeros_initializer()
      self.b = tf.Variable(
          initial_value=b_init(shape=(self.units,), dtype='float32'),
          trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b

  # Instantiates the layer.
  linear_layer = SimpleDense(4)

  # This will also call `build(input_shape)` and create the weights.
  y = linear_layer(tf.ones((2, 2)))
  assert len(linear_layer.weights) == 2

  # These weights are trainable, so they're listed in `trainable_weights`:
  assert len(linear_layer.trainable_weights) == 2
  ```

  Note that the method `add_weight()` offers a shortcut to create weights:

  ```python
  class SimpleDense(Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
  ```

  Besides trainable weights, updated via backpropagation during training,
  layers can also have non-trainable weights. These weights are meant to
  be updated manually during `call()`. Here's a example layer that computes
  the running sum of its inputs:

  ```python
  class ComputeSum(Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

  my_sum = ComputeSum(2)
  x = tf.ones((2, 2))

  y = my_sum(x)
  print(y.numpy())  # [2. 2.]

  y = my_sum(x)
  print(y.numpy())  # [4. 4.]

  assert my_sum.weights == [my_sum.total]
  assert my_sum.non_trainable_weights == [my_sum.total]
  assert my_sum.trainable_weights == []
  ```

  For more information about creating layers, see the guide
  [Making new Layers and Models via subclassing](
    https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  """

  # See tf.Module for the usage of this property.
  # The key for _obj_reference_counts_dict is a Trackable, which could be a
  # variable or layer etc. tf.Module._flatten will fail to flatten the key
  # since it is trying to convert Trackable to a string. This attribute can be
  # ignored even after the fix of nest lib, since the trackable object should
  # already been available as individual attributes. _obj_reference_counts_dict
  # just contains a copy of them.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_obj_reference_counts_dict',),
      module.Module._TF_MODULE_IGNORED_PROPERTIES
  ))

  # When loading from a SavedModel, Layers typically can be revived into a
  # generic Layer wrapper. Sometimes, however, layers may implement methods
  # that go beyond this wrapper, as in the case of PreprocessingLayers'
  # `adapt` method. When this is the case, layer implementers can override
  # must_restore_from_config to return True; layers with this property must
  # be restored into their actual objects (and will fail if the object is
  # not available to the restoration code).
  _must_restore_from_config = False

  def _get_cell_name(self):
    canonical_name = get_canonical_name_for_symbol(
        self.__class__, api_name='keras', add_prefix_to_v1_names=True)
    if canonical_name is not None:
      return 'tf.{}'.format(canonical_name)
    return self.__class__.__module__ + '.' + self.__class__.__name__

  def _instrument_layer_creation(self):
    self._instrumented_keras_api = False
    self._instrumented_keras_layer_class = False
    self._instrumented_keras_model_class = False
    if not getattr(self, '_disable_keras_instrumentation', False):
      self._instrumented_keras_api = True
      if getattr(self, '_is_model_for_instrumentation', False):
        self._instrumented_keras_model_class = True
      else:
        self._instrumented_keras_layer_class = True

  @trackable.no_automatic_dependency_tracking
  def __init__(self,
               trainable=True,
               name=None,
               dtype=None,
               dynamic=False,
               **kwargs):
    self._instrument_layer_creation()

    # These properties should be set by the user via keyword arguments.
    # note that 'dtype', 'input_shape' and 'batch_input_shape'
    # are only applicable to input layers: do not pass these keywords
    # to non-input layers.
    allowed_kwargs = {
        'input_dim',
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
        'autocast',
        'implementation',
    }
    # Validate optional keyword arguments.
    generic_utils.validate_kwargs(kwargs, allowed_kwargs)

    # Mutable properties
    # Indicates whether the layer's weights are updated during training
    # and whether the layer's updates are run during training.
    self._trainable = trainable
    # A stateful layer is a layer whose updates are run during inference too,
    # for instance stateful RNNs.
    self._stateful = False
    # Indicates whether `build` needs to be called upon layer call, to create
    # the layer's weights.
    self.built = False
    # Provides information about which inputs are compatible with the layer.
    self._input_spec = None

    # SavedModel-related attributes.
    # Record the build input shape for loading purposes.
    # TODO(kathywu): Move this to Layer._set_save_spec once cl/290121460 is
    # submitted.
    self._build_input_shape = None
    self._saved_model_inputs_spec = None

    # `Layer.compute_mask` will be called at the end of `Layer.__call__` if
    # `Layer.compute_mask` is overridden, or if the `Layer` subclass sets
    # `self.supports_masking=True`.
    self._supports_masking = not generic_utils.is_default(self.compute_mask)

    self._init_set_name(name)
    self._activity_regularizer = regularizers.get(
        kwargs.pop('activity_regularizer', None))
    self._maybe_create_attribute('_trainable_weights', [])
    self._maybe_create_attribute('_non_trainable_weights', [])
    self._updates = []
    # Object to store all thread local layer properties.
    self._thread_local = threading.local()
    # A list of zero-argument lambdas which return Tensors, used for variable
    # regularizers.
    self._callable_losses = []
    # A list of symbolic Tensors containing activity regularizers and losses
    # manually added through `add_loss` in graph-building mode.
    self._losses = []
    # A list of metric instances corresponding to the symbolic metric tensors
    # added using the `add_metric` API.
    self._metrics = []
    # Ensures the same metric is not added multiple times in `MirroredStrategy`.
    self._metrics_lock = threading.Lock()

    # Both graph and subclassed networks have a dtype policy. For graph
    # networks, the policy's compute and variable dtypes are ignored. Such
    # networks only use the policy if it is a PolicyV1, in which case it uses
    # the PolicyV1's loss_scale (Policy does not have a loss_scale). For
    # subclassed networks, the compute and variable dtypes are used as like any
    # ordinary layer.
    self._set_dtype_policy(dtype)
    # Boolean indicating whether the layer automatically casts its inputs to the
    # layer's compute_dtype.
    self._autocast = kwargs.get('autocast',
                                base_layer_utils.v2_dtype_behavior_enabled())

    # Tracks `TrackableDataStructure`s, `Module`s, and `Layer`s.
    # Ordered by when the object was assigned as an attr.
    # Entries are unique.
    self._maybe_create_attribute('_self_tracked_trackables', [])

    # These lists will be filled via successive calls
    # to self._add_inbound_node().
    # Used in symbolic mode only, only in conjunction with graph-networks
    self._inbound_nodes_value = []
    self._outbound_nodes_value = []

    self._init_call_fn_args()

    # Whether the `call` method can be used to build a TF graph without issues.
    # This attribute has no effect if the model is created using the Functional
    # API. Instead, `model.dynamic` is determined based on the internal layers.
    self._dynamic = dynamic

    # Manage input shape information if passed.
    if 'input_dim' in kwargs and 'input_shape' not in kwargs:
      # Backwards compatibility: alias 'input_dim' to 'input_shape'.
      kwargs['input_shape'] = (kwargs['input_dim'],)
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
    self._initial_weights = kwargs.get('weights', None)

    # Whether the layer will track any layers that is set as attribute on itself
    # as sub-layers, the weights from the sub-layers will be included in the
    # parent layer's variables() as well.
    # Default to True, which means auto tracking is turned on. Certain subclass
    # might want to turn it off, like Sequential model.
    self._auto_track_sub_layers = True

    # For backwards compat reasons, most built-in layers do not guarantee
    # That they will 100% preserve the structure of input args when saving
    # / loading configs. E.g. they may un-nest an arg that is
    # a list with one element.
    self._preserve_input_structure_in_config = False

  @trackable.no_automatic_dependency_tracking
  @generic_utils.default
  def build(self, input_shape):
    """Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Args:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    """
    # Only record the build input shapes of overridden build methods.
    if not hasattr(self.build, '_is_default'):
      self._build_input_shape = input_shape
    self.built = True

  @doc_controls.for_subclass_implementers
  def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
    """This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Args:
      inputs: Input tensor, or dict/list/tuple of input tensors.
        The first positional `inputs` argument is subject to special rules:
        - `inputs` must be explicitly passed. A layer cannot have zero
          arguments, and `inputs` cannot be provided via the default value
          of a keyword argument.
        - NumPy array or Python scalar values in `inputs` get cast as tensors.
        - Keras mask metadata is only collected from `inputs`.
        - Layers are built (`build(input_shape)` method)
          using shape info from `inputs` only.
        - `input_spec` compatibility is only checked against `inputs`.
        - Mixed precision input casting is only applied to `inputs`.
          If a layer has tensor arguments in `*args` or `**kwargs`, their
          casting behavior in mixed precision should be handled manually.
        - The SavedModel input specification is generated using `inputs` only.
        - Integration with various ecosystem packages like TFMOT, TFLite,
          TF.js, etc is only supported for `inputs` and not for tensors in
          positional and keyword arguments.
      *args: Additional positional arguments. May contain tensors, although
        this is not recommended, for the reasons above.
      **kwargs: Additional keyword arguments. May contain tensors, although
        this is not recommended, for the reasons above.
        The following optional keyword arguments are reserved:
        - `training`: Boolean scalar tensor of Python boolean indicating
          whether the `call` is meant for training or inference.
        - `mask`: Boolean input mask. If the layer's `call()` method takes a
          `mask` argument, its default value will be set to the mask generated
          for `inputs` by the previous layer (if `input` did come from a layer
          that generated a corresponding mask, i.e. if it came from a Keras
          layer with masking support).

    Returns:
      A tensor or list/tuple of tensors.
    """
    return inputs

  @doc_controls.for_subclass_implementers
  def _add_trackable(self, trackable_object, trainable):
    """Adds a Trackable object to this layer's state.

    Args:
      trackable_object: The tf.tracking.Trackable object to add.
      trainable: Boolean, whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases) or
        "non_trainable_variables" (e.g. BatchNorm mean and variance).

    Returns:
      The TrackableWeightHandler used to track this object.
    """
    if isinstance(trackable_object, base_layer_utils.TrackableWeightHandler):
      handler = trackable_object
    else:
      handler = base_layer_utils.TrackableWeightHandler(trackable_object)
    if trainable:
      self._trainable_weights.append(handler)
    else:
      self._non_trainable_weights.append(handler)
    return handler

  @doc_controls.for_subclass_implementers
  def add_weight(self,
                 name=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 use_resource=None,
                 synchronization=tf_variables.VariableSynchronization.AUTO,
                 aggregation=tf_variables.VariableAggregation.NONE,
                 **kwargs):
    """Adds a new variable to the layer.

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype`.
      initializer: Initializer instance (callable).
      regularizer: Regularizer instance (callable).
      trainable: Boolean, whether the variable should be part of the layer's
        "trainable_variables" (e.g. variables, biases)
        or "non_trainable_variables" (e.g. BatchNorm mean and variance).
        Note that `trainable` cannot be `True` if `synchronization`
        is set to `ON_READ`.
      constraint: Constraint instance (callable).
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
      **kwargs: Additional keyword arguments. Accepted values are `getter`,
        `collections`, `experimental_autocast` and `caching_device`.

    Returns:
      The variable created.

    Raises:
      ValueError: When giving unsupported dtype and no initializer or when
        trainable has been set to True with synchronization set as `ON_READ`.
    """
    if shape is None:
      shape = ()
    kwargs.pop('partitioner', None)  # Ignored.
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in ['collections', 'experimental_autocast',
                       'caching_device', 'getter']:
        raise TypeError('Unknown keyword argument:', kwarg)
    collections_arg = kwargs.pop('collections', None)
    # 'experimental_autocast' can be set to False by the caller to indicate an
    # AutoCastVariable should never be created.
    autocast = kwargs.pop('experimental_autocast', True)
    # See the docstring for tf.Variable about the details for caching_device.
    caching_device = kwargs.pop('caching_device', None)

    if dtype is None:
      dtype = self.dtype or backend.floatx()
    dtype = dtypes.as_dtype(dtype)
    if self._dtype_policy.variable_dtype is None:
      # The policy is "_infer", so we infer the policy from the variable dtype.
      self._set_dtype_policy(policy.Policy(dtype.base_dtype.name))
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    constraint = constraints.get(constraint)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
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

    # Initialize variable when no initializer provided
    if initializer is None:
      # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
      if dtype.is_floating:
        initializer = initializers.get('glorot_uniform')
      # If dtype is DT_INT/DT_UINT, provide a default value `zero`
      # If dtype is DT_BOOL, provide a default value `FALSE`
      elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        initializer = initializers.get('zeros')
      # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
      elif 'getter' not in kwargs:
        # When `getter` is specified, it's possibly fine for `initializer` to be
        # None since it's up to the custom `getter` to raise error in case it
        # indeed needs `initializer`.
        raise ValueError('An initializer for variable %s of type %s is required'
                         ' for layer %s' % (name, dtype.base_dtype, self.name))

    getter = kwargs.pop('getter', base_layer_utils.make_variable)
    if (autocast and
        self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype
        and dtype.is_floating):
      old_getter = getter
      # Wrap variable constructor to return an AutoCastVariable.
      def getter(*args, **kwargs):  # pylint: disable=function-redefined
        variable = old_getter(*args, **kwargs)
        return autocast_variable.create_autocast_variable(variable)
      # Also the caching_device does not work with the mixed precision API,
      # disable it if it is specified.
      # TODO(b/142020079): Reenable it once the bug is fixed.
      if caching_device is not None:
        tf_logging.warning(
            '`caching_device` does not work with mixed precision API. Ignoring '
            'user specified `caching_device`.')
        caching_device = None

    variable = self._add_variable_with_custom_getter(
        name=name,
        shape=shape,
        # TODO(allenl): a `make_variable` equivalent should be added as a
        # `Trackable` method.
        getter=getter,
        # Manage errors in Layer rather than Trackable.
        overwrite=True,
        initializer=initializer,
        dtype=dtype,
        constraint=constraint,
        trainable=trainable,
        use_resource=use_resource,
        collections=collections_arg,
        synchronization=synchronization,
        aggregation=aggregation,
        caching_device=caching_device)
    if regularizer is not None:
      # TODO(fchollet): in the future, this should be handled at the
      # level of variable creation, and weight regularization losses
      # should be variable attributes.
      name_in_scope = variable.name[:variable.name.find(':')]
      self._handle_weight_regularization(name_in_scope,
                                         variable,
                                         regularizer)
    if base_layer_utils.is_split_variable(variable):
      for v in variable:
        backend.track_variable(v)
        if trainable:
          self._trainable_weights.append(v)
        else:
          self._non_trainable_weights.append(v)
    else:
      backend.track_variable(variable)
      if trainable:
        self._trainable_weights.append(variable)
      else:
        self._non_trainable_weights.append(variable)
    return variable

  @generic_utils.default
  def get_config(self):
    """Returns the config of the layer.

    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).

    Note that `get_config()` does not guarantee to return a fresh copy of dict
    every time it is called. The callers should make a copy of the returned dict
    if they want to modify it.

    Returns:
        Python dictionary.
    """
    all_args = tf_inspect.getfullargspec(self.__init__).args
    config = {
        'name': self.name,
        'trainable': self.trainable,
    }
    if hasattr(self, '_batch_input_shape'):
      config['batch_input_shape'] = self._batch_input_shape
    config['dtype'] = policy.serialize(self._dtype_policy)
    if hasattr(self, 'dynamic'):
      # Only include `dynamic` in the `config` if it is `True`
      if self.dynamic:
        config['dynamic'] = self.dynamic
      elif 'dynamic' in all_args:
        all_args.remove('dynamic')
    expected_args = config.keys()
    # Finds all arguments in the `__init__` that are not in the config:
    extra_args = [arg for arg in all_args if arg not in expected_args]
    # Check that either the only argument in the `__init__` is  `self`,
    # or that `get_config` has been overridden:
    if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
      raise NotImplementedError('Layer %s has arguments in `__init__` and '
                                'therefore must override `get_config`.' %
                                self.__class__.__name__)
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Args:
        config: A Python dictionary, typically the
            output of get_config.

    Returns:
        A layer instance.
    """
    return cls(**config)

  def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Args:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.

    Returns:
        An input shape tuple.
    """
    if context.executing_eagerly():
      # In this case we build the model first in order to do shape inference.
      # This is acceptable because the framework only calls
      # `compute_output_shape` on shape values that the layer would later be
      # built for. It would however cause issues in case a user attempts to
      # use `compute_output_shape` manually with shapes that are incompatible
      # with the shape the Layer will be called on (these users will have to
      # implement `compute_output_shape` themselves).
      self._maybe_build(input_shape)
      with func_graph.FuncGraph(str(self.name) + '_scratch_graph').as_default():
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
        def _make_placeholder_like(shape):
          ph = backend.placeholder(shape=shape, dtype=self.dtype)
          ph._keras_mask = None
          return ph
        inputs = nest.map_structure(_make_placeholder_like, input_shape)
        try:
          outputs = self(inputs, training=False)
        except TypeError as e:
          raise NotImplementedError(
              'We could not automatically infer the static shape of the '
              'layer\'s output. Please implement the '
              '`compute_output_shape` method on your layer (%s).' %
              self.__class__.__name__) from e
      return nest.map_structure(lambda t: t.shape, outputs)
    raise NotImplementedError(
        'Please run in eager mode or implement the `compute_output_shape` '
        'method on your layer (%s).' % self.__class__.__name__)

  @doc_controls.for_subclass_implementers
  def compute_output_signature(self, input_signature):
    """Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
      input_signature: Single TensorSpec or nested structure of TensorSpec
        objects, describing a candidate input for the layer.

    Returns:
      Single TensorSpec or nested structure of TensorSpec objects, describing
        how the layer would transform the provided input.

    Raises:
      TypeError: If input_signature contains a non-TensorSpec object.
    """
    def check_type_return_shape(s):
      if not isinstance(s, tensor_lib.TensorSpec):
        raise TypeError('Only TensorSpec signature types are supported, '
                        'but saw signature entry: {}.'.format(s))
      return s.shape
    input_shape = nest.map_structure(check_type_return_shape, input_signature)
    output_shape = self.compute_output_shape(input_shape)
    dtype = self._compute_dtype
    if dtype is None:
      input_dtypes = [s.dtype for s in nest.flatten(input_signature)]
      # Default behavior when self.dtype is None, is to use the first input's
      # dtype.
      dtype = input_dtypes[0]
    return nest.map_structure(
        lambda s: tensor_lib.TensorSpec(dtype=dtype, shape=s),
        output_shape)

  def _keras_tensor_symbolic_call(self, inputs, input_masks, args, kwargs):
    if self.dynamic:
      # We will use static shape inference to return symbolic tensors
      # matching the specifications of the layer outputs.
      # Since `self.dynamic` is True, we will never attempt to
      # run the underlying TF graph (which is disconnected).
      # TODO(fchollet): consider py_func as an alternative, which
      # would enable us to run the underlying graph if needed.
      input_signature = nest.map_structure(
          lambda x: tensor_lib.TensorSpec(shape=x.shape, dtype=x.dtype),
          inputs)
      output_signature = self.compute_output_signature(input_signature)
      return nest.map_structure(keras_tensor.KerasTensor, output_signature)
    else:
      return self._infer_output_signature(inputs, args, kwargs, input_masks)

  def _infer_output_signature(self, inputs, args, kwargs, input_masks):
    """TODO(kaftan): Docstring."""

    call_fn = self.call
    # Wrapping `call` function in autograph to allow for dynamic control
    # flow and control dependencies in call. We are limiting this to
    # subclassed layers as autograph is strictly needed only for
    # subclassed layers and models.
    # tf_convert will respect the value of autograph setting in the
    # enclosing tf.function, if any.
    if (base_layer_utils.is_subclassed(self) and
        not base_layer_utils.from_saved_model(self)):
      call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())

    # We enter a scratch graph and build placeholder inputs inside of it that
    # match the input args.
    # We then call the layer inside of the scratch graph to identify the
    # output signatures, then we build KerasTensors corresponding to those
    # outputs.
    scratch_graph = func_graph.FuncGraph(str(self.name) + '_scratch_graph')
    with scratch_graph.as_default():
      inputs = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, inputs)
      args = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, args)
      kwargs = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, kwargs)
      input_masks = nest.map_structure(
          keras_tensor.keras_tensor_to_placeholder, input_masks)

      with backend.name_scope(self._name_scope()):  # pylint: disable=not-callable
        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
          # Build layer if applicable (if the `build` method has been
          # overridden).
          # TODO(kaftan): do we maybe_build here, or have we already done it?
          self._maybe_build(inputs)
          inputs = self._maybe_cast_inputs(inputs)
          outputs = call_fn(inputs, *args, **kwargs)

        self._handle_activity_regularization(inputs, outputs)
      self._set_mask_metadata(inputs, outputs, input_masks,
                              build_graph=False)
      outputs = nest.map_structure(
          keras_tensor.keras_tensor_from_tensor, outputs)

    if hasattr(self, '_set_inputs') and not self.inputs:
      # TODO(kaftan): figure out if we need to do this at all
      # Subclassed network: explicitly set metadata normally set by
      # a call to self._set_inputs().
      self._set_inputs(inputs, outputs)
    del scratch_graph
    return outputs

  @generic_utils.default
  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    """Computes an output mask tensor.

    Args:
        inputs: Tensor or list of tensors.
        mask: Tensor or list of tensors.

    Returns:
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    """
    if not self._supports_masking:
      if any(m is not None for m in nest.flatten(mask)):
        raise TypeError('Layer ' + self.name + ' does not support masking, '
                        'but was passed an input_mask: ' + str(mask))
      # masking not explicitly supported: return None as mask.
      return None
    # if masking is explicitly supported, by default
    # carry over the input mask
    return mask

  def __call__(self, *args, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Args:
      *args: Positional arguments to be passed to `self.call`.
      **kwargs: Keyword arguments to be passed to `self.call`.

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
      - If the layer is not built, the method will call `build`.

    Raises:
      ValueError: if the layer's `call` method returns None (an invalid value).
      RuntimeError: if `super().__init__()` was not called in the constructor.
    """
    if not hasattr(self, '_thread_local'):
      raise RuntimeError(
          'You must call `super().__init__()` in the layer constructor.')

    # `inputs` (the first arg in the method spec) is special cased in
    # layer call due to historical reasons.
    # This special casing currently takes the form of:
    # - 'inputs' must be explicitly passed. A layer cannot have zero arguments,
    #   and inputs cannot have been provided via the default value of a kwarg.
    # - numpy/scalar values in `inputs` get converted to tensors
    # - implicit masks / mask metadata are only collected from 'inputs`
    # - Layers are built using shape info from 'inputs' only
    # - input_spec compatibility is only checked against `inputs`
    # - mixed precision casting (autocast) is only applied to `inputs`,
    #   not to any other argument.
    # - setting the SavedModel saving spec.
    inputs, args, kwargs = self._split_out_first_arg(args, kwargs)
    input_list = nest.flatten(inputs)

    # Functional Model construction mode is invoked when `Layer`s are called on
    # symbolic `KerasTensor`s, i.e.:
    # >> inputs = tf.keras.Input(10)
    # >> outputs = MyLayer()(inputs)  # Functional construction mode.
    # >> model = tf.keras.Model(inputs, outputs)
    if _in_functional_construction_mode(self, inputs, args, kwargs, input_list):
      return self._functional_construction_call(inputs, args, kwargs,
                                                input_list)

    # Maintains info about the `Layer.call` stack.
    call_context = base_layer_utils.call_context()

    # Accept NumPy and scalar inputs by converting to Tensors.
    if any(isinstance(x, (
        np_arrays.ndarray, np.ndarray, float, int)) for x in input_list):
      inputs = nest.map_structure(_convert_numpy_or_python_types, inputs)
      input_list = nest.flatten(inputs)

    # Handle `mask` propagation from previous layer to current layer. Masks can
    # be propagated explicitly via the `mask` argument, or implicitly via
    # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
    # explicitly take priority.
    input_masks, mask_is_implicit = self._get_input_masks(
        inputs, input_list, args, kwargs)
    if self._expects_mask_arg and mask_is_implicit:
      kwargs['mask'] = input_masks

    # Training mode for `Layer.call` is set via (in order of priority):
    # (1) The `training` argument passed to this `Layer.call`, if it is not None
    # (2) The training mode of an outer `Layer.call`.
    # (3) The default mode set by `tf.keras.backend.set_learning_phase` (if set)
    # (4) Any non-None default value for `training` specified in the call
    #  signature
    # (5) False (treating the layer as if it's in inference)
    args, kwargs, training_mode = self._set_training_mode(
        args, kwargs, call_context)

    # Losses are cleared for all sublayers on the outermost `Layer.call`.
    # Losses are not cleared on inner `Layer.call`s, because sublayers can be
    # called multiple times.
    if not call_context.in_call:
      self._clear_losses()

    eager = context.executing_eagerly()
    with call_context.enter(
        layer=self,
        inputs=inputs,
        build_graph=not eager,
        training=training_mode):

      input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
      if eager:
        call_fn = self.call
        name_scope = self._name
      else:
        name_scope = self._name_scope()  # Avoid autoincrementing.  # pylint: disable=not-callable
        call_fn = self._autographed_call()

      with ops.name_scope_v2(name_scope):
        if not self.built:
          self._maybe_build(inputs)

        if self._autocast:
          inputs = self._maybe_cast_inputs(inputs, input_list)

        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
          outputs = call_fn(inputs, *args, **kwargs)

        if self._activity_regularizer:
          self._handle_activity_regularization(inputs, outputs)
        if self._supports_masking:
          self._set_mask_metadata(inputs, outputs, input_masks, not eager)
        if self._saved_model_inputs_spec is None:
          self._set_save_spec(inputs)

        return outputs

  def _functional_construction_call(self, inputs, args, kwargs, input_list):
    call_context = base_layer_utils.call_context()

    # Accept NumPy and scalar inputs by converting to Tensors.
    if any(isinstance(x, (
        np_arrays.ndarray, np.ndarray, float, int)) for x in input_list):

      def _convert_non_tensor(x):
        # Don't call `ops.convert_to_tensor` on all `inputs` because
        # `SparseTensors` can't be converted to `Tensor`.
        if isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)):
          return tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
        return x

      inputs = nest.map_structure(_convert_non_tensor, inputs)
      input_list = nest.flatten(inputs)

    # Handle `mask` propagation from previous layer to current layer. Masks can
    # be propagated explicitly via the `mask` argument, or implicitly via
    # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
    # explicitly take priority.
    mask_arg_passed_by_framework = False
    input_masks, mask_is_implicit = self._get_input_masks(
        inputs, input_list, args, kwargs)
    if self._expects_mask_arg and mask_is_implicit:
      kwargs['mask'] = input_masks
      mask_arg_passed_by_framework = True

    # If `training` argument is None or not explicitly passed,
    # propagate `training` value from this layer's calling layer.
    training_value = None
    training_arg_passed_by_framework = False
    # Priority 1: `training` was explicitly passed a non-None value.
    if self._call_arg_was_passed('training', args, kwargs):
      training_value = self._get_call_arg_value('training', args, kwargs)
      if not self._expects_training_arg:
        kwargs.pop('training')

    if training_value is None:
      # Priority 2: `training` was passed to a parent layer.
      if call_context.training is not None:
        training_value = call_context.training
      # Priority 3: `learning_phase()` has been set.
      elif backend.global_learning_phase_is_set():
        training_value = backend.learning_phase()
        # Force the training_value to be bool type which matches to the contract
        # for layer/model call args.
        if tensor_util.is_tf_type(training_value):
          training_value = math_ops.cast(training_value, dtypes.bool)
        else:
          training_value = bool(training_value)
      # Priority 4: trace layer with the default training argument specified
      # in the `call` signature (or in inference mode if the `call` signature
      # specifies no non-None default).
      else:
        training_value = self._default_training_arg
      # In cases (2), (3), (4) the training argument is passed automatically
      # by the framework, and will not be hard-coded into the model.
      if self._expects_training_arg:
        args, kwargs = self._set_call_arg_value('training', training_value,
                                                args, kwargs)
        training_arg_passed_by_framework = True

    with call_context.enter(
        layer=self, inputs=inputs, build_graph=True, training=training_value):
      # Check input assumptions set after layer building, e.g. input shape.
      outputs = self._keras_tensor_symbolic_call(
          inputs, input_masks, args, kwargs)

      if outputs is None:
        raise ValueError('A layer\'s `call` method should return a '
                         'Tensor or a list of Tensors, not None '
                         '(layer: ' + self.name + ').')
      if training_arg_passed_by_framework:
        args, kwargs = self._set_call_arg_value(
            'training', None, args, kwargs, pop_kwarg_if_none=True)
      if mask_arg_passed_by_framework:
        kwargs.pop('mask')
      # Node connectivity does not special-case the first argument.
      outputs = self._set_connectivity_metadata((inputs,) + args, kwargs,
                                                outputs)
      return outputs

  def _set_training_mode(self, args, kwargs, call_context):
    training_mode = None
    if self._expects_training_arg:
      # (1) `training` was passed to this `Layer.call`.
      if self._call_arg_was_passed('training', args, kwargs):
        training_mode = self._get_call_arg_value('training', args, kwargs)
      # If no `training` arg was passed, or `None` was explicitly passed,
      # the framework will make a decision about the training mode is.
      if training_mode is None:
        call_ctx_training = call_context.training
        # (2) `training` mode is inferred from an outer `Layer.call`.
        if call_ctx_training is not None:
          training_mode = call_ctx_training
        # (3) User set `tf.keras.backend.set_learning_phase`.
        elif backend.global_learning_phase_is_set():
          training_mode = backend.learning_phase()
          # Ensure value is a `bool` or `tf.bool`.
          if isinstance(training_mode, bool):
            pass
          elif tensor_util.is_tf_type(training_mode):
            training_mode = math_ops.cast(training_mode, dtypes.bool)
          else:
            training_mode = bool(training_mode)
        # (4) We default to using `call`'s default value for `training`,
        # or treating the layer as if it is in inference if no non-None default
        # is specified in the `call` signature.
        else:
          training_mode = self._default_training_arg

        # For case (2), (3), (4) `training` arg is passed by framework.
        args, kwargs = self._set_call_arg_value('training', training_mode, args,
                                                kwargs)
    else:
      if 'training' in kwargs:
        # `training` was passed to this `Layer` but is not needed for
        # `Layer.call`. It will set the default mode for inner `Layer.call`s.
        training_mode = kwargs.pop('training')
      else:
        # Grab the current `training` mode from any outer `Layer.call`.
        training_mode = call_context.training

    return args, kwargs, training_mode

  def _autographed_call(self):
    # Wrapping `call` function in autograph to allow for dynamic control
    # flow and control dependencies in call. We are limiting this to
    # subclassed layers as autograph is strictly needed only for
    # subclassed layers and models.
    # tf_convert will respect the value of autograph setting in the
    # enclosing tf.function, if any.
    if (base_layer_utils.is_subclassed(self) and
        not base_layer_utils.from_saved_model(self)):
      return autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
    else:
      return self.call

  @property
  def dtype(self):
    """The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.
    """
    return self._dtype_policy.variable_dtype

  @property
  def name(self):
    """Name of the layer (string), set in the constructor."""
    return self._name

  @property
  def supports_masking(self):
    """Whether this layer supports computing a mask using `compute_mask`."""
    return self._supports_masking

  @supports_masking.setter
  def supports_masking(self, value):
    self._supports_masking = value

  @property
  def dynamic(self):
    """Whether the layer is dynamic (eager-only); set in the constructor."""
    return any(layer._dynamic for layer in self._flatten_layers())

  @property
  @doc_controls.do_not_doc_inheritable
  def stateful(self):
    return any(layer._stateful for layer in self._flatten_layers())

  @stateful.setter
  def stateful(self, value):
    self._stateful = value

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    for layer in self._flatten_layers():
      layer._trainable = value

  @property
  def activity_regularizer(self):
    """Optional regularizer function for the output of this layer."""
    return self._activity_regularizer

  @activity_regularizer.setter
  def activity_regularizer(self, regularizer):
    """Optional regularizer function for the output of this layer."""
    self._activity_regularizer = regularizer

  @property
  def input_spec(self):
    """`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
      A `tf.keras.layers.InputSpec` instance, or nested structure thereof.
    """
    return self._input_spec

  @input_spec.setter
  # Must be decorated to prevent tracking, since the input_spec can be nested
  # InputSpec objects.
  @trackable.no_automatic_dependency_tracking
  def input_spec(self, value):
    for v in nest.flatten(value):
      if v is not None and not isinstance(v, InputSpec):
        raise TypeError('Layer input_spec must be an instance of InputSpec. '
                        'Got: {}'.format(v))
    self._input_spec = value

  @property
  def trainable_weights(self):
    """List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Returns:
      A list of trainable variables.
    """
    if self.trainable:
      children_weights = self._gather_children_attribute('trainable_variables')
      return self._dedup_weights(self._trainable_weights + children_weights)
    else:
      return []

  @property
  def non_trainable_weights(self):
    """List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Returns:
      A list of non-trainable variables.
    """
    if self.trainable:
      children_weights = self._gather_children_attribute(
          'non_trainable_variables')
      non_trainable_weights = self._non_trainable_weights + children_weights
    else:
      children_weights = self._gather_children_attribute('variables')
      non_trainable_weights = (
          self._trainable_weights + self._non_trainable_weights +
          children_weights)
    return self._dedup_weights(non_trainable_weights)

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self.trainable_weights + self.non_trainable_weights

  @property
  @doc_controls.do_not_generate_docs
  def updates(self):
    warnings.warn('`layer.updates` will be removed in a future version. '
                  'This property should not be used in TensorFlow 2.0, '
                  'as `updates` are applied automatically.')
    return []

  @property
  def losses(self):
    """List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
      A list of tensors.
    """
    collected_losses = []
    for layer in self._flatten_layers():
      # If any eager losses are present, we assume the model to be part of an
      # eager training loop (either a custom one or the one used when
      # `run_eagerly=True`) and so we always return just the eager losses.
      if layer._eager_losses:
        # Filter placeholder losses that may have been added by revived layers.
        # (see base_layer_utils for details).
        if (layer._eager_losses[0] is
            not base_layer_utils.REVIVED_LOSS_PLACEHOLDER):
          collected_losses.extend(layer._eager_losses)
      else:
        collected_losses.extend(layer._losses)
      for regularizer in layer._callable_losses:
        loss_tensor = regularizer()
        if loss_tensor is not None:
          collected_losses.append(loss_tensor)
    return collected_losses

  def add_loss(self, losses, **kwargs):
    """Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
      def call(self, inputs):
        self.add_loss(tf.abs(tf.reduce_mean(inputs)))
        return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Args:
      losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
        may also be zero-argument callables which create a loss tensor.
      **kwargs: Additional keyword arguments for backward compatibility.
        Accepted values:
          inputs - Deprecated, will be automatically inferred.
    """
    kwargs.pop('inputs', None)
    if kwargs:
      raise TypeError('Unknown keyword arguments: %s' % (kwargs.keys(),))

    def _tag_callable(loss):
      """Tags callable loss tensor as `_unconditional_loss`."""
      if callable(loss):
        # We run the loss without autocasting, as regularizers are often
        # numerically unstable in float16.
        with autocast_variable.enable_auto_cast_variables(None):
          loss = loss()
      if loss is None:
        return None  # Will be filtered out when computing the .losses property
      if not tensor_util.is_tf_type(loss):
        loss = tensor_conversion.convert_to_tensor_v2_with_dispatch(
            loss, dtype=backend.floatx()
        )
      loss._unconditional_loss = True  # pylint: disable=protected-access
      return loss

    losses = nest.flatten(losses)

    callable_losses = []
    eager_losses = []
    symbolic_losses = []
    for loss in losses:
      if callable(loss):
        callable_losses.append(functools.partial(_tag_callable, loss))
        continue
      if loss is None:
        continue
      if not tensor_util.is_tf_type(loss) and not isinstance(
          loss, keras_tensor.KerasTensor):
        loss = tensor_conversion.convert_to_tensor_v2_with_dispatch(
            loss, dtype=backend.floatx()
        )
      # TF Functions should take the eager path.
      if ((tf_utils.is_symbolic_tensor(loss) or
           isinstance(loss, keras_tensor.KerasTensor)) and
          not base_layer_utils.is_in_tf_function()):
        symbolic_losses.append(loss)
      elif tensor_util.is_tf_type(loss):
        eager_losses.append(loss)

    self._callable_losses.extend(callable_losses)

    in_call_context = base_layer_utils.call_context().in_call
    if eager_losses and not in_call_context:
      raise ValueError(
          'Expected a symbolic Tensors or a callable for the loss value. '
          'Please wrap your loss computation in a zero argument `lambda`.')

    self._eager_losses.extend(eager_losses)

    for symbolic_loss in symbolic_losses:
      if getattr(self, '_is_graph_network', False):
        self._graph_network_add_loss(symbolic_loss)
      else:
        # Possible a loss was added in a Layer's `build`.
        self._losses.append(symbolic_loss)

  def _clear_losses(self):
    """Used every step in eager to reset losses."""
    # Set to thread local directly to avoid Layer.__setattr__ overhead.
    if not getattr(self, '_self_tracked_trackables',
                   None):  # Fast path for single Layer.
      self._thread_local._eager_losses = []
    else:
      for layer in self._flatten_layers():
        layer._thread_local._eager_losses = []

  @property
  def metrics(self):
    """List of metrics added using the `add_metric()` API.

    Example:

    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
      A list of `Metric` objects.
    """
    collected_metrics = []
    for layer in self._flatten_layers():
      with layer._metrics_lock:
        collected_metrics.extend(layer._metrics)
    return collected_metrics

  def add_metric(self, value, name=None, **kwargs):
    """Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
      def __init__(self):
        super(MyMetricLayer, self).__init__(name='my_metric_layer')
        self.mean = tf.keras.metrics.Mean(name='metric_1')

      def call(self, inputs):
        self.add_metric(self.mean(inputs))
        self.add_metric(tf.reduce_sum(inputs), name='metric_2')
        return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
      value: Metric tensor.
      name: String metric name.
      **kwargs: Additional keyword arguments for backward compatibility.
        Accepted values:
        `aggregation` - When the `value` tensor provided is not the result of
        calling a `keras.Metric` instance, it will be aggregated by default
        using a `keras.Metric.Mean`.
    """
    kwargs_keys = list(kwargs.keys())
    if (len(kwargs_keys) > 1 or
        (len(kwargs_keys) == 1 and kwargs_keys[0] != 'aggregation')):
      raise TypeError('Unknown keyword arguments: ', str(kwargs.keys()))

    from_metric_obj = hasattr(value, '_metric_obj')
    is_symbolic = isinstance(value, keras_tensor.KerasTensor)
    in_call_context = base_layer_utils.call_context().in_call

    if name is None and not from_metric_obj:
      # Eg. `self.add_metric(math_ops.reduce_sum(x))`
      # In eager mode, we use metric name to lookup a metric. Without a name,
      # a new Mean metric wrapper will be created on every model/layer call.
      # So, we raise an error when no name is provided.
      # We will do the same for symbolic mode for consistency although a name
      # will be generated if no name is provided.

      # We will not raise this error in the foll use case for the sake of
      # consistency as name in provided in the metric constructor.
      # mean = metrics.Mean(name='my_metric')
      # model.add_metric(mean(outputs))
      raise ValueError('Please provide a name for your metric like '
                       '`self.add_metric(tf.reduce_sum(inputs), '
                       'name=\'mean_activation\')`')
    elif from_metric_obj:
      name = value._metric_obj.name

    if not in_call_context and not is_symbolic:
      raise ValueError('Expected a symbolic Tensor for the metric value, '
                       'received: ' + str(value))

    # If a metric was added in a Layer's `call` or `build`.
    if in_call_context or not getattr(self, '_is_graph_network', False):
      # TF Function path should take the eager path.

      # If the given metric is available in `metrics` list we just update state
      # on it, otherwise we create a new metric instance and
      # add it to the `metrics` list.
      metric_obj = getattr(value, '_metric_obj', None)
      # Tensors that come from a Metric object already updated the Metric state.
      should_update_state = not metric_obj
      name = metric_obj.name if metric_obj else name

      with self._metrics_lock:
        match = self._get_existing_metric(name)
        if match:
          metric_obj = match
        elif metric_obj:
          self._metrics.append(metric_obj)
        else:
          # Build the metric object with the value's dtype if it defines one
          metric_obj = metrics_mod.Mean(
              name=name, dtype=getattr(value, 'dtype', None))
          self._metrics.append(metric_obj)

      if should_update_state:
        metric_obj(value)
    else:
      if from_metric_obj:
        raise ValueError('Using the result of calling a `Metric` object '
                         'when calling `add_metric` on a Functional '
                         'Model is not supported. Please pass the '
                         'Tensor to monitor directly.')

      # Insert layers into the Keras Graph Network.
      aggregation = None if from_metric_obj else 'mean'
      self._graph_network_add_metric(value, aggregation, name)

  @doc_controls.do_not_doc_inheritable
  def add_update(self, updates, inputs=None):
    """Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Args:
      updates: Update op, or list/tuple of update ops, or zero-arg callable
        that returns an update op. A zero-arg callable should be passed in
        order to disable running the updates by setting `trainable=False`
        on this Layer, when executing in Eager mode.
      inputs: Deprecated, will be automatically inferred.
    """
    if inputs is not None:
      tf_logging.warning(
          '`add_update` `inputs` kwarg has been deprecated. You no longer need '
          'to pass a value to `inputs` as it is being automatically inferred.')
    call_context = base_layer_utils.call_context()
    # No need to run updates during Functional API construction.
    if call_context.in_keras_graph:
      return

    # Callable updates are disabled by setting `trainable=False`.
    if not call_context.frozen:
      for update in nest.flatten(updates):
        if callable(update):
          update()  # pylint: disable=not-callable

  def set_weights(self, weights):
    """Sets the weights of the layer, from NumPy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function, by calling
    the layer.

    For example, a `Dense` layer returns a list of two values: the kernel matrix
    and the bias vector. These can be used to set the weights of another
    `Dense` layer:

    >>> layer_a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> layer_a.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> layer_b.get_weights()
    [array([[2.],
           [2.],
           [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b.set_weights(layer_a.get_weights())
    >>> layer_b.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]

    Args:
      weights: a list of NumPy arrays. The number
        of arrays and their shape must match
        number of the dimensions of the weights
        of the layer (i.e. it should match the
        output of `get_weights`).

    Raises:
      ValueError: If the provided weights list does not match the
        layer's specifications.
    """
    params = self.weights

    expected_num_weights = 0
    for param in params:
      if isinstance(param, base_layer_utils.TrackableWeightHandler):
        expected_num_weights += param.num_tensors
      else:
        expected_num_weights += 1

    if expected_num_weights != len(weights):
      raise ValueError(
          'You called `set_weights(weights)` on layer "%s" '
          'with a weight list of length %s, but the layer was '
          'expecting %s weights. Provided weights: %s...' %
          (self.name, len(weights), expected_num_weights, str(weights)[:50]))

    weight_index = 0
    weight_value_tuples = []
    for param in params:
      if isinstance(param, base_layer_utils.TrackableWeightHandler):
        num_tensors = param.num_tensors
        tensors = weights[weight_index:weight_index + num_tensors]
        param.set_weights(tensors)
        weight_index += num_tensors
      else:
        weight = weights[weight_index]
        weight_shape = weight.shape if hasattr(weight, 'shape') else ()
        ref_shape = param.shape
        if not ref_shape.is_compatible_with(weight_shape):
          raise ValueError(
              'Layer weight shape %s not compatible with provided weight '
              'shape %s' % (ref_shape, weight_shape))
        weight_value_tuples.append((param, weight))
        weight_index += 1

    backend.batch_set_value(weight_value_tuples)

    # Perform any layer defined finalization of the layer state.
    for layer in self._flatten_layers():
      layer.finalize_state()

  def get_weights(self):
    """Returns the current weights of the layer, as NumPy arrays.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of NumPy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a `Dense` layer returns a list of two values: the kernel matrix
    and the bias vector. These can be used to set the weights of another
    `Dense` layer:

    >>> layer_a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> layer_a.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> layer_b.get_weights()
    [array([[2.],
           [2.],
           [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> layer_b.set_weights(layer_a.get_weights())
    >>> layer_b.get_weights()
    [array([[1.],
           [1.],
           [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
        Weights values as a list of NumPy arrays.
    """
    weights = self.weights
    output_weights = []
    for weight in weights:
      if isinstance(weight, base_layer_utils.TrackableWeightHandler):
        output_weights.extend(weight.get_tensors())
      else:
        output_weights.append(weight)
    return backend.batch_get_value(output_weights)

  @doc_controls.do_not_generate_docs
  def finalize_state(self):
    """Finalizes the layers state after updating layer weights.

    This function can be subclassed in a layer and will be called after updating
    a layer weights. It can be overridden to finalize any additional layer state
    after a weight update.
    """
    pass

  @doc_controls.do_not_generate_docs
  def get_updates_for(self, inputs):
    """Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Args:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of update ops of the layer that depend on `inputs`.
    """
    warnings.warn('`layer.get_updates_for` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.updates` method instead.')
    return self.updates

  @doc_controls.do_not_generate_docs
  def get_losses_for(self, inputs):
    """Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Args:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.
    """
    warnings.warn('`layer.get_losses_for` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.losses` instead.')
    return self.losses

  @doc_controls.do_not_doc_inheritable
  def get_input_mask_at(self, node_index):
    """Retrieves the input mask tensor(s) of a layer at a given node.

    Args:
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

  @doc_controls.do_not_doc_inheritable
  def get_output_mask_at(self, node_index):
    """Retrieves the output mask tensor(s) of a layer at a given node.

    Args:
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
  @doc_controls.do_not_doc_inheritable
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
  @doc_controls.do_not_doc_inheritable
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

  @doc_controls.do_not_doc_inheritable
  def get_input_shape_at(self, node_index):
    """Retrieves the input shape(s) of a layer at a given node.

    Args:
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

  @doc_controls.do_not_doc_inheritable
  def get_output_shape_at(self, node_index):
    """Retrieves the output shape(s) of a layer at a given node.

    Args:
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

  @doc_controls.do_not_doc_inheritable
  def get_input_at(self, node_index):
    """Retrieves the input tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first input node of the layer.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    """
    return self._get_node_attribute_at_index(node_index, 'input_tensors',
                                             'input')

  @doc_controls.do_not_doc_inheritable
  def get_output_at(self, node_index):
    """Retrieves the output tensor(s) of a layer at a given node.

    Args:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first output node of the layer.

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
  @doc_controls.do_not_doc_inheritable
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
      return self._inbound_nodes[0].input_shapes
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
      if getattr(self, '_is_graph_network', False):
        with tf_utils.maybe_init_scope(self):
          self._maybe_build(self.inputs)
      else:
        raise ValueError('You tried to call `count_params` on ' + self.name +
                         ', but the layer isn\'t built. '
                         'You can build it manually via: `' + self.name +
                         '.build(batch_input_shape)`.')
    return layer_utils.count_params(self.weights)

  @property
  @doc_controls.do_not_doc_inheritable
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
      return self._inbound_nodes[0].output_shapes
    else:
      raise AttributeError('The layer "%s"'
                           ' has multiple inbound nodes, '
                           'with different output shapes. Hence '
                           'the notion of "output shape" is '
                           'ill-defined for the layer. '
                           'Use `get_output_shape_at(node_index)` '
                           'instead.' % self.name)

  @property
  @doc_controls.do_not_doc_inheritable
  def inbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._inbound_nodes

  @property
  @doc_controls.do_not_doc_inheritable
  def outbound_nodes(self):
    """Deprecated, do NOT use! Only for compatibility with external Keras."""
    return self._outbound_nodes

  ##############################################################################
  # Methods & attributes below are public aliases of other methods.            #
  ##############################################################################

  @doc_controls.do_not_doc_inheritable
  def apply(self, inputs, *args, **kwargs):
    """Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Args:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    warnings.warn('`layer.apply` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.__call__` method instead.')
    return self.__call__(inputs, *args, **kwargs)

  @doc_controls.do_not_doc_inheritable
  def add_variable(self, *args, **kwargs):
    """Deprecated, do NOT use! Alias for `add_weight`."""
    warnings.warn('`layer.add_variable` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `layer.add_weight` method instead.')
    return self.add_weight(*args, **kwargs)

  @property
  @doc_controls.do_not_generate_docs
  def variables(self):
    """Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
      A list of variables.
    """
    return self.weights

  @property
  @doc_controls.do_not_generate_docs
  def trainable_variables(self):
    return self.trainable_weights

  @property
  @doc_controls.do_not_generate_docs
  def non_trainable_variables(self):
    return self.non_trainable_weights

  ##############################################################################
  # Methods & attributes below are all private and only used by the framework. #
  ##############################################################################

  @property
  def _inbound_nodes(self):
    return self._inbound_nodes_value

  @_inbound_nodes.setter
  @trackable.no_automatic_dependency_tracking
  def _inbound_nodes(self, value):
    self._inbound_nodes_value = value

  @property
  def _outbound_nodes(self):
    return self._outbound_nodes_value

  @_outbound_nodes.setter
  @trackable.no_automatic_dependency_tracking
  def _outbound_nodes(self, value):
    self._outbound_nodes_value = value

  def _set_dtype_policy(self, dtype):
    """Sets self._dtype_policy."""
    if isinstance(dtype, policy.Policy):
      self._dtype_policy = dtype
    elif isinstance(dtype, dict):
      self._dtype_policy = policy.deserialize(dtype)
    elif isinstance(dtype, str) and dtype in ('mixed_float16',
                                              'mixed_bfloat16'):
      # The isinstance check is required since np.dtype raises an error if
      # compared to a non-dtype string.
      self._dtype_policy = policy.Policy(dtype)
    elif dtype:
      self._dtype_policy = policy.Policy(dtypes.as_dtype(dtype).name)
    else:
      self._dtype_policy = policy.global_policy()
    if (self._dtype_policy.name == 'mixed_float16' and
        not loss_scale_optimizer.strategy_supports_loss_scaling()):
      # Although only loss scaling doesn't support certain strategies, to avoid
      # confusion, we disallow the 'mixed_float16' policy with unsupported
      # strategies. This is because 'mixed_float16' requires loss scaling for
      # numeric stability.
      strategy = distribute_lib.get_strategy()
      raise ValueError('Mixed precision is not supported with the '
                       'tf.distribute.Strategy: %s. Either stop using mixed '
                       'precision by removing the use of the "%s" policy or '
                       'use a different Strategy, e.g. a MirroredStrategy.' %
                       (strategy.__class__.__name__, self._dtype_policy.name))

    # Performance optimization: cache the compute dtype as a Dtype object or
    # None, so that str to Dtype conversion doesn't happen in Layer.__call__.
    # TODO(b/157486353): Investigate returning DTypes in Policy.
    if self._dtype_policy.compute_dtype:
      self._compute_dtype_object = dtypes.as_dtype(
          self._dtype_policy.compute_dtype)
    else:
      self._compute_dtype_object = None

  @property
  def dtype_policy(self):
    """The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.
    """
    return self._dtype_policy

  @property
  def compute_dtype(self):
    """The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
      The layer's compute dtype.
    """
    return self._dtype_policy.compute_dtype

  @property
  def _compute_dtype(self):
    """Deprecated alias of `compute_dtype`."""
    return self._dtype_policy.compute_dtype

  @property
  def variable_dtype(self):
    """Alias of `Layer.dtype`, the dtype of the weights."""
    return self.dtype

  def _maybe_cast_inputs(self, inputs, input_list=None):
    """Maybe casts the inputs to the compute dtype.

    If self._compute_dtype is floating-point, and self_autocast is True,
    floating-point inputs are casted to self._compute_dtype.

    Args:
      inputs: Input tensor, or structure of input tensors.
      input_list: Flat list of input tensors.

    Returns:
      `inputs`, but tensors may have been casted to self._compute_dtype
    """
    if not input_list:
      input_list = nest.flatten(inputs)

    compute_dtype_object = self._compute_dtype_object
    should_autocast = (
        self._autocast and compute_dtype_object and
        compute_dtype_object.is_floating)

    if (should_autocast and
        any(map(self._should_cast_single_input, input_list))):
      # Only perform expensive `nest` operation when needed.
      return nest.map_structure(self._cast_single_input, inputs)
    else:
      return inputs

  def _should_cast_single_input(self, x):
    if isinstance(x, _AUTOCAST_TYPES):
      return (self._compute_dtype_object and
              x.dtype != self._compute_dtype_object and x.dtype.is_floating)
    return False

  def _cast_single_input(self, x):
    """Cast a single Tensor or TensorSpec to the compute dtype."""
    if self._should_cast_single_input(x):
      return math_ops.cast(x, self._compute_dtype_object)
    else:
      return x

  # _dtype used to be an attribute set in the constructor. We still expose it
  # because some clients still use it.
  # TODO(reedwm): Deprecate, then remove the _dtype property.
  @property
  def _dtype(self):
    # This is equivalent to returning self.dtype . We do not return self.dtype
    # as it would cause infinite recursion in a few subclasses, which override
    # "dtype" to return self._dtype.
    return self._dtype_policy.variable_dtype

  @_dtype.setter
  def _dtype(self, value):
    value = dtypes.as_dtype(value).name
    self._set_dtype_policy(policy.Policy(value))

  def _name_scope(self):  # pylint: disable=method-hidden
    if not tf2.enabled():
      return self.name
    name_scope = self.name
    current_name_scope = ops.get_name_scope()
    if current_name_scope:
      name_scope = current_name_scope + '/' + name_scope
    if name_scope:
      # Note that the trailing `/` prevents autogenerated
      # numerical suffixes to get appended. It will also fully reset
      # nested name scope (i.e. the outer name scope has no effect).
      name_scope += '/'
    return name_scope

  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      backend.observe_object_name(name)
      self._name = name

  def _get_existing_metric(self, name=None):
    match = [m for m in self._metrics if m.name == name]
    if not match:
      return
    if len(match) > 1:
      raise ValueError(
          'Please provide different names for the metrics you have added. '
          'We found {} metrics with the name: "{}"'.format(len(match), name))
    return match[0]

  def _handle_weight_regularization(self, name, variable, regularizer):
    """Create lambdas which compute regularization losses."""

    def _loss_for_variable(v):
      """Creates a regularization loss `Tensor` for variable `v`."""
      with backend.name_scope(name + '/Regularizer'):
        regularization = regularizer(v)
      return regularization

    if base_layer_utils.is_split_variable(variable):
      for v in variable:
        self.add_loss(functools.partial(_loss_for_variable, v))
    else:
      self.add_loss(functools.partial(_loss_for_variable, variable))

  def _handle_activity_regularization(self, inputs, outputs):
    # Apply activity regularization.
    # Note that it should be applied every time the layer creates a new
    # output, since it is output-specific.
    if self._activity_regularizer:
      output_list = nest.flatten(outputs)
      with backend.name_scope('ActivityRegularizer'):
        for output in output_list:
          activity_loss = self._activity_regularizer(output)
          batch_size = math_ops.cast(
              array_ops.shape(output)[0], activity_loss.dtype)
          # Make activity regularization strength batch-agnostic.
          mean_activity_loss = activity_loss / batch_size
          self.add_loss(mean_activity_loss)

  def _set_mask_metadata(self, inputs, outputs, previous_mask, build_graph):
    # Many `Layer`s don't need to call `compute_mask`.
    # This method is optimized to do as little work as needed for the common
    # case.
    if not self._supports_masking:
      return

    flat_outputs = nest.flatten(outputs)

    mask_already_computed = (
        getattr(self, '_compute_output_and_mask_jointly', False) or
        all(getattr(x, '_keras_mask', None) is not None for x in flat_outputs))
    if mask_already_computed:
      if build_graph:
        self._set_mask_keras_history_checked(flat_outputs)
      return

    output_masks = self.compute_mask(inputs, previous_mask)
    if output_masks is None:
      return

    flat_masks = nest.flatten(output_masks)
    for tensor, mask in zip(flat_outputs, flat_masks):
      try:
        tensor._keras_mask = mask
      except AttributeError:
        # C Type such as np.ndarray.
        pass

    if build_graph:
      self._set_mask_keras_history_checked(flat_outputs)

  def _set_mask_keras_history_checked(self, flat_outputs):
    for output in flat_outputs:
      if getattr(output, '_keras_mask', None) is not None:
        # Do not track masks for `TensorFlowOpLayer` construction.
        output._keras_mask._keras_history_checked = True

  def _get_input_masks(self, inputs, input_list, args, kwargs):
    if not self._supports_masking and not self._expects_mask_arg:
      # Input masks only need to be retrieved if they are needed for `call`
      # or `compute_mask`.
      input_masks = None
      implicit_mask = False
    elif self._call_arg_was_passed('mask', args, kwargs):
      input_masks = self._get_call_arg_value('mask', args, kwargs)
      implicit_mask = False
    else:
      input_masks = [getattr(t, '_keras_mask', None) for t in input_list]
      if all(mask is None for mask in input_masks):
        input_masks = None
        implicit_mask = False
      else:
        # Only do expensive `nest` op when masking is actually being used.
        input_masks = nest.pack_sequence_as(inputs, input_masks)
        implicit_mask = True
    return input_masks, implicit_mask

  def _call_arg_was_passed(self, arg_name, args, kwargs, inputs_in_args=False):
    # Performance optimization: do no work in most common case.
    if not args and not kwargs:
      return False

    if arg_name in kwargs:
      return True
    call_fn_args = self._call_fn_args
    if not inputs_in_args:
      # Ignore `inputs` arg.
      call_fn_args = call_fn_args[1:]
    return arg_name in dict(zip(call_fn_args, args))

  def _get_call_arg_value(self, arg_name, args, kwargs, inputs_in_args=False):
    if arg_name in kwargs:
      return kwargs[arg_name]
    call_fn_args = self._call_fn_args
    if not inputs_in_args:
      # Ignore `inputs` arg.
      call_fn_args = call_fn_args[1:]
    args_dict = dict(zip(call_fn_args, args))
    return args_dict[arg_name]

  def _set_call_arg_value(
      self, arg_name, new_value, args,
      kwargs, inputs_in_args=False, pop_kwarg_if_none=False):
    arg_pos = self._call_fn_arg_positions.get(arg_name, None)
    if arg_pos is not None:
      if not inputs_in_args:
        # Ignore `inputs` arg.
        arg_pos = arg_pos - 1
      if len(args) > arg_pos:
        args = list(args)
        args[arg_pos] = new_value
        return tuple(args), kwargs
    if new_value is None and pop_kwarg_if_none:
      kwargs.pop(arg_name, None)
    else:
      kwargs[arg_name] = new_value
    return args, kwargs

  def _set_connectivity_metadata(self, args, kwargs, outputs):
    # If the layer returns tensors from its inputs unmodified,
    # we copy them to avoid loss of KerasHistory metadata.
    flat_outputs = nest.flatten(outputs)
    flat_inputs = nest.flatten((args, kwargs))
    input_ids_set = {id(i) for i in flat_inputs}
    outputs_copy = []
    for x in flat_outputs:
      if id(x) in input_ids_set:
        with backend.name_scope(self.name):
          x = array_ops.identity(x)
      outputs_copy.append(x)
    outputs = nest.pack_sequence_as(outputs, outputs_copy)

    # Create node, Node wires itself to inbound and outbound layers.
    # The Node constructor actually updates this layer's self._inbound_nodes,
    # sets _keras_history on the outputs, and adds itself to the
    # `_outbound_nodes` of the layers that produced the inputs to this
    # layer call.
    node_module.Node(self, call_args=args, call_kwargs=kwargs, outputs=outputs)
    return outputs

  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
    """Private utility to retrieves an attribute (e.g. inputs) from a node.

    This is used to implement the methods:
        - get_input_shape_at
        - get_output_shape_at
        - get_input_at
        etc...

    Args:
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
    if isinstance(values, list) and len(values) == 1:
      return values[0]
    else:
      return values

  def _maybe_build(self, inputs):
    # Check input assumptions set before layer building, e.g. input rank.
    if not self.built:
      input_spec.assert_input_compatibility(
          self.input_spec, inputs, self.name)
      input_list = nest.flatten(inputs)
      if input_list and self._dtype_policy.compute_dtype is None:
        try:
          dtype = input_list[0].dtype.base_dtype.name
        except AttributeError:
          pass
        else:
          self._set_dtype_policy(policy.Policy(dtype))
      input_shapes = None
      # Converts Tensors / CompositeTensors to TensorShapes.
      if all(hasattr(x, 'shape') for x in input_list):
        input_shapes = tf_utils.get_shapes(inputs)
      else:
        # Converts input shape to TensorShapes.
        try:
          input_shapes = tf_utils.convert_shapes(inputs, to_tuples=False)
        except ValueError:
          pass
      # Only call `build` if the user has manually overridden the build method.
      if not hasattr(self.build, '_is_default'):
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
          self.build(input_shapes)  # pylint:disable=not-callable
      # We must set also ensure that the layer is marked as built, and the build
      # shape is stored since user defined build functions may not be calling
      # `super.build()`
      Layer.build(self, input_shapes)

    # Optionally load weight values specified at layer instantiation.
    if self._initial_weights is not None:
      with ops.init_scope():
        # Using `init_scope` since we want variable assignment in
        # `set_weights` to be treated like variable initialization.
        self.set_weights(self._initial_weights)
      self._initial_weights = None

  def _symbolic_call(self, inputs):
    input_shapes = nest.map_structure(lambda x: x.shape, inputs)
    output_shapes = self.compute_output_shape(input_shapes)
    # Convert to TensorShape so that nest.map_structure will not map into
    # individual dim of the shape.
    output_shapes = tf_utils.convert_shapes(output_shapes, to_tuples=False)

    def _make_placeholder_like(shape):
      ph = backend.placeholder(shape=shape, dtype=self.dtype)
      ph._keras_mask = None
      return ph
    return nest.map_structure(_make_placeholder_like, output_shapes)

  def _get_trainable_state(self):
    """Get the `trainable` state of each sublayer.

    Returns:
      A dict mapping all sublayers to their `trainable` value.
    """
    trainable_state = weakref.WeakKeyDictionary()
    for layer in self._flatten_layers():
      trainable_state[layer] = layer.trainable
    return trainable_state

  def _set_trainable_state(self, trainable_state):
    """Set `trainable` state for each sublayer."""
    for layer in self._flatten_layers():
      if layer in trainable_state:
        layer.trainable = trainable_state[layer]

  @property
  def _obj_reference_counts(self):
    """A dictionary counting the number of attributes referencing an object."""
    self._maybe_create_attribute('_obj_reference_counts_dict',
                                 object_identity.ObjectIdentityDictionary())
    return self._obj_reference_counts_dict

  @trackable.no_automatic_dependency_tracking
  def _maybe_create_attribute(self, name, default_value):
    """Create the attribute with the default value if it hasn't been created.

    This is useful for fields that is used for tracking purpose,
    _trainable_weights, or _layers. Note that user could create a layer subclass
    and assign an internal field before invoking the Layer.__init__(), the
    __setattr__() need to create the tracking fields and __init__() need to not
    override them.

    Args:
      name: String, the name of the attribute.
      default_value: Object, the default value of the attribute.
    """
    if not hasattr(self, name):
      self.__setattr__(name, default_value)

  def __delattr__(self, name):
    # For any super.__delattr__() call, we will directly use the implementation
    # in Trackable and skip the behavior in AutoTrackable. The Layer was
    # originally use Trackable as base class, the change of using Module as base
    # class forced us to have AutoTrackable in the class hierarchy.
    #
    # TODO(b/180760306) Keeping the status quo of skipping _delattr__ and
    # __setattr__ in AutoTrackable may be unsustainable.
    existing_value = getattr(self, name, None)

    # If this value is replacing an existing object assigned to an attribute, we
    # should clean it out to avoid leaking memory. First we check if there are
    # other attributes referencing it.
    reference_counts = self._obj_reference_counts
    if existing_value not in reference_counts:
      super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call
      return

    reference_count = reference_counts[existing_value]
    if reference_count > 1:
      # There are other remaining references. We can't remove this object from
      # _layers etc.
      reference_counts[existing_value] = reference_count - 1
      super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call
      return
    else:
      # This is the last remaining reference.
      del reference_counts[existing_value]

    super(autotrackable.AutoTrackable, self).__delattr__(name)  # pylint: disable=bad-super-call

    if (isinstance(existing_value, Layer)
        or base_layer_utils.has_weights(existing_value)):
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_self_tracked_trackables',
          [l for l in self._self_tracked_trackables if l is not existing_value])
    if isinstance(existing_value, tf_variables.Variable):
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_trainable_weights',
          [w for w in self._trainable_weights if w is not existing_value])
      super(autotrackable.AutoTrackable, self).__setattr__(  # pylint: disable=bad-super-call
          '_non_trainable_weights',
          [w for w in self._non_trainable_weights if w is not existing_value])

  def __setattr__(self, name, value):
    if (name == '_self_setattr_tracking' or
        not getattr(self, '_self_setattr_tracking', True) or
        # Exclude @property.setters from tracking
        hasattr(self.__class__, name)):
      try:
        super(autotrackable.AutoTrackable, self).__setattr__(name, value)  # pylint: disable=bad-super-call
      except AttributeError:
        raise AttributeError(
            ('Can\'t set the attribute "{}", likely because it conflicts with '
             'an existing read-only @property of the object. Please choose a '
             'different name.').format(name))
      return

    # Wraps data structures in `Trackable`, unwraps `NoDependency` objects.
    value = data_structures.sticky_attribute_assignment(
        trackable=self, value=value, name=name)

    reference_counts = self._obj_reference_counts
    reference_counts[value] = reference_counts.get(value, 0) + 1

    # Clean out the old attribute, which clears _layers and _trainable_weights
    # if necessary.
    try:
      self.__delattr__(name)
    except AttributeError:
      pass

    # Keep track of metric instance created in subclassed layer.
    for val in nest.flatten(value):
      if isinstance(val, metrics_mod.Metric) and hasattr(self, '_metrics'):
        self._metrics.append(val)

    # Append value to self._self_tracked_trackables if relevant
    if (getattr(self, '_auto_track_sub_layers', True) and
        (isinstance(value, module.Module) or
         base_layer_utils.has_weights(value))):
      self._maybe_create_attribute('_self_tracked_trackables', [])
      # We need to check object identity to avoid de-duplicating empty
      # container types which compare equal.
      if not any((layer is value for layer in self._self_tracked_trackables)):
        self._self_tracked_trackables.append(value)
        if hasattr(value, '_use_resource_variables'):
          # Legacy layers (V1 tf.layers) must always use
          # resource variables.
          value._use_resource_variables = True

    # Append value to list of trainable / non-trainable weights if relevant
    # TODO(b/125122625): This won't pick up on any variables added to a
    # list/dict after creation.
    for val in nest.flatten(value, expand_composites=True):
      if not isinstance(val, tf_variables.Variable):
        continue

      # Users may add extra weights/variables
      # simply by assigning them to attributes (invalid for graph networks)
      self._maybe_create_attribute('_trainable_weights', [])
      self._maybe_create_attribute('_non_trainable_weights', [])
      if val.trainable:
        if any(val is w for w in self._trainable_weights):
          continue
        self._trainable_weights.append(val)
      else:
        if any(val is w for w in self._non_trainable_weights):
          continue
        self._non_trainable_weights.append(val)

      backend.track_variable(val)

    # TODO(b/180760306) Skip the auto trackable from tf.Module to keep status
    # quo. See the comment at __delattr__.
    super(autotrackable.AutoTrackable, self).__setattr__(name, value)  # pylint: disable=bad-super-call

  def _gather_children_attribute(self, attribute):
    assert attribute in {
        'variables', 'trainable_variables', 'non_trainable_variables'
    }
    if hasattr(self, '_self_tracked_trackables'):
      nested_layers = self._flatten_modules(include_self=False, recursive=False)
      return list(
          itertools.chain.from_iterable(
              getattr(layer, attribute) for layer in nested_layers))
    return []

  def _flatten_layers(self, recursive=True, include_self=True):
    for m in self._flatten_modules(
        recursive=recursive, include_self=include_self):
      if isinstance(m, Layer):
        yield m

  def _flatten_modules(self, recursive=True, include_self=True):
    """Flattens `tf.Module` instances (excluding `Metrics`).

    Args:
      recursive: Whether to recursively flatten through submodules.
      include_self: Whether to include this `Layer` instance.

    Yields:
      `tf.Module` instance tracked by this `Layer`.
    """
    if include_self:
      yield self

    # Only instantiate set and deque if needed.
    trackables = getattr(self, '_self_tracked_trackables', None)
    if trackables:
      seen_object_ids = set()
      deque = collections.deque(trackables)
      while deque:
        trackable_obj = deque.popleft()
        trackable_id = id(trackable_obj)
        if trackable_id in seen_object_ids:
          continue
        seen_object_ids.add(trackable_id)

        # Metrics are not considered part of the Layer's topology.
        if (isinstance(trackable_obj, module.Module) and
            not isinstance(trackable_obj, metrics_mod.Metric)):
          yield trackable_obj
          # Introspect recursively through sublayers.
          if recursive:
            subtrackables = getattr(trackable_obj, '_self_tracked_trackables',
                                    None)
            if subtrackables:
              deque.extendleft(reversed(subtrackables))
        elif isinstance(trackable_obj, data_structures.TrackableDataStructure):
          # Data structures are introspected even with `recursive=False`.
          tracked_values = trackable_obj._values
          if tracked_values:
            deque.extendleft(reversed(tracked_values))

  # This is a hack so that the is_layer (within
  # training/trackable/layer_utils.py) check doesn't get the weights attr.
  # TODO(b/110718070): Remove when fixed.
  def _is_layer(self):
    return True

  def _init_call_fn_args(self, expects_training_arg=None):
    # Clear cached call function arguments.
    self.__class__._call_full_argspec.fget.cache.pop(self, None)
    self.__class__._call_fn_args.fget.cache.pop(self, None)
    self.__class__._call_accepts_kwargs.fget.cache.pop(self, None)

    call_fn_args = self._call_fn_args
    call_fn_args += self._call_full_argspec.kwonlyargs or []
    if expects_training_arg is None:
      self._expects_training_arg = ('training' in call_fn_args or
                                    self._call_accepts_kwargs)
    else:
      # Use value encoded into the metadata when loading from the SavedModel.
      self._expects_training_arg = expects_training_arg
    # The default training arg will be any (non-None) default specified in the
    # method signature, or None if no value is specified.
    call_fn_arg_defaults = self._call_fn_arg_defaults.copy()
    call_fn_arg_defaults.update(self._call_full_argspec.kwonlydefaults or {})
    self._default_training_arg = call_fn_arg_defaults.get('training')

    self._expects_mask_arg = ('mask' in call_fn_args or
                              self._call_accepts_kwargs)

  @property
  @layer_utils.cached_per_instance
  def _call_full_argspec(self):
    # Argspec inspection is expensive and the call spec is used often, so it
    # makes sense to cache the result.
    return tf_inspect.getfullargspec(self.call)

  @property
  @layer_utils.cached_per_instance
  def _call_fn_args(self):
    all_args = self._call_full_argspec.args
    # Scrub `self` that appears if a decorator was applied.
    if all_args and all_args[0] == 'self':
      return all_args[1:]
    return all_args

  @property
  @layer_utils.cached_per_instance
  def _call_fn_arg_defaults(self):
    call_fn_args = self._call_fn_args
    call_fn_defaults = self._call_full_argspec.defaults or []
    defaults = dict()

    # The call arg defaults are an n-tuple of the last n elements of the args
    # list. (n = # of elements that have a default argument)
    for i in range(-1 * len(call_fn_defaults), 0):
      defaults[call_fn_args[i]] = call_fn_defaults[i]
    return defaults

  @property
  @layer_utils.cached_per_instance
  def _call_fn_arg_positions(self):
    call_fn_arg_positions = dict()
    for pos, arg in enumerate(self._call_fn_args):
      call_fn_arg_positions[arg] = pos
    return call_fn_arg_positions

  @property
  @layer_utils.cached_per_instance
  def _call_accepts_kwargs(self):
    return self._call_full_argspec.varkw is not None

  @property
  def _eager_losses(self):
    # A list of loss values containing activity regularizers and losses
    # manually added through `add_loss` during eager execution. It is cleared
    # after every batch.
    # Because we plan on eventually allowing a same model instance to be trained
    # in eager mode or graph mode alternatively, we need to keep track of
    # eager losses and symbolic losses via separate attributes.
    if not hasattr(self._thread_local, '_eager_losses'):
      self._thread_local._eager_losses = []
    return self._thread_local._eager_losses

  @_eager_losses.setter
  def _eager_losses(self, losses):
    self._thread_local._eager_losses = losses

  def _dedup_weights(self, weights):
    """Dedupe weights while maintaining order as much as possible."""
    output, seen_ids = [], set()
    for w in weights:
      if id(w) not in seen_ids:
        output.append(w)
        # Track the Variable's identity to avoid __eq__ issues.
        seen_ids.add(id(w))

    return output

  def _split_out_first_arg(self, args, kwargs):
    # Grab the argument corresponding to the first argument in the
    # layer's `call` method spec. This will either be the first positional
    # argument, or it will be provided as a keyword argument.
    if args:
      inputs = args[0]
      args = args[1:]
    elif self._call_fn_args[0] in kwargs:
      kwargs = copy.copy(kwargs)
      inputs = kwargs.pop(self._call_fn_args[0])
    else:
      raise ValueError(
          'The first argument to `Layer.call` must always be passed.')
    return inputs, args, kwargs

  # SavedModel properties. Please see keras/saving/saved_model for details.

  @trackable.no_automatic_dependency_tracking
  def _set_save_spec(self, inputs):
    if self._saved_model_inputs_spec is not None:
      return  # Already set.

    self._saved_model_inputs_spec = nest.map_structure(tf_utils.get_tensor_spec,
                                                       inputs)

  def _get_save_spec(self, dynamic_batch=True):
    if self._saved_model_inputs_spec is None:
      return None

    return nest.map_structure(
        lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=dynamic_batch),
        self._saved_model_inputs_spec)

  @property
  def _trackable_saved_model_saver(self):
    return layer_serialization.LayerSavedModelSaver(self)

  @property
  def _object_identifier(self):
    return self._trackable_saved_model_saver.object_identifier

  @property
  def _tracking_metadata(self):
    """Info about this layer to be saved into the SavedModel."""
    return self._trackable_saved_model_saver.tracking_metadata

  def _trackable_children(self, save_type='checkpoint', **kwargs):
    if save_type == 'savedmodel':
      cache = kwargs['cache']
      # TODO(b/213628533): This must be called before super() to ensure
      # that any input shape changes are applied before getting the config of
      # the model.
      children = self._trackable_saved_model_saver.trackable_children(cache)
    else:
      children = {}
    children.update(super()._trackable_children(save_type, **kwargs))
    return children

  @property
  def _use_input_spec_as_call_signature(self):
    # Whether input spec can be used as the call signature when tracing the
    # Layer for SavedModel. By default, this is set to `True` for layers
    # exported from the Keras library, because the layers more rigidly define
    # the `input_specs` property (many custom layers only set the `ndims`)
    return get_canonical_name_for_symbol(type(self),
                                         api_name='keras') is not None

  def __getstate__(self):
    # Override to support `copy.deepcopy` and pickling.
    # Thread-local objects cannot be copied in Python 3, so pop these.
    # Thread-local objects are used to cache losses in MirroredStrategy, and
    # so shouldn't be copied.
    state = self.__dict__.copy()
    state.pop('_thread_local', None)
    state.pop('_metrics_lock', None)
    return state

  def __setstate__(self, state):
    state['_thread_local'] = threading.local()
    state['_metrics_lock'] = threading.Lock()
    # Bypass Trackable logic as `__dict__` already contains this info.
    object.__setattr__(self, '__dict__', state)


class TensorFlowOpLayer(Layer):
  """Wraps a TensorFlow Operation in a Layer.

  This class is used internally by the Functional API. When a user
  uses a raw TensorFlow Operation on symbolic tensors originating
  from an `Input` Layer, the resultant operation will be wrapped
  with this Layer object in order to make the operation compatible
  with the Keras API.

  This Layer will create a new, identical operation (except for inputs
  and outputs) every time it is called. If `run_eagerly` is `True`,
  the op creation and calculation will happen inside an Eager function.

  Instances of this Layer are created when `autolambda` is called, which
  is whenever a Layer's `__call__` encounters symbolic inputs that do
  not have Keras metadata, or when a Network's `__init__` encounters
  outputs that do not have Keras metadata.

  Attributes:
    node_def: String, the serialized NodeDef of the Op this layer will wrap.
    name: String, the name of the Layer.
    constants: Dict of NumPy arrays, the values of any Tensors needed for this
      Operation that do not originate from a Keras `Input` Layer. Since all
      placeholders must come from Keras `Input` Layers, these Tensors must be
      treated as constant in the Functional API.
    trainable: Bool, whether this Layer is trainable. Currently Variables are
      not supported, and so this parameter has no effect.
    dtype: The default dtype of this Layer. Inherited from `Layer` and has no
      effect on this class, however is used in `get_config`.
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self,
               node_def,
               name,
               constants=None,
               trainable=True,
               dtype=None):
    # Pass autocast=False, as if inputs are cast, input types might not match
    # Operation type.
    super(TensorFlowOpLayer, self).__init__(
        name=_TF_OP_LAYER_NAME_PREFIX + name, trainable=trainable, dtype=dtype,
        autocast=False)
    if isinstance(node_def, dict):
      self.node_def = json_format.ParseDict(node_def, node_def_pb2.NodeDef())
    else:
      if not isinstance(node_def, bytes):
        node_def = node_def.encode('utf-8')
      self.node_def = node_def_pb2.NodeDef.FromString(node_def)
    # JSON serialization stringifies keys which are integer input indices.
    self.constants = ({
        int(index): constant for index, constant in constants.items()
    } if constants is not None else {})
    # Layer uses original op unless it is called on new inputs.
    # This means `built` is not set in `__call__`.
    self.built = True

    # Do not individually trace TensorflowOpLayers in the SavedModel.
    self._must_restore_from_config = True

  def call(self, inputs):
    if context.executing_eagerly():
      return self._defun_call(inputs)
    return self._make_op(inputs)

  def _make_node_def(self, graph):
    node_def = node_def_pb2.NodeDef()
    node_def.CopyFrom(self.node_def)
    # Used in TPUReplicateContext to indicate whether this node has been cloned
    # and to not add TPU attributes.
    node_def.attr['_cloned'].b = True
    node_def.name = graph.unique_name(node_def.name)
    return node_def

  def _make_op(self, inputs):
    inputs = nest.flatten(inputs)
    graph = inputs[0].graph
    node_def = self._make_node_def(graph)
    with graph.as_default():
      for index, constant in self.constants.items():
        # Recreate constant in graph to add distribution context.
        value = tensor_util.constant_value(constant)
        if value is not None:
          constant = constant_op.constant(value, name=node_def.input[index])
        inputs.insert(index, constant)
      # TODO(b/183990973): We should drop or consolidate these private api calls
      # for adding an op to the graph and recording its gradient.
      c_op = ops._create_c_op(graph, node_def, inputs, control_inputs=[])
      op = graph._create_op_from_tf_operation(c_op)
      op._control_flow_post_processing()

      # Record the gradient because custom-made ops don't go through the
      # code-gen'd eager call path
      op_type = compat.as_str(op.op_def.name)
      attr_names = [compat.as_str(attr.name) for attr in op.op_def.attr]
      attrs = []
      for attr_name in attr_names:
        attrs.append(attr_name)
        attrs.append(op.get_attr(attr_name))
      attrs = tuple(attrs)
      backprop.record_gradient(op_type, op.inputs, attrs, op.outputs)

      if len(op.outputs) == 1:
        return op.outputs[0]
      return op.outputs

  @def_function.function
  def _defun_call(self, inputs):
    """Wraps the op creation method in an Eager function for `run_eagerly`."""
    return self._make_op(inputs)

  def get_config(self):
    config = super(TensorFlowOpLayer, self).get_config()
    config.update({
        # `__init__` prefixes the name. Revert to the constructor argument.
        'name': config['name'][len(_TF_OP_LAYER_NAME_PREFIX):],
        'node_def': json_format.MessageToDict(self.node_def),
        'constants': {
            i: backend.get_value(c) for i, c in self.constants.items()
        }
    })
    return config


class AddLoss(Layer):
  """Adds its inputs as a loss.

  Attributes:
    unconditional: Whether or not the loss should be conditioned on the inputs.
  """

  def __init__(self, unconditional, **kwargs):
    # Pass autocast=False, as there is no reason to cast loss to a different
    # dtype.
    kwargs['autocast'] = False
    super(AddLoss, self).__init__(**kwargs)
    self.unconditional = unconditional

  def call(self, inputs):
    self.add_loss(inputs, inputs=(not self.unconditional))
    return inputs

  def get_config(self):
    config = super(AddLoss, self).get_config()
    config.update({'unconditional': self.unconditional})
    return config


class AddMetric(Layer):
  """Adds its inputs as a metric.

  Attributes:
    aggregation: 'mean' or None. How the inputs should be aggregated.
    metric_name: The name to use for this metric.
  """

  def __init__(self, aggregation=None, metric_name=None, **kwargs):
    super(AddMetric, self).__init__(**kwargs)
    self.aggregation = aggregation
    self.metric_name = metric_name

  def call(self, inputs):
    self.add_metric(inputs, aggregation=self.aggregation, name=self.metric_name)
    return inputs

  def get_config(self):
    config = super(AddMetric, self).get_config()
    config.update({
        'aggregation': self.aggregation,
        'metric_name': self.metric_name
    })
    return config


def _in_functional_construction_mode(layer, inputs, args, kwargs, input_list):  # pylint: disable=unused-argument
  """Check the arguments to see if we are constructing a functional model."""
  # We are constructing a functional model if any of the inputs
  # are KerasTensors
  return any(
      isinstance(tensor, keras_tensor.KerasTensor)
      for tensor in nest.flatten([inputs, args, kwargs]))


def _convert_numpy_or_python_types(x):
  if isinstance(x, (np_arrays.ndarray, np.ndarray, float, int)):
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(x)
  return x


# Avoid breaking users who directly import this symbol from this file.
# TODO(fchollet): remove this.
InputSpec = input_spec.InputSpec  # pylint:disable=invalid-name
