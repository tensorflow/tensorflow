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
"""Training-related part of the Keras engine."""

import copy
import itertools
import json
import os
import warnings
import weakref

from tensorflow.python.autograph.lang import directives
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import constants as sm_constants
from tensorflow.python.saved_model import loader_impl as sm_loader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.tools.docs import doc_controls


# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None
# pylint: enable=g-import-not-at-top


def disable_multi_worker(method):
  """Decorator that disallows multi-worker use of `method`."""

  def _method_wrapper(self, *args, **kwargs):
    if self._in_multi_worker_mode():  # pylint: disable=protected-access
      raise ValueError('{} is not supported in multi-worker mode.'.format(
          method.__name__))
    return method(self, *args, **kwargs)

  return tf_decorator.make_decorator(
      target=method, decorator_func=_method_wrapper)


def inject_functional_model_class(cls):
  """Inject `Functional` into the hierarchy of this class if needed."""
  from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.engine import training_v1  # pylint: disable=g-import-not-at-top
  if cls == Model or cls == training_v1.Model:
    return functional.Functional
  # In case there is any multiple inheritance, we stop injecting the
  # class if keras model is not in its class hierarchy.
  if cls == object:
    return object

  cls.__bases__ = tuple(inject_functional_model_class(base)
                        for base in cls.__bases__)
  # Trigger any `__new__` class swapping that needed to happen on `Functional`
  # but did not because functional was not in the class hierarchy.
  cls.__new__(cls)

  return cls


def is_functional_model_init_params(args, kwargs):
  return (len(args) == 2 or
          len(args) == 1 and 'outputs' in kwargs or
          'inputs' in kwargs and 'outputs' in kwargs)


class Model(base_layer.Layer, version_utils.ModelVersionSelector):
  """`Model` groups layers into an object with training and inference features.

  Args:
      inputs: The input(s) of the model: a `keras.Input` object or list of
          `keras.Input` objects.
      outputs: The output(s) of the model. See Functional API example below.
      name: String, the name of the model.

  There are two ways to instantiate a `Model`:

  1 - With the "Functional API", where you start from `Input`,
  you chain layer calls to specify the model's forward pass,
  and finally you create your model from inputs and outputs:

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  Note: Only dicts, lists, and tuples of input tensors are supported. Nested
  inputs are not supported (e.g. lists of list or dicts of dict).

  2 - By subclassing the `Model` class: in that case, you should define your
  layers in `__init__` and you should implement the model's forward pass
  in `call`.

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)

  model = MyModel()
  ```

  If you subclass `Model`, you can optionally have
  a `training` argument (boolean) in `call`, which you can use to specify
  a different behavior in training and inference:

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
      x = self.dense1(inputs)
      if training:
        x = self.dropout(x, training=training)
      return self.dense2(x)

  model = MyModel()
  ```

  Once the model is created, you can config the model with losses and metrics
  with `model.compile()`, train the model with `model.fit()`, or use the model
  to do prediction with `model.predict()`.
  """
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(
      itertools.chain(('_train_counter', '_test_counter', '_predict_counter',
                       '_steps_per_execution'),
                      base_layer.Layer._TF_MODULE_IGNORED_PROPERTIES))  # pylint: disable=protected-access

  def __new__(cls, *args, **kwargs):
    # Signature detection
    if is_functional_model_init_params(args, kwargs) and cls == Model:
      # Functional model
      from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
      return functional.Functional(skip_init=True, *args, **kwargs)
    else:
      return super(Model, cls).__new__(cls, *args, **kwargs)

  @trackable.no_automatic_dependency_tracking
  def __init__(self, *args, **kwargs):
    self._is_model_for_instrumentation = True

    # Special case for Subclassed Functional Model, which we couldn't detect
    # when __new__ is called. We only realize it is a functional model when it
    # calls super.__init__ with input and output tensor.
    from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
    if (is_functional_model_init_params(args, kwargs) and
        not isinstance(self, functional.Functional)):
      # Filter the kwargs for multiple inheritance.
      supported_kwargs = ['inputs', 'outputs', 'name', 'trainable', 'skip_init']
      model_kwargs = {k: kwargs[k] for k in kwargs if k in supported_kwargs}
      other_kwargs = {k: kwargs[k] for k in kwargs if k not in supported_kwargs}
      inject_functional_model_class(self.__class__)
      functional.Functional.__init__(self, *args, **model_kwargs)

      # In case there is any multiple inheritance here, we need to call the
      # __init__ for any class that appears after the Functional class.
      clz_to_init = []
      found_functional_class = False
      for clz in self.__class__.__bases__:
        if issubclass(clz, functional.Functional):
          found_functional_class = True
          continue
        if found_functional_class:
          clz_to_init.append(clz)

      if clz_to_init:
        for clz in clz_to_init:
          clz.__init__(self, *args, **other_kwargs)
      elif other_kwargs:
        # In case there are unused kwargs, we should raise an error to user, in
        # case they have a typo in the param name.
        raise TypeError(
            'The following keyword arguments aren\'t supported: {}'.format(
                other_kwargs))
      return

    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # `inputs` / `outputs` will only appear in kwargs if either are misspelled.
    generic_utils.validate_kwargs(kwargs, {
        'trainable', 'dtype', 'dynamic', 'name', 'autocast', 'inputs', 'outputs'
    })
    super(Model, self).__init__(**kwargs)
    # By default, Model is a subclass model, which is not in graph network.
    self._is_graph_network = False

    self.inputs = None
    self.outputs = None
    self.input_names = None
    self.output_names = None
    # stop_training is used by callback to stop training when error happens
    self.stop_training = False
    self.history = None
    # These objects are used in the default `Model.compile`. They are not
    # guaranteed to be set after `Model.compile` is called, as users can
    # override compile with custom logic.
    self.compiled_loss = None
    self.compiled_metrics = None

    # This is True for Sequential networks and Functional networks.
    self._compute_output_and_mask_jointly = False

    # Don't reset compilation if already done. This may occur if calling
    # `__init__` (or `_init_graph_network`) on an already-compiled model
    # such as a Sequential model. Sequential models may need to rebuild
    # themselves after compilation.
    self._maybe_create_attribute('_is_compiled', False)
    self._maybe_create_attribute('optimizer', None)

    # Model must be created under scope of DistStrat it will be trained with.
    if distribute_lib.has_strategy():
      self._distribution_strategy = distribute_lib.get_strategy()
    else:
      self._distribution_strategy = None

    self._cluster_coordinator = None

    # Defaults to value of `tf.config.experimental_functions_run_eagerly`.
    self._run_eagerly = None
    # Initialize cache attrs.
    self._reset_compile_cache()

    # Fault-tolerance handler. Set in `ModelCheckpoint`.
    self._training_state = None
    self._saved_model_inputs_spec = None
    self._checkpoint = trackable_utils.Checkpoint(root=weakref.ref(self))

    self._steps_per_execution = None

    self._init_batch_counters()
    self._base_model_initialized = True

  @trackable.no_automatic_dependency_tracking
  def _init_batch_counters(self):
    # Untracked Variables, used to keep track of mini-batches seen in `fit`,
    # `evaluate`, and `predict`.
    agg = variables.VariableAggregationV2.ONLY_FIRST_REPLICA
    self._train_counter = variables.Variable(0, dtype='int64', aggregation=agg)
    self._test_counter = variables.Variable(0, dtype='int64', aggregation=agg)
    self._predict_counter = variables.Variable(
        0, dtype='int64', aggregation=agg)

  def __setattr__(self, name, value):
    if not getattr(self, '_self_setattr_tracking', True):
      super(Model, self).__setattr__(name, value)
      return

    if all(
        isinstance(v, (base_layer.Layer, variables.Variable)) or
        base_layer_utils.has_weights(v) for v in nest.flatten(value)):
      try:
        self._base_model_initialized
      except AttributeError:
        raise RuntimeError(
            'It looks like you are subclassing `Model` and you '
            'forgot to call `super().__init__()`.'
            ' Always start with this line.')

    super(Model, self).__setattr__(name, value)

  @generic_utils.default
  def build(self, input_shape):
    """Builds the model based on input shapes received.

    This is to be used for subclassed models, which do not know at instantiation
    time what their inputs look like.

    This method only exists for users who want to call `model.build()` in a
    standalone way (as a substitute for calling the model on real data to
    build it). It will never be called by the framework (and thus it will
    never throw unexpected errors in an unrelated workflow).

    Args:
     input_shape: Single tuple, TensorShape, or list/dict of shapes, where
         shapes are tuples, integers, or TensorShapes.

    Raises:
      ValueError:
        1. In case of invalid user-provided data (not of type tuple,
           list, TensorShape, or dict).
        2. If the model requires call arguments that are agnostic
           to the input shapes (positional or kwarg in call signature).
        3. If not all layers were properly built.
        4. If float type inputs are not supported within the layers.

      In each of these cases, the user should build their model by calling it
      on real tensor data.
    """
    if self._is_graph_network:
      super(Model, self).build(input_shape)
      return

    if input_shape is None:
      raise ValueError('Input shape must be defined when calling build on a '
                       'model subclass network.')
    valid_types = (tuple, list, tensor_shape.TensorShape, dict)
    if not isinstance(input_shape, valid_types):
      raise ValueError('Specified input shape is not one of the valid types. '
                       'Please specify a batch input shape of type tuple or '
                       'list of input shapes. User provided '
                       'input type: {}'.format(type(input_shape)))

    if input_shape and not self.inputs:
      # We create placeholders for the `None`s in the shape and build the model
      # in a Graph. Since tf.Variable is compatible with both eager execution
      # and graph building, the variables created after building the model in
      # a Graph are still valid when executing eagerly.
      if context.executing_eagerly():
        graph = func_graph.FuncGraph('build_graph')
      else:
        graph = backend.get_graph()
      with graph.as_default():
        if (isinstance(input_shape, list) and
            all(d is None or isinstance(d, int) for d in input_shape)):
          input_shape = tuple(input_shape)
        if isinstance(input_shape, list):
          x = [base_layer_utils.generate_placeholders_from_shape(shape)
               for shape in input_shape]
        elif isinstance(input_shape, dict):
          x = {
              k: base_layer_utils.generate_placeholders_from_shape(shape)
              for k, shape in input_shape.items()
          }
        else:
          x = base_layer_utils.generate_placeholders_from_shape(input_shape)

        kwargs = {}
        call_signature = self._call_full_argspec
        call_args = call_signature.args
        # Exclude `self`, `inputs`, and any argument with a default value.
        if len(call_args) > 2:
          if call_signature.defaults:
            call_args = call_args[2:-len(call_signature.defaults)]
          else:
            call_args = call_args[2:]
          for arg in call_args:
            if arg == 'training':
              # Case where `training` is a positional arg with no default.
              kwargs['training'] = False
            else:
              # Has invalid call signature with unknown positional arguments.
              raise ValueError(
                  'Currently, you cannot build your model if it has '
                  'positional or keyword arguments that are not '
                  'inputs to the model, but are required for its '
                  '`call` method. Instead, in order to instantiate '
                  'and build your model, `call` your model on real '
                  'tensor data with all expected call arguments.')
        elif len(call_args) < 2:
          # Signature without `inputs`.
          raise ValueError('You can only call `build` on a model if its `call` '
                           'method accepts an `inputs` argument.')
        try:
          self.call(x, **kwargs)
        except (errors.InvalidArgumentError, TypeError):
          raise ValueError('You cannot build your model by calling `build` '
                           'if your layers do not support float type inputs. '
                           'Instead, in order to instantiate and build your '
                           'model, `call` your model on real tensor data (of '
                           'the correct dtype).')
    super(Model, self).build(input_shape)

  @doc_controls.doc_in_current_and_subclasses
  def call(self, inputs, training=None, mask=None):
    """Calls the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Note: This method should not be called directly. It is only meant to be
    overridden when subclassing `tf.keras.Model`.
    To call a model on an input, always use the `__call__` method,
    i.e. `model(inputs)`, which relies on the underlying `call` method.

    Args:
        inputs: Input tensor, or dict/list/tuple of input tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    raise NotImplementedError('When subclassing the `Model` class, you should '
                              'implement a `call` method.')

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    """Configures the model for training.

    Args:
        optimizer: String (name of optimizer) or optimizer instance. See
          `tf.keras.optimizers`.
        loss: String (name of objective function), objective function or
          `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective
          function is any callable with the signature `loss = fn(y_true,
          y_pred)`, where y_true = ground truth values with shape =
          `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse
          categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`.
          y_pred = predicted values with shape = `[batch_size, d0, .. dN]`. It
          returns a weighted loss float tensor. If a custom `Loss` instance is
          used and reduction is set to `None`, return value has the shape
          `[batch_size, d0, .. dN-1]` i.e. per-sample or per-timestep loss
          values; otherwise, it is a scalar. If the model has multiple outputs,
          you can use a different loss on each output by passing a dictionary
          or a list of losses. The loss value that will be minimized by the
          model will then be the sum of all individual losses, unless
          `loss_weights` is specified.
        metrics: List of metrics to be evaluated by the model during training
          and testing. Each of this can be a string (name of a built-in
          function), function or a `tf.keras.metrics.Metric` instance. See
          `tf.keras.metrics`. Typically you will use `metrics=['accuracy']`. A
          function is any callable with the signature `result = fn(y_true,
          y_pred)`. To specify different metrics for different outputs of a
          multi-output model, you could also pass a dictionary, such as
          `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
          You can also pass a list to specify a metric or a list of metrics
          for each output, such as `metrics=[['accuracy'], ['accuracy', 'mse']]`
          or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
          strings 'accuracy' or 'acc', we convert this to one of
          `tf.keras.metrics.BinaryAccuracy`,
          `tf.keras.metrics.CategoricalAccuracy`,
          `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss
          function used and the model output shape. We do a similar
          conversion for the strings 'crossentropy' and 'ce' as well.
        loss_weights: Optional list or dictionary specifying scalar coefficients
          (Python floats) to weight the loss contributions of different model
          outputs. The loss value that will be minimized by the model will then
          be the *weighted sum* of all individual losses, weighted by the
          `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping to the model's
              outputs. If a dict, it is expected to map output names (strings)
              to scalar coefficients.
        weighted_metrics: List of metrics to be evaluated and weighted by
          `sample_weight` or `class_weight` during training and testing.
        run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
          logic will not be wrapped in a `tf.function`. Recommended to leave
          this as `None` unless your `Model` cannot be run inside a
          `tf.function`. `run_eagerly=True` is not supported when using
          `tf.distribute.experimental.ParameterServerStrategy`.
        steps_per_execution: Int. Defaults to 1. The number of batches to
          run during each `tf.function` call. Running multiple batches
          inside a single `tf.function` call can greatly improve performance
          on TPUs or small models with a large Python overhead.
          At most, one full epoch will be run each
          execution. If a number larger than the size of the epoch is passed,
          the execution will be truncated to the size of the epoch.
          Note that if `steps_per_execution` is set to `N`,
          `Callback.on_batch_begin` and `Callback.on_batch_end` methods
          will only be called every `N` batches
          (i.e. before/after each `tf.function` execution).
        **kwargs: Arguments supported for backwards compatibility only.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss` or `metrics`.
    """
    with self.distribute_strategy.scope():
      if 'experimental_steps_per_execution' in kwargs:
        logging.warning('The argument `steps_per_execution` is no longer '
                        'experimental. Pass `steps_per_execution` instead of '
                        '`experimental_steps_per_execution`.')
        if not steps_per_execution:
          steps_per_execution = kwargs.pop('experimental_steps_per_execution')

      # When compiling from an already-serialized model, we do not want to
      # reapply some processing steps (e.g. metric renaming for multi-output
      # models, which have prefixes added for each corresponding output name).
      from_serialized = kwargs.pop('from_serialized', False)

      self._validate_compile(optimizer, metrics, **kwargs)
      self._run_eagerly = run_eagerly

      self.optimizer = self._get_optimizer(optimizer)
      self.compiled_loss = compile_utils.LossesContainer(
          loss, loss_weights, output_names=self.output_names)
      self.compiled_metrics = compile_utils.MetricsContainer(
          metrics, weighted_metrics, output_names=self.output_names,
          from_serialized=from_serialized)

      self._configure_steps_per_execution(steps_per_execution or 1)

      # Initializes attrs that are reset each time `compile` is called.
      self._reset_compile_cache()
      self._is_compiled = True

      self.loss = loss or {}  # Backwards compat.

  def _get_optimizer(self, optimizer):
    """Wraps `optimizer` in `LossScaleOptimizer` if necessary."""
    # The deprecated PolicyV1 has a loss_scale, which we use for backwards
    # compatibility to match TF 2.3 behavior. The new Policy does not have a
    # loss_scale, so we use dynamic loss scaling if the mixed_float16 policy is
    # used.
    if isinstance(self._dtype_policy, policy.PolicyV1):
      loss_scale = self._dtype_policy.loss_scale
    elif self._dtype_policy.name == 'mixed_float16':
      loss_scale = 'dynamic'
    else:
      loss_scale = None

    def _get_single_optimizer(opt):
      opt = optimizers.get(opt)
      if (loss_scale is not None and
          not isinstance(opt, lso.LossScaleOptimizer)):
        if loss_scale == 'dynamic':
          opt = lso.LossScaleOptimizer(opt)
        else:
          opt = lso.LossScaleOptimizerV1(opt, loss_scale)
      return opt

    return nest.map_structure(_get_single_optimizer, optimizer)

  @trackable.no_automatic_dependency_tracking
  def _reset_compile_cache(self):
    self.train_function = None
    self.test_function = None
    self.predict_function = None
    # Used to cache the `tf.function`'ed `train_function` to be logged in
    # TensorBoard, since the original `train_function` is not necessarily
    # a `tf.function` (e.g., with ParameterServerStrategy, the `train_function`
    # is a scheduling of the actual training function to a remote worker).
    self.train_tf_function = None

    # Used to cache `trainable` attr of `Layer`s for `fit`.
    self._compiled_trainable_state = self._get_trainable_state()

  @trackable.no_automatic_dependency_tracking
  def _configure_steps_per_execution(self, steps_per_execution):
    self._steps_per_execution = variables.Variable(
        steps_per_execution,
        dtype='int64',
        aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)

  @property
  def _should_compute_mask(self):
    return False

  @property
  def metrics(self):
    """Returns the model's metrics added using `compile`, `add_metric` APIs.

    Note: Metrics passed to `compile()` are available only after a `keras.Model`
    has been trained/evaluated on actual data.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    >>> [m.name for m in model.metrics]
    []

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> model.fit(x, y)
    >>> [m.name for m in model.metrics]
    ['loss', 'mae']

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2, name='out')
    >>> output_1 = d(inputs)
    >>> output_2 = d(inputs)
    >>> model = tf.keras.models.Model(
    ...    inputs=inputs, outputs=[output_1, output_2])
    >>> model.add_metric(
    ...    tf.reduce_sum(output_2), name='mean', aggregation='mean')
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    >>> model.fit(x, (y, y))
    >>> [m.name for m in model.metrics]
    ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
    'out_1_acc', 'mean']

    """
    metrics = []
    if self._is_compiled:
      # TODO(omalleyt): Track `LossesContainer` and `MetricsContainer` objects
      # so that attr names are not load-bearing.
      if self.compiled_loss is not None:
        metrics += self.compiled_loss.metrics
      if self.compiled_metrics is not None:
        metrics += self.compiled_metrics.metrics

    for l in self._flatten_layers():
      metrics.extend(l._metrics)  # pylint: disable=protected-access
    return metrics

  @property
  def metrics_names(self):
    """Returns the model's display labels for all outputs.

    Note: `metrics_names` are available only after a `keras.Model` has been
    trained/evaluated on actual data.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    >>> model.metrics_names
    []

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> model.fit(x, y)
    >>> model.metrics_names
    ['loss', 'mae']

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2, name='out')
    >>> output_1 = d(inputs)
    >>> output_2 = d(inputs)
    >>> model = tf.keras.models.Model(
    ...    inputs=inputs, outputs=[output_1, output_2])
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    >>> model.fit(x, (y, y))
    >>> model.metrics_names
    ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
    'out_1_acc']

    """

    # This property includes all output names including `loss` and per-output
    # losses for backward compatibility.
    return [m.name for m in self.metrics]

  @property
  def distribute_strategy(self):
    """The `tf.distribute.Strategy` this model was created under."""
    return self._distribution_strategy or distribute_lib.get_strategy()

  @property
  def run_eagerly(self):
    """Settable attribute indicating whether the model should run eagerly.

    Running eagerly means that your model will be run step by step,
    like Python code. Your model might run slower, but it should become easier
    for you to debug it by stepping into individual layer calls.

    By default, we will attempt to compile your model to a static graph to
    deliver the best execution performance.

    Returns:
      Boolean, whether the model should run eagerly.
    """
    if self.dynamic and self._run_eagerly is False:  # pylint:disable=g-bool-id-comparison
      # TODO(fchollet): consider using py_func to enable this.
      raise ValueError('Your model contains layers that can only be '
                       'successfully run in eager execution (layers '
                       'constructed with `dynamic=True`). '
                       'You cannot set `run_eagerly=False`.')

    if self._cluster_coordinator and self._run_eagerly:
      raise ValueError('When using `Model` with `ParameterServerStrategy`, '
                       '`run_eagerly` is not supported.')

    # Run eagerly logic, by priority:
    # (1) Dynamic models must be run eagerly.
    # (2) Explicitly setting run_eagerly causes a Model to be run eagerly.
    # (3) Not explicitly setting run_eagerly defaults to TF's global setting.
    return (self.dynamic or self._run_eagerly or
            (def_function.functions_run_eagerly() and
             self._run_eagerly is None))

  @run_eagerly.setter
  def run_eagerly(self, value):
    self._run_eagerly = value

  def train_step(self, data):
    """The logic for one training step.

    This method can be overridden to support custom training logic.
    For concrete examples of how to override this method see
    [Customizing what happens in
    fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
    This method is called by `Model.make_train_function`.

    This method should contain the mathematical logic for one step of training.
    This typically includes the forward pass, loss calculation, backpropagation,
    and metric updates.

    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_train_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # These are the only transformations `Model.fit` applies to user-input
    # data when a `tf.data.Dataset` is provided.
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    # Run forward pass.
    with backprop.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
    # Run backwards pass.
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def make_train_function(self):
    """Creates a function that executes one step of training.

    This method can be overridden to support custom training logic.
    This method is called by `Model.fit` and `Model.train_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual training
    logic to `Model.train_step`.

    This function is cached the first time `Model.fit` or
    `Model.train_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self.train_function is not None:
      return self.train_function

    def step_function(model, iterator):
      """Runs a single training step."""

      def run_step(data):
        outputs = model.train_step(data)
        # Ensure counter is updated only if `train_step` succeeds.
        with ops.control_dependencies(_minimum_control_deps(outputs)):
          model._train_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='first')
      write_scalar_summaries(outputs, step=model._train_counter)  # pylint: disable=protected-access
      return outputs

    if self._steps_per_execution.numpy().item() == 1:

      def train_function(iterator):
        """Runs a training execution with one step."""
        return step_function(self, iterator)

    else:

      def train_function(iterator):
        """Runs a training execution with multiple steps."""
        for _ in math_ops.range(self._steps_per_execution):
          outputs = step_function(self, iterator)
        return outputs

    if not self.run_eagerly:
      train_function = def_function.function(
          train_function, experimental_relax_shapes=True)
      self.train_tf_function = train_function

    self.train_function = train_function

    if self._cluster_coordinator:
      self.train_function = lambda iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
          train_function, args=(iterator,))

    return self.train_function

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose='auto',
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    """Trains the model for a fixed number of epochs (iterations on a dataset).

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
            callable that takes a single argument of type
            `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
            `DatasetCreator` should be used when users prefer to specify the
            per-replica batching and sharding logic for the `Dataset`.
            See `tf.keras.utils.experimental.DatasetCreator` doc for more
            information.
          A more detailed description of unpacking behavior for iterator types
          (Dataset, generator, Sequence) is given below. If using
          `tf.distribute.experimental.ParameterServerStrategy`, only
          `DatasetCreator` type is supported for `x`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 'auto', 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so verbose=2 is
            recommended when not running interactively (eg, in a production
            environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
            and `tf.keras.callbacks.History` callbacks are created automatically
            and need not be passed into `model.fit`.
            `tf.keras.callbacks.ProgbarLogger` is created or not based on
            `verbose` argument to `model.fit`.
            Callbacks with batch-level calls are currently unsupported with
            `tf.distribute.experimental.ParameterServerStrategy`, and users are
            advised to implement epoch-level calls instead with an appropriate
            `steps_per_epoch` value.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
            `validation_split` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact
            that the validation loss of data provided using `validation_split`
            or `validation_data` is not affected by regularization layers like
            noise and dropout.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
              - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
              - A `tf.data.Dataset`.
              - A Python generator or `keras.utils.Sequence` returning
              `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            `validation_data` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is ignored
            when `x` is a generator or an object of tf.data.Dataset.
            'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample. This
            argument is not supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            When passing an infinitely repeating dataset, you must specify the
            `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
            will run indefinitely with an infinitely repeating dataset.
            This argument is not supported with array inputs.
            When using `tf.distribute.experimental.ParameterServerStrategy`:
              * `steps_per_epoch=None` is not supported.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of an infinitely repeated dataset, it will run into an
            infinite loop. If 'validation_steps' is specified and only part of
            the dataset will be consumed, the evaluation will start from the
            beginning of the dataset at each epoch. This ensures that the same
            validation samples are used every time.
        validation_batch_size: Integer or `None`.
            Number of samples per validation batch.
            If unspecified, will default to `batch_size`.
            Do not specify the `validation_batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

    Unpacking behavior for iterator-like inputs:
        A common pattern is to pass a tf.data.Dataset, generator, or
      tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
      yield not only features (x) but optionally targets (y) and sample weights.
      Keras requires that the output of such iterator-likes be unambiguous. The
      iterator should return a tuple of length 1, 2, or 3, where the optional
      second and third elements will be used for y and sample_weight
      respectively. Any other type provided will be wrapped in a length one
      tuple, effectively treating everything as 'x'. When yielding dicts, they
      should still adhere to the top-level tuple structure.
      e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
      features, targets, and weights from the keys of a single dict.
        A notable unsupported data type is the namedtuple. The reason is that
      it behaves like both an ordered datatype (tuple) and a mapping
      datatype (dict). So given a namedtuple of the form:
          `namedtuple("example_tuple", ["y", "x"])`
      it is ambiguous whether to reverse the order of the elements when
      interpreting the value. Even worse is a tuple of the form:
          `namedtuple("other_tuple", ["x", "y", "z"])`
      where it is unclear if the tuple was intended to be unpacked into x, y,
      and sample_weight or passed through as a single element to `x`. As a
      result the data processing code will simply raise a ValueError if it
      encounters a namedtuple. (Along with instructions to remedy the issue.)

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: 1. If the model was never compiled or,
        2. If `model.fit` is  wrapped in `tf.function`.

        ValueError: In case of mismatch between the provided input data
            and what the model expects or when the input data is empty.
    """
    # Legacy graph support is contained in `training_v1.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')
    _disallow_inside_tf_function('fit')

    if verbose == 'auto':
      if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        verbose = 2  # Default to epoch-level logging for PSStrategy.
      else:
        verbose = 1  # Default to batch-level logging otherwise.

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y, sample_weight), validation_data = (
          data_adapter.train_validation_split(
              (x, y, sample_weight), validation_split=validation_split))

    if validation_data:
      val_x, val_y, val_sample_weight = (
          data_adapter.unpack_x_y_sample_weight(validation_data))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
          self.distribute_strategy)

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.get_data_handler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          initial_epoch=initial_epoch,
          epochs=epochs,
          shuffle=shuffle,
          class_weight=class_weight,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps)

      self.stop_training = False
      self.train_function = self.make_train_function()
      self._train_counter.assign(0)
      callbacks.on_train_begin()
      training_logs = None
      # Handle fault-tolerance for multi-worker.
      # TODO(omalleyt): Fix the ordering issues that mean this has to
      # happen after `callbacks.on_train_begin`.
      data_handler._initial_epoch = (  # pylint: disable=protected-access
          self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
      logs = None
      for epoch, iterator in data_handler.enumerate_epochs():
        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with trace.Trace(
                'train',
                epoch_num=epoch,
                step_num=step,
                batch_size=batch_size,
                _r=1):
              callbacks.on_train_batch_begin(step)
              tmp_logs = self.train_function(iterator)
              if data_handler.should_sync:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              end_step = step + data_handler.step_increment
              callbacks.on_train_batch_end(end_step, logs)
              if self.stop_training:
                break

        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        if logs is None:
          raise ValueError('Expect x to be a non-empty array or dataset.')
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._eval_data_handler = data_adapter.get_data_handler(
                x=val_x,
                y=val_y,
                sample_weight=val_sample_weight,
                batch_size=validation_batch_size or batch_size,
                steps_per_epoch=validation_steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)
          val_logs = self.evaluate(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size,
              steps=validation_steps,
              callbacks=callbacks,
              max_queue_size=max_queue_size,
              workers=workers,
              use_multiprocessing=use_multiprocessing,
              return_dict=True,
              _use_cached_eval_dataset=True)
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if self.stop_training:
          break

      # If eval data_handler exists, delete it after all epochs are done.
      if getattr(self, '_eval_data_handler', None) is not None:
        del self._eval_data_handler
      callbacks.on_train_end(logs=training_logs)
      return self.history

  def test_step(self, data):
    """The logic for one evaluation step.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.make_test_function`.

    This function should contain the mathematical logic for one step of
    evaluation.
    This typically includes the forward pass, loss calculation, and metrics
    updates.

    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_test_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values that will be passed to
      `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
      values of the `Model`'s metrics are returned.
    """
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    # Collect metrics to return
    return_metrics = {}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def make_test_function(self):
    """Creates a function that executes one step of evaluation.

    This method can be overridden to support custom evaluation logic.
    This method is called by `Model.evaluate` and `Model.test_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.test_step`.

    This function is cached the first time `Model.evaluate` or
    `Model.test_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_test_batch_end`.
    """
    if self.test_function is not None:
      return self.test_function

    def step_function(model, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = model.test_step(data)
        # Ensure counter is updated only if `test_step` succeeds.
        with ops.control_dependencies(_minimum_control_deps(outputs)):
          model._test_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='first')
      return outputs

    if self._steps_per_execution.numpy().item() == 1:

      def test_function(iterator):
        """Runs an evaluation execution with one step."""
        return step_function(self, iterator)

    else:

      def test_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        for _ in math_ops.range(self._steps_per_execution):
          outputs = step_function(self, iterator)
        return outputs

    if not self.run_eagerly:
      test_function = def_function.function(
          test_function, experimental_relax_shapes=True)

    self.test_function = test_function

    if self._cluster_coordinator:
      self.test_function = lambda iterator: self._cluster_coordinator.schedule(  # pylint: disable=g-long-lambda
          test_function, args=(iterator,))

    return self.test_function

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False,
               return_dict=False,
               **kwargs):
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          A more detailed description of unpacking behavior for iterator types
          (Dataset, generator, Sequence) is given in the `Unpacking behavior
          for iterator-like inputs` section of `Model.fit`.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely). If
          `x` is a dataset, generator or `keras.utils.Sequence` instance, `y`
          should not be specified (since targets will be obtained from the
          iterator/dataset).
        batch_size: Integer or `None`. Number of samples per batch of
          computation. If unspecified, `batch_size` will default to 32. Do not
          specify the `batch_size` if your data is in the form of a dataset,
          generators, or `keras.utils.Sequence` instances (since they generate
          batches).
        verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
        sample_weight: Optional Numpy array of weights for the test samples,
          used for weighting the loss function. You can either pass a flat (1D)
          Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples), or in the case of
              temporal data, you can pass a 2D array with shape `(samples,
              sequence_length)`, to apply a different weight to every timestep
              of every sample. This argument is not supported when `x` is a
              dataset, instead pass sample weights as the third element of `x`.
        steps: Integer or `None`. Total number of steps (batches of samples)
          before declaring the evaluation round finished. Ignored with the
          default value of `None`. If x is a `tf.data` dataset and `steps` is
          None, 'evaluate' will run until the dataset is exhausted. This
          argument is not supported with array inputs.
        callbacks: List of `keras.callbacks.Callback` instances. List of
          callbacks to apply during evaluation. See
          [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
          input only. Maximum size for the generator queue. If unspecified,
          `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
          only. Maximum number of processes to spin up when using process-based
          threading. If unspecified, `workers` will default to 1.
        use_multiprocessing: Boolean. Used for generator or
          `keras.utils.Sequence` input only. If `True`, use process-based
          threading. If unspecified, `use_multiprocessing` will default to
          `False`. Note that because this implementation relies on
          multiprocessing, you should not pass non-picklable arguments to the
          generator as they can't be passed easily to children processes.
        return_dict: If `True`, loss and metric results are returned as a dict,
          with each key being the name of the metric. If `False`, they are
          returned as a list.
        **kwargs: Unused at this time.

    See the discussion of `Unpacking behavior for iterator-like inputs` for
    `Model.fit`.

    `Model.evaluate` is not yet supported with
    `tf.distribute.experimental.ParameterServerStrategy`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: If `model.evaluate` is wrapped in `tf.function`.
        ValueError: in case of invalid arguments.
    """
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')
    _disallow_inside_tf_function('evaluate')
    use_cached_eval_dataset = kwargs.pop('_use_cached_eval_dataset', False)
    if kwargs:
      raise TypeError('Invalid keyword arguments: %s' % (kwargs,))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
          self.distribute_strategy)

    with self.distribute_strategy.scope():
      # Use cached evaluation data only when it's called in `Model.fit`
      if (use_cached_eval_dataset
          and getattr(self, '_eval_data_handler', None) is not None):
        data_handler = self._eval_data_handler
      else:
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = data_adapter.get_data_handler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      logs = {}
      self.test_function = self.make_test_function()
      self._test_counter.assign(0)
      callbacks.on_test_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        self.reset_metrics()
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with trace.Trace('test', step_num=step, _r=1):
              callbacks.on_test_batch_begin(step)
              tmp_logs = self.test_function(iterator)
              if data_handler.should_sync:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              end_step = step + data_handler.step_increment
              callbacks.on_test_batch_end(end_step, logs)
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      callbacks.on_test_end(logs=logs)

      if return_dict:
        return logs
      else:
        return flatten_metrics_in_order(logs, self.metrics_names)

  def predict_step(self, data):
    """The logic for one inference step.

    This method can be overridden to support custom inference logic.
    This method is called by `Model.make_predict_function`.

    This method should contain the mathematical logic for one step of inference.
    This typically includes the forward pass.

    Configuration details for *how* this logic is run (e.g. `tf.function` and
    `tf.distribute.Strategy` settings), should be left to
    `Model.make_predict_function`, which can also be overridden.

    Args:
      data: A nested structure of `Tensor`s.

    Returns:
      The result of one inference step, typically the output of calling the
      `Model` on data.
    """
    data = data_adapter.expand_1d(data)
    x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
    return self(x, training=False)

  def make_predict_function(self):
    """Creates a function that executes one step of inference.

    This method can be overridden to support custom inference logic.
    This method is called by `Model.predict` and `Model.predict_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual evaluation
    logic to `Model.predict_step`.

    This function is cached the first time `Model.predict` or
    `Model.predict_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return the outputs of the `Model`.
    """
    if self.predict_function is not None:
      return self.predict_function

    def step_function(model, iterator):
      """Runs a single evaluation step."""

      def run_step(data):
        outputs = model.predict_step(data)
        # Ensure counter is updated only if `test_step` succeeds.
        with ops.control_dependencies(_minimum_control_deps(outputs)):
          model._predict_counter.assign_add(1)  # pylint: disable=protected-access
        return outputs

      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='concat')
      return outputs

    if (self._steps_per_execution is None or
        self._steps_per_execution.numpy().item() == 1):

      def predict_function(iterator):
        """Runs an evaluation execution with one step."""
        return step_function(self, iterator)

    else:

      def predict_function(iterator):
        """Runs an evaluation execution with multiple steps."""
        outputs = step_function(self, iterator)
        for _ in math_ops.range(self._steps_per_execution - 1):
          directives.set_loop_options(
              shape_invariants=[(
                  t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape)
                                for t in nest.flatten(outputs)])
          step_outputs = step_function(self, iterator)
          outputs = nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs,
                                       step_outputs)
        return outputs

    if not self.run_eagerly:
      predict_function = def_function.function(
          predict_function, experimental_relax_shapes=True)

    self.predict_function = predict_function
    return self.predict_function

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    """Generates output predictions for the input samples.

    Computation is done in batches. This method is designed for performance in
    large scale inputs. For small amount of inputs that fit in one batch,
    directly using `__call__` is recommended for faster execution, e.g.,
    `model(x)`, or `model(x, training=False)` if you have layers such as
    `tf.keras.layers.BatchNormalization` that behaves differently during
    inference. Also, note the fact that test loss is not affected by
    regularization layers like noise and dropout.

    Args:
        x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
          A more detailed description of unpacking behavior for iterator types
          (Dataset, generator, Sequence) is given in the `Unpacking behavior
          for iterator-like inputs` section of `Model.fit`.
        batch_size: Integer or `None`.
            Number of samples per batch.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of dataset, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`. If x is a `tf.data`
            dataset and `steps` is None, `predict` will
            run until the input dataset is exhausted.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during prediction.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

    See the discussion of `Unpacking behavior for iterator-like inputs` for
    `Model.fit`. Note that Model.predict uses the same interpretation rules as
    `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for all
    three methods.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        RuntimeError: If `model.predict` is wrapped in `tf.function`.
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    version_utils.disallow_legacy_graph('Model', 'predict')
    self._check_call_args('predict')
    _disallow_inside_tf_function('predict')

    # TODO(yashkatariya): Cache model on the coordinator for faster prediction.
    # If running under PSS, then swap it with OneDeviceStrategy so that
    # execution will run on the coordinator.
    original_pss_strategy = None
    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      original_pss_strategy = self.distribute_strategy
      self._distribution_strategy = None

    # Cluster coordinator is set by `.fit()` and `.evaluate()` which is not
    # needed in `.predict()` because all the predictions happen on the
    # coordinator/locally.
    if self._cluster_coordinator:
      self._cluster_coordinator = None

    outputs = None
    with self.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      dataset_types = (data_types.DatasetV1, data_types.DatasetV2)
      if (self._in_multi_worker_mode() or _is_tpu_multi_host(
          self.distribute_strategy)) and isinstance(x, dataset_types):
        try:
          options = options_lib.Options()
          data_option = options_lib.AutoShardPolicy.DATA
          options.experimental_distribute.auto_shard_policy = data_option
          x = x.with_options(options)
        except ValueError:
          warnings.warn('Using Model.predict with '
                        'MultiWorkerDistributionStrategy or TPUStrategy and '
                        'AutoShardPolicy.FILE might lead to out-of-order result'
                        '. Consider setting it to AutoShardPolicy.DATA.')

      data_handler = data_adapter.get_data_handler(
          x=x,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      self.predict_function = self.make_predict_function()
      self._predict_counter.assign(0)
      callbacks.on_predict_begin()
      batch_outputs = None
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            tmp_batch_outputs = self.predict_function(iterator)
            if data_handler.should_sync:
              context.async_wait()
            batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
            if outputs is None:
              outputs = nest.map_structure(lambda batch_output: [batch_output],
                                           batch_outputs)
            else:
              nest.map_structure_up_to(
                  batch_outputs,
                  lambda output, batch_output: output.append(batch_output),
                  outputs, batch_outputs)
            end_step = step + data_handler.step_increment
            callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
      if batch_outputs is None:
        raise ValueError('Expect x to be a non-empty array or dataset.')
      callbacks.on_predict_end()
    all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)

    # If originally PSS strategy was used, then replace it back since predict
    # is running under `OneDeviceStrategy` after the swap and once its done
    # we need to replace it back to PSS again.
    if original_pss_strategy is not None:
      self._distribution_strategy = original_pss_strategy

    return tf_utils.sync_to_numpy_or_python_type(all_outputs)

  def reset_metrics(self):
    """Resets the state of all the metrics in the model.

    Examples:

    >>> inputs = tf.keras.layers.Input(shape=(3,))
    >>> outputs = tf.keras.layers.Dense(2)(inputs)
    >>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    >>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    >>> x = np.random.random((2, 3))
    >>> y = np.random.randint(0, 2, (2, 2))
    >>> _ = model.fit(x, y, verbose=0)
    >>> assert all(float(m.result()) for m in model.metrics)

    >>> model.reset_metrics()
    >>> assert all(float(m.result()) == 0 for m in model.metrics)

    """
    for m in self.metrics:
      m.reset_state()

  def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True,
                     return_dict=False):
    """Runs a single gradient update on a single batch of data.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
              (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case of
          temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample.
        class_weight: Optional dictionary mapping class indices (integers) to a
          weight (float) to apply to the model's loss for the samples from this
          class during training. This can be useful to tell the model to "pay
          more attention" to samples from an under-represented class.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.
        return_dict: If `True`, loss and metric results are returned as a dict,
          with each key being the name of the metric. If `False`, they are
          returned as a list.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
      RuntimeError: If `model.train_on_batch` is wrapped in `tf.function`.
      ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('train_on_batch')
    _disallow_inside_tf_function('train_on_batch')
    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x,
                                                    y, sample_weight,
                                                    class_weight)
      self.train_function = self.make_train_function()
      logs = self.train_function(iterator)

    if reset_metrics:
      self.reset_metrics()
    logs = tf_utils.sync_to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      return flatten_metrics_in_order(logs, self.metrics_names)

  def test_on_batch(self,
                    x,
                    y=None,
                    sample_weight=None,
                    reset_metrics=True,
                    return_dict=False):
    """Test the model on a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays (in case the
              model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors (in case the model has
              multiple inputs).
          - A dict mapping input names to the corresponding array/tensors, if
              the model has named inputs.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case of
          temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.
        return_dict: If `True`, loss and metric results are returned as a dict,
          with each key being the name of the metric. If `False`, they are
          returned as a list.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: If `model.test_on_batch` is wrapped in `tf.function`.
        ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('test_on_batch')
    _disallow_inside_tf_function('test_on_batch')
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x,
                                                    y, sample_weight)
      self.test_function = self.make_test_function()
      logs = self.test_function(iterator)

    if reset_metrics:
      self.reset_metrics()
    logs = tf_utils.sync_to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      return flatten_metrics_in_order(logs, self.metrics_names)

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays (in case the
              model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors (in case the model has
              multiple inputs).

    Returns:
        Numpy array(s) of predictions.

    Raises:
        RuntimeError: If `model.predict_on_batch` is wrapped in `tf.function`.
        ValueError: In case of mismatch between given number of inputs and
          expectations of the model.
    """
    self._check_call_args('predict_on_batch')
    _disallow_inside_tf_function('predict_on_batch')
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x)
      self.predict_function = self.make_predict_function()
      outputs = self.predict_function(iterator)
    return tf_utils.sync_to_numpy_or_python_type(outputs)

  def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0):
    """Fits the model on data yielded batch-by-batch by a Python generator.

    DEPRECATED:
      `Model.fit` now supports generators, so there is no longer any need to use
      this endpoint.
    """
    warnings.warn('`Model.fit_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.fit`, which supports generators.')
    return self.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch)

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         verbose=0):
    """Evaluates the model on a data generator.

    DEPRECATED:
      `Model.evaluate` now supports generators, so there is no longer any need
      to use this endpoint.
    """
    warnings.warn('`Model.evaluate_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.evaluate`, which supports generators.')
    self._check_call_args('evaluate_generator')

    return self.evaluate(
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,
        callbacks=callbacks)

  def predict_generator(self,
                        generator,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0):
    """Generates predictions for the input samples from a data generator.

    DEPRECATED:
      `Model.predict` now supports generators, so there is no longer any need
      to use this endpoint.
    """
    warnings.warn('`Model.predict_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.predict`, which supports generators.')
    return self.predict(
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,
        callbacks=callbacks)

  ######################################################################
  # Functions below are not training related. They are for model weights
  # tracking, save/load, serialization, etc.
  ######################################################################

  @property
  def trainable_weights(self):
    self._assert_weights_created()
    if not self._trainable:
      return []
    trainable_variables = []
    for trackable_obj in self._self_tracked_trackables:
      trainable_variables += trackable_obj.trainable_variables
    trainable_variables += self._trainable_weights
    return self._dedup_weights(trainable_variables)

  @property
  def non_trainable_weights(self):
    self._assert_weights_created()
    non_trainable_variables = []
    for trackable_obj in self._self_tracked_trackables:
      non_trainable_variables += trackable_obj.non_trainable_variables

    if not self._trainable:
      # Return order is all trainable vars, then all non-trainable vars.
      trainable_variables = []
      for trackable_obj in self._self_tracked_trackables:
        trainable_variables += trackable_obj.trainable_variables

      non_trainable_variables = (
          trainable_variables + self._trainable_weights +
          non_trainable_variables + self._non_trainable_weights)
    else:
      non_trainable_variables = (
          non_trainable_variables + self._non_trainable_weights)

    return self._dedup_weights(non_trainable_variables)

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    with self.distribute_strategy.scope():
      return super(Model, self).get_weights()

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None,
           save_traces=True):
    # pylint: disable=line-too-long
    """Saves the model to Tensorflow SavedModel or a single HDF5 file.

    Please see `tf.keras.models.save_model` or the
    [Serialization and Saving guide](https://keras.io/guides/serialization_and_saving/)
    for details.

    Args:
        filepath: String, PathLike, path to SavedModel or H5 file to save the
            model.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
        save_format: Either `'tf'` or `'h5'`, indicating whether to save the
            model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X,
            and 'h5' in TF 1.X.
        signatures: Signatures to save with the SavedModel. Applicable to the
            'tf' format only. Please see the `signatures` argument in
            `tf.saved_model.save` for details.
        options: (only applies to SavedModel format)
            `tf.saved_model.SaveOptions` object that specifies options for
            saving to SavedModel.
        save_traces: (only applies to SavedModel format) When enabled, the
            SavedModel will store the function traces for each layer. This
            can be disabled, so that only the configs of each layer are stored.
            Defaults to `True`. Disabling this will decrease serialization time
            and reduce file size, but it requires that all custom layers/models
            implement a `get_config()` method.

    Example:

    ```python
    from keras.models import load_model

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')
    ```
    """
    # pylint: enable=line-too-long
    save.save_model(self, filepath, overwrite, include_optimizer, save_format,
                    signatures, options, save_traces)

  def save_weights(self,
                   filepath,
                   overwrite=True,
                   save_format=None,
                   options=None):
    """Saves all layer weights.

    Either saves in HDF5 or in TensorFlow format based on the `save_format`
    argument.

    When saving in HDF5 format, the weight file has:
      - `layer_names` (attribute), a list of strings
          (ordered names of model layers).
      - For every layer, a `group` named `layer.name`
          - For every such layer group, a group attribute `weight_names`,
              a list of strings
              (ordered names of weights tensor of the layer).
          - For every weight in the layer, a dataset
              storing the weight value, named after the weight tensor.

    When saving in TensorFlow format, all objects referenced by the network are
    saved in the same format as `tf.train.Checkpoint`, including any `Layer`
    instances or `Optimizer` instances assigned to object attributes. For
    networks constructed from inputs and outputs using `tf.keras.Model(inputs,
    outputs)`, `Layer` instances used by the network are tracked/saved
    automatically. For user-defined classes which inherit from `tf.keras.Model`,
    `Layer` instances must be assigned to object attributes, typically in the
    constructor. See the documentation of `tf.train.Checkpoint` and
    `tf.keras.Model` for details.

    While the formats are the same, do not mix `save_weights` and
    `tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should be
    loaded using `Model.load_weights`. Checkpoints saved using
    `tf.train.Checkpoint.save` should be restored using the corresponding
    `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
    `save_weights` for training checkpoints.

    The TensorFlow format matches objects and variables by starting at a root
    object, `self` for `save_weights`, and greedily matching attribute
    names. For `Model.save` this is the `Model`, and for `Checkpoint.save` this
    is the `Checkpoint` even if the `Checkpoint` has a model attached. This
    means saving a `tf.keras.Model` using `save_weights` and loading into a
    `tf.train.Checkpoint` with a `Model` attached (or vice versa) will not match
    the `Model`'s variables. See the [guide to training
    checkpoints](https://www.tensorflow.org/guide/checkpoint) for details
    on the TensorFlow format.

    Args:
        filepath: String or PathLike, path to the file to save the weights to.
            When saving in TensorFlow format, this is the prefix used for
            checkpoint files (multiple files are generated). Note that the '.h5'
            suffix causes weights to be saved in HDF5 format.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
            '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
            `None` defaults to 'tf'.
        options: Optional `tf.train.CheckpointOptions` object that specifies
            options for saving weights.

    Raises:
        ImportError: If h5py is not available when attempting to save in HDF5
            format.
        ValueError: For invalid/unknown format arguments.
    """
    self._assert_weights_created()
    filepath = path_to_string(filepath)
    filepath_is_h5 = saving_utils.is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        hdf5_format.save_weights_to_hdf5_group(f, self.layers)
    else:
      if not context.executing_eagerly():
        # Call `get_session` to initialize any uninitialized variables.
        backend.get_session()
      self._checkpoint.write(filepath, options=options)
      # Record this checkpoint so it's visible from tf.train.latest_checkpoint.
      checkpoint_management.update_checkpoint_state_internal(
          save_dir=os.path.dirname(filepath),
          model_checkpoint_path=filepath,
          save_relative_paths=True,
          all_model_checkpoint_paths=[filepath])

  def load_weights(self,
                   filepath,
                   by_name=False,
                   skip_mismatch=False,
                   options=None):
    """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.

    If `by_name` is False weights are loaded based on the network's
    topology. This means the architecture should be the same as when the weights
    were saved.  Note that layers that don't have weights are not taken into
    account in the topological ordering, so adding or removing layers is fine as
    long as they don't have weights.

    If `by_name` is True, weights are loaded into layers only if they share the
    same name. This is useful for fine-tuning or transfer-learning models where
    some of the layers have changed.

    Only topological loading (`by_name=False`) is supported when loading weights
    from the TensorFlow format. Note that topological loading differs slightly
    between TensorFlow and HDF5 formats for user-defined classes inheriting from
    `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
    TensorFlow format loads based on the object-local names of attributes to
    which layers are assigned in the `Model`'s constructor.

    Args:
        filepath: String, path to the weights file to load. For weight files in
            TensorFlow format, this is the file prefix (the same as was passed
            to `save_weights`). This can also be a path to a SavedModel
            saved from `model.save`.
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.
        skip_mismatch: Boolean, whether to skip loading of layers where there is
            a mismatch in the number of weights, or a mismatch in the shape of
            the weight (only valid when `by_name=True`).
        options: Optional `tf.train.CheckpointOptions` object that specifies
            options for loading weights.

    Returns:
        When loading a weight file in TensorFlow format, returns the same status
        object as `tf.train.Checkpoint.restore`. When graph building, restore
        ops are run automatically as soon as the network is built (on first call
        for user-defined classes inheriting from `Model`, immediately if it is
        already built).

        When loading weights in HDF5 format, returns `None`.

    Raises:
        ImportError: If h5py is not available and the weight file is in HDF5
            format.
        ValueError: If `skip_mismatch` is set to `True` when `by_name` is
          `False`.
    """
    if backend.is_tpu_strategy(self._distribution_strategy):
      if (self._distribution_strategy.extended.steps_per_run > 1 and
          (not saving_utils.is_hdf5_filepath(filepath))):
        raise ValueError('Load weights is not yet supported with TPUStrategy '
                         'with steps_per_run greater than 1.')
    if skip_mismatch and not by_name:
      raise ValueError(
          'When calling model.load_weights, skip_mismatch can only be set to '
          'True when by_name is True.')

    filepath, save_format = _detect_save_format(filepath)
    if save_format == 'tf':
      status = self._checkpoint.read(filepath, options)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        # Restore existing variables (if any) immediately, and set up a
        # streaming restore for any variables created in the future.
        trackable_utils.streaming_restore(status=status, session=session)
      status.assert_nontrivial_match()
    else:
      status = None
      if h5py is None:
        raise ImportError(
            '`load_weights` requires h5py when loading weights from HDF5.')
      if not self._is_graph_network and not self.built:
        raise ValueError(
            'Unable to load weights saved in HDF5 format into a subclassed '
            'Model which has not created its variables yet. Call the Model '
            'first, then load the weights.')
      self._assert_weights_created()
      with h5py.File(filepath, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
          f = f['model_weights']
        if by_name:
          hdf5_format.load_weights_from_hdf5_group_by_name(
              f, self.layers, skip_mismatch=skip_mismatch)
        else:
          hdf5_format.load_weights_from_hdf5_group(f, self.layers)

    # Perform any layer defined finalization of the layer state.
    for layer in self.layers:
      layer.finalize_state()
    return status

  def _updated_config(self):
    """Util shared between different serialization methods.

    Returns:
        Model config with Keras version information added.
    """
    from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': backend.backend()
    }
    return model_config

  def get_config(self):
    raise NotImplementedError

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # `from_config` assumes `cls` is either `Functional` or a child class of
    # `Functional`. In the case that `cls` is meant to behave like a child class
    # of `Functional` but only inherits from the `Model` class, we have to call
    # `cls(...)` instead of `Functional.from_config`.
    from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
    with generic_utils.SharedObjectLoadingScope():
      input_tensors, output_tensors, created_layers = (
          functional.reconstruct_from_config(config, custom_objects))
      # Initialize a model belonging to `cls`, which can be user-defined or
      # `Functional`.
      model = cls(inputs=input_tensors, outputs=output_tensors,
                  name=config.get('name'))
      functional.connect_ancillary_layers(model, created_layers)
      return model

  def to_json(self, **kwargs):
    """Returns a JSON string containing the network configuration.

    To load a network from a JSON save file, use
    `keras.models.model_from_json(json_string, custom_objects={})`.

    Args:
        **kwargs: Additional keyword arguments
            to be passed to `json.dumps()`.

    Returns:
        A JSON string.
    """
    model_config = self._updated_config()
    return json.dumps(
        model_config, default=json_utils.get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    """Returns a yaml string containing the network configuration.

    Note: Since TF 2.6, this method is no longer supported and will raise a
    RuntimeError.

    To load a network from a yaml save file, use
    `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

    `custom_objects` should be a dictionary mapping
    the names of custom losses / layers / etc to the corresponding
    functions / classes.

    Args:
        **kwargs: Additional keyword arguments
            to be passed to `yaml.dump()`.

    Returns:
        A YAML string.

    Raises:
        RuntimeError: announces that the method poses a security risk
    """
    raise RuntimeError(
        'Method `model.to_yaml()` has been removed due to security risk of '
        'arbitrary code execution. Please use `model.to_json()` instead.'
    )

  def reset_states(self):
    for layer in self.layers:
      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
        layer.reset_states()

  @property
  @doc_controls.do_not_generate_docs
  def state_updates(self):
    """Deprecated, do NOT use!

    Returns the `updates` from all layers that are stateful.

    This is useful for separating training updates and
    state updates, e.g. when we need to update a layer's internal state
    during prediction.

    Returns:
        A list of update ops.
    """
    warnings.warn('`Model.state_updates` will be removed in a future version. '
                  'This property should not be used in TensorFlow 2.0, '
                  'as `updates` are applied automatically.')
    state_updates = []
    for layer in self.layers:
      if getattr(layer, 'stateful', False):
        if hasattr(layer, 'updates'):
          state_updates += layer.updates
    return state_updates

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
      A list of variables.
    """
    return self._dedup_weights(self._undeduplicated_weights)

  @property
  def _undeduplicated_weights(self):
    """Returns the undeduplicated list of all layer variables/weights."""
    self._assert_weights_created()
    weights = []
    for layer in self._self_tracked_trackables:
      weights += layer.variables
    weights += (self._trainable_weights + self._non_trainable_weights)
    return weights

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Prints a string summary of the network.

    Args:
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements
            in each line. If not provided,
            defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use. Defaults to `print`.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.

    Raises:
        ValueError: if `summary()` is called before the model is built.
    """
    if not self.built:
      raise ValueError('This model has not yet been built. '
                       'Build the model first by calling `build()` or calling '
                       '`fit()` with some data, or specify '
                       'an `input_shape` argument in the first layer(s) for '
                       'automatic build.')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

  @property
  def layers(self):
    return list(self._flatten_layers(include_self=False, recursive=False))

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    If `name` and `index` are both provided, `index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Args:
        name: String, name of layer.
        index: Integer, index of layer.

    Returns:
        A layer instance.

    Raises:
        ValueError: In case of invalid layer name or index.
    """
    # TODO(fchollet): We could build a dictionary based on layer names
    # since they are constant, but we have not done that yet.
    if index is not None and name is not None:
      raise ValueError('Provide only a layer name or a layer index.')

    if index is not None:
      if len(self.layers) <= index:
        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                         ' but model only has ' + str(len(self.layers)) +
                         ' layers.')
      else:
        return self.layers[index]

    if name is not None:
      for layer in self.layers:
        if layer.name == name:
          return layer
      raise ValueError('No such layer: ' + name + '.')
    raise ValueError('Provide either a layer name or layer index.')

  @trackable.no_automatic_dependency_tracking
  def _set_save_spec(self, inputs):
    if self._saved_model_inputs_spec is not None:
      return  # Already set.

    input_names = self.input_names
    if not input_names:
      input_names = compile_utils.create_pseudo_input_names(inputs)

    flat_inputs = nest.flatten(inputs)
    specs = []
    for name, tensor in zip(input_names, flat_inputs):
      specs.append(
          tf_utils.get_tensor_spec(tensor, dynamic_batch=False, name=name))
    specs = nest.pack_sequence_as(inputs, specs)

    self._saved_model_inputs_spec = specs

    # Store the input shapes
    if (self.__class__.__name__ == 'Sequential' and
        self._build_input_shape is None):
      self._build_input_shape = nest.map_structure(
          lambda x: None if x is None else x.shape, specs)

  def _assert_weights_created(self):
    """Asserts that all the weights for the model have been created.

    For a non-dynamic model, the weights must already be created after the
    layer has been called. For a dynamic model, the exact list of weights can
    never be known for certain since it may change at any time during execution.

    We run this check right before accessing weights or getting the Numpy value
    for the current weights. Otherwise, if the layer has never been called,
    the user would just get an empty list, which is misleading.

    Raises:
      ValueError: if the weights of the network has not yet been created.
    """
    if self.dynamic:
      return

    if ('build' in self.__class__.__dict__ and
        self.__class__ != Model and
        not self.built):
      # For any model that has customized build() method but hasn't
      # been invoked yet, this will cover both sequential and subclass model.
      # Also make sure to exclude Model class itself which has build() defined.
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  def _check_call_args(self, method_name):
    """Check that `call` has only one positional arg."""
    # Always allow first arg, regardless of arg name.
    fullargspec = self._call_full_argspec
    if fullargspec.defaults:
      positional_args = fullargspec.args[:-len(fullargspec.defaults)]
    else:
      positional_args = fullargspec.args
    if 'training' in positional_args:
      positional_args.remove('training')

    # self and first arg can be positional.
    if len(positional_args) > 2:
      extra_args = positional_args[2:]
      raise ValueError(
          'Models passed to `' + method_name + '` can only have `training` '
          'and the first argument in `call` as positional arguments, '
          'found: ' + str(extra_args) + '.')

  def _validate_compile(self, optimizer, metrics, **kwargs):
    """Performs validation checks for the default `compile`."""
    if any(
        isinstance(opt, optimizer_v1.Optimizer)
        for opt in nest.flatten(optimizer)):
      raise ValueError(
          '`tf.compat.v1.keras` Optimizer (', optimizer, ') is '
          'not supported when eager execution is enabled. Use a '
          '`tf.keras` Optimizer instead, or disable eager '
          'execution.')

    kwargs.pop('cloning', None)  # Legacy DistStrat argument, never used.
    kwargs.pop('experimental_run_tf_function', None)  # Always `True`.
    if kwargs.pop('distribute', None) is not None:
      raise ValueError(
          'Distribute argument in compile is not available in TF 2.0 please '
          'create the model under the distribution strategy scope.')
    if kwargs.pop('target_tensors', None) is not None:
      raise ValueError(
          'target_tensors argument is not supported when executing eagerly.')
    invalid_kwargs = set(kwargs) - {'sample_weight_mode'}
    if invalid_kwargs:
      raise TypeError('Invalid keyword argument(s) in `compile`: %s' %
                      (invalid_kwargs,))

    # Model must be created and compiled with the same DistStrat.
    if self.built and distribute_lib.has_strategy():
      strategy = distribute_lib.get_strategy()
      for v in self.variables:
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
              'Variable (%s) was not created in the distribution strategy '
              'scope of (%s). It is most likely due to not all layers or '
              'the model or optimizer being created outside the distribution '
              'strategy scope. Try to make sure your code looks similar '
              'to the following.\n'
              'with strategy.scope():\n'
              '  model=_create_model()\n'
              '  model.compile(...)' % (v, strategy))

    # Model metrics must be created in the same distribution strategy scope
    # as the model.
    strategy = self.distribute_strategy
    for metric in nest.flatten(metrics):
      for v in getattr(metric, 'variables', []):
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
              'Metric (%s) passed to model.compile was created inside of a '
              'different distribution strategy scope than the model. All '
              'metrics must be created in the same distribution strategy '
              'scope as the model (in this case %s). If you pass in a string '
              'identifier for a metric to compile the metric will '
              'automatically be created in the correct distribution '
              'strategy scope.' % (metric, strategy)
          )

    # Model metrics must be created in the same distribution strategy scope
    # as the model.
    for opt in nest.flatten(optimizer):
      for v in getattr(opt, '_weights', []):
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
              'Optimizer (%s) passed to model.compile was created inside of a '
              'different distribution strategy scope than the model. All '
              'optimizers must be created in the same distribution strategy '
              'scope as the model (in this case %s). If you pass in a string '
              'identifier for an optimizer to compile the optimizer will '
              'automatically be created in the correct distribution '
              'strategy scope.' % (opt, strategy))

  def _maybe_load_initial_epoch_from_ckpt(self, initial_epoch):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    Refer to tensorflow/python/keras/distribute/worker_training_state.py
    for more information.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    if self._training_state is not None:
      return self._training_state.maybe_load_initial_epoch_from_ckpt(
          initial_epoch, mode=ModeKeys.TRAIN)
    return initial_epoch

  def _assert_compile_was_called(self):
    # Checks whether `compile` has been called. If it has been called,
    # then the optimizer is set. This is different from whether the
    # model is compiled
    # (i.e. whether the model is built and its inputs/outputs are set).
    if not self._is_compiled:
      raise RuntimeError('You must compile your model before '
                         'training/testing. '
                         'Use `model.compile(optimizer, loss)`.')

  def _set_inputs(self, inputs, outputs=None, training=None):
    """This method is for compat with Modelv1. Only inputs are needed here."""
    self._set_save_spec(inputs)

  @property
  def _trackable_saved_model_saver(self):
    return model_serialization.ModelSavedModelSaver(self)

  def _trackable_children(self, save_type='checkpoint', **kwargs):
    if save_type == 'savedmodel':
      # SavedModel needs to ignore the execution functions.
      train_function = self.train_function
      test_function = self.test_function
      predict_function = self.predict_function
      train_tf_function = self.train_tf_function
      self.train_function = None
      self.test_function = None
      self.predict_function = None
      self.train_tf_function = None

    children = super(Model, self)._trackable_children(save_type, **kwargs)

    if save_type == 'savedmodel':
      self.train_function = train_function
      self.test_function = test_function
      self.predict_function = predict_function
      self.train_tf_function = train_tf_function

    return children

  def _should_eval(self, epoch, validation_freq):
    epoch = epoch + 1  # one-index the user-facing epoch.
    if isinstance(validation_freq, int):
      return epoch % validation_freq == 0
    elif isinstance(validation_freq, list):
      return epoch in validation_freq
    else:
      raise ValueError('Expected `validation_freq` to be a list or int.')

  ######################################################################
  # Functions below exist only as v1 / v2 compatibility shims.
  ######################################################################

  def _get_compile_args(self, user_metrics=True):
    """Used for saving or cloning a Model.

    Args:
      user_metrics: Whether to return user-supplied metrics or `Metric` objects.
        Defaults to returning the user-supplied metrics.

    Returns:
      Dictionary of arguments that were used when compiling the model.
    """
    self._assert_compile_was_called()
    # pylint: disable=protected-access

    saved_metrics = self.compiled_metrics._user_metrics
    saved_weighted_metrics = self.compiled_metrics._user_weighted_metrics

    if not user_metrics:
      if saved_metrics is not None:
        saved_metrics = self.compiled_metrics._metrics
      if saved_weighted_metrics is not None:
        saved_weighted_metrics = self.compiled_metrics._weighted_metrics

    compile_args = {
        'optimizer': self.optimizer,
        'loss': self.compiled_loss._user_losses,
        'metrics': saved_metrics,
        'weighted_metrics': saved_weighted_metrics,
        'loss_weights': self.compiled_loss._user_loss_weights,
    }
    # pylint: enable=protected-access
    return compile_args

  def _get_callback_model(self):
    return self

  def _in_multi_worker_mode(self):
    return self.distribute_strategy.extended._in_multi_worker_mode()  # pylint: disable=protected-access

  @property
  def _compile_was_called(self):
    return self._is_compiled


def reduce_per_replica(values, strategy, reduction='first'):
  """Reduce PerReplica objects.

  Args:
    values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
      returned as-is.
    strategy: `tf.distribute.Strategy` object.
    reduction: One of 'first', 'concat'.

  Returns:
    Structure of `Tensor`s.
  """

  def _reduce(v):
    """Reduce a single `PerReplica` object."""
    if reduction == 'concat' and _collective_all_reduce_multi_worker(strategy):
      return _multi_worker_concat(v, strategy)
    if not _is_per_replica_instance(v):
      return v
    elif reduction == 'first':
      return strategy.unwrap(v)[0]
    elif reduction == 'concat':
      if _is_tpu_multi_host(strategy):
        return _tpu_multi_host_concat(v, strategy)
      else:
        return concat(strategy.unwrap(v))
    else:
      raise ValueError('`reduction` must be "first" or "concat".')

  return nest.map_structure(_reduce, values)


def concat(tensors, axis=0):
  """Concats `tensor`s along `axis`."""
  if isinstance(tensors[0], sparse_tensor.SparseTensor):
    return sparse_ops.sparse_concat_v2(axis=axis, sp_inputs=tensors)
  elif _is_scalar(tensors[0]):
    return array_ops_stack.stack(tensors, axis=axis)
  else:
    return array_ops.concat(tensors, axis=axis)


def _is_tpu_multi_host(strategy):
  return (backend.is_tpu_strategy(strategy) and
          strategy.extended.num_hosts > 1)


def _tpu_multi_host_concat(v, strategy):
  """Correctly order TPU PerReplica objects."""
  replicas = strategy.unwrap(v)
  # When distributed datasets are created from Tensors / NumPy,
  # TPUStrategy.experimental_distribute_dataset shards data in
  # (Replica, Host) order, and TPUStrategy.unwrap returns it in
  # (Host, Replica) order.
  # TODO(b/150317897): Figure out long-term plan here.
  num_replicas_per_host = strategy.extended.num_replicas_per_host
  ordered_replicas = []
  for replica_id in range(num_replicas_per_host):
    ordered_replicas += replicas[replica_id::num_replicas_per_host]
  return concat(ordered_replicas)


def _collective_all_reduce_multi_worker(strategy):
  return (isinstance(strategy,
                     collective_all_reduce_strategy.CollectiveAllReduceStrategy)
         ) and strategy.extended._in_multi_worker_mode()  # pylint: disable=protected-access


# TODO(wxinyi): merge this with _tpu_multi_host_concat once we have all_gather
# for all strategies
def _multi_worker_concat(v, strategy):
  """Order PerReplica objects for CollectiveAllReduceStrategy and concat."""
  replicas = strategy.gather(v, axis=0)
  # v might not have the same shape on different replicas
  if _is_per_replica_instance(v):
    shapes = array_ops.concat([
        array_ops.expand_dims_v2(array_ops.shape(single_value)[0], axis=0)
        for single_value in v.values
    ],
                              axis=0)
    all_shapes = strategy.gather(shapes, axis=0)
  else:
    # v is a tensor. This may happen when, say, we have 2x1 multi-worker.
    all_shapes = strategy.gather(
        array_ops.expand_dims_v2(array_ops.shape(v)[0], axis=0), axis=0)

  replicas = array_ops.split(
      replicas,
      num_or_size_splits=all_shapes,
      num=strategy.num_replicas_in_sync)
  ordered_replicas = []
  num_replicas_per_worker = len(strategy.extended.worker_devices)
  for replica_id in range(num_replicas_per_worker):
    ordered_replicas += replicas[replica_id::num_replicas_per_worker]
  return concat(ordered_replicas)


def _is_scalar(x):
  return isinstance(
      x, (tensor_lib.Tensor, variables.Variable)) and x.shape.rank == 0


def write_scalar_summaries(logs, step):
  for name, value in logs.items():
    if _is_scalar(value):
      summary_ops_v2.scalar('batch_' + name, value, step=step)


def _minimum_control_deps(outputs):
  """Returns the minimum control dependencies to ensure step succeeded."""
  if context.executing_eagerly():
    return []  # Control dependencies not needed.
  outputs = nest.flatten(outputs, expand_composites=True)
  for out in outputs:
    # Variables can't be control dependencies.
    if not isinstance(out, variables.Variable):
      return [out]  # Return first Tensor or Op from outputs.
  return []  # No viable Tensor or Op to use for control deps.


def _disallow_inside_tf_function(method_name):
  if ops.inside_function():
    error_msg = (
        'Detected a call to `Model.{method_name}` inside a `tf.function`. '
        '`Model.{method_name} is a high-level endpoint that manages its own '
        '`tf.function`. Please move the call to `Model.{method_name}` outside '
        'of all enclosing `tf.function`s. Note that you can call a `Model` '
        'directly on `Tensor`s inside a `tf.function` like: `model(x)`.'
    ).format(method_name=method_name)
    raise RuntimeError(error_msg)


def _detect_save_format(filepath):
  """Returns path to weights file and save format."""

  filepath = path_to_string(filepath)
  if saving_utils.is_hdf5_filepath(filepath):
    return filepath, 'h5'

  # Filepath could be a TensorFlow checkpoint file prefix or SavedModel
  # directory. It's possible for filepath to be both a prefix and directory.
  # Prioritize checkpoint over SavedModel.
  if _is_readable_tf_checkpoint(filepath):
    save_format = 'tf'
  elif sm_loader.contains_saved_model(filepath):
    ckpt_path = os.path.join(filepath, sm_constants.VARIABLES_DIRECTORY,
                             sm_constants.VARIABLES_FILENAME)
    if _is_readable_tf_checkpoint(ckpt_path):
      filepath = ckpt_path
      save_format = 'tf'
    else:
      raise ValueError('Unable to load weights. filepath {} appears to be a '
                       'SavedModel directory, but checkpoint either doesn\'t '
                       'exist, or is incorrectly formatted.'.format(filepath))
  else:
    # Not a TensorFlow checkpoint. This filepath is likely an H5 file that
    # doesn't have the hdf5/keras extensions.
    save_format = 'h5'
  return filepath, save_format


def _is_readable_tf_checkpoint(filepath):
  try:
    py_checkpoint_reader.NewCheckpointReader(filepath)
    return True
  except errors_impl.DataLossError:
    # The checkpoint is not readable in TensorFlow format.
    return False


def flatten_metrics_in_order(logs, metrics_names):
  """Turns the `logs` dict into a list as per key order of `metrics_names`."""
  results = []
  for name in metrics_names:
    if name in logs:
      results.append(logs[name])
  for key in sorted(logs.keys()):
    if key not in metrics_names:
      results.append(logs[key])
  if len(results) == 1:
    return results[0]
  return results


def _is_per_replica_instance(obj):
  return (isinstance(obj, ds_values.DistributedValues) and
          isinstance(obj, composite_tensor.CompositeTensor))
