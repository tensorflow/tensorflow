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
"""V1 Training-related part of the Keras engine."""

import collections
import warnings

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest

try:
  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
  issparse = None


class Model(training_lib.Model):
  """`Model` groups layers into an object with training and inference features.

  There are two ways to instantiate a `Model`:

  1 - With the "functional API", where you start from `Input`,
  you chain layer calls to specify the model's forward pass,
  and finally you create your model from inputs and outputs:

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

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
  """

  def __init__(self, *args, **kwargs):
    super(Model, self).__init__(*args, **kwargs)
    # initializing _distribution_strategy here since it is possible to call
    # predict on a model without compiling it.
    self._distribution_strategy = None
    self._compile_time_distribution_strategy = None
    if (ops.executing_eagerly_outside_functions() and
        distribution_strategy_context.has_strategy()):
      self._set_strategy(
          distribution_strategy_context.get_strategy())

    # This flag is used to track if the user is using the deprecated path of
    # passing distribution strategy to compile rather than creating the model
    # under distribution strategy scope.
    self._compile_distribution = False

    self._run_eagerly = None
    self._experimental_run_tf_function = (
        ops.executing_eagerly_outside_functions())

    self._v1_compile_was_called = False

  def _init_batch_counters(self):
    pass  # Batch counters should not be created in legacy graph mode.

  @trackable.no_automatic_dependency_tracking
  def _set_strategy(self, strategy):
    self._compile_time_distribution_strategy = strategy

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    strategy = (self._distribution_strategy or
                self._compile_time_distribution_strategy)
    if strategy:
      with strategy.scope():
        return base_layer.Layer.get_weights(self)
    return base_layer.Layer.get_weights(self)

  def load_weights(self, filepath, by_name=False, skip_mismatch=False):
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
            to `save_weights`).
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.
        skip_mismatch: Boolean, whether to skip loading of layers where there is
            a mismatch in the number of weights, or a mismatch in the shape of
            the weight (only valid when `by_name=True`).

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
          (not saving_utils.is_hdf5_filepath(filepath))):  # pylint: disable=protected-access
        raise ValueError('Load weights is not yet supported with TPUStrategy '
                         'with steps_per_run greater than 1.')
    return super(Model, self).load_weights(filepath, by_name, skip_mismatch)

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
    """Configures the model for training.

    Args:
        optimizer: String (name of optimizer) or optimizer instance.
            See `tf.keras.optimizers`.
        loss: String (name of objective function), objective function or
            `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective
            function is any callable with the signature
            `scalar_loss = fn(y_true, y_pred)`. If the model has multiple
            outputs, you can use a different loss on each output by passing a
            dictionary or a list of losses. The loss value that will be
            minimized by the model will then be the sum of all individual
            losses.
        metrics: List of metrics to be evaluated by the model during training
            and testing. Typically you will use `metrics=['accuracy']`.
            To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary, such as
            `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
            You can also pass a list (len = len(outputs)) of lists of metrics
            such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
            `metrics=['accuracy', ['accuracy', 'mse']]`.
        loss_weights: Optional list or dictionary specifying scalar
            coefficients (Python floats) to weight the loss contributions
            of different model outputs.
            The loss value that will be minimized by the model
            will then be the *weighted sum* of all individual losses,
            weighted by the `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping
            to the model's outputs. If a tensor, it is expected to map
            output names (strings) to scalar coefficients.
        sample_weight_mode: If you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted
            by sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create placeholders for the
            model's target, which will be fed with the target data during
            training. If instead you would like to use your own
            target tensors (in turn, Keras will not expect external
            Numpy data for these targets at training time), you
            can specify them via the `target_tensors` argument. It can be
            a single tensor (for a single-output model), a list of tensors,
            or a dict mapping output names to target tensors.
        distribute: NOT SUPPORTED IN TF 2.0, please create and compile the
            model under distribution strategy scope instead of passing it to
            compile.
        **kwargs: Any additional arguments.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    self._assert_built_as_v1()
    self._run_eagerly = kwargs.pop('run_eagerly', None)
    self._experimental_run_tf_function = kwargs.pop(
        'experimental_run_tf_function', True)
    self._v1_compile_was_called = True

    # Prepare Session arguments (legacy).
    kwargs.pop('cloning', None)  # Legacy DistStrat argument, never used.
    self._from_serialized = kwargs.pop('from_serialized', False)
    allowed_kwargs = {'feed_dict', 'fetches', 'options', 'run_metadata'}
    unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
    if unknown_kwargs:
      raise TypeError(
          'Invalid keyword argument(s) in `compile`: %s' % (unknown_kwargs,))
    self._function_kwargs = kwargs
    if self._function_kwargs:
      self._experimental_run_tf_function = False
      if self.run_eagerly:
        raise ValueError(
            'Session keyword arguments are not supported '
            'when `run_eagerly=True`. You passed the following '
            'Session arguments: %s' % (self._function_kwargs,))

    self._set_optimizer(optimizer)
    is_any_keras_optimizer_v1 = any(
        (isinstance(opt, optimizer_v1.Optimizer)
         and not isinstance(opt, optimizer_v1.TFOptimizer)
        ) for opt in nest.flatten(self.optimizer))

    if is_any_keras_optimizer_v1 and ops.executing_eagerly_outside_functions():
      raise ValueError('`tf.compat.v1.keras` Optimizer (', optimizer, ') is '
                       'not supported when eager execution is enabled. Use a '
                       '`tf.keras` Optimizer instead, or disable eager '
                       'execution.')

    if ((target_tensors is not None)
        or not ops.executing_eagerly_outside_functions()):
      # Fallback out of things that aren't supported with v2 loops
      self._experimental_run_tf_function = False

    if distribute is not None:
      if tf2.enabled() or self._experimental_run_tf_function:
        raise ValueError(
            'Distribute argument in compile is not available in TF 2.0 please '
            'create the model under the distribution strategy scope.')
      logging.warning('Distribute argument in compile is deprecated please '
                      'create the model under the distribution strategy scope.')
      self._distribution_strategy = distribute
      self._compile_distribution = True
    else:
      if distribution_strategy_context.has_strategy():
        # When the user builds the model in the DS scope and cross replica
        # context we want distribution strategy to be set but when building the
        # replica copies of the models internally we should not be compiling
        # with distribution strategy and use the default compilation path.
        if distribution_strategy_context.in_cross_replica_context():
          self._distribution_strategy = (
              distribution_strategy_context.get_strategy())

    if isinstance(self._distribution_strategy,
                  parameter_server_strategy.ParameterServerStrategyV1):
      raise NotImplementedError(
          '`tf.compat.v1.distribute.experimental.ParameterServerStrategy` '
          'currently only works with the tf.Estimator API')

    if isinstance(self._distribution_strategy,
                  parameter_server_strategy_v2.ParameterServerStrategyV2):
      raise NotImplementedError(
          '`tf.distribute.experimental.ParameterServerStrategy` is only '
          'supported in TF2.')

    if not self._experimental_run_tf_function:
      self._validate_compile_param_for_distribution_strategy(self.run_eagerly,
                                                             sample_weight_mode,
                                                             target_tensors,
                                                             weighted_metrics)
    # We've disabled automatic dependency tracking for this method, but do want
    # to add a checkpoint dependency on the optimizer if it's trackable.
    if isinstance(self.optimizer, trackable.Trackable):
      self._track_trackable(
          self.optimizer, name='optimizer', overwrite=True)
    self.loss = loss or {}
    self.loss_weights = loss_weights
    self.sample_weight_mode = sample_weight_mode
    self._compile_metrics = metrics or []
    self._compile_weighted_metrics = weighted_metrics
    if self.run_eagerly and target_tensors is not None:
      raise ValueError(
          'target_tensors argument is not supported when '
          'running a model eagerly.')

    # _training_endpoints contains a list of _TrainingEndpoint object, which has
    # all the model output/target/loss and related metadata.
    self._training_endpoints = []

    # Used to freeze the behavior of the Model once `compile` has been called.
    self._compiled_trainable_state = self._get_trainable_state()

    # Set tf.distribute.Strategy specific parameters.
    self._distributed_model_cache = {}
    self._distributed_function_cache = {}

    # Clear any `_eager_losses` that was added.
    self._clear_losses()

    if (not context.executing_eagerly() and
        self._distribution_strategy is not None):
      # Ensures a Session is created and configured correctly for Distribution
      # Strategy.
      backend.configure_and_create_distributed_session(
          self._distribution_strategy)
    # Initialize model metric attributes.
    self._init_metric_attributes()
    if not self.built or not self.inputs or not self.outputs:
      # Model is not compilable because it does not know its number of inputs
      # and outputs, nor their shapes and names. We will compile after the first
      # time the model gets called on training data.
      return
    self._is_compiled = True
    base_layer.keras_api_gauge.get_cell('compile').set(True)

    # Prepare list of loss functions, same size of model outputs.
    self.loss_functions = training_utils_v1.prepare_loss_functions(
        self.loss, self.output_names)

    target_tensors = self._process_target_tensor_for_compile(target_tensors)

    for o, n, l, t in zip(self.outputs, self.output_names,
                          self.loss_functions, target_tensors):
      endpoint = _TrainingEndpoint(o, n, l)
      endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
      self._training_endpoints.append(endpoint)

    # Prepare list loss weights, same size of model outputs.
    training_utils_v1.prepare_loss_weights(self._training_endpoints,
                                           loss_weights)

    # Initialization for Eager mode execution.
    if self.run_eagerly:
      self._compile_eagerly(metrics, weighted_metrics, sample_weight_mode)
      return

    with backend.get_graph().as_default():
      # Save all metric attributes per output of the model.
      self._cache_output_metric_attributes(metrics, weighted_metrics)

      # Set metric attributes on model.
      self._set_metric_attributes()

      # Invoke metric functions (unweighted) for all the outputs.
      self._handle_metrics(
          self.outputs,
          targets=self._targets,
          skip_target_masks=self._prepare_skip_target_masks(),
          masks=self._prepare_output_masks())

      # Prepare sample weight modes. List with the same length as model outputs.
      training_utils_v1.prepare_sample_weight_modes(
          self._training_endpoints, sample_weight_mode)

      # Creates the model loss and weighted metrics sub-graphs.
      self._compile_weights_loss_and_weighted_metrics()

      # Functions for train, test and predict will
      # be compiled lazily when required.
      # This saves time when the user is not using all functions.
      self.train_function = None
      self.test_function = None
      self.predict_function = None

      # Collected trainable weights, sorted in topological order.
      self._collected_trainable_weights = self.trainable_weights

      # Validate all variables were correctly created in distribution scope.
      if self._distribution_strategy and not self._compile_distribution:
        for v in self.variables:
          strategy = self._distribution_strategy
          if not strategy.extended.variable_created_in_scope(v):
            raise ValueError(
                'Variable (%s) was not created in the distribution strategy '
                'scope of (%s). It is most likely due to not all layers or '
                'the model or optimizer being created outside the distribution '
                'strategy scope. Try to make sure your code looks similar '
                'to the following.\n'
                'with strategy.scope():\n'
                '  model=_create_model()\n'
                '  model.compile(...)'% (v, strategy))

  @trackable.no_automatic_dependency_tracking
  def _init_distributed_function_cache_if_not_compiled(self):
    if not hasattr(self, '_distributed_function_cache'):
      self._distributed_function_cache = {}

  @property
  def metrics(self):
    """Returns the model's metrics added using `compile`, `add_metric` APIs."""
    metrics = []
    if self._is_compiled:
      if not hasattr(self, '_v1_compile_was_called'):
        # See b/155687393 for more details, the model is created as a v2
        # instance but converted to v1. Fallback to use base Model to retrieve
        # the metrics.
        return super(Model, self).metrics
      metrics += self._compile_metric_functions
    metrics.extend(self._metrics)
    metrics.extend(
        _get_metrics_from_layers(
            list(self._flatten_layers(include_self=False, recursive=False))))
    return metrics

  @property
  def metrics_names(self):
    """Returns the model's display labels for all outputs."""

    # This property includes all output names including `loss` and per-output
    # losses for backward compatibility.
    metrics_names = ['loss']
    if self._is_compiled:
      if not hasattr(self, '_v1_compile_was_called'):
        # See b/155687393 for more details, the model is created as a v2
        # instance but converted to v1. Fallback to use base Model to retrieve
        # the metrics name
        return super(Model, self).metrics_names

      # Add output loss metric names to the metric names list.
      if len(self._training_endpoints) > 1:
        metrics_names.extend([
            e.loss_name()
            for e in self._training_endpoints
            if not e.should_skip_target()
        ])

    # Add all metric names.
    metrics_names += [m.name for m in self.metrics]
    return metrics_names

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
    if self._run_eagerly is True and not context.executing_eagerly():
      raise ValueError('You can only set `run_eagerly=True` if eager execution '
                       'is enabled.')
    if not self.dynamic:
      if self._run_eagerly is None:
        # Respect `tf.config.run_functions_eagerly` unless
        # `run_eagerly` was explicitly passed to `compile`.
        return def_function.functions_run_eagerly()
      else:
        return self._run_eagerly
    else:
      if not context.executing_eagerly():
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You must enable eager execution with '
                         '`tf.enable_eager_execution()`.')
      if self._run_eagerly is False:
        # TODO(fchollet): consider using py_func to enable this.
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You cannot set `run_eagerly=False`.')
      return context.executing_eagerly()

  @run_eagerly.setter
  def run_eagerly(self, value):
    self._run_eagerly = value

  def _select_training_loop(self, inputs):
    """Select training loop for fit/eval/predict based on the inputs."""
    # TODO(kaftan) or TODO(scottzhu): This check should eventually be nicely
    #  integrated into the data adapters in the v2 loop. We can't do this yet
    #  because we currently have to fall back for unhandled data types.
    if isinstance(inputs, (iterator_ops.Iterator,
                           iterator_ops.IteratorBase)):
      raise ValueError('For performance reasons Keras `fit`, `evaluate` and'
                       '`predict` accept tf.data `Datasets` as input but not '
                       'iterators that have been manually generated from '
                       'Datasets by users. Please directly pass in the '
                       'original `Dataset` object instead of passing in '
                       '`iter(dataset)`.')

    # Case 1: distribution strategy.
    if self._distribution_strategy:
      if self._in_multi_worker_mode():
        return training_distributed_v1.DistributionMultiWorkerTrainingLoop(
            training_distributed_v1.DistributionSingleWorkerTrainingLoop())
      else:
        return training_distributed_v1.DistributionSingleWorkerTrainingLoop()

    # Case 2: generator-like. Input is Python generator, or Sequence object,
    # or a non-distributed Dataset or iterator in eager execution.
    if data_utils.is_generator_or_sequence(inputs):
      return training_generator_v1.GeneratorOrSequenceTrainingLoop()
    if training_utils_v1.is_eager_dataset_or_iterator(inputs):
      return training_generator_v1.EagerDatasetOrIteratorTrainingLoop()

    # Case 3: Symbolic tensors or Numpy array-like.
    # This includes Datasets and iterators in graph mode (since they
    # generate symbolic tensors).
    if self.run_eagerly:
      return training_generator_v1.GeneratorLikeTrainingLoop()
    else:
      return training_arrays_v1.ArrayLikeTrainingLoop()

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
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
            or `(inputs, targets, sample weights)`.
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
            form of symbolic tensors, datasets,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`.
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
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset
            For the first two cases, `batch_size` must be provided.
            For the last case, `validation_steps` could be provided.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch').
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
            Has no effect when `steps_per_epoch` is not `None`.
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
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, generator, or
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
            This argument is not supported with array inputs.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of a infinite dataset, it will run into a infinite loop.
            If 'validation_steps' is specified and only part of the dataset
            will be consumed, the evaluation will start from the beginning of
            the dataset at each epoch. This ensures that the same validation
            samples are used every time.
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
            will default to 1. If 0, will execute the generator on the main
            thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        **kwargs: Used for backwards compatibility.

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    self._assert_built_as_v1()
    base_layer.keras_api_gauge.get_cell('fit').set(True)
    # Legacy support
    if 'nb_epoch' in kwargs:
      logging.warning(
          'The `nb_epoch` argument in `fit` has been renamed `epochs`.')
      epochs = kwargs.pop('nb_epoch')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    self._assert_compile_was_called()
    self._check_call_args('fit')

    func = self._select_training_loop(x)
    return func.fit(
        self,
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

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
               use_multiprocessing=False):
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
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely).
          If `x` is a dataset, generator or
          `keras.utils.Sequence` instance, `y` should not be specified (since
          targets will be obtained from the iterator/dataset).
        batch_size: Integer or `None`.
            Number of samples per batch of computation.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        verbose: 0 or 1. Verbosity mode.
            0 = silent, 1 = progress bar.
        sample_weight: Optional Numpy array of weights for
            the test samples, used for weighting the loss function.
            You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, instead pass
            sample weights as the third element of `x`.
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
            If x is a `tf.data` dataset and `steps` is
            None, 'evaluate' will run until the dataset is exhausted.
            This argument is not supported with array inputs.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during evaluation.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    self._assert_built_as_v1()
    base_layer.keras_api_gauge.get_cell('evaluate').set(True)
    self._assert_compile_was_called()
    self._check_call_args('evaluate')

    func = self._select_training_loop(x)
    return func.evaluate(
        self,
        x=x,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight,
        steps=steps,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

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

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
        batch_size: Integer or `None`.
            Number of samples per batch of computation.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
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
            to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.


    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    self._assert_built_as_v1()
    base_layer.keras_api_gauge.get_cell('predict').set(True)
    self._check_call_args('predict')

    func = self._select_training_loop(x)
    return func.predict(
        self,
        x=x,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def reset_metrics(self):
    """Resets the state of metrics."""
    metrics = self._get_training_eval_metrics()
    for m in metrics:
      m.reset_state()

    # Reset metrics on all the distributed (cloned) models.
    if self._distribution_strategy:
      distributed_training_utils_v1._reset_metrics(self)  # pylint: disable=protected-access

  def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True):
    """Runs a single gradient update on a single batch of data.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
              (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
          - A `tf.data` dataset.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely). If
          `x` is a dataset, `y` should not be specified
          (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case of
          temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample. In this case you should make sure to specify
          sample_weight_mode="temporal" in compile(). This argument is not
          supported when `x` is a dataset.
        class_weight: Optional dictionary mapping class indices (integers) to a
          weight (float) to apply to the model's loss for the samples from this
          class during training. This can be useful to tell the model to "pay
          more attention" to samples from an under-represented class.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
      ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('train_on_batch')

    # If at this point we are in the replica context, then it is okay to execute
    # the Eager code path.  The expected way to get here is to call `fit` that
    # calls `train_on_batch` on each replica.
    if (self._distribution_strategy and
        distribution_strategy_context.in_cross_replica_context()):
      raise NotImplementedError('`train_on_batch` is not supported for models '
                                'distributed with tf.distribute.Strategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, class_weight=class_weight,
        extract_tensors_from_dataset=True)

    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point because of the check above.  `train_on_batch` is being run
    # for each replica by `self._distribution_strategy` and the same code path
    # as Eager is expected to be taken.
    if self.run_eagerly or self._distribution_strategy:
      output_dict = training_eager_v1.train_on_batch(
          self,
          x,
          y,
          sample_weights=sample_weights,
          output_loss_metrics=self._output_loss_metrics)
      outputs = (output_dict['total_loss'] + output_dict['output_losses']
                 + output_dict['metrics'])
      outputs = [_non_none_constant_value(v) for v in outputs]  # pylint: disable=protected-access
    else:
      x = training_utils_v1.ModelInputs(x).as_list()
      ins = x + list(y or []) + list(sample_weights or [])

      if not isinstance(backend.symbolic_learning_phase(), int):
        ins += [True]  # Add learning phase value.

      self._update_sample_weight_modes(sample_weights=sample_weights)
      self._make_train_function()
      outputs = self.train_function(ins)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
    """Test the model on a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset `y` should
          not be specified (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile(). This argument is not
            supported when `x` is a dataset.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('test_on_batch')

    if (self._distribution_strategy and
        distribution_strategy_context.in_cross_replica_context()):
      raise NotImplementedError('`test_on_batch` is not supported for models '
                                'distributed with tf.distribute.Strategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, extract_tensors_from_dataset=True)

    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point.
    if self.run_eagerly or self._distribution_strategy:
      output_dict = training_eager_v1.test_on_batch(
          self,
          x,
          y,
          sample_weights=sample_weights,
          output_loss_metrics=self._output_loss_metrics)
      outputs = (output_dict['total_loss'] + output_dict['output_losses']
                 + output_dict['metrics'])
      outputs = [_non_none_constant_value(v) for v in outputs]  # pylint: disable=protected-access
    else:
      x = training_utils_v1.ModelInputs(x).as_list()
      inputs = x + list(y or []) + list(sample_weights or [])

      self._update_sample_weight_modes(sample_weights=sample_weights)
      self._make_test_function()
      outputs = self.test_function(inputs)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between given number of inputs and
          expectations of the model.
    """
    self._check_call_args('predict_on_batch')

    if (self._distribution_strategy and
        distribution_strategy_context.in_cross_replica_context()):
      raise NotImplementedError(
          '`predict_on_batch` is not supported for models distributed with'
          ' tf.distribute.Strategy.')
    # Validate and standardize user data.
    inputs, _, _ = self._standardize_user_data(
        x, extract_tensors_from_dataset=True)
    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point.
    if self.run_eagerly or self._distribution_strategy:
      inputs = training_utils_v1.cast_if_floating_dtype(inputs)
      if isinstance(inputs, collections.abc.Sequence):
        # Unwrap lists with only one input, as we do when training on batch
        if len(inputs) == 1:
          inputs = inputs[0]

      return self(inputs)  # pylint: disable=not-callable

    self._make_predict_function()
    outputs = self.predict_function(inputs)

    if len(outputs) == 1:
      return outputs[0]
    return outputs

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
    warnings.warn('`model.fit_generator` is deprecated and '
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

  def _set_optimizer(self, optimizer):
    """Sets self.optimizer.

    Sets self.optimizer to `optimizer`, potentially wrapping it with a
    LossScaleOptimizer.

    Args:
      optimizer: The optimizer(s) to assign to self.optimizer.
    """
    if isinstance(optimizer, (list, tuple)):
      self.optimizer = [optimizers.get(opt) for opt in optimizer]
    else:
      self.optimizer = optimizers.get(optimizer)

    if isinstance(self._dtype_policy, policy.PolicyV1):
      loss_scale = self._dtype_policy.loss_scale
    elif self._dtype_policy.name == 'mixed_float16':
      loss_scale = 'dynamic'
    else:
      loss_scale = None

    if (loss_scale is not None and
        not isinstance(self.optimizer,
                       loss_scale_optimizer.LossScaleOptimizer)):
      if isinstance(self.optimizer, list):
        raise ValueError('When a dtype policy with a loss scale is used, you '
                         'can only pass a single optimizer. Using policy %s '
                         'and got optimizers: %s' %
                         self._dtype_policy, self.optimizer)
      if not isinstance(self.optimizer, optimizer_v2.OptimizerV2):
        raise ValueError('"optimizer" must be an instance of '
                         'tf.keras.optimizers.Optimizer when a dype policy '
                         'with a loss scale  used, but got: %s. Using policy: '
                         '%s' %
                         (self.optimizer, self._dtype_policy))
      if loss_scale == 'dynamic':
        self.optimizer = loss_scale_optimizer.LossScaleOptimizer(self.optimizer)
      else:
        self.optimizer = loss_scale_optimizer.LossScaleOptimizerV1(
            self.optimizer, loss_scale)

  def _prepare_validation_data(self, validation_data, batch_size,
                               validation_steps):
    """Unpack and check the validation data."""
    val_x, val_y, val_sample_weights = training_utils_v1.unpack_validation_data(
        validation_data)
    return self._standardize_user_data(
        val_x,
        val_y,
        sample_weight=val_sample_weights,
        batch_size=batch_size,
        steps=validation_steps,
        steps_name='validation_steps')

  def _validate_compile_param_for_distribution_strategy(
      self, run_eagerly, sample_weight_mode, target_tensors, weighted_metrics):
    # Validate that arguments passed by the user to `compile` are supported by
    # tf.distribute.Strategy.
    if self._distribution_strategy:
      if sample_weight_mode:
        raise NotImplementedError('sample_weight_mode is not supported with '
                                  'tf.distribute.Strategy.')
      if weighted_metrics:
        raise NotImplementedError('weighted_metrics is not supported with '
                                  'tf.distribute.Strategy.')
      if target_tensors:
        raise ValueError('target_tensors is not supported with '
                         'tf.distribute.Strategy.')

      if run_eagerly:
        raise ValueError(
            'We currently do not support enabling `run_eagerly` with '
            'distribution strategy.')

      if (distributed_training_utils_v1.is_distributing_by_cloning(self) and
          (not self.built or not self.inputs or not self.outputs)):
        raise ValueError(
            'We currently do not support distribution strategy with a '
            '`Sequential` model that is created without `input_shape`/'
            '`input_dim` set in its first layer or a subclassed model.')

  def _process_target_tensor_for_compile(self, target_tensors):
    if self.run_eagerly:
      # target tensor is not supported with run_eagerly. Create a list with None
      # as placeholder for each output.
      return [None for _ in self.output_names]

    if target_tensors is not None and not (isinstance(target_tensors, list) and
                                           target_tensors == []):  # pylint: disable=g-explicit-bool-comparison
      if isinstance(target_tensors, list):
        if len(target_tensors) != len(self.outputs):
          raise ValueError(
              'When passing a list as `target_tensors`, '
              'it should have one entry per model output. '
              'The model has %s outputs, but you passed target_tensors=%s' %
              (len(self.outputs), target_tensors))
      elif isinstance(target_tensors, dict):
        unexpected_target_tensor_names = set(target_tensors.keys()).difference(
            self.output_names)
        if unexpected_target_tensor_names:
          raise ValueError(
              'Unknown entry in `target_tensors` dictionary: "{name}". '
              'Only expected the following keys: {keys}'.format(
                  name=unexpected_target_tensor_names,
                  keys=str(self.output_names)))
        tmp_target_tensors = []
        for name in self.output_names:
          tmp_target_tensors.append(target_tensors.get(name, None))
        target_tensors = tmp_target_tensors
      elif tensor_util.is_tf_type(target_tensors):
        target_tensors = [target_tensors]
      else:
        raise TypeError('Expected `target_tensors` to be a list or tuple or '
                        'dict or a single tensor, but got:', target_tensors)
    else:
      # In case target tensor is empty or None, create a list with Nones
      # that has same length as self.output_names. With that, the None check of
      # target tensor can be skipped downstream.
      target_tensors = [None for _ in self.output_names]
    return target_tensors

  def _compile_eagerly(self, metrics, weighted_metrics, sample_weight_mode):
    # Prepare sample weight modes. List with the same length as model outputs.
    training_utils_v1.prepare_sample_weight_modes(
        self._training_endpoints, sample_weight_mode)
    # Prepare sample weights.
    self._prepare_sample_weights()
    # Save all metric attributes per output of the model.
    self._cache_output_metric_attributes(metrics, weighted_metrics)
    self.total_loss = None
    # Set metric attributes on model.
    self._set_metric_attributes()

    self._collected_trainable_weights = self.trainable_weights

  def _update_sample_weight_modes(self, sample_weights=None):
    """Updates sample weight modes based on training/eval inputs.

    Sample weight placeholders will be created for all or no outputs
    based on whether sample_weight is provided for any output.

    If model contains `_sample_weight_modes` we check if the input
    `sample_weights` corresponds to the sample weight modes.
      1. Set sample weight mode to be 'temporal' for output i, if `compile`
        sample_weight_mode was set to `temporal` and sample weight inputs
        are given for one or more outputs.
      2. Set sample weight mode to be 'samplewise' for output i, if `compile`
        sample_weight_mode was not set and sample weight inputs are given for
        one or more outputs.
      3. Reset sample weight mode to None for output i if sample weight mode
        was set but there is no sample weight input.

    Args:
      sample_weights: List of sample weights of the same length as model outputs
        or None.
    """
    if not self._is_compiled:
      return
    if sample_weights and any(s is not None for s in sample_weights):
      for endpoint in self._training_endpoints:
        endpoint.sample_weight_mode = (
            endpoint.sample_weight_mode or 'samplewise')
    else:
      for endpoint in self._training_endpoints:
        endpoint.sample_weight_mode = None

  def _recompile_weights_loss_and_weighted_metrics(self):
    if not self._is_compiled:
      return False
    recompile = any(
        e.sample_weights_mismatch() for e in self._training_endpoints)

    if recompile:
      self._compile_weights_loss_and_weighted_metrics()
    return recompile

  @trackable.no_automatic_dependency_tracking
  def _compile_weights_loss_and_weighted_metrics(self, sample_weights=None):
    """Compiles the model loss and weighted metric sub-graphs.

    This may be used to set graph tensors as sample weights (instead of creating
    placeholders). This functionality is necessary for
    `tf.keras.estimator.model_to_estimator`, which calls Keras models in a v1
    graph, and creates iterator tensors for inputs, targets, and sample weights.

    Args:
      sample_weights: List of tensors to use as the sample weights. Must be the
        same length as the number of outputs. If left as `None`, placeholders
        are used instead.
    """
    with backend.get_graph().as_default():
      if sample_weights is not None:
        self._update_sample_weight_modes(sample_weights)
      self._prepare_sample_weights(sample_weights)

      masks = self._prepare_output_masks()

      # Compute weighted metrics.
      self._handle_metrics(
          self.outputs,
          targets=self._targets,
          skip_target_masks=self._prepare_skip_target_masks(),
          sample_weights=self.sample_weights,
          masks=masks,
          return_weighted_metrics=True)

      # Compute total loss.
      # Used to keep track of the total loss value (stateless).
      # eg., total_loss = loss_weight_1 * output_1_loss_fn(...) +
      #                   loss_weight_2 * output_2_loss_fn(...) +
      #                   layer losses.
      self.total_loss = self._prepare_total_loss(masks)

  def _prepare_skip_target_masks(self):
    """Boolean mask for whether the target in the output list should be skipped.

    If the loss function corresponding to a model output is None, then this
    output will be skipped during total loss calculation and feed targets
    preparation.

    Returns:
      A boolean list for whether the corresponding target in the output list
      should be skipped during loss calculation.
    """
    return [l is None for l in self.loss_functions]

  def _prepare_output_masks(self):
    """Returns masks corresponding to model outputs."""
    return [getattr(x, '_keras_mask', None) for x in self.outputs]

  def _prepare_total_loss(self, masks):
    """Computes total loss from loss functions.

    Args:
        masks: List of mask values corresponding to each model output.

    Returns:
        A list of loss weights of python floats.

    Raises:
        TypeError: If model run_eagerly is True.
    """
    if self.run_eagerly:
      raise TypeError('total loss can not be computed when compiled with '
                      'run_eagerly = True.')
    loss_list = []
    with backend.name_scope('loss'):
      for endpoint, mask in zip(self._training_endpoints, masks):
        if endpoint.should_skip_target():
          continue
        y_true = endpoint.training_target.target
        y_pred = endpoint.output
        loss_fn = endpoint.loss_fn
        loss_weight = endpoint.loss_weight
        loss_name = endpoint.loss_name()
        sample_weight = endpoint.sample_weight

        with backend.name_scope(loss_name):
          if mask is not None:
            mask = math_ops.cast(mask, y_pred.dtype)
            # Update weights with mask.
            if sample_weight is None:
              sample_weight = mask
            else:
              # Update dimensions of weights to match with mask if possible.
              mask, _, sample_weight = (
                  losses_utils.squeeze_or_expand_dimensions(
                      mask, sample_weight=sample_weight))
              sample_weight *= mask

          if hasattr(loss_fn, 'reduction'):
            per_sample_losses = loss_fn.call(y_true, y_pred)
            weighted_losses = losses_utils.compute_weighted_loss(
                per_sample_losses,
                sample_weight=sample_weight,
                reduction=losses_utils.ReductionV2.NONE)
            loss_reduction = loss_fn.reduction

            # `AUTO` loss reduction defaults to `SUM_OVER_BATCH_SIZE` for all
            # compile use cases.
            if loss_reduction == losses_utils.ReductionV2.AUTO:
              loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

            # Compute the stateless loss value.
            output_loss = losses_utils.reduce_weighted_loss(
                weighted_losses, reduction=loss_reduction)
          else:
            # Compute the stateless loss value for a custom loss class.
            # Here we assume that the class takes care of loss reduction
            # because if this class returns a vector value we cannot
            # differentiate between use case where a custom optimizer
            # expects a vector loss value vs unreduced per-sample loss value.
            output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

        if len(self.outputs) > 1:
          # Keep track of stateful result tensor for the loss.
          endpoint.output_loss_metric(output_loss)

        # Scale output loss for distribution. For custom losses we assume
        # reduction was mean.
        if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
          output_loss = losses_utils.scale_loss_for_distribution(output_loss)

        loss_list.append(loss_weight * output_loss)
      if not loss_list and not self.losses:
        raise ValueError('The model cannot be compiled '
                         'because it has no loss to optimize.')

      # Add regularization penalties and other layer-specific losses.
      custom_losses = self.get_losses_for(None) + self.get_losses_for(
          self.inputs)
      if custom_losses:
        total_custom_loss = math_ops.add_n(
            losses_utils.cast_losses_to_common_dtype(custom_losses))
        loss_list.append(
            losses_utils.scale_loss_for_distribution(total_custom_loss))

      loss_list = losses_utils.cast_losses_to_common_dtype(loss_list)
      if loss_list:
        total_loss = math_ops.add_n(loss_list)
      else:
        total_loss = 0.
    return total_loss

  def _get_callback_model(self):
    """Returns the Callback Model for this Model."""

    if hasattr(self, '_replicated_model') and self._replicated_model:
      # When using training_distributed, we set the callback model
      # to an instance of the `DistributedModel` that we create in
      # the `compile` call. The `DistributedModel` is initialized
      # with the first replicated model. We need to set the callback
      # model to a DistributedModel to allow us to override saving
      # and loading weights when we checkpoint the model during training.
      return self._replicated_model
    if hasattr(self, 'callback_model') and self.callback_model:
      return self.callback_model
    return self

  @trackable.no_automatic_dependency_tracking
  def _make_callback_model(self, grouped_model):
    first_replicated_model = self._distribution_strategy.unwrap(
        grouped_model)[0]
    # We initialize the callback model with the first replicated model.
    self._replicated_model = DistributedCallbackModel(first_replicated_model)
    self._replicated_model.set_original_model(self)

  def _validate_or_infer_batch_size(self, batch_size, steps, x):
    """Validates that the `batch_size` provided is consistent with InputLayer.

    It's possible that the user specified a static batch size in their
    InputLayer. If so, this method checks the provided `batch_size` and `x`
    arguments are consistent with this static batch size. Also, if
    `batch_size` is `None`, this method will attempt to infer the batch size
    from the static batch size of the InputLayer. Lastly, ValueError will be
    raised if `x` is a tf.data.Dataset and `batch_size` is specified as we
    expect users to provide batched datasets.

    Args:
      batch_size: The batch_size provided as an argument to
        fit/evaluate/predict.
      steps: The steps provided as an argument to fit/evaluate/predict.
      x: The data passed as `x` to fit/evaluate/predict.

    Returns:
      The validated batch_size, auto-inferred from the first layer if not
      provided.
    """
    if (isinstance(x, (dataset_ops.DatasetV1,
                       dataset_ops.DatasetV2,
                       data_utils.Sequence)) or
        tf_inspect.isgenerator(x)):
      if batch_size is not None:
        raise ValueError(
            'The `batch_size` argument must not be specified for the given '
            'input type. Received input: {}, batch_size: {}'.format(
                x, batch_size))
      return

    # Avoids the override in Sequential.layers which filters Input layers.
    # (Which are often the very layers that we're after.)
    layers = self._flatten_layers(include_self=False, recursive=False)
    first_layer = next(layers, None)
    if first_layer:
      # The per-replica static batch size.
      static_batch_size = training_utils.get_static_batch_size(first_layer)
      if static_batch_size is not None:

        # Determine number of times the user-supplied batch size will be split.
        if (self._distribution_strategy and
            distributed_training_utils.global_batch_size_supported(
                self._distribution_strategy)):
          num_splits_for_ds = self._distribution_strategy.num_replicas_in_sync
        else:
          num_splits_for_ds = 1

        # Check `batch_size` argument is consistent with InputLayer.
        if batch_size is not None:
          if batch_size % num_splits_for_ds != 0:
            raise ValueError('The `batch_size` argument ({}) must be divisible '
                             'the by number of replicas ({})'.format(
                                 batch_size, num_splits_for_ds))
          per_replica_batch_size = batch_size // num_splits_for_ds

          if per_replica_batch_size != static_batch_size:
            raise ValueError('The `batch_size` argument value {} is '
                             'incompatible with the specified batch size of '
                             'your Input Layer: {}'.format(
                                 per_replica_batch_size, static_batch_size))

        # Check Dataset/Iterator batch size is consistent with InputLayer.
        if isinstance(x, (dataset_ops.DatasetV2, iterator_ops.Iterator,
                          iterator_ops.IteratorBase)):
          ds_batch_size = tensor_shape.Dimension(
              nest.flatten(dataset_ops.get_legacy_output_shapes(x))[0][0]).value
          if ds_batch_size is not None:
            if ds_batch_size % num_splits_for_ds != 0:
              raise ValueError(
                  'The batch output shape of your `Dataset` {} '
                  'cannot be divisible by number of replicas {}'.format(
                      ds_batch_size, num_splits_for_ds))

            ds_per_replica_batch_size = ds_batch_size // num_splits_for_ds
            if ds_per_replica_batch_size != static_batch_size:
              raise ValueError('The batch output shape of your `Dataset` is '
                               '{}, which is incompatible with the specified '
                               'batch size of your Input Layer: {}'.format(
                                   ds_per_replica_batch_size,
                                   static_batch_size))

        # Set inferred batch size from the InputLayer.
        if steps is None:
          batch_size = static_batch_size * num_splits_for_ds

    if batch_size is None and steps is None:
      # Backwards compatibility
      batch_size = 32
    return batch_size

  def _prepare_sample_weights(self, sample_weights=None):
    """Sets sample weight attribute on the model."""
    # List with the same length as model outputs.
    if sample_weights is not None:
      if len(sample_weights) != len(self._training_endpoints):
        raise ValueError('Provided sample weights must have same length as the '
                         'number of outputs. Expected: {}, got: {}.'.format(
                             len(self._training_endpoints),
                             len(sample_weights)))
    else:
      sample_weights = [None] * len(self._training_endpoints)
    for endpoint, weight in zip(self._training_endpoints, sample_weights):
      endpoint.populate_sample_weight(weight, endpoint.sample_weight_mode)

  def _cache_output_metric_attributes(self, metrics, weighted_metrics):
    """Caches metric name and function attributes for every model output."""
    output_shapes = []
    for output in self.outputs:
      if output is None or output.shape.rank is None:
        output_shapes.append(None)
      else:
        output_shapes.append(output.shape.as_list())
    self._per_output_metrics = training_utils_v1.collect_per_output_metric_info(
        metrics, self.output_names, output_shapes, self.loss_functions,
        from_serialized=self._from_serialized)
    self._per_output_weighted_metrics = (
        training_utils_v1.collect_per_output_metric_info(
            weighted_metrics,
            self.output_names,
            output_shapes,
            self.loss_functions,
            from_serialized=self._from_serialized,
            is_weighted=True))

  def _add_unique_metric_name(self, metric_name, metric_fn, output_index):
    """Makes the metric name unique.

      If there are multiple outputs for which the metrics are calculated, the
      metric names have to be made unique by appending an integer.

    Args:
      metric_name: Metric name that corresponds to the metric specified by the
          user. For example: 'acc'.
      metric_fn: The Metric object.
      output_index: The index of the model output for which the metric name is
        being added.

    Returns:
      string, name of the model's unique metric name
    """
    # For multi-output models, prepend the output names to the metric name.
    if len(self.output_names) > 1:
      # If we're loading from an already-serialized model, we've already
      # prepended the output name, and we don't want to do it again.
      #
      # Alternatively, we may be receiving a stateless metric (e.g. the string
      # "accuracy") rather than a `Metric` object, in which case we want to
      # prepend the output name even if we are loading a serialized model.
      if not getattr(metric_fn, '_from_serialized', False):
        metric_name = '%s_%s' % (self.output_names[output_index], metric_name)

    j = 1
    base_metric_name = metric_name
    while metric_name in self.metrics_names:
      metric_name = '%s_%d' % (base_metric_name, j)
      j += 1

    return metric_name

  def _init_metric_attributes(self):
    """Initialized model metric attributes."""
    # List of stateful metric functions. Used for resetting metric state during
    # training/eval.
    self._compile_metric_functions = []

  def _set_per_output_metric_attributes(self, metrics_dict, output_index):
    """Sets the metric attributes on the model for the given output.

    Args:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      output_index: The index of the model output for which the metric
        attributes are added.

    Returns:
      Metrics dict updated with unique metric names as keys.
    """
    updated_metrics_dict = collections.OrderedDict()
    for metric_name, metric_fn in metrics_dict.items():
      metric_name = self._add_unique_metric_name(
          metric_name, metric_fn, output_index)

      # Update the name on the metric class to be the unique generated name.
      metric_fn._name = metric_name  # pylint: disable=protected-access
      updated_metrics_dict[metric_name] = metric_fn
      # Keep track of metric name and function.
      self._compile_metric_functions.append(metric_fn)
    return updated_metrics_dict

  def _set_metric_attributes(self):
    """Sets the metric attributes on the model for all the model outputs."""
    updated_per_output_metrics = []
    updated_per_output_weighted_metrics = []
    for i, endpoint in enumerate(self._training_endpoints):
      if endpoint.should_skip_target():
        updated_per_output_metrics.append(self._per_output_metrics[i])
        updated_per_output_weighted_metrics.append(
            self._per_output_weighted_metrics[i])
        continue
      updated_per_output_metrics.append(
          self._set_per_output_metric_attributes(self._per_output_metrics[i],
                                                 i))
      updated_per_output_weighted_metrics.append(
          self._set_per_output_metric_attributes(
              self._per_output_weighted_metrics[i], i))

    # Create a metric wrapper for each output loss. This computes mean of an
    # output loss across mini-batches (irrespective of how we reduce within a
    # batch).
    if len(self._training_endpoints) > 1:
      for endpoint in self._training_endpoints:
        if not endpoint.should_skip_target():
          endpoint.output_loss_metric = metrics_module.Mean(
              name=endpoint.loss_name())

    self._per_output_metrics = updated_per_output_metrics
    self._per_output_weighted_metrics = updated_per_output_weighted_metrics

  def _handle_per_output_metrics(self,
                                 metrics_dict,
                                 y_true,
                                 y_pred,
                                 mask,
                                 weights=None):
    """Calls metric functions for a single output.

    Args:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      y_true: Target output.
      y_pred: Predicted output.
      mask: Computed mask value for the current output.
      weights: Weights to be applied on the current output.

    Returns:
      A list of metric result tensors.
    """
    metric_results = []
    for metric_name, metric_fn in metrics_dict.items():
      with backend.name_scope(metric_name):
        metric_result = training_utils_v1.call_metric_function(
            metric_fn, y_true, y_pred, weights=weights, mask=mask)
        metric_results.append(metric_result)
    return metric_results

  def _handle_metrics(self,
                      outputs,
                      targets=None,
                      skip_target_masks=None,
                      sample_weights=None,
                      masks=None,
                      return_weighted_metrics=False,
                      return_weighted_and_unweighted_metrics=False):
    """Handles calling metric functions.

    Args:
      outputs: List of outputs (predictions).
      targets: List of targets.
      skip_target_masks: Optional. List of boolean for whether the corresponding
        target should be ignored or not.
      sample_weights: Optional list of sample weight arrays.
      masks: List of computed output mask values.
      return_weighted_metrics: Flag that indicates whether weighted metrics
        should be computed instead of unweighted metrics. This flag is ignored
        when `return_weighted_and_unweighted_metrics` is enabled.
      return_weighted_and_unweighted_metrics: Flag that is used to indicate
        whether both weighted and unweighted metrics should be computed. When
        this is not enabled, we use `return_weighted_metrics` param to indicate
        whether weighted or unweighted metrics should be returned.

    Returns:
      A list of metric result tensors.
    """
    # TODO(scottzhu): Update this to use the new training_endpoints. Currently
    # the eager and graph logic is bit different.
    skip_target_masks = skip_target_masks or [False] * len(outputs)
    metric_results = []
    with backend.name_scope('metrics'):
      # Invoke all metrics added using `compile`.
      for i in range(len(outputs)):
        if skip_target_masks[i]:
          continue
        output = outputs[i] if outputs else None
        target = targets[i] if targets else None
        output_mask = masks[i] if masks else None

        if (return_weighted_and_unweighted_metrics or
            not return_weighted_metrics):
          metric_results.extend(
              self._handle_per_output_metrics(self._per_output_metrics[i],
                                              target, output, output_mask))
        if return_weighted_and_unweighted_metrics or return_weighted_metrics:
          metric_results.extend(
              self._handle_per_output_metrics(
                  self._per_output_weighted_metrics[i],
                  target,
                  output,
                  output_mask,
                  weights=sample_weights[i] if sample_weights else None))
    return metric_results

  def _check_trainable_weights_consistency(self):
    """Check trainable weights count consistency.

    This will raise a warning if `trainable_weights` and
    `_collected_trainable_weights` are inconsistent (i.e. have different
    number of parameters).
    Inconsistency will typically arise when one modifies `model.trainable`
    without calling `model.compile` again.
    """
    if not hasattr(self, '_collected_trainable_weights'):
      return

    if len(self.trainable_weights) != len(self._collected_trainable_weights):
      logging.log_first_n(
          logging.WARN, 'Discrepancy between trainable weights and collected'
          ' trainable weights, did you set `model.trainable`'
          ' without calling `model.compile` after ?', 1)

  def _make_train_function(self):
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    self._check_trainable_weights_consistency()
    if isinstance(self.optimizer, list):
      raise ValueError('The `optimizer` in `compile` should be a single '
                       'optimizer.')
    # If we have re-compiled the loss/weighted metric sub-graphs then create
    # train function even if one exists already. This is because
    # `_feed_sample_weights` list has been updated on re-compile.
    if getattr(self, 'train_function', None) is None or has_recompiled:
      # Restore the compiled trainable state.
      current_trainable_state = self._get_trainable_state()
      self._set_trainable_state(self._compiled_trainable_state)

      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if not isinstance(backend.symbolic_learning_phase(), int):
        inputs += [backend.symbolic_learning_phase()]

      with backend.get_graph().as_default():
        with backend.name_scope('training'):
          # Training updates
          updates = self.optimizer.get_updates(
              params=self._collected_trainable_weights, loss=self.total_loss)
          # Unconditional updates
          updates += self.get_updates_for(None)
          # Conditional updates relevant to this model
          updates += self.get_updates_for(self.inputs)

        metrics = self._get_training_eval_metrics()
        metrics_tensors = [
            m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
        ]

      with backend.name_scope('training'):
        # Gets loss and metrics. Updates weights at each call.
        fn = backend.function(
            inputs, [self.total_loss] + metrics_tensors,
            updates=updates,
            name='train_function',
            **self._function_kwargs)
        setattr(self, 'train_function', fn)

      # Restore the current trainable state
      self._set_trainable_state(current_trainable_state)

  def _make_test_function(self):
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    # If we have re-compiled the loss/weighted metric sub-graphs then create
    # test function even if one exists already. This is because
    # `_feed_sample_weights` list has been updated on re-compile.
    if getattr(self, 'test_function', None) is None or has_recompiled:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)

      with backend.get_graph().as_default():
        metrics = self._get_training_eval_metrics()
        metrics_tensors = [
            m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
        ]

      with backend.name_scope('evaluation'):
        updates = self.state_updates
        # Return loss and metrics, no gradient updates.
        # Does update the network states.
        fn = backend.function(
            inputs, [self.total_loss] + metrics_tensors,
            updates=updates,
            name='test_function',
            **self._function_kwargs)
        setattr(self, 'test_function', fn)

  def _make_predict_function(self):
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      kwargs = getattr(self, '_function_kwargs', {})
      with backend.name_scope(ModeKeys.PREDICT):
        self.predict_function = backend.function(
            inputs,
            self.outputs,
            updates=self.state_updates,
            name='predict_function',
            **kwargs)

  def _make_execution_function(self, mode):
    if mode == ModeKeys.TRAIN:
      self._make_train_function()
      return self.train_function
    if mode == ModeKeys.TEST:
      self._make_test_function()
      return self.test_function
    if mode == ModeKeys.PREDICT:
      self._make_predict_function()
      return self.predict_function

  def _distribution_standardize_user_data(self,
                                          x,
                                          y=None,
                                          sample_weight=None,
                                          class_weight=None,
                                          batch_size=None,
                                          validation_split=0,
                                          shuffle=False,
                                          epochs=1,
                                          allow_partial_batch=False):
    """Runs validation checks on input and target data passed by the user.

    This is called when using tf.distribute.Strategy to train, evaluate or serve
    the model.

    Args:
      x: Input data. A numpy array or `tf.data` dataset.
      y: Target data. A numpy array or None if x is a `tf.data` dataset.
      sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`.
      class_weight: An optional class-weight array by the user to
        weight the importance of samples in `x` based on the class they belong
        to, as conveyed by `y`.
      batch_size: Integer batch size. If provided, it is used to run additional
        validation checks on stateful models.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      epochs: Integer epochs. If > 1, repeat the numpy training data epochs
        times when converting to training dataset.
      allow_partial_batch: Boolean whether to enforce that all batches have the
        same size.

    Returns:
      Dataset instance.

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if class_weight:
      raise NotImplementedError('`class_weight` is currently not supported '
                                'when using tf.distribute.Strategy.')

    if (sample_weight is not None and sample_weight.all() and
        backend.is_tpu_strategy(self._distribution_strategy)):
      raise NotImplementedError('`sample_weight` is currently not supported '
                                'when using TPUStrategy.')

    # Validates `steps` and `shuffle` arguments right at the beginning
    # since we use it to construct the dataset object.
    # TODO(anjalisridhar): Remove this check once we refactor the
    # _standardize_user_data code path. This check is already present elsewhere
    # in the codebase.
    if isinstance(x, dataset_ops.DatasetV2):
      if shuffle:
        training_utils_v1.verify_dataset_shuffled(x)

    strategy = self._distribution_strategy
    with strategy.scope():
      # We should be sure to call get_session() inside the strategy.scope()
      # so the strategy can affect the session options.
      if ops.executing_eagerly_outside_functions():
        session = None
      else:
        session = backend.get_session()

      first_x_value = nest.flatten(x)[0]
      if isinstance(first_x_value, np.ndarray):
        x = training_utils.list_to_tuple(x)
        if y is not None:
          y = training_utils.list_to_tuple(y)
          if sample_weight is not None:
            sample_weight = training_utils.list_to_tuple(sample_weight)
            in_tuple = (x, y, sample_weight)
          else:
            in_tuple = (x, y)
        else:
          in_tuple = x

        ds = strategy.extended.experimental_make_numpy_dataset(in_tuple,
                                                               session=session)
        if shuffle:
          # We want a buffer size that is larger than the batch size provided by
          # the user and provides sufficient randomness. Note that larger
          # numbers introduce more memory usage based on the size of each
          # sample.
          ds = ds.shuffle(max(1024, batch_size * 8))
        if epochs > 1:
          ds = ds.repeat(epochs)

        # We need to use the drop_remainder argument to get a known static
        # input shape which is required for TPUs.
        drop_remainder = (not allow_partial_batch and
                          strategy.extended.experimental_require_static_shapes)

        # TODO(b/131720208): We still drop remainder here if number of examples
        # is divisible by batch size, as sometimes dynamic padder will time out
        # with keras.metrics.CategoricalAccuracy() metric.
        if backend.is_tpu_strategy(strategy) and not drop_remainder:
          dataset_size = first_x_value.shape[0]
          if dataset_size % batch_size == 0:
            drop_remainder = True

        x = ds.batch(batch_size, drop_remainder=drop_remainder)
      else:
        assert isinstance(x, dataset_ops.DatasetV2)
        training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                                 validation_split)
    return x

  def _standardize_user_data(self,
                             x,
                             y=None,
                             sample_weight=None,
                             class_weight=None,
                             batch_size=None,
                             check_steps=False,
                             steps_name='steps',
                             steps=None,
                             validation_split=0,
                             shuffle=False,
                             extract_tensors_from_dataset=False):
    """Runs validation checks on input and target data passed by the user.

    Also standardizes the data to lists of arrays, in order.

    Also builds and compiles the model on the fly if it is a subclassed model
    that has never been called before (and thus has no inputs/outputs).

    This is a purely internal method, subject to refactoring at any time.

    Args:
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors,
          if the model has named inputs.
        - A `tf.data` dataset.
      y: Target data. Like the input data `x`,
        it could be either Numpy array(s) or TensorFlow tensor(s).
        It should be consistent with `x` (you cannot have Numpy inputs and
        tensor targets, or inversely). If `x` is a dataset, `y` should not be
        specified (since targets will be obtained from the iterator).
      sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`.
      class_weight: An optional class-weight array by the user to
        weight the importance of samples in `x` based on the class they belong
        to, as conveyed by `y`. If both `sample_weight` and `class_weight` are
        provided, the weights are multiplied.
      batch_size: Integer batch size. If provided, it is used to run additional
        validation checks on stateful models.
      check_steps: boolean, True if we want to check for validity of `steps` and
        False, otherwise. For example, when we are standardizing one batch of
        data for train_on_batch/predict_on_batch/test_on_batch APIs, `steps`
        value is not required and we should not check for its validity in these
        cases.
      steps_name: The public API's parameter name for `steps`.
      steps: Integer or `None`. Total number of steps (batches of samples) to
        execute.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      extract_tensors_from_dataset: Boolean. When `x` is a dataset instance,
        this indicates whether to extract actual tensors from the dataset or
        instead output the dataset instance itself.
        Set to True when calling from `train_on_batch`/etc.

    Returns:
      A tuple of 3: inputs (arrays or dicts, depending on whether `x` was a dict
      or not), target arrays, sample-weight arrays.
      If the model's input and targets are symbolic, these lists are empty
      (since the model takes no user-provided data, instead the data comes
      from the symbolic inputs/targets).

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
      # Graph mode dataset. We'll pass the dataset as-is (unless
      # `extract_tensors_from_dataset` is True, in which case we extract
      # the tensors from the dataset and we output them.
      training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                               validation_split)
      if shuffle:
        training_utils_v1.verify_dataset_shuffled(x)

      is_dataset = True
      if extract_tensors_from_dataset:
        # We do this for `train_on_batch`/etc.
        x, y, sample_weight = training_utils_v1.extract_tensors_from_dataset(x)
    elif isinstance(x, iterator_ops.Iterator):
      # Graph mode iterator. We extract the symbolic tensors.
      training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                               validation_split)
      iterator = x
      x, y, sample_weight = training_utils_v1.unpack_iterator_input(iterator)
      is_dataset = True
    else:
      is_dataset = False

    # Validates `steps` argument based on x's type.
    if check_steps:
      training_utils_v1.check_steps_argument(x, steps, steps_name)

    # First, we build the model on the fly if necessary.
    if not self.inputs:
      all_inputs, y_input, dict_inputs = self._build_model_with_inputs(x, y)
      is_build_called = True
    else:
      all_inputs = []
      # Whether this is a subclassed model that expects dictionary inputs
      # rather than list inputs (e.g. FeatureColumn-based models).
      dict_inputs = isinstance(self.inputs, dict)
      is_build_called = False
      y_input = y

    # Second, we compile the model on the fly if necessary, mostly for subclass
    # models.
    is_compile_called = False
    if not self._is_compiled and self.optimizer:
      self._compile_from_inputs(all_inputs, y_input, x, y)
      is_compile_called = True

    # In graph mode, if we had just set inputs and targets as symbolic tensors
    # by invoking build and compile on the model respectively, we do not have to
    # feed anything to the model. Model already has input and target data as
    # part of the graph.
    # Note: in this case, `any` and `all` are equivalent since we disallow
    # mixed symbolic/value inputs.

    # self.run_eagerly is not free to compute, so we want to reuse the value.
    run_eagerly = self.run_eagerly

    if (not run_eagerly and is_build_called and is_compile_called and
        not is_dataset  and any(_is_symbolic_tensor(v) for v in all_inputs)):
      return [], [], None

    return self._standardize_tensors(
        x, y, sample_weight,
        run_eagerly=run_eagerly,
        dict_inputs=dict_inputs,
        is_dataset=is_dataset,
        class_weight=class_weight,
        batch_size=batch_size)

  def _standardize_tensors(self, x, y, sample_weight, run_eagerly, dict_inputs,
                           is_dataset, class_weight=None, batch_size=None):
    if run_eagerly:
      # In eager mode, do not do shape validation
      # since the network has no input nodes (placeholders) to be fed.
      feed_input_names = self.input_names
      feed_input_shapes = None
    elif not self._is_graph_network:
      # Case: symbolic-mode subclassed network. Do not do shape validation.
      feed_input_names = self._feed_input_names
      feed_input_shapes = None
    else:
      # Case: symbolic-mode graph network.
      # In this case, we run extensive shape validation checks.
      feed_input_names = self._feed_input_names
      feed_input_shapes = self._feed_input_shapes

    # Standardize the inputs.
    if not isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
      # TODO(fchollet): run static checks with dataset output shape(s).
      x = training_utils_v1.standardize_input_data(
          x,
          feed_input_names,
          feed_input_shapes,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='input')

    # Get typespecs for the input data and sanitize it if necessary.
    # TODO(momernick): This should be capable of doing full input validation
    # at all times - validate that this is so and refactor the standardization
    # code.
    if isinstance(x, dataset_ops.DatasetV2):
      x_shapes = dataset_ops.get_structure(x)
      if isinstance(x_shapes, tuple):
        # If the output of a Dataset is a tuple, we assume it's either of the
        # form (x_data, y_data) or (x_data, y_data, sample_weights). In either
        # case, we only care about x_data here.
        x_shapes = x_shapes[0]
    else:
      flat_inputs = nest.flatten(x, expand_composites=False)
      flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
      converted_x = []
      for (a, b) in zip(flat_inputs, flat_expected_inputs):
        converted_x.append(_convert_scipy_sparse_tensor(a, b))
      x = nest.pack_sequence_as(x, converted_x, expand_composites=False)

      def _type_spec_from_value(value):
        """Grab type_spec without converting array-likes to tensors."""
        if tf_utils.is_extension_type(value):
          return value._type_spec  # pylint: disable=protected-access
        # Get a TensorSpec for array-like data without
        # converting the data to a Tensor
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
          return tensor_spec.TensorSpec(value.shape, value.dtype)
        else:
          return type_spec.type_spec_from_value(value)

      x_shapes = nest.map_structure(_type_spec_from_value, x)

    flat_inputs = nest.flatten(x_shapes, expand_composites=False)
    flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
    for (a, b) in zip(flat_inputs, flat_expected_inputs):
      nest.assert_same_structure(a, b, expand_composites=True)

    if y is not None:
      # Prepare self._sample_weight_modes. List with the same length as
      # model outputs.
      training_utils_v1.prepare_sample_weight_modes(self._training_endpoints,
                                                    self.sample_weight_mode)
      feed_output_names = self._feed_output_names
      feed_sample_weight_modes = self._sample_weight_modes
      if not self._is_graph_network:
        feed_output_shapes = None
      else:
        feed_output_shapes = self._feed_output_shapes

      # Standardize the outputs.
      y = training_utils_v1.standardize_input_data(
          y,
          feed_output_names,
          # Don't enforce target shapes to match output shapes.
          # Precise checks will be run in `check_loss_and_target_compatibility`.
          shapes=None,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='target')

      # Generate sample-wise weight values given the `sample_weight` and
      # `class_weight` arguments.
      sample_weights = training_utils_v1.standardize_sample_weights(
          sample_weight, feed_output_names)
      class_weights = training_utils_v1.standardize_class_weights(
          class_weight, feed_output_names)

      sample_weights = [
          training_utils_v1.standardize_weights(ref, sw, cw, mode)
          for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights,
                                         feed_sample_weight_modes)
      ]
      # Check that all arrays have the same length.
      if not self._distribution_strategy:
        training_utils_v1.check_array_lengths(x, y, sample_weights)
        if self._is_graph_network and not run_eagerly:
          # Additional checks to avoid users mistakenly using improper loss fns.
          training_utils_v1.check_loss_and_target_compatibility(
              y, self._feed_loss_fns, feed_output_shapes)

      sample_weights, _, _ = training_utils.handle_partial_sample_weights(
          y, sample_weights, feed_sample_weight_modes, check_all_flat=True)
    else:
      y = []
      sample_weights = None

    if self.stateful and batch_size and not is_dataset:
      # Check that for stateful networks, number of samples is a multiple
      # of the static batch size.
      if x[0].shape[0] % batch_size != 0:
        raise ValueError('In a stateful network, '
                         'you should only pass inputs with '
                         'a number of samples that can be '
                         'divided by the batch size. Found: ' +
                         str(x[0].shape[0]) + ' samples')

    # If dictionary inputs were provided, we return a dictionary as well.
    if dict_inputs and not isinstance(x, (dataset_ops.DatasetV1,
                                          dataset_ops.DatasetV2)):
      x = dict(zip(feed_input_names, x))
    return x, y, sample_weights

  def _build_model_with_inputs(self, inputs, targets):
    """Build the model (set model inputs/outputs), mainly for subclass model."""
    processed_inputs = []
    is_dict_inputs = False
    orig_inputs = inputs
    # We need to use `inputs` to set the model inputs.
    # If input data is a dataset iterator in graph mode or if it is an eager
    # iterator and only one batch of samples is required, we fetch the data
    # tensors from the iterator and then standardize them.
    if isinstance(inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
      inputs, targets, _ = training_utils_v1.extract_tensors_from_dataset(
          inputs)
    # We type-check that `inputs` and `targets` are either single arrays
    # or lists of arrays, and extract a flat list of inputs from the passed
    # structure.
    training_utils_v1.validate_input_types(inputs, orig_inputs)

    if isinstance(inputs, (list, tuple)):
      processed_inputs += list(inputs)
    elif isinstance(inputs, dict):
      is_dict_inputs = True
      keys = sorted(inputs.keys())
      processed_inputs = [inputs[k] for k in keys]
    else:
      processed_inputs.append(inputs)
    # Now that we have a flat set of inputs, we make sure that none of them
    # are CompositeTensors or CompositeTensorValues of any type (or scipy
    # sparse arrays, which we treat as SparseTensor values). We cannot safely
    # infer input data from an arbitrary composite tensor, so we don't try -
    # users should explicitly add composite tensor inputs to their subclassed
    # models.
    for input_tensor in processed_inputs:
      if training_utils_v1.is_composite_or_composite_value(input_tensor):
        # TODO(b/132691975): Document subclass-model CT input handling.
        raise ValueError(
            'All SparseTensor and RaggedTensor inputs must be explicitly '
            'declared using a keras.Input() with sparse=True or ragged=True. '
            'We found an undeclared input %s. For Sequential models, please '
            'add a keras.Input() as your first Layer. For subclassed models, '
            'please call self._set_inputs() on your input set, which you can '
            'create using keras.Input() for each input to your model.' %
            (input_tensor,))
    # Build the model using the retrieved inputs (value or symbolic).
    # If values are generated from a dataset, then in symbolic-mode
    # placeholders will be created to match the value shapes.
    if isinstance(orig_inputs, (dataset_ops.DatasetV1, dataset_ops.DatasetV2,
                                iterator_ops.Iterator)):
      if not self.inputs:
        # For subclassed models, a robust input spec is not available so we
        # must cast to the model dtype.
        inputs = training_utils_v1.cast_if_floating_dtype(inputs, self.dtype)

      def create_tensor_spec(t):
        return tensor_spec.TensorSpec(t.shape, t.dtype)

      cast_inputs = nest.map_structure(create_tensor_spec, inputs)
    elif training_utils_v1.has_tensors(inputs):
      cast_inputs = training_utils_v1.cast_if_floating_dtype(inputs)
    else:
      cast_inputs = inputs
    self._set_inputs(cast_inputs)
    return processed_inputs, targets, is_dict_inputs

  def _compile_from_inputs(self, all_inputs, target, orig_inputs, orig_target):
    if target is not None:
      # We need to use `y` to set the model targets.
      if training_utils_v1.has_tensors(target):
        target = training_utils_v1.cast_if_floating_dtype_and_mismatch(
            target, self.outputs)
      training_utils_v1.validate_input_types(
          target, orig_target, allow_dict=False, field_name='target')
      if isinstance(target, (list, tuple)):
        all_inputs += list(target)
      else:
        all_inputs.append(target)
    # Type check that all inputs are *either* value *or* symbolic.
    # TODO(fchollet): this check could be removed in Eager mode?
    if any(tensor_util.is_tf_type(v) for v in all_inputs):
      if not all(tensor_util.is_tf_type(v) for v in all_inputs):
        raise ValueError('Do not pass inputs that mix Numpy arrays and '
                         'TensorFlow tensors. '
                         'You passed: x=' + str(orig_inputs) +
                         '; y=' + str(orig_target))
    is_dataset = isinstance(orig_inputs, (dataset_ops.DatasetV1,
                                          dataset_ops.DatasetV2,
                                          iterator_ops.Iterator))
    if is_dataset or context.executing_eagerly():
      target_tensors = None
    else:
      # Handle target tensors if any passed.
      if target is not None:
        if not isinstance(target, (list, tuple)):
          target = [target]
        target_tensors = [v for v in target if _is_symbolic_tensor(v)]
      else:
        target_tensors = None

    self.compile(
        optimizer=self.optimizer,
        loss=self.loss,
        metrics=self._compile_metrics,
        weighted_metrics=self._compile_weighted_metrics,
        loss_weights=self.loss_weights,
        target_tensors=target_tensors,
        sample_weight_mode=self.sample_weight_mode,
        run_eagerly=self.run_eagerly,
        experimental_run_tf_function=self._experimental_run_tf_function)

  # TODO(omalleyt): Consider changing to a more descriptive function name.
  def _set_inputs(self, inputs, outputs=None, training=None):
    """Set model's input and output specs based on the input data received.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    Args:
      inputs: Single array, or list of arrays. The arrays could be placeholders,
        Numpy arrays, data tensors, or TensorSpecs.
        - if placeholders: the model is built on top of these placeholders,
          and we expect Numpy data to be fed for them when calling `fit`/etc.
        - if Numpy data or TensorShapes: we create placeholders matching the
          TensorShapes or shapes of the Numpy arrays. We expect Numpy data to be
          fed for these placeholders when calling `fit`/etc.
        - if data tensors: the model is built on top of these tensors.
          We do not expect any Numpy data to be provided when calling `fit`/etc.
      outputs: None, a data tensor, or a list of tensors. If None, the
        outputs will be determined by invoking `self.call()`, otherwise the
        provided value will be used.
      training: Boolean or None. Only relevant in symbolic mode. Specifies
        whether to build the model's graph in inference mode (False), training
        mode (True), or using the Keras learning phase (None).
    Raises:
      ValueError: If dict inputs are passed to a Sequential Model where the
        first layer isn't FeatureLayer.
    """
    self._set_save_spec(inputs)
    inputs = self._set_input_attrs(inputs)

    if outputs is None:
      kwargs = {}
      if self._expects_training_arg:
        # In V2 mode, feeding `training=None` is not allowed because any value
        # explicitly passed by the user is respected, even `None`.`
        if training is None and not ops.executing_eagerly_outside_functions():
          training = backend.learning_phase()
        if training is not None:
          kwargs['training'] = training
      try:
        outputs = self(inputs, **kwargs)
      except NotImplementedError:
        # This Model or a submodel is dynamic and hasn't overridden
        # `compute_output_shape`.
        outputs = None

    self._set_output_attrs(outputs)

  @trackable.no_automatic_dependency_tracking
  def _set_input_attrs(self, inputs):
    """Sets attributes related to the inputs of the Model."""
    if self.inputs:
      raise ValueError('Model inputs are already set.')

    if self.__class__.__name__ == 'Sequential' and not self.built:
      if tensor_util.is_tf_type(inputs):
        input_shape = (None,) + tuple(inputs.shape.as_list()[1:])
      elif isinstance(inputs, tensor_shape.TensorShape):
        input_shape = (None,) + tuple(inputs.as_list()[1:])
      elif isinstance(inputs, dict):
        # We assert that the first layer is a FeatureLayer.
        if not training_utils_v1.is_feature_layer(self.layers[0]):
          raise ValueError('Passing a dictionary input to a Sequential Model '
                           'which doesn\'t have FeatureLayer as the first layer'
                           ' is an error.')
        input_shape = (None,)
      else:
        input_shape = (None,) + tuple(inputs.shape[1:])
      self._build_input_shape = input_shape

    # Cast inputs to the compute dtype. This is primarily used
    # when saving to determine the correct dtype in the input signature.
    inputs = self._maybe_cast_inputs(inputs)

    # On-the-fly setting of symbolic model inputs (either by using the tensor
    # provided, or by creating a placeholder if Numpy data was provided).
    model_inputs = training_utils_v1.ModelInputs(inputs)
    inputs = model_inputs.get_symbolic_inputs()
    self.inputs = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.input_names = model_inputs.get_input_names()

    self._feed_inputs = []
    self._feed_input_names = []
    self._feed_input_shapes = []

    for k, v in model_inputs.as_dict():
      if backend.is_placeholder(v):
        self._feed_input_names.append(k)
        self._feed_inputs.append(v)
        self._feed_input_shapes.append(backend.int_shape(v))

    return inputs

  @trackable.no_automatic_dependency_tracking
  def _set_output_attrs(self, outputs):
    """Sets attributes related to the outputs of the Model."""
    # NOTE(taylorrobie): This convention cannot be changed without updating the
    #                    data adapter since it assumes nest.flatten ordering.
    outputs = nest.flatten(outputs)
    self.outputs = outputs
    self.output_names = training_utils_v1.generic_output_names(outputs)
    # TODO(scottzhu): Should we cleanup the self._training_endpoints here?
    self.built = True

  @property
  def _targets(self):
    """The output target tensors for the model."""
    return [
        e.training_target.target
        for e in self._training_endpoints
        if e.has_training_target()
    ]

  @property
  def _feed_targets(self):
    return [
        e.training_target.target
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_output_names(self):
    return [
        e.output_name
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_output_shapes(self):
    return [
        e.feed_output_shape
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_loss_fns(self):
    return [
        e.loss_fn
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _loss_weights_list(self):
    return [e.loss_weight for e in self._training_endpoints]

  @property
  def _output_loss_metrics(self):
    if hasattr(self, '_training_endpoints'):
      return [
          e.output_loss_metric
          for e in self._training_endpoints
          if e.output_loss_metric is not None
      ]
    return None

  @property
  def sample_weights(self):
    return [e.sample_weight for e in self._training_endpoints]

  @property
  def _sample_weight_modes(self):
    return [e.sample_weight_mode for e in self._training_endpoints]

  @property
  def _feed_sample_weights(self):
    return [e.sample_weight for e in self._training_endpoints
            if e.sample_weight is not None]

  def _maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    Refer to tensorflow/python/keras/distribute/worker_training_state.py
    for more information.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    if self._training_state is not None:
      return self._training_state.maybe_load_initial_epoch_from_ckpt(
          initial_epoch, mode)
    return initial_epoch

  def _get_training_eval_metrics(self):
    """Returns all the metrics that are to be reported.

    This includes the output loss metrics, compile metrics/weighted metrics,
    add_metric metrics.
    """
    metrics = []
    metrics.extend(getattr(self, '_output_loss_metrics', None) or [])
    metrics.extend(getattr(self, 'metrics', None) or [])
    return metrics

  def _assert_compile_was_called(self):
    # Checks whether `compile` has been called. If it has been called,
    # then the optimizer is set. This is different from whether the
    # model is compiled
    # (i.e. whether the model is built and its inputs/outputs are set).
    if not self._compile_was_called:
      raise RuntimeError('You must compile your model before '
                         'training/testing. '
                         'Use `model.compile(optimizer, loss)`.')

  def _in_multi_worker_mode(self):
    """Method to infer if this `Model` is working in multi-worker settings.

    Multi-worker training refers to the setup where the training is
    distributed across multiple workers, as opposed to the case where
    only a local process performs the training. This function is
    used to infer for example whether or not a distribute coordinator
    should be run, and thus TensorFlow servers should be started for
    communication with other servers in the cluster, or whether or not
    saving/restoring checkpoints is relevant for preemption fault tolerance.

    Experimental. Signature and implementation are subject to change.

    Returns:
      Whether this model indicates it's working in multi-worker settings.
    """
    strategy = self._distribution_strategy

    # Otherwise, use the strategy whose scope this is in.
    if not strategy and distribution_strategy_context.has_strategy():
      strategy = distribution_strategy_context.get_strategy()
    return strategy and strategy.extended._in_multi_worker_mode()  # pylint: disable=protected-access

  @property
  def _trackable_saved_model_saver(self):
    return model_serialization.ModelSavedModelSaver(self)

  def _get_compile_args(self, user_metrics=True):
    del user_metrics
    self._assert_compile_was_called()
    kwargs = {
        'loss': self.loss,
        'metrics': self._compile_metrics,
        'loss_weights': self.loss_weights,
        'sample_weight_mode': self.sample_weight_mode,
        'weighted_metrics': self._compile_weighted_metrics,
    }
    return kwargs

  @property
  def _compile_was_called(self):
    return self._v1_compile_was_called


class DistributedCallbackModel(Model):
  """Model that is used for callbacks with tf.distribute.Strategy."""

  def __init__(self, model):
    super(DistributedCallbackModel, self).__init__()
    self.optimizer = model.optimizer

  def set_original_model(self, orig_model):
    self._original_model = orig_model

  def save_weights(self, filepath, overwrite=True, save_format=None):
    self._replicated_model.save_weights(filepath, overwrite=overwrite,
                                        save_format=save_format)

  def save(self, filepath, overwrite=True, include_optimizer=True):
    # save weights from the distributed model to the original model
    distributed_model_weights = self.get_weights()
    self._original_model.set_weights(distributed_model_weights)
    # TODO(anjalisridhar): Do we need to save the original model here?
    # Saving the first replicated model works as well.
    self._original_model.save(filepath, overwrite=True, include_optimizer=False)

  def load_weights(self, filepath, by_name=False):
    self._original_model.load_weights(filepath, by_name=False)
    # Copy the weights from the original model to each of the replicated models.
    orig_model_weights = self._original_model.get_weights()
    distributed_training_utils_v1.set_weights(
        self._original_model._distribution_strategy, self,  # pylint: disable=protected-access
        orig_model_weights)

  def __getattr__(self, item):
    # Allowed attributes of the model that can be accessed by the user
    # during a callback.
    if item not in ('_setattr_tracking', '_layers'):
      logging.warning('You are accessing attribute ' + item + ' of the '
                      'DistributedCallbackModel that may not have been set '
                      'correctly.')
    return super(DistributedCallbackModel, self).__getattr__(item)


class _TrainingEndpoint(object):
  """A container for the training output/target and related entities.

  In the case of model with multiple outputs, there is a one-to-one mapping
  between model output (y_pred), model target (y_true), loss, metrics etc.
  By unifying these entities into one class, different entity can access
  information between each other, rather than currently access different list of
  attributes of the model.
  """

  def __init__(self,
               output,
               output_name,
               loss_fn,
               loss_weight=None,
               training_target=None,
               output_loss_metric=None,
               sample_weight=None,
               sample_weight_mode=None):
    """Initialize the _TrainingEndpoint.

    Note that the output and output_name should be stable as long as the model
    structure doesn't change. The training_target suppose to be mutable since
    the information is provided via `compile()`

    Args:
      output: the output tensor of the model.
      output_name: the unique name of the output tensor.
      loss_fn: the loss function for the output tensor.
      loss_weight: float, the weights for the loss.
      training_target: the _TrainingTarget for the model.
      output_loss_metric: the metric object for the loss function.
      sample_weight: the weights for how a sample is weighted during metric and
        loss calculation. Could be None.
      sample_weight_mode: string, 'temporal', 'samplewise' or None. The mode for
        how the sample_weight is populated.
    """
    self._output = output
    self._output_name = output_name
    self._loss_fn = loss_fn
    self._loss_weight = loss_weight
    self._training_target = training_target
    self._output_loss_metric = output_loss_metric
    self._sample_weight = sample_weight
    self._sample_weight_mode = sample_weight_mode

  @property
  def output(self):
    return self._output

  @property
  def output_name(self):
    return self._output_name

  @property
  def shape(self):
    return backend.int_shape(self.output)

  @property
  def loss_fn(self):
    return self._loss_fn

  @property
  def loss_weight(self):
    return self._loss_weight

  @loss_weight.setter
  def loss_weight(self, value):
    self._loss_weight = value

  @property
  def training_target(self):
    return self._training_target

  @training_target.setter
  def training_target(self, value):
    self._training_target = value

  def create_training_target(self, target, run_eagerly=False):
    """Create training_target instance and update the self.training_target.

    Note that the input target should just be a tensor or None, and
    corresponding training target will be created based on the output and
    loss_fn.

    Args:
      target: the target tensor for the current output. Could be None.
      run_eagerly: boolean, whether the model is in run_eagerly mode.

    Raises:
      ValueError if the training_target field for the current instance has
      already been populated.
    """
    if self.has_training_target():
      raise ValueError('The training_target field for the _TrainingEndpoint '
                       'instance has already been populated')
    if run_eagerly:
      # When run_eagerly, the target tensor is ignored, and the None placeholder
      # is created instead.
      self.training_target = _TrainingTarget(
          None, feedable=True, skip_target_weights=False)
      return

    if self.should_skip_target():
      self.training_target = _TrainingTarget(None)
    else:
      if target is not None and not backend.is_placeholder(target):
        feedable = False
        skip_target_weights = True
      else:
        feedable = True
        skip_target_weights = False

      if target is None:
        target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(
            self.loss_fn, backend.dtype(self.output))

        target = backend.placeholder(
            ndim=len(self.shape),
            name=self.output_name + '_target',
            sparse=backend.is_sparse(self.output),
            dtype=target_dtype)

      self.training_target = _TrainingTarget(
          target,
          feedable=feedable,
          skip_target_weights=skip_target_weights)

  @property
  def output_loss_metric(self):
    return self._output_loss_metric

  @output_loss_metric.setter
  def output_loss_metric(self, value):
    self._output_loss_metric = value

  @property
  def sample_weight(self):
    return self._sample_weight

  @sample_weight.setter
  def sample_weight(self, value):
    self._sample_weight = value

  @property
  def sample_weight_mode(self):
    return self._sample_weight_mode

  @sample_weight_mode.setter
  def sample_weight_mode(self, value):
    self._sample_weight_mode = value

  def should_skip_target(self):
    return self._loss_fn is None

  def should_skip_target_weights(self):
    return (self.should_skip_target() or self.training_target is None or
            self.training_target.skip_target_weights)

  def has_training_target(self):
    return self.training_target is not None

  def has_feedable_training_target(self):
    return (not self.should_skip_target() and
            self.training_target is not None and self.training_target.feedable)

  def loss_name(self):
    if self._loss_fn is not None:
      return self._output_name + '_loss'
    return None

  @property
  def feed_output_shape(self):
    """The output shape for the feedable target."""
    if not self.has_feedable_training_target():
      return None

    if ((isinstance(self.loss_fn, losses.LossFunctionWrapper) and
         self.loss_fn.fn == losses.sparse_categorical_crossentropy)) or (
             isinstance(self.loss_fn, losses.SparseCategoricalCrossentropy)):
      if backend.image_data_format() == 'channels_first':
        return (self.shape[0], 1) + self.shape[2:]
      else:
        return self.shape[:-1] + (1,)
    elif (not isinstance(self.loss_fn, losses.Loss) or
          (isinstance(self.loss_fn, losses.LossFunctionWrapper) and
           (getattr(losses, self.loss_fn.fn.__name__, None) is None))):
      # If the given loss is not an instance of the `Loss` class (custom
      # class) or if the loss function that is wrapped is not in the
      # `losses` module, then it is a user-defined loss and we make no
      # assumptions about it.
      return None
    else:
      return self.shape

  def sample_weights_mismatch(self):
    """Check if the sample weight and the mode match or not."""
    # If there is a mismatch between sample weight mode and the placeholders
    # created, then recompile the sub-graphs that depend on sample weights.
    return (
        (self.sample_weight_mode is not None and self.sample_weight is None) or
        (self.sample_weight_mode is None and self.sample_weight is not None))

  def populate_sample_weight(self, sample_weight, sample_weight_mode):
    """Populate the sample weight and based on the sample weight mode."""
    if (sample_weight is None and
        (self.should_skip_target_weights() or sample_weight_mode is None or
         context.executing_eagerly())):
      self._sample_weight = None
      return

    assert sample_weight_mode in ['temporal', 'samplewise']
    if sample_weight_mode == 'temporal':
      default_value = [[1.]]
      shape = [None, None]
    else:
      # sample_weight_mode == 'samplewise'
      default_value = [1.]
      shape = [None]

    if sample_weight is not None:
      if not sample_weight.shape.is_compatible_with(shape):
        raise ValueError('Received sample weight with shape {}. Expected shape '
                         '{}.'.format(sample_weight.shape, shape))
      self._sample_weight = sample_weight
    else:
      self._sample_weight = array_ops.placeholder_with_default(
          constant_op.constant(default_value, dtype=backend.floatx()),
          shape=shape,
          name=self.output_name + '_sample_weights')


class _TrainingTarget(object):
  """Container for a target tensor (y_true) and its metadata (shape, loss...).

  Args:
    target: A target tensor for the model. It may be `None` if the
      output is excluded from loss computation. It is still kept as None
      since each output of the model should have a corresponding target. If
      the target is None, the rest of the attributes will be None as well.
    feedable: Boolean, whether the target is feedable (requires data to be
      passed in `fit` or `train_on_batch`), or not (model compiled with
      `target_tensors` argument).
    skip_target_weights: Boolean, whether the target should be skipped during
      weights calculation.
  """

  def __init__(self, target, feedable=False, skip_target_weights=True):
    self._target = target
    self._feedable = feedable
    self._skip_target_weights = skip_target_weights

  @property
  def target(self):
    return self._target

  @property
  def feedable(self):
    return self._feedable

  @property
  def skip_target_weights(self):
    return self._skip_target_weights


def _is_symbolic_tensor(x):
  return tensor_util.is_tf_type(x)


def _convert_scipy_sparse_tensor(value, expected_input):
  """Handle scipy sparse tensor conversions.

  This method takes a value 'value' and returns the proper conversion. If
  value is a scipy sparse tensor and the expected input is a dense tensor,
  we densify 'value'. If value is a scipy sparse tensor and the expected input
  is a TF SparseTensor, we convert 'value' to a SparseTensor. If 'value' is
  not a scipy sparse tensor, or scipy is not imported, we pass it through
  unchanged.

  Args:
    value: An object that may be a scipy sparse tensor
    expected_input: The expected input placeholder.

  Returns:
    The possibly-converted 'value'.
  """
  if issparse is not None and issparse(value):
    if backend.is_sparse(expected_input):
      sparse_coo = value.tocoo()
      row, col = sparse_coo.row, sparse_coo.col
      data, shape = sparse_coo.data, sparse_coo.shape
      indices = np.concatenate((np.expand_dims(row, 1), np.expand_dims(col, 1)),
                               1)
      return sparse_tensor.SparseTensor(indices, data, shape)
    else:
      if ops.executing_eagerly_outside_functions():
        # In TF2 we do not silently densify sparse matrices.
        raise ValueError('A SciPy sparse matrix was passed to a model '
                         'that expects dense inputs. Please densify your '
                         'inputs first, such as by calling `x.toarray().')
      return value.toarray()
  else:
    return value


def _get_metrics_from_layers(layers):
  """Returns list of metrics from the given layers.

  This will not include the `compile` metrics of a model layer.

  Args:
    layers: List of layers.

  Returns:
    List of metrics.
  """
  metrics = []
  layers = layer_utils.filter_empty_layer_containers(layers)
  for layer in layers:
    if isinstance(layer, Model):
      # We cannot call 'metrics' on the model because we do not want to
      # include the metrics that were added in compile API of a nested model.
      metrics.extend(layer._metrics)  # pylint: disable=protected-access
      metrics.extend(_get_metrics_from_layers(layer.layers))
    else:
      metrics.extend(layer.metrics)
  return metrics


def _non_none_constant_value(v):
  constant_value = tensor_util.constant_value(v)
  return constant_value if constant_value is not None else v
