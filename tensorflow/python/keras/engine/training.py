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
"""Training-related part of the Keras engine.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.keras.engine import training_arrays
from tensorflow.python.keras.engine import training_distributed
from tensorflow.python.keras.engine import training_eager
from tensorflow.python.keras.engine import training_generator
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer as tf_optimizer_module
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.mode_keys import ModeKeys
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.models.Model', 'keras.Model')
class Model(Network):
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
    # This flag is used to track if the user is using the deprecated path of
    # passing distribution strategy to compile rather than creating the model
    # under distribution strategy scope.
    self._compile_distribution = False

    self.run_eagerly = None

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    if self._distribution_strategy:
      with self._distribution_strategy.scope():
        return super(Model, self).get_weights()
    return super(Model, self).get_weights()

  @checkpointable.no_automatic_dependency_tracking
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: String (name of optimizer) or optimizer instance.
            See `tf.keras.optimizers`.
        loss: String (name of objective function) or objective function.
            See `tf.losses`. If the model has multiple outputs, you can use a
            different loss on each output by passing a dictionary or a list of
            losses. The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model
            during training and testing.
            Typically you will use `metrics=['accuracy']`.
            To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary,
            such as `metrics={'output_a': 'accuracy'}`.
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
    run_eagerly = kwargs.pop('run_eagerly', None)
    self._run_eagerly = run_eagerly
    optimizer = optimizers.get(optimizer)

    if distribute is not None:
      if tf2.enabled():
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

    # Validate that arguments passed by the user to `compile` are supported by
    # DistributionStrategy.
    if self._distribution_strategy:
      if not isinstance(optimizer,
                        (tf_optimizer_module.Optimizer, optimizers.TFOptimizer,
                         optimizer_v2.OptimizerV2)):
        raise NotImplementedError(
            'optimizer must be an instance of '
            'tf.train.Optimizer, not a %s' % type(optimizer))
      if sample_weight_mode:
        raise NotImplementedError('sample_weight_mode is not supported with '
                                  'DistributionStrategy.')
      if weighted_metrics:
        raise NotImplementedError('weighted_metrics is not supported with '
                                  'DistributionStrategy.')
      if target_tensors:
        raise ValueError('target_tensors is not supported with '
                         'DistributionStrategy.')

    loss = loss or {}
    if self.run_eagerly and not isinstance(
        optimizer, (tf_optimizer_module.Optimizer, optimizers.TFOptimizer,
                    optimizer_v2.OptimizerV2)):
      raise ValueError(
          'When running a model in eager execution, the optimizer must be an '
          'instance of tf.train.Optimizer. Received: '
          '%s' % optimizer)

    self.optimizer = optimizer
    # We've disabled automatic dependency tracking for this method, but do want
    # to add a checkpoint dependency on the optimizer if it's checkpointable.
    if isinstance(self.optimizer, checkpointable.Checkpointable):
      self._track_checkpointable(
          self.optimizer, name='optimizer', overwrite=True)
    self.loss = loss
    self._compile_metrics = metrics or []
    self.loss_weights = loss_weights
    self.sample_weight_mode = sample_weight_mode
    self._compile_weighted_metrics = weighted_metrics
    if self.run_eagerly and target_tensors is not None:
      raise ValueError(
          'target_tensors argument is not supported when '
          'running a model eagerly.')
    self.target_tensors = target_tensors

    # Set DistributionStrategy specific parameters.
    self._distributed_model_cache = {}

    if self._distribution_strategy is not None:
      # Ensures a Session is created and configured correctly for Distribution
      # Strategy.
      K.configure_and_create_distributed_session(self._distribution_strategy)
    # Initialize model metric attributes.
    self._init_metric_attributes()
    if not self.built or not self.inputs or not self.outputs:
      # Model is not compilable because it does not know its number of inputs
      # and outputs, nor their shapes and names. We will compile after the first
      # time the model gets called on training data.
      return
    self._is_compiled = True

    # Prepare loss functions.
    if isinstance(loss, dict):
      for name in loss:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in loss '
              'dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      loss_functions = []
      for name in self.output_names:
        if name not in loss:
          logging.warning(
              'Output "' + name +
              '" missing from loss dictionary. We assume '
              'this was done on purpose. The fit and evaluate APIs will not be '
              'expecting any data to be passed to "' + name + '".')
        loss_functions.append(training_utils.get_loss_function(loss.get(name)))
    elif isinstance(loss, list):
      if len(loss) != len(self.outputs):
        raise ValueError('When passing a list as loss, '
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed loss=' + str(loss))
      loss_functions = [training_utils.get_loss_function(l) for l in loss]
    else:
      loss_function = training_utils.get_loss_function(loss)
      loss_functions = [loss_function for _ in range(len(self.outputs))]
    self.loss_functions = loss_functions

    skip_target_indices = []
    skip_target_weighing_indices = []
    self._feed_outputs = []
    self._feed_output_names = []
    self._feed_output_shapes = []
    self._feed_loss_fns = []
    for i in range(len(loss_functions)):
      if loss_functions[i] is None:
        skip_target_indices.append(i)
        skip_target_weighing_indices.append(i)

    # Prepare output masks.
    if not self.run_eagerly:
      masks = [getattr(x, '_keras_mask', None) for x in self.outputs]
      if not isinstance(masks, list):
        masks = [masks]

    # Prepare loss weights.
    if loss_weights is None:
      loss_weights_list = [1. for _ in range(len(self.outputs))]
    elif isinstance(loss_weights, dict):
      for name in loss_weights:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in loss_weights '
              'dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      loss_weights_list = []
      for name in self.output_names:
        loss_weights_list.append(loss_weights.get(name, 1.))
    elif isinstance(loss_weights, list):
      if len(loss_weights) != len(self.outputs):
        raise ValueError(
            'When passing a list as loss_weights, '
            'it should have one entry per model output. '
            'The model has ' + str(len(self.outputs)) +
            ' outputs, but you passed loss_weights=' + str(loss_weights))
      loss_weights_list = loss_weights
    else:
      raise TypeError('Could not interpret loss_weights argument: ' +
                      str(loss_weights) + ' - expected a list of dicts.')
    self.loss_weights_list = loss_weights_list

    # Initialization for Eager mode execution.
    if self.run_eagerly:
      # Prepare sample weights.
      self._set_sample_weight_attributes(sample_weight_mode,
                                         skip_target_weighing_indices)
      # Save all metric attributes per output of the model.
      self._cache_output_metric_attributes(metrics, weighted_metrics)

      if target_tensors is not None:
        raise ValueError('target_tensors are not currently supported in Eager '
                         'mode.')
      self.total_loss = None
      for i in range(len(self.outputs)):
        if len(self.outputs) > 1:
          self._compile_metrics_names.append(self.output_names[i] + '_loss')

      # Set metric attributes on model.
      self._set_metric_attributes(
          self.outputs,
          skip_target_indices=skip_target_indices,
      )

      self.targets = []
      for i in range(len(self.outputs)):
        self._feed_output_names.append(self.output_names[i])
      self._collected_trainable_weights = self.trainable_weights
      return

    with K.get_graph().as_default():
      # Prepare targets of model.
      self.targets = []
      self._feed_targets = []
      if target_tensors not in (None, []):
        if isinstance(target_tensors, list):
          if len(target_tensors) != len(self.outputs):
            raise ValueError(
                'When passing a list as `target_tensors`, '
                'it should have one entry per model output. '
                'The model has %s outputs, but you passed target_tensors=%s' %
                (len(self.outputs), target_tensors))
        elif isinstance(target_tensors, dict):
          for name in target_tensors:
            if name not in self.output_names:
              raise ValueError(
                  'Unknown entry in `target_tensors` '
                  'dictionary: "' + name + '". '
                  'Only expected the following keys: ' + str(self.output_names))
          tmp_target_tensors = []
          for name in self.output_names:
            tmp_target_tensors.append(target_tensors.get(name, None))
          target_tensors = tmp_target_tensors
        elif tensor_util.is_tensor(target_tensors):
          target_tensors = [target_tensors]
        else:
          raise TypeError('Expected `target_tensors` to be a list or tuple or '
                          'dict or a single tensor, but got:', target_tensors)

      for i in range(len(self.outputs)):
        if i in skip_target_indices:
          self.targets.append(None)
        else:
          shape = K.int_shape(self.outputs[i])
          name = self.output_names[i]
          if target_tensors not in (None, []):
            target = target_tensors[i]
          else:
            target = None
          if target is None or K.is_placeholder(target):
            if target is None:
              target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(
                  self.loss_functions[i],
                  K.dtype(self.outputs[i]))

              target = K.placeholder(
                  ndim=len(shape),
                  name=name + '_target',
                  sparse=K.is_sparse(self.outputs[i]),
                  dtype=target_dtype)
            self._feed_targets.append(target)
            self._feed_outputs.append(self.outputs[i])
            self._feed_output_names.append(name)
            self._feed_output_shapes.append(shape)
            self._feed_loss_fns.append(self.loss_functions[i])
          else:
            skip_target_weighing_indices.append(i)
          self.targets.append(target)

      # Prepare sample weights.
      self._set_sample_weight_attributes(sample_weight_mode,
                                         skip_target_weighing_indices)
      # Save all metric attributes per output of the model.
      self._cache_output_metric_attributes(metrics, weighted_metrics)

      # Compute total loss.
      total_loss = None
      with K.name_scope('loss'):
        for i in range(len(self.outputs)):
          if i in skip_target_indices:
            continue
          y_true = self.targets[i]
          y_pred = self.outputs[i]
          loss_fn = loss_functions[i]
          sample_weight = self.sample_weights[i]
          mask = masks[i]
          loss_weight = loss_weights_list[i]
          with K.name_scope(self.output_names[i] + '_loss'):
            if mask is not None:
              mask = math_ops.cast(mask, y_pred.dtype)
              # Update weights with mask.
              if sample_weight is None:
                sample_weight = mask
              else:
                # Update dimensions of weights to match with mask if possible.
                mask, _, sample_weight = squeeze_or_expand_dimensions(
                    mask, None, sample_weight)
                sample_weight *= mask

            output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)

          if len(self.outputs) > 1:
            # Keep track of the un-aggregated loss result tensor.
            self._compile_metrics_tensors[self.output_names[i] +
                                          '_loss'] = output_loss

            # Keep track of stateful result tensor and function for the loss.
            mean_wrapped_loss = metrics_module.MeanMetricWrapper(
                loss_fn, name=loss_fn.name)
            result_tensor = self._call_metric_fn(mean_wrapped_loss, y_true,
                                                 y_pred, sample_weight, mask)
            self._compile_stateful_metrics_tensors[self.output_names[i] +
                                                   '_loss'] = result_tensor
            self._compile_stateful_metric_functions.append(mean_wrapped_loss)

            self._compile_metrics_names.append(self.output_names[i] + '_loss')
          if total_loss is None:
            total_loss = loss_weight * output_loss
          else:
            total_loss += loss_weight * output_loss
        if total_loss is None:
          if not self.losses:
            raise ValueError('The model cannot be compiled '
                             'because it has no loss to optimize.')
          else:
            total_loss = 0.

        # Add regularization penalties
        # and other layer-specific losses.
        for loss_tensor in self.losses:
          total_loss += loss_tensor

      # Set metric attributes on model.
      self._set_metric_attributes(
          self.outputs,
          skip_target_indices=skip_target_indices,
      )
      # Invoke metric functions for all the outputs.
      self._handle_metrics(
          self.outputs,
          masks=masks,
          targets=self.targets,
          skip_target_indices=skip_target_indices,
          sample_weights=self.sample_weights)

      # Prepare gradient updates and state updates.
      self.total_loss = total_loss

      # Functions for train, test and predict will
      # be compiled lazily when required.
      # This saves time when the user is not using all functions.
      self._function_kwargs = kwargs

      self._fit_function = None
      self._eval_function = None
      self.train_function = None
      self.test_function = None
      self.predict_function = None

      # Collected trainable weights, sorted in topological order.
      trainable_weights = self.trainable_weights
      self._collected_trainable_weights = trainable_weights

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

  @property
  def metrics(self):
    """Returns the model's metrics added using `compile`, `add_metric` APIs."""
    metrics = []
    if self._is_compiled:
      metrics += self._compile_stateful_metric_functions
    return metrics + super(Model, self).metrics

  @property
  def metrics_names(self):
    """Returns the model's display labels for all outputs."""
    metrics_names = []
    if self._is_compiled:
      metrics_names += self._compile_metrics_names  # Includes names of losses.

    # Add metric names from layers.
    for layer in self.layers:
      metrics_names += [m.name for m in layer._metrics]  # pylint: disable=protected-access
    metrics_names += [m.name for m in self._metrics]
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
        return False
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

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset or a dataset iterator. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample weights)`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, dataset
          iterator, generator, or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset, dataset iterators,
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
        verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
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
            not supported when `x` is a dataset, dataset iterator, generator or
           `keras.utils.Sequence` instance.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset or a dataset iterator
            For the first two cases, `batch_size` must be provided.
            For the last case, `validation_steps` must be provided.
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
            supported when `x` is a dataset, dataset iterator, generator, or
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
            the batch size, or 1 if that cannot be determined.
        validation_steps: Only relevant if `validation_data` is provided and
            is a dataset or dataset iterator. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.Container` instance (e.g. list, tuple, etc.). If an
            integer, specifies how many training epochs to run before a new
            validation run is performed, e.g. `validation_freq=2` runs
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
    # Legacy support
    if 'nb_epoch' in kwargs:
      logging.warning(
          'The `nb_epoch` argument in `fit` '
          'has been renamed `epochs`.')
      epochs = kwargs.pop('nb_epoch')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    # When the model expects dictionary inputs (i.e. FeatureColumn-based
    # models), set run_eagerly to True as there's no support for graph
    # functions.
    training_utils.set_run_eagerly_for_dict_structure(self, x)

    # Case 1: distribution strategy.
    if self._distribution_strategy:
      if K.in_multi_worker_mode():
        # Multi-Worker mode runs the Keras training loop on multiple
        # servers via the Distribute Coordinator.
        def _worker_fn(_):
          """Run training inside the distributed coordinator."""
          return training_distributed.fit_distributed(
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
              validation_freq=validation_freq)

        # Independent worker only for now.
        return dc.run_distribute_coordinator(
            _worker_fn,
            self._distribution_strategy,
            mode=dc.CoordinatorMode.INDEPENDENT_WORKER)
      else:
        return training_distributed.fit_distributed(
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
            validation_freq=validation_freq)

    batch_size = self._validate_or_infer_batch_size(
        batch_size, steps_per_epoch, x)

    # Case 2: generator-like. Input is Python generator, or Sequence object,
    # or a non-distributed Dataset or iterator in eager execution.
    if data_utils.is_generator_or_sequence(x):
      training_utils.check_generator_arguments(
          y, sample_weight, validation_split=validation_split)
      return self.fit_generator(
          x,
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
    if training_utils.is_eager_dataset_or_iterator(x):
      # Make sure that y, sample_weights, validation_split are not passed.
      training_utils.validate_dataset_input(x, y, sample_weight,
                                            validation_split)
      if (isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))
          and shuffle):
        training_utils.verify_dataset_shuffled(x)

      return self.fit_generator(
          x,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          verbose=verbose,
          callbacks=callbacks,
          validation_data=validation_data,
          validation_steps=validation_steps,
          validation_freq=validation_freq,
          class_weight=class_weight,
          workers=0,
          shuffle=shuffle,
          initial_epoch=initial_epoch)

    # Case 3: Symbolic tensors or Numpy array-like.
    # This includes Datasets and iterators in graph mode (since they
    # generate symbolic tensors).
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps_per_epoch',
        steps=steps_per_epoch,
        validation_split=validation_split,
        shuffle=shuffle)

    # Prepare validation data.
    if validation_data:
      val_x, val_y, val_sample_weights = self._unpack_validation_data(
          validation_data)
      val_x, val_y, val_sample_weights = self._standardize_user_data(
          val_x,
          val_y,
          sample_weight=val_sample_weights,
          batch_size=batch_size,
          steps=validation_steps,
          steps_name='validation_steps')
    elif validation_split and 0. < validation_split < 1.:
      if training_utils.has_symbolic_tensors(x):
        raise ValueError('If your data is in the form of symbolic tensors, '
                         'you cannot use `validation_split`.')
      if hasattr(x[0], 'shape'):
        split_at = int(x[0].shape[0] * (1. - validation_split))
      else:
        split_at = int(len(x[0]) * (1. - validation_split))
      x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
      y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))
      sample_weights, val_sample_weights = (slice_arrays(
          sample_weights, 0, split_at), slice_arrays(sample_weights, split_at))
    elif validation_steps:
      val_x = []
      val_y = []
      val_sample_weights = []
    else:
      val_x = None
      val_y = None
      val_sample_weights = None

    if self.run_eagerly:
      return training_generator.fit_generator(
          self, (x, y, sample_weights),
          steps_per_epoch=steps_per_epoch,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          callbacks=callbacks,
          validation_data=validation_data,
          validation_steps=validation_steps,
          validation_freq=validation_freq,
          workers=0,
          shuffle=shuffle,
          initial_epoch=initial_epoch,
          steps_name='steps_per_epoch')
    else:
      return training_arrays.fit_loop(
          self,
          x,
          y,
          sample_weights=sample_weights,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          callbacks=callbacks,
          val_inputs=val_x,
          val_targets=val_y,
          val_sample_weights=val_sample_weights,
          shuffle=shuffle,
          initial_epoch=initial_epoch,
          steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps,
          validation_freq=validation_freq,
          steps_name='steps_per_epoch')

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

    Computation is done in batches.

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset or a dataset iterator.
          - A generator or `keras.utils.Sequence` instance.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely).
          If `x` is a dataset, dataset iterator, generator or
          `keras.utils.Sequence` instance, `y` should not be specified (since
          targets will be obtained from the iterator/dataset).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` is your data is in the
            form of symbolic tensors, dataset, dataset iterators,
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
            supported when `x` is a dataset or a dataset iterator, instead pass
            sample weights as the third element of `x`.
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
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
    # Case 1: distribution strategy.
    if self._distribution_strategy:
      if K.in_multi_worker_mode():
        # Multi-Worker mode runs the Keras evaluation loop on multiple
        # servers via the Distribute Coordinator.
        def _worker_fn(_):
          """Run training inside the distributed coordinator."""
          return training_distributed.evaluate_distributed(
              self,
              x=x,
              y=y,
              batch_size=batch_size,
              verbose=verbose,
              sample_weight=sample_weight,
              steps=steps,
              callbacks=callbacks)

        # Independent worker only for now.
        return dc.run_distribute_coordinator(
            _worker_fn,
            self._distribution_strategy,
            mode=dc.CoordinatorMode.INDEPENDENT_WORKER)
      else:
        return training_distributed.evaluate_distributed(
            self,
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks)

    batch_size = self._validate_or_infer_batch_size(batch_size, steps, x)

    # Case 2: generator-like. Input is Python generator, or Sequence object,
    # or a non-distributed Dataset or iterator in eager execution.
    if data_utils.is_generator_or_sequence(x):
      training_utils.check_generator_arguments(y, sample_weight)
      return self.evaluate_generator(
          x,
          steps=steps,
          verbose=verbose,
          callbacks=callbacks,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing)
    if training_utils.is_eager_dataset_or_iterator(x):
      # Make sure that y, sample_weights are not passed.
      training_utils.validate_dataset_input(x, y, sample_weight)
      return training_generator.evaluate_generator(
          self, x,
          steps=steps,
          batch_size=batch_size,
          verbose=verbose,
          workers=0,
          callbacks=callbacks)

    # Case 3: Symbolic tensors or Numpy array-like.
    # This includes Datasets and iterators in graph mode (since they
    # generate symbolic tensors).
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps',
        steps=steps)

    if self.run_eagerly:
      return training_generator.evaluate_generator(
          self, (x, y, sample_weights),
          steps=steps,
          batch_size=batch_size,
          verbose=verbose,
          workers=0,
          callbacks=callbacks)
    else:
      return training_arrays.test_loop(
          self,
          inputs=x,
          targets=y,
          sample_weights=sample_weights,
          batch_size=batch_size,
          verbose=verbose,
          steps=steps,
          callbacks=callbacks)

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

    Computation is done in batches.

    Arguments:
         x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset or a dataset iterator.
          - A generator or `keras.utils.Sequence` instance.
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` is your data is in the
            form of symbolic tensors, dataset, dataset iterators,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.
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
    # Case 1: distribution strategy.
    if self._distribution_strategy:
      return training_distributed.predict_distributed(self,
                                                      x=x,
                                                      batch_size=batch_size,
                                                      verbose=verbose,
                                                      steps=steps,
                                                      callbacks=callbacks)

    batch_size = self._validate_or_infer_batch_size(batch_size, steps, x)

    # Case 2: generator-like. Input is Python generator, or Sequence object,
    # or a non-distributed Dataset or iterator in eager execution.
    if data_utils.is_generator_or_sequence(x):
      return self.predict_generator(
          x,
          steps=steps,
          verbose=verbose,
          callbacks=callbacks,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing)
    if training_utils.is_eager_dataset_or_iterator(x):
      return training_generator.predict_generator(
          self,
          x,
          steps=steps,
          batch_size=batch_size,
          verbose=verbose,
          workers=0,
          callbacks=callbacks)

    # Case 3: Symbolic tensors or Numpy array-like.
    # This includes Datasets and iterators in graph mode (since they
    # generate symbolic tensors).
    x, _, _ = self._standardize_user_data(
        x, check_steps=True, steps_name='steps', steps=steps)

    if self.run_eagerly:
      return training_generator.predict_generator(
          self,
          x,
          steps=steps,
          batch_size=batch_size,
          verbose=verbose,
          workers=0,
          callbacks=callbacks)
    else:
      return training_arrays.predict_loop(
          self,
          x,
          batch_size=batch_size,
          verbose=verbose,
          steps=steps,
          callbacks=callbacks)

  def reset_metrics(self):
    """Resets the state of metrics."""
    if hasattr(self, 'metrics'):
      for m in self.metrics:
        m.reset_states()
      if self._distribution_strategy:
        distributed_training_utils._reset_metrics(self)  # pylint: disable=protected-access

  def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True):
    """Runs a single gradient update on a single batch of data.

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
              (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
          - A `tf.data` dataset or a dataset iterator.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely). If
          `x` is a dataset or a dataset iterator, `y` should not be specified
          (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case of
          temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample. In this case you should make sure to specify
          sample_weight_mode="temporal" in compile(). This argument is not
          supported when `x` is a dataset or a dataset iterator.
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
    if self._distribution_strategy:
      raise NotImplementedError('`train_on_batch` is not supported for models '
                                'compiled with DistributionStrategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, class_weight=class_weight,
        extract_tensors_from_dataset=True)

    if self.run_eagerly:
      outputs = training_eager.train_on_batch(
          self, x, y, sample_weights=sample_weights)
    else:
      if not isinstance(K.symbolic_learning_phase(), int):
        ins = x + y + sample_weights + [True]
      else:
        ins = x + y + sample_weights

      if reset_metrics:
        self._make_train_function()
        outputs = self.train_function(ins)  # pylint: disable=not-callable
      else:
        self._make_fit_function()
        outputs = self._fit_function(ins)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
    """Test the model on a single batch of samples.

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset or a dataset iterator.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset or a
          dataset iterator, `y` should not be specified
          (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile(). This argument is not
            supported when `x` is a dataset or a dataset iterator.
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
    if self._distribution_strategy:
      raise NotImplementedError('`test_on_batch` is not supported for models '
                                'compiled with DistributionStrategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, extract_tensors_from_dataset=True)

    if self.run_eagerly:
      outputs = training_eager.test_on_batch(
          self, x, y, sample_weights=sample_weights)
    else:
      inputs = x + y + sample_weights
      if reset_metrics:
        self._make_test_function()
        outputs = self.test_function(inputs)  # pylint: disable=not-callable
      else:
        self._make_eval_function()
        outputs = self._eval_function(inputs)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Arguments:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset or a dataset iterator.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between given number of inputs and
          expectations of the model.
    """
    if self._distribution_strategy:
      raise NotImplementedError('`predict_on_batch` is not supported for '
                                'models compiled with DistributionStrategy.')
    # Validate and standardize user data.
    inputs, _, _ = self._standardize_user_data(
        x, extract_tensors_from_dataset=True)
    if self.run_eagerly:
      if (isinstance(inputs, iterator_ops.EagerIterator) or
          (isinstance(inputs, dataset_ops.DatasetV2))):
        inputs = training_utils.cast_if_floating_dtype(inputs)
      elif isinstance(inputs, collections.Sequence):
        inputs = [
            ops.convert_to_tensor(val, dtype=K.floatx()) for val in inputs]

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

    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.

    The use of `keras.utils.Sequence` guarantees the ordering
    and guarantees the single use of every input per epoch when
    using `use_multiprocessing=True`.

    Arguments:
        generator: A generator or an instance of `Sequence`
          (`keras.utils.Sequence`)
            object in order to avoid duplicate data
            when using multiprocessing.
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.
            This tuple (a single output of the generator) makes a single batch.
            Therefore, all arrays in this tuple must have the same length (equal
            to the size of this batch). Different batches may have different
              sizes.
            For example, the last batch of the epoch is commonly smaller than
              the
            others, if the size of the dataset is not divisible by the batch
              size.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of samples of your dataset
            divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        epochs: Integer, total number of iterations on the data.
        verbose: Verbosity mode, 0, 1, or 2.
        callbacks: List of callbacks to be called during training.
        validation_data: This can be either
            - a generator for the validation data
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if `validation_data`
            is a generator. Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(validation_data)` as a number of steps.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.Container` instance (e.g. list, tuple, etc.). If an
            integer, specifies how many training epochs to run before a new
            validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        class_weight: Dictionary mapping class indices to a weight
            for the class.
        max_queue_size: Integer. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Maximum number of processes to spin up
            when using process-based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean.
            If `True`, use process-based threading.
            If unspecified, `use_multiprocessing` will default to `False`.
            Note that because this implementation relies on multiprocessing,
            you should not pass non-picklable arguments to the generator
            as they can't be passed easily to children processes.
        shuffle: Boolean. Whether to shuffle the order of the batches at
            the beginning of each epoch. Only used with instances
            of `Sequence` (`keras.utils.Sequence`).
            Has no effect when `steps_per_epoch` is not `None`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)

    Returns:
        A `History` object.

    Example:

    ```python
        def generate_arrays_from_file(path):
            while 1:
                f = open(path)
                for line in f:
                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    x1, x2, y = process_line(line)
                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                f.close()

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
    ```
    Raises:
        ValueError: In case the generator yields data in an invalid format.
    """
    if self._distribution_strategy:
      raise NotImplementedError('`fit_generator` is not supported for '
                                'models compiled with DistributionStrategy.')
    return training_generator.fit_generator(
        self,
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
        initial_epoch=initial_epoch,
        steps_name='steps_per_epoch')

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         verbose=0):
    """Evaluates the model on a data generator.

    The generator should return the same kind of data
    as accepted by `test_on_batch`.

    Arguments:
        generator: Generator yielding tuples (inputs, targets)
            or (inputs, targets, sample_weights)
            or an instance of `keras.utils.Sequence`
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during evaluation.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: maximum size for the generator queue
        workers: Integer. Maximum number of processes to spin up
            when using process-based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean.
            If `True`, use process-based threading.
            If unspecified, `use_multiprocessing` will default to `False`.
            Note that because this implementation relies on multiprocessing,
            you should not pass non-picklable arguments to the generator
            as they can't be passed easily to children processes.
        verbose: Verbosity mode, 0 or 1.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.

    Raises:
        ValueError: In case the generator yields data in an invalid format.
    """
    if self._distribution_strategy:
      raise NotImplementedError('`evaluate_generator` is not supported for '
                                'models compiled with DistributionStrategy.')
    return training_generator.evaluate_generator(
        self,
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

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: Generator yielding batches of input samples
            or an instance of `keras.utils.Sequence` object in order to
            avoid duplicate data when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during prediction.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Maximum size for the generator queue.
        workers: Integer. Maximum number of processes to spin up
            when using process-based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean.
            If `True`, use process-based threading.
            If unspecified, `use_multiprocessing` will default to `False`.
            Note that because this implementation relies on multiprocessing,
            you should not pass non-picklable arguments to the generator
            as they can't be passed easily to children processes.
        verbose: verbosity mode, 0 or 1.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case the generator yields data in an invalid format.
    """
    if self._distribution_strategy:
      raise NotImplementedError('`predict_generator` is not supported for '
                                'models compiled with DistributionStrategy.')
    return training_generator.predict_generator(
        self,
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,
        callbacks=callbacks)

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

    Arguments:
      batch_size: The batch_size provided as an argument to
        fit/evaluate/predict.
      steps: The steps provided as an argument to fit/evaluate/predict.
      x: The data passed as `x` to fit/evaluate/predict.

    Returns:
      The validated batch_size, auto-inferred from the first layer if not
      provided.
    """
    if batch_size is not None and isinstance(x, dataset_ops.DatasetV2):
      raise ValueError('The `batch_size` argument must not be specified when'
                       ' using dataset as an input.')

    layers = super(Model, self).layers  # Avoids the override in Sequential.
    if layers:
      first_layer = layers[0]
      static_batch_size = training_utils.get_static_batch_size(first_layer)
      if static_batch_size is not None:

        # Check `batch_size` argument is consistent with InputLayer.
        if batch_size is not None and batch_size != static_batch_size:
          raise ValueError('The `batch_size` argument value {} is incompatible '
                           'with the specified batch size of your Input Layer: '
                           '{}'.format(batch_size, static_batch_size))

        # Check Dataset/Iterator batch size is consistent with InputLayer.
        if isinstance(x, (dataset_ops.DatasetV2, iterator_ops.Iterator,
                          iterator_ops.EagerIterator)):
          ds_batch_size = tensor_shape.as_dimension(
              nest.flatten(x.output_shapes)[0][0]).value
          if ds_batch_size is not None and ds_batch_size != static_batch_size:
            raise ValueError('The batch output shape of your `Dataset` is {}, '
                             'which is incompatible with the specified batch '
                             'size of your Input Layer: {}'.format(
                                 ds_batch_size, static_batch_size))

        # Set inferred batch size from the InputLayer.
        if steps is None:
          batch_size = static_batch_size

    if batch_size is None and steps is None:
      # Backwards compatibility
      batch_size = 32
    return batch_size

  @property
  def _default_save_signature(self):
    return saving_utils.trace_model_call(self)

  def _set_sample_weight_attributes(self, sample_weight_mode,
                                    skip_target_weighing_indices):
    """Sets sample weight related attributes on the model."""
    sample_weights, sample_weight_modes = training_utils.prepare_sample_weights(
        self.output_names, sample_weight_mode, skip_target_weighing_indices)
    self.sample_weights = sample_weights
    self.sample_weight_modes = sample_weight_modes
    self._feed_sample_weight_modes = [
        sample_weight_modes[i]
        for i in range(len(self.outputs))
        if i not in skip_target_weighing_indices
    ]
    self._feed_sample_weights = [
        sample_weights[i]
        for i in range(len(sample_weights))
        if i not in skip_target_weighing_indices
    ]

  def _cache_output_metric_attributes(self, metrics, weighted_metrics):
    """Caches metric name and function attributes for every model output."""
    output_shapes = []
    for output in self.outputs:
      if output is None or output.shape.rank is None:
        output_shapes.append(None)
      else:
        output_shapes.append(output.shape.as_list())
    self._per_output_metrics = training_utils.collect_per_output_metric_info(
        metrics, self.output_names, output_shapes, self.loss_functions)
    self._per_output_weighted_metrics = \
        training_utils.collect_per_output_metric_info(
            weighted_metrics, self.output_names, output_shapes,
            self.loss_functions, self.sample_weights)

  def _add_unique_metric_name(self, metric_name, output_index):
    """Makes the metric name unique and adds it to the model's metric name list.

      If there are multiple outputs for which the metrics are calculated, the
      metric names have to be made unique by appending an integer.

    Arguments:
      metric_name: Metric name that corresponds to the metric specified by the
          user. For example: 'acc'.
      output_index: The index of the model output for which the metric name is
        being added.

    Returns:
      string, name of the model's unique metric name
    """
    if len(self.output_names) > 1:
      metric_name = '%s_%s' % (self.output_names[output_index], metric_name)
    j = 1
    base_metric_name = metric_name
    while metric_name in self._compile_metrics_names:
      metric_name = '%s_%d' % (base_metric_name, j)
      j += 1

    return metric_name

  @property
  def _all_metrics_tensors(self):
    """Returns the network's symbolic metric tensors."""
    metrics_tensors = {}
    if self._is_compiled:
      metrics_tensors.update(self._compile_metrics_tensors)
    metrics_tensors.update(super(Model, self)._all_metrics_tensors)
    return metrics_tensors

  @property
  def _all_stateful_metrics_tensors(self):
    """Returns the network's symbolic metric tensors."""
    metrics_tensors = {}
    if self._is_compiled:
      metrics_tensors.update(self._compile_stateful_metrics_tensors)
    metrics_tensors.update(super(Model, self)._all_metrics_tensors)
    return metrics_tensors

  def _init_metric_attributes(self):
    """Initialized model metric attributes."""
    # List of all metric names in the model.
    self._compile_metrics_names = ['loss']
    # List of stateful metric functions. Used for resetting metric state during
    # training/eval.
    # This includes loss functions when there are multiple outputs.
    self._compile_stateful_metric_functions = []
    # Dict of all aggregated metric result tensors. This includes aggregated
    # loss result tensors when there are multiple outputs.
    self._compile_stateful_metrics_tensors = {}
    # Dict of all metric result tensors (aggregated or not - based on the
    # values given in compile.). This includes aggregated loss result tensors
    # when there are multiple outputs.
    self._compile_metrics_tensors = {}

  def _set_per_output_metric_attributes(self, metrics_dict, output_index):
    """Sets the metric attributes on the model for the given output.

    Arguments:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      output_index: The index of the model output for which the metric
        attributes are added.

    Returns:
      Metrics dict updated with unique metric names as keys.
    """
    updated_metrics_dict = collections.OrderedDict()
    for metric_name, (metric_fn, stateful_metric_fn) in metrics_dict.items():
      metric_name = self._add_unique_metric_name(metric_name, output_index)
      updated_metrics_dict[metric_name] = (metric_fn, stateful_metric_fn)
      # Keep track of metric name, function and stateful function.
      self._compile_metrics_names.append(metric_name)
      self._compile_stateful_metric_functions.append(stateful_metric_fn)
    return updated_metrics_dict

  def _set_metric_attributes(self, outputs, skip_target_indices=None):
    """Sets the metric attributes on the model for all the model outputs."""
    skip_target_indices = skip_target_indices or []
    updated_per_output_metrics = []
    updated_per_output_weighted_metrics = []
    for i in range(len(outputs)):
      if i in skip_target_indices:
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

    self._per_output_metrics = updated_per_output_metrics
    self._per_output_weighted_metrics = updated_per_output_weighted_metrics

  def _call_metric_fn(self, fn, y_true, y_pred, weights, mask):
    """Helper function to call metric function with distribution strategy."""
    # TODO(b/120571621): We want to avoid metric reductions here since
    # since TPUStrategy does not implement replica local variables.
    # Remove this hack once we support TPUReplicaLocalVariables.
    is_tpu = distributed_training_utils.is_tpu_strategy(
        self._distribution_strategy)
    if ((not is_tpu) and self._distribution_strategy and
        distribution_strategy_context.in_cross_replica_context()):
      with self._distribution_strategy.scope():
        return self._distribution_strategy.extended.call_for_each_replica(
            training_utils.call_metric_function,
            (fn, y_true, y_pred, weights, mask))
    return training_utils.call_metric_function(
        fn, y_true, y_pred, weights=weights, mask=mask)

  def _handle_per_output_metrics(self,
                                 metrics_dict,
                                 y_true,
                                 y_pred,
                                 mask,
                                 weights=None,
                                 return_stateful_result=True):
    """Calls metric functions for a single output.

    Arguments:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      y_true: Target output.
      y_pred: Predicted output.
      mask: Computed mask value for the current output.
      weights: Weights to be applied on the current output.
      return_stateful_result: Boolean, indicates whether the stateful
        (aggregated)/stateless metric result should be returned.

    Returns:
      A list of metric result tensors.
    """
    metric_results = []
    for metric_name, (metric_fn, stateful_fn) in metrics_dict.items():
      with K.name_scope(metric_name):

        def _call_stateful_fn(fn):
          """Create stateful metrics correctly."""
          return self._call_metric_fn(fn, y_true, y_pred, weights, mask)

        def _call_stateless_fn(fn):
          weighted_metric_fn = training_utils.weighted_masked_objective(fn)
          return weighted_metric_fn(y_true, y_pred, weights=weights, mask=mask)

        def _track_metric_tensors(name, stateless_result, stateful_result):
          self._compile_metrics_tensors[name] = stateless_result
          self._compile_stateful_metrics_tensors[name] = stateful_result

        if isinstance(metric_fn, metrics_module.Metric):
          # If the given metric fn is stateful, call the fn and return result.
          metric_result = _call_stateful_fn(metric_fn)
          metric_results.append(metric_result)
          if not self.run_eagerly:
            _track_metric_tensors(metric_name, metric_result, metric_result)
        elif self.run_eagerly:
          # In eager mode, if the given metric fn is not stateful, we invoke the
          # given fn or its stateful version based on the given flag.
          if return_stateful_result:
            metric_result = _call_stateful_fn(stateful_fn)
          else:
            metric_result = _call_stateless_fn(metric_fn)
          metric_results.append(metric_result)
        else:
          # In graph mode, we build the sub-graph for both the stateful and the
          # stateless fns.
          stateful_metric_result = _call_stateful_fn(stateful_fn)
          metric_result = _call_stateless_fn(metric_fn)
          _track_metric_tensors(metric_name, metric_result,
                                stateful_metric_result)

    return metric_results

  def _handle_metrics(self,
                      outputs,
                      skip_target_indices=None,
                      targets=None,
                      sample_weights=None,
                      masks=None,
                      return_stateful_result=True):
    """Handles calling metric functions.

    Arguments:
      outputs: List of outputs (predictions).
      skip_target_indices: Optional. List of target ids to skip.
      targets: List of targets.
      sample_weights: Optional list of sample weight arrays.
      masks: List of computed output mask values.
      return_stateful_result: Boolean, indicates whether the stateful
        (aggregated)/stateless metric result should be returned.

    Returns:
      A list of metric result tensors.
    """
    skip_target_indices = skip_target_indices or []
    metric_results = []
    with K.name_scope('metrics'):
      # Invoke all metrics added using `compile`.
      for i in range(len(outputs)):
        if i in skip_target_indices:
          continue
        output = outputs[i] if outputs else None
        target = targets[i] if targets else None
        output_mask = masks[i] if masks else None
        metric_results.extend(
            self._handle_per_output_metrics(
                self._per_output_metrics[i],
                target,
                output,
                output_mask,
                return_stateful_result=return_stateful_result))
        metric_results.extend(
            self._handle_per_output_metrics(
                self._per_output_weighted_metrics[i],
                target,
                output,
                output_mask,
                weights=sample_weights[i],
                return_stateful_result=return_stateful_result))

    # Add metric results from the `add_metric` metrics in eager mode.
    if context.executing_eagerly():
      for m in self.metrics:
        if m not in self._compile_stateful_metric_functions:
          metric_results.append(m.result())
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

  def _make_train_function_helper(self, fn_name, outputs, metric_updates=None):
    if not self._is_compiled:
      raise RuntimeError('You must compile your model before using it.')
    self._check_trainable_weights_consistency()
    if getattr(self, fn_name) is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if not isinstance(K.symbolic_learning_phase(), int):
        inputs += [K.symbolic_learning_phase()]

      with K.get_graph().as_default():
        with K.name_scope('training'):
          with K.name_scope(self.optimizer.__class__.__name__):
            # Training updates
            updates = self.optimizer.get_updates(
                params=self._collected_trainable_weights, loss=self.total_loss)
      # Unconditional updates
      updates += self.get_updates_for(None)
      # Conditional updates relevant to this model
      updates += self.get_updates_for(self.inputs)
      # Add stateful metrics updates.
      if metric_updates is not None:
        updates += metric_updates

      with K.name_scope('training'):
        # Gets loss and metrics. Updates weights at each call.
        fn = K.function(
            inputs,
            outputs,
            updates=updates,
            name='train_function',
            **self._function_kwargs)
        setattr(self, fn_name, fn)

  def _make_train_function(self):
    metrics_tensors = [
        self._all_metrics_tensors[m] for m in self.metrics_names[1:]
    ]
    self._make_train_function_helper('train_function',
                                     [self.total_loss] + metrics_tensors)

  def _make_fit_function(self):
    metrics_tensors = [
        self._all_stateful_metrics_tensors[m] for m in self.metrics_names[1:]
    ]
    self._make_train_function_helper(
        '_fit_function', [self.total_loss] + metrics_tensors)

  def _make_test_function_helper(self, fn_name, outputs, metric_updates=None):
    if not self._is_compiled:
      raise RuntimeError('You must compile your model before using it.')
    if getattr(self, fn_name) is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)

      with K.name_scope('evaluation'):
        updates = self.state_updates
        # Add stateful metrics updates.
        if metric_updates is not None:
          updates += metric_updates
        # Return loss and metrics, no gradient updates.
        # Does update the network states.
        fn = K.function(
            inputs,
            outputs,
            updates=updates,
            name='test_function',
            **self._function_kwargs)
        setattr(self, fn_name, fn)

  def _make_test_function(self):
    metrics_tensors = [
        self._all_metrics_tensors[m] for m in self.metrics_names[1:]
    ]
    self._make_test_function_helper('test_function',
                                    [self.total_loss] + metrics_tensors)

  def _make_eval_function(self):
    metrics_tensors = [
        self._all_stateful_metrics_tensors[m] for m in self.metrics_names[1:]
    ]
    self._make_test_function_helper(
        '_eval_function', [self.total_loss] + metrics_tensors)

  def _make_predict_function(self):
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      kwargs = getattr(self, '_function_kwargs', {})
      with K.name_scope(ModeKeys.PREDICT):
        self.predict_function = K.function(
            inputs,
            self.outputs,
            updates=self.state_updates,
            name='predict_function',
            **kwargs)

  def _make_execution_function(self, mode):
    if mode == ModeKeys.TRAIN:
      self._make_fit_function()
      return self._fit_function
    if mode == ModeKeys.TEST:
      self._make_eval_function()
      return self._eval_function
    if mode == ModeKeys.PREDICT:
      self._make_predict_function()
      return self.predict_function

  def _distribution_standardize_user_data(self,
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
                                          repeat=True,
                                          allow_partial_batch=False):
    """Runs validation checks on input and target data passed by the user.

    This is called when using DistributionStrategy to train, evaluate or serve
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
      check_steps: boolean, True if we want to check for validity of `steps` and
        False, otherwise.
      steps_name: The public API's parameter name for `steps`.
      steps: Integer or `None`. Total number of steps (batches of samples) to
        execute.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      repeat: Boolean whether to repeat the numpy training data when converting
        to training dataset.
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
                                'when using DistributionStrategy.')

    if (sample_weight is not None and sample_weight.all() and
        distributed_training_utils.is_tpu_strategy(
            self._distribution_strategy)):
      raise NotImplementedError('`sample_weight` is currently not supported '
                                'when using TPUStrategy.')

    if (self.stateful and distributed_training_utils.is_tpu_strategy(
        self._distribution_strategy) and self._distribution_strategy.
        num_replicas_in_sync != 1):
      raise ValueError('Single core must be used for computation on '
                       'stateful models. Consider adding `device_assignment` '
                       'parameter to TPUStrategy using\n'
                       'topology = tf.contrib.distribute.'
                       'initialize_tpu_system()\n'
                       'device_assignment = tf.contrib.tpu.DeviceAssignment('
                       'topology, core_assignment=tf.contrib.tpu.'
                       'SINGLE_CORE_ASSIGNMENT)\n'
                       'tpu_strategy = tf.contrib.distribute.TPUStrategy('
                       'device_assignment=device_assignment)')

    # Validates `steps` and `shuffle` arguments right at the beginning
    # since we use it to construct the dataset object.
    # TODO(anjalisridhar): Remove this check once we refactor the
    # _standardize_user_data code path. This check is already present elsewhere
    # in the codebase.
    if isinstance(x, dataset_ops.DatasetV2):
      if shuffle:
        training_utils.verify_dataset_shuffled(x)

      if check_steps and steps is None:
        raise ValueError('When using Datasets as input, '
                         'you should specify the `{steps_name}` argument.'
                         .format(steps_name=steps_name))

    if ops.executing_eagerly_outside_functions():
      session = None
    else:
      session = K.get_session()

    strategy = self._distribution_strategy
    with strategy.scope():
      first_x_value = nest.flatten(x)[0]
      if isinstance(first_x_value, np.ndarray):
        x = distributed_training_utils.list_to_tuple(x)
        if y is not None:
          y = distributed_training_utils.list_to_tuple(y)
          if sample_weight is not None:
            sample_weight = distributed_training_utils.list_to_tuple(
                sample_weight)
            in_tuple = (x, y, sample_weight)
          else:
            in_tuple = (x, y)
        else:
          in_tuple = x

        if shuffle:
          # 1024 is a good buffer size since it is much larger than the average
          # batch size provided by the user and provides sufficient randomness.
          # One thing to keep in mind is the memory usage based on the size of
          # each sample.
          shuffle_buffer = 1024
        else:
          shuffle_buffer = None
        ds = strategy.extended.experimental_make_numpy_dataset(in_tuple,
                                                               session=session)
        if shuffle_buffer:
          ds = ds.shuffle(shuffle_buffer)
        if repeat:
          ds = ds.repeat()

        # We need to use the drop_remainder argument to get a known static
        # input shape which is required for TPUs.
        drop_remainder = (not allow_partial_batch and
                          strategy.extended.experimental_require_static_shapes)
        x = ds.batch(batch_size, drop_remainder=drop_remainder)
      else:
        assert isinstance(x, dataset_ops.DatasetV2)
        training_utils.validate_dataset_input(x, y, sample_weight,
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
        - A `tf.data` dataset or a dataset iterator.
      y: Target data. Like the input data `x`,
        it could be either Numpy array(s) or TensorFlow tensor(s).
        It should be consistent with `x` (you cannot have Numpy inputs and
        tensor targets, or inversely). If `x` is a dataset or a
        dataset iterator, `y` should not be specified
        (since targets will be obtained from the iterator).
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
      training_utils.validate_dataset_input(x, y, sample_weight,
                                            validation_split)
      if shuffle:
        training_utils.verify_dataset_shuffled(x)

      is_dataset = True
      if extract_tensors_from_dataset:
        # We do this for `train_on_batch`/etc.
        x, y, sample_weight = training_utils.extract_tensors_from_dataset(x)
    elif isinstance(x, iterator_ops.Iterator):
      # Graph mode iterator. We extract the symbolic tensors.
      training_utils.validate_dataset_input(x, y, sample_weight,
                                            validation_split)
      iterator = x
      x, y, sample_weight = training_utils.unpack_iterator_input(iterator)
      is_dataset = True
    else:
      is_dataset = False

    # Validates `steps` argument based on x's type.
    if check_steps:
      training_utils.check_steps_argument(x, steps, steps_name)

    # First, we build/compile the model on the fly if necessary.
    all_inputs = []
    is_build_called = False
    is_compile_called = False
    # Whether this is a subclassed model that expects dictionary inputs
    # rather than list inputs (e.g. FeatureColumn-based models).
    dict_inputs = False
    if not self.inputs:
      # We need to use `x_input` to set the model inputs.

      # If input data is a dataset iterator in graph mode or if it is an eager
      # iterator and only one batch of samples is required, we fetch the data
      # tensors from the iterator and then standardize them.
      if isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2)):
        x_input, y_input, _ = training_utils.extract_tensors_from_dataset(x)
      else:
        x_input = x
        y_input = y
      # We type-check that `x_input` and `y_input` are either single arrays
      # or lists of arrays.
      if isinstance(x_input, (list, tuple)):
        if not all(isinstance(v, np.ndarray) or
                   tensor_util.is_tensor(v) for v in x_input):
          raise ValueError('Please provide as model inputs either a single '
                           'array or a list of arrays. You passed: x=' + str(x))
        all_inputs += list(x_input)
      elif isinstance(x_input, dict):
        dict_inputs = True
        keys = sorted(x_input.keys())
        all_inputs = [x_input[k] for k in keys]
      else:
        if (not isinstance(x_input, np.ndarray) and
            not tensor_util.is_tensor(x_input)):
          raise ValueError('Please provide as model inputs either a single '
                           'array or a list of arrays. You passed: x=' + str(x))
        all_inputs.append(x_input)

      # Build the model using the retrieved inputs (value or symbolic).
      # If values or generated from a dataset, then in symbolic-mode
      # placeholders will be created to match the value shapes.
      is_build_called = True
      if is_dataset:
        cast_inputs = nest.map_structure(lambda v: v.shape, x_input)
      elif training_utils.has_tensors(x_input):
        cast_inputs = training_utils.cast_if_floating_dtype(x_input)
      else:
        cast_inputs = x_input
      self._set_inputs(cast_inputs)
    else:
      y_input = y
      dict_inputs = isinstance(self.inputs, dict)

    if y_input is not None:
      if not self.optimizer:
        raise RuntimeError('You must compile a model before '
                           'training/testing. '
                           'Use `model.compile(optimizer, loss)`.')
      if not self._is_compiled:
        # On-the-fly compilation of the model.
        # We need to use `y` to set the model targets.
        if training_utils.has_tensors(y_input):
          y_input = training_utils.cast_if_floating_dtype(y_input)
        if isinstance(y_input, (list, tuple)):
          if not all(isinstance(v, np.ndarray) or
                     tensor_util.is_tensor(v) for v in y_input):
            raise ValueError('Please provide as model targets either a single '
                             'array or a list of arrays. '
                             'You passed: y=' + str(y))
          all_inputs += list(y_input)
        elif isinstance(y_input, dict):
          raise ValueError('You cannot pass a dictionary as model targets.')
        else:
          if (not isinstance(y_input, np.ndarray) and
              not tensor_util.is_tensor(y_input)):
            raise ValueError('Please provide as model targets either a single '
                             'array or a list of arrays. '
                             'You passed: y=' + str(y))
          all_inputs.append(y_input)

        # Typecheck that all inputs are *either* value *or* symbolic.
        # TODO(fchollet): this check could be removed in Eager mode?
        if any(tensor_util.is_tensor(v) for v in all_inputs):
          if not all(tensor_util.is_tensor(v) for v in all_inputs):
            raise ValueError('Do not pass inputs that mix Numpy arrays and '
                             'TensorFlow tensors. '
                             'You passed: x=' + str(x) + '; y=' + str(y))

        if is_dataset or context.executing_eagerly():
          target_tensors = None
        else:
          # Handle target tensors if any passed.
          if not isinstance(y_input, (list, tuple)):
            y_input = [y_input]
          target_tensors = [v for v in y_input if _is_symbolic_tensor(v)]
        is_compile_called = True
        self.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self._compile_metrics,
            weighted_metrics=self._compile_weighted_metrics,
            loss_weights=self.loss_weights,
            target_tensors=target_tensors,
            run_eagerly=self.run_eagerly)

    # In graph mode, if we had just set inputs and targets as symbolic tensors
    # by invoking build and compile on the model respectively, we do not have to
    # feed anything to the model. Model already has input and target data as
    # part of the graph.
    # Note: in this case, `any` and `all` are equivalent since we disallow
    # mixed symbolic/value inputs.
    if (not self.run_eagerly and is_build_called and is_compile_called and
        not is_dataset  and any(_is_symbolic_tensor(v) for v in all_inputs)):
      return [], [], []

    # What follows is input validation and standardization to list format,
    # in the case where all inputs are value arrays.

    if self.run_eagerly:
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
      x = training_utils.standardize_input_data(
          x,
          feed_input_names,
          feed_input_shapes,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='input')

    if y is not None:
      if not self._is_graph_network:
        feed_output_names = self._feed_output_names
        feed_output_shapes = None
        # Sample weighting not supported in this case.
        # TODO(fchollet): consider supporting it.
        feed_sample_weight_modes = [None for _ in self.outputs]
      else:
        feed_output_names = self._feed_output_names
        feed_sample_weight_modes = self._feed_sample_weight_modes
        feed_output_shapes = []
        for output_shape, loss_fn in zip(self._feed_output_shapes,
                                         self._feed_loss_fns):
          if ((isinstance(loss_fn, losses.LossFunctionWrapper) and
               loss_fn.fn == losses.sparse_categorical_crossentropy)) or (
                   isinstance(loss_fn, losses.SparseCategoricalCrossentropy)):
            if K.image_data_format() == 'channels_first':
              feed_output_shapes.append(
                  (output_shape[0], 1) + output_shape[2:])
            else:
              feed_output_shapes.append(output_shape[:-1] + (1,))
          elif (not isinstance(loss_fn, losses.Loss) or
                (isinstance(loss_fn, losses.LossFunctionWrapper) and
                 (getattr(losses, loss_fn.fn.__name__, None) is None))):
            # If the given loss is not an instance of the `Loss` class (custom
            # class) or if the loss function that is wrapped is not in the
            # `losses` module, then it is a user-defined loss and we make no
            # assumptions about it.
            feed_output_shapes.append(None)
          else:
            feed_output_shapes.append(output_shape)

      # Standardize the outputs.
      y = training_utils.standardize_input_data(
          y,
          feed_output_names,
          # Don't enforce target shapes to match output shapes.
          # Precise checks will be run in `check_loss_and_target_compatibility`.
          shapes=None,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='target')

      # Generate sample-wise weight values given the `sample_weight` and
      # `class_weight` arguments.
      sample_weights = training_utils.standardize_sample_weights(
          sample_weight, feed_output_names)
      class_weights = training_utils.standardize_class_weights(
          class_weight, feed_output_names)
      sample_weights = [
          training_utils.standardize_weights(ref, sw, cw, mode)
          for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights,
                                         feed_sample_weight_modes)
      ]
      # Check that all arrays have the same length.
      if not self._distribution_strategy:
        training_utils.check_array_lengths(x, y, sample_weights)
        if self._is_graph_network and not self.run_eagerly:
          # Additional checks to avoid users mistakenly using improper loss fns.
          training_utils.check_loss_and_target_compatibility(
              y, self._feed_loss_fns, feed_output_shapes)
    else:
      y = []
      sample_weights = []

    if self.stateful and batch_size:
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

  def _unpack_validation_data(self, validation_data):
    if (isinstance(validation_data, (iterator_ops.Iterator,
                                     iterator_ops.EagerIterator,
                                     dataset_ops.DatasetV2))):
      val_x = validation_data
      val_y = None
      val_sample_weight = None
    elif len(validation_data) == 2:
      val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
      val_sample_weight = None
    elif len(validation_data) == 3:
      val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
    else:
      raise ValueError(
          'When passing a `validation_data` argument, '
          'it must contain either 2 items (x_val, y_val), '
          'or 3 items (x_val, y_val, val_sample_weights), '
          'or alternatively it could be a dataset or a '
          'dataset or a dataset iterator. '
          'However we received `validation_data=%s`' % validation_data)
    return val_x, val_y, val_sample_weight

  @checkpointable.no_automatic_dependency_tracking
  def _set_inputs(self, inputs, outputs=None, training=None):
    """Set model's input and output specs based on the input data received.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    Args:
      inputs: Single array, or list of arrays. The arrays could be placeholders,
        Numpy arrays, data tensors, or TensorShapes.
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
    if self.inputs:
      raise ValueError('Model inputs are already set.')

    if self.__class__.__name__ == 'Sequential' and not self.built:
      if tensor_util.is_tensor(inputs):
        input_shape = (None,) + tuple(inputs.shape.as_list()[1:])
      elif isinstance(inputs, tensor_shape.TensorShape):
        input_shape = (None,) + tuple(inputs.as_list()[1:])
      elif isinstance(inputs, dict):
        # We assert that the first layer is a FeatureLayer.
        if not training_utils.is_feature_layer(self.layers[0]):
          raise ValueError('Passing a dictionary input to a Sequential Model '
                           'which doesn\'t have FeatureLayer as the first layer'
                           ' is an error.')
        input_shape = (None,)
      else:
        input_shape = (None,) + tuple(inputs.shape[1:])
      self._build_input_shape = input_shape

    # On-the-fly setting of symbolic model inputs (either by using the tensor
    # provided, or by creating a placeholder if Numpy data was provided).
    model_inputs = training_utils.ModelInputs(inputs)
    inputs = model_inputs.get_symbolic_inputs()
    self.inputs = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.input_names = model_inputs.get_input_names()

    self._feed_inputs = []
    self._feed_input_names = []
    self._feed_input_shapes = []

    for k, v in model_inputs.as_dict():
      if K.is_placeholder(v):
        self._feed_input_names.append(k)
        self._feed_inputs.append(v)
        self._feed_input_shapes.append(K.int_shape(v))

    # TODO(fchollet): consider calling `_maybe_build` before calling the model.
    if outputs is None:
      if not self._dynamic:
        # The network may include dynamic layers but its `call`
        # itself isn't dynamic.
        # Obtain symbolic outputs by calling the model.
        with K.get_graph().as_default():
          if self._expects_training_arg:
            outputs = self.call(inputs, training=training)
          else:
            outputs = self.call(inputs)
      else:
        # Case: network's `call` is dynamic.
        try:
          outputs = self._symbolic_call(inputs)
        except NotImplementedError:
          # Static shape inference was not implemented for this dynamic net.
          # Do not specify symbolic outputs.
          outputs = None

    outputs = nest.flatten(outputs)
    self.outputs = outputs
    self.output_names = training_utils.generic_output_names(outputs)
    self.built = True


class DistributedCallbackModel(Model):
  """Model that is used for callbacks with DistributionStrategy."""

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
    distributed_training_utils.set_weights(
        self._original_model._distribution_strategy, self,  # pylint: disable=protected-access
        orig_model_weights)

  def __getattr__(self, item):
    # Whitelisted atttributes of the model that can be accessed by the user
    # during a callback.
    if item not in ['_setattr_tracking']:
      logging.warning('You are accessing attribute ' + item + ' of the '
                      'DistributedCallbackModel that may not have been set '
                      'correctly.')


def _is_symbolic_tensor(x):
  return tensor_util.is_tensor(x) and not isinstance(x, ops.EagerTensor)
