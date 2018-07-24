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

import weakref
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training_arrays
from tensorflow.python.keras.engine import training_eager
from tensorflow.python.keras.engine import training_generator
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer as tf_optimizer_module
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.models.Model', 'keras.Model')
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
    # Create a cache for iterator get_next op.
    self._iterator_get_next = weakref.WeakKeyDictionary()
    # Create a cache for dataset - uninitialized iterators
    self._dataset_iterator_cache = weakref.WeakKeyDictionary()

  @checkpointable.no_automatic_dependency_tracking
  def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: String (name of optimizer) or optimizer instance.
            See [optimizers](/optimizers).
        loss: String (name of objective function) or objective function.
            See [losses](/losses).
            If the model has multiple outputs, you can use a different loss
            on each output by passing a dictionary or a list of losses.
            The loss value that will be minimized by the model
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
        **kwargs: These arguments are passed to `tf.Session.run`.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    loss = loss or {}
    if context.executing_eagerly() and not isinstance(
        optimizer, (tf_optimizer_module.Optimizer, optimizers.TFOptimizer)):
      raise ValueError('Only TF native optimizers are supported in Eager mode.')

    self.optimizer = optimizers.get(optimizer)
    # We've disabled automatic dependency tracking for this method, but do want
    # to add a checkpoint dependency on the optimizer if it's checkpointable.
    if isinstance(self.optimizer, checkpointable.CheckpointableBase):
      self._track_checkpointable(
          self.optimizer, name='optimizer', overwrite=True)
    self.loss = loss
    self.metrics = metrics or []
    self.loss_weights = loss_weights
    if context.executing_eagerly() and sample_weight_mode is not None:
      raise ValueError('sample_weight_mode is not supported in Eager mode.')
    self.sample_weight_mode = sample_weight_mode
    if context.executing_eagerly() and weighted_metrics is not None:
      raise ValueError('weighted_metrics is not supported in Eager mode.')
    self.weighted_metrics = weighted_metrics
    if context.executing_eagerly() and target_tensors is not None:
      raise ValueError('target_tensors is not supported in Eager mode.')
    self.target_tensors = target_tensors

    if not self.built:
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
              'Output "' + name + '" missing from loss dictionary. We assume '
              'this was done on purpose. The fit and evaluate APIs will not be '
              'expecting any data to be passed to "' + name + '".')
        loss_functions.append(losses.get(loss.get(name)))
    elif isinstance(loss, list):
      if len(loss) != len(self.outputs):
        raise ValueError('When passing a list as loss, '
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed loss=' + str(loss))
      loss_functions = [losses.get(l) for l in loss]
    else:
      loss_function = losses.get(loss)
      loss_functions = [loss_function for _ in range(len(self.outputs))]
    self.loss_functions = loss_functions

    weighted_losses = [training_utils.weighted_masked_objective(fn)
                       for fn in loss_functions]
    skip_target_indices = []
    skip_target_weighing_indices = []
    self._feed_outputs = []
    self._feed_output_names = []
    self._feed_output_shapes = []
    self._feed_loss_fns = []
    for i in range(len(weighted_losses)):
      if weighted_losses[i] is None:
        skip_target_indices.append(i)
        skip_target_weighing_indices.append(i)

    # Prepare output masks.
    if not context.executing_eagerly():
      masks = self.compute_mask(self.inputs, mask=None)
      if masks is None:
        masks = [None for _ in self.outputs]
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

    # initialization for Eager mode execution
    if context.executing_eagerly():
      if target_tensors is not None:
        raise ValueError('target_tensors are not currently supported in Eager '
                         'mode.')
      self.total_loss = None
      self.metrics_tensors = []
      self.metrics_names = ['loss']
      for i in range(len(self.outputs)):
        if len(self.outputs) > 1:
          self.metrics_names.append(self.output_names[i] + '_loss')
      self.nested_metrics = training_utils.collect_metrics(metrics,
                                                           self.output_names)
      # TODO(fchollet): support stateful metrics in eager execution.
      self.stateful_metric_functions = []
      self.stateful_metric_names = []

      with K.name_scope('metrics'):
        training_utils.populate_metric_names(self)
      self._feed_sample_weight_modes = []
      for i in range(len(self.outputs)):
        self._feed_sample_weight_modes.append(None)
      self.sample_weights = []
      self.targets = []
      for i in range(len(self.outputs)):
        self._feed_output_names.append(self.output_names[i])
      self._collected_trainable_weights = self.trainable_weights
      return

    # Prepare targets of model.
    self.targets = []
    self._feed_targets = []
    if target_tensors not in (None, []):
      if isinstance(target_tensors, list):
        if len(target_tensors) != len(self.outputs):
          raise ValueError(
              'When passing a list as `target_tensors`, '
              'it should have one entry per model output. '
              'The model has ' + str(len(self.outputs)) +
              ' outputs, but you passed target_tensors=' + str(target_tensors))
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
      else:
        raise TypeError('Expected `target_tensors` to be '
                        'a list or dict, but got:', target_tensors)

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
            target = K.placeholder(
                ndim=len(shape),
                name=name + '_target',
                sparse=K.is_sparse(self.outputs[i]),
                dtype=K.dtype(self.outputs[i]))
          self._feed_targets.append(target)
          self._feed_outputs.append(self.outputs[i])
          self._feed_output_names.append(name)
          self._feed_output_shapes.append(shape)
          self._feed_loss_fns.append(self.loss_functions[i])
        else:
          skip_target_weighing_indices.append(i)
        self.targets.append(target)

    # Prepare sample weights.
    sample_weights = []
    sample_weight_modes = []
    if isinstance(sample_weight_mode, dict):
      for name in sample_weight_mode:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in '
              'sample_weight_mode dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      for i, name in enumerate(self.output_names):
        if i in skip_target_weighing_indices:
          weight = None
          sample_weight_modes.append(None)
        else:
          if name not in sample_weight_mode:
            raise ValueError(
                'Output "' + name + '" missing from sample_weight_modes '
                'dictionary')
          if sample_weight_mode.get(name) == 'temporal':
            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
            sample_weight_modes.append('temporal')
          else:
            weight = K.placeholder(ndim=1, name=name + 'sample_weights')
            sample_weight_modes.append(None)
        sample_weights.append(weight)
    elif isinstance(sample_weight_mode, list):
      if len(sample_weight_mode) != len(self.outputs):
        raise ValueError('When passing a list as sample_weight_mode, '
                         'it should have one entry per model output. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed '
                         'sample_weight_mode=' + str(sample_weight_mode))
      for i in range(len(self.output_names)):
        if i in skip_target_weighing_indices:
          weight = None
          sample_weight_modes.append(None)
        else:
          mode = sample_weight_mode[i]
          name = self.output_names[i]
          if mode == 'temporal':
            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
            sample_weight_modes.append('temporal')
          else:
            weight = K.placeholder(ndim=1, name=name + '_sample_weights')
            sample_weight_modes.append(None)
        sample_weights.append(weight)
    else:
      for i, name in enumerate(self.output_names):
        if i in skip_target_weighing_indices:
          sample_weight_modes.append(None)
          sample_weights.append(None)
        else:
          if sample_weight_mode == 'temporal':
            sample_weights.append(array_ops.placeholder_with_default(
                constant_op.constant([[1.]], dtype=K.floatx()),
                shape=[None, None], name=name + '_sample_weights'))
            sample_weight_modes.append('temporal')
          else:
            sample_weights.append(array_ops.placeholder_with_default(
                constant_op.constant([1.], dtype=K.floatx()),
                shape=[None], name=name + '_sample_weights'))
            sample_weight_modes.append(None)
    self.sample_weight_modes = sample_weight_modes
    self._feed_sample_weight_modes = []
    for i in range(len(self.outputs)):
      if i not in skip_target_weighing_indices:
        self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

    # Prepare metrics.
    self.weighted_metrics = weighted_metrics
    self.metrics_names = ['loss']
    self.metrics_tensors = []

    # Compute total loss.
    total_loss = None
    with K.name_scope('loss'):
      for i in range(len(self.outputs)):
        if i in skip_target_indices:
          continue
        y_true = self.targets[i]
        y_pred = self.outputs[i]
        weighted_loss = weighted_losses[i]
        sample_weight = sample_weights[i]
        mask = masks[i]
        loss_weight = loss_weights_list[i]
        with K.name_scope(self.output_names[i] + '_loss'):
          output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
        if len(self.outputs) > 1:
          self.metrics_tensors.append(output_loss)
          self.metrics_names.append(self.output_names[i] + '_loss')
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

    # List of same size as output_names.
    # contains tuples (metrics for output, names of metrics).
    nested_metrics = training_utils.collect_metrics(metrics, self.output_names)
    nested_weighted_metrics = training_utils.collect_metrics(weighted_metrics,
                                                             self.output_names)
    self.metrics_updates = []
    self.stateful_metric_names = []
    self.stateful_metric_functions = []
    with K.name_scope('metrics'):
      for i in range(len(self.outputs)):
        if i in skip_target_indices:
          continue

        y_true = self.targets[i]
        y_pred = self.outputs[i]
        weights = sample_weights[i]
        output_metrics = nested_metrics[i]
        output_weighted_metrics = nested_weighted_metrics[i]

        def handle_metrics(metrics, weights=None):

          for metric in metrics:
            if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
              # custom handling of accuracy/crossentropy
              # (because of class mode duality)
              output_shape = self.outputs[i].get_shape().as_list()
              if (output_shape[-1] == 1 or
                  self.loss_functions[i] == losses.binary_crossentropy):
                # case: binary accuracy/crossentropy
                if metric in ('accuracy', 'acc'):
                  metric_fn = metrics_module.binary_accuracy
                elif metric in ('crossentropy', 'ce'):
                  metric_fn = metrics_module.binary_crossentropy
              elif self.loss_functions[
                  i] == losses.sparse_categorical_crossentropy:
                # case: categorical accuracy/crossentropy with sparse targets
                if metric in ('accuracy', 'acc'):
                  metric_fn = metrics_module.sparse_categorical_accuracy
                elif metric in ('crossentropy', 'ce'):
                  metric_fn = metrics_module.sparse_categorical_crossentropy
              else:
                # case: categorical accuracy/crossentropy
                if metric in ('accuracy', 'acc'):
                  metric_fn = metrics_module.categorical_accuracy
                elif metric in ('crossentropy', 'ce'):
                  metric_fn = metrics_module.categorical_crossentropy
              weighted_metric_fn = training_utils.weighted_masked_objective(
                  metric_fn)
            else:
              metric_fn = metrics_module.get(metric)
              weighted_metric_fn = training_utils.weighted_masked_objective(
                  metric_fn)
            metric_name = training_utils.get_base_metric_name(
                metric, weighted=weights is not None)
            with K.name_scope(metric_name):
              metric_result = weighted_metric_fn(
                  y_true, y_pred, weights=weights, mask=masks[i])

            training_utils.add_metric_name(self, metric_name, i)
            self.metrics_tensors.append(metric_result)

            # Keep track of state updates created by
            # stateful metrics (i.e. metrics layers).
            if isinstance(metric_fn, base_layer.Layer) and metric_fn.stateful:
              self.stateful_metric_names.append(metric_name)
              self.stateful_metric_functions.append(metric_fn)
              self.metrics_updates += metric_fn.updates

        handle_metrics(output_metrics)
        handle_metrics(output_weighted_metrics, weights=weights)

    # Prepare gradient updates and state updates.
    self.total_loss = total_loss
    self.sample_weights = sample_weights
    self._feed_sample_weights = []
    for i in range(len(self.sample_weights)):
      if i not in skip_target_weighing_indices:
        self._feed_sample_weights.append(self.sample_weights[i])

    # Functions for train, test and predict will
    # be compiled lazily when required.
    # This saves time when the user is not using all functions.
    self._function_kwargs = kwargs

    self.train_function = None
    self.test_function = None
    self.predict_function = None

    # Collected trainable weights, sorted in topological order.
    trainable_weights = self.trainable_weights
    self._collected_trainable_weights = trainable_weights

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
      logging.warning(
          UserWarning(
              'Discrepancy between trainable weights and collected trainable'
              ' weights, did you set `model.trainable` without calling'
              ' `model.compile` after ?'))

  def _make_train_function(self):
    if not hasattr(self, 'train_function'):
      raise RuntimeError('You must compile your model before using it.')
    self._check_trainable_weights_consistency()
    if self.train_function is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

      with K.name_scope('training'):
        with K.name_scope(self.optimizer.__class__.__name__):
          # Training updates
          updates = self.optimizer.get_updates(
              params=self._collected_trainable_weights, loss=self.total_loss)
        # Unconditional updates
        updates += self.get_updates_for(None)
        # Conditional updates relevant to this model
        updates += self.get_updates_for(self.inputs)
        # Stateful metrics updates
        updates += self.metrics_updates
        # Gets loss and metrics. Updates weights at each call.
        self.train_function = K.function(
            inputs, [self.total_loss] + self.metrics_tensors,
            updates=updates,
            name='train_function',
            **self._function_kwargs)
    self._post_build_cleanup()

  def _make_test_function(self):
    if not hasattr(self, 'test_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.test_function is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
      # Return loss and metrics, no gradient updates.
      # Does update the network states.
      self.test_function = K.function(
          inputs, [self.total_loss] + self.metrics_tensors,
          updates=self.state_updates + self.metrics_updates,
          name='test_function',
          **self._function_kwargs)
    self._post_build_cleanup()

  def _make_predict_function(self):
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs = self._feed_inputs + [K.learning_phase()]
      else:
        inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      kwargs = getattr(self, '_function_kwargs', {})
      self.predict_function = K.function(
          inputs,
          self.outputs,
          updates=self.state_updates,
          name='predict_function',
          **kwargs)
    self._post_build_cleanup()

  def _get_iterator_get_next_tensors(self, iterator):
    get_next_op = self._iterator_get_next.get(iterator, None)
    if get_next_op is None:
      get_next_op = iterator.get_next()
      self._iterator_get_next[iterator] = get_next_op
    return get_next_op

  def _standardize_user_data(self,
                             x,
                             y=None,
                             sample_weight=None,
                             class_weight=None,
                             batch_size=None,
                             check_steps=False,
                             steps_name='steps',
                             steps=None,
                             validation_split=0):
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
        to, as conveyed by `y`.
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

    Returns:
      A tuple of 3 lists: input arrays, target arrays, sample-weight arrays.
      If the model's input and targets are symbolic, these lists are empty
      (since the model takes no user-provided data, instead the data comes
      from the symbolic inputs/targets).

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if isinstance(x, dataset_ops.Dataset):
      if context.executing_eagerly():
        x = x.make_one_shot_iterator()
      else:
        if x in self._dataset_iterator_cache:
          x = self._dataset_iterator_cache[x]
        else:
          iterator = x.make_initializable_iterator()
          self._dataset_iterator_cache[x] = iterator
          x = iterator
        K.get_session().run(x.initializer)

    # Validates `steps` argument based on x's type.
    if check_steps:
      training_utils.check_steps_argument(x, steps, steps_name)

    is_x_eager_iterator = isinstance(x, iterator_ops.EagerIterator)
    is_x_iterator = isinstance(x, iterator_ops.Iterator)

    # Validate user inputs when data is given as a dataset or dataset iterator.
    if is_x_iterator or is_x_eager_iterator:
      training_utils.validate_iterator_input(x, y, sample_weight,
                                             validation_split)

    # For eager iterators, when we have to process multiple batches of samples,
    # we will standardize the data when we actually loop over iterator and get
    # the batches. For now, we just return the iterator as is.
    if is_x_eager_iterator and steps is not None:
      return x, y, sample_weight

    # If input data is a dataset iterator in graph mode or if it is an eager
    # iterator and only one batch of samples is required, we fetch the data
    # tensors from the iterator and then standardize them.
    if is_x_iterator or is_x_eager_iterator:
      try:
        if is_x_iterator:
          next_element = self._get_iterator_get_next_tensors(x)
        else:
          next_element = x.get_next()
      except errors.OutOfRangeError:
        raise RuntimeError('Your dataset iterator ran out of data; '
                           'Make sure that your dataset can generate '
                           'required number of samples.')

      if not isinstance(next_element, (list, tuple)) or len(next_element) != 2:
        raise ValueError('Please provide data as a list or tuple of 2 elements '
                         ' - input and target pair. Received %s' % next_element)
      x, y = next_element

    # First, we build/compile the model on the fly if necessary.
    all_inputs = []
    is_build_called = False
    is_compile_called = False
    if not self.built:
      # We need to use `x` to set the model inputs.
      # We type-check that `x` and `y` are either single arrays
      # or lists of arrays.
      if isinstance(x, (list, tuple)):
        if not all(isinstance(v, np.ndarray) or
                   tensor_util.is_tensor(v) for v in x):
          raise ValueError('Please provide as model inputs either a single '
                           'array or a list of arrays. You passed: x=' + str(x))
        all_inputs += list(x)
      elif isinstance(x, dict):
        raise ValueError('Please do not pass a dictionary as model inputs.')
      else:
        if not isinstance(x, np.ndarray) and not tensor_util.is_tensor(x):
          raise ValueError('Please provide as model inputs either a single '
                           'array or a list of arrays. You passed: x=' + str(x))
        all_inputs.append(x)

      # Build the model using the retrieved inputs (value or symbolic).
      # If values, then in symbolic-mode placeholders will be created
      # to match the value shapes.
      if not self.inputs:
        is_build_called = True
        self._set_inputs(x)

    if y is not None:
      if not self.optimizer:
        raise RuntimeError('You must compile a model before '
                           'training/testing. '
                           'Use `model.compile(optimizer, loss)`.')
      if not self._is_compiled:
        # On-the-fly compilation of the model.
        # We need to use `y` to set the model targets.
        if isinstance(y, (list, tuple)):
          if not all(isinstance(v, np.ndarray) or
                     tensor_util.is_tensor(v) for v in y):
            raise ValueError('Please provide as model targets either a single '
                             'array or a list of arrays. '
                             'You passed: y=' + str(y))
          all_inputs += list(y)
        elif isinstance(y, dict):
          raise ValueError('Please do not pass a dictionary as model targets.')
        else:
          if not isinstance(y, np.ndarray) and not tensor_util.is_tensor(y):
            raise ValueError('Please provide as model targets either a single '
                             'array or a list of arrays. '
                             'You passed: y=' + str(y))
          all_inputs.append(y)

        # Typecheck that all inputs are *either* value *or* symbolic.
        # TODO(fchollet): this check could be removed in Eager mode?
        if any(tensor_util.is_tensor(v) for v in all_inputs):
          if not all(tensor_util.is_tensor(v) for v in all_inputs):
            raise ValueError('Do not pass inputs that mix Numpy arrays and '
                             'TensorFlow tensors. '
                             'You passed: x=' + str(x) + '; y=' + str(y))

        if context.executing_eagerly():
          target_tensors = None
        else:
          # Handle target tensors if any passed.
          if not isinstance(y, (list, tuple)):
            y = [y]
          target_tensors = [v for v in y if tensor_util.is_tensor(v)]
        is_compile_called = True
        self.compile(optimizer=self.optimizer,
                     loss=self.loss,
                     metrics=self.metrics,
                     loss_weights=self.loss_weights,
                     target_tensors=target_tensors)

    # In graph mode, if we had just set inputs and targets as symbolic tensors
    # by invoking build and compile on the model respectively, we do not have to
    # feed anything to the model. Model already has input and target data as
    # part of the graph.
    # Note: in this case, `any` and `all` are equivalent since we disallow
    # mixed symbolic/value inputs.
    if (not context.executing_eagerly() and is_build_called and
        is_compile_called and
        any(tensor_util.is_tensor(v) for v in all_inputs)):
      return [], [], []

    # What follows is input validation and standardization to list format,
    # in the case where all inputs are value arrays.

    if context.executing_eagerly():
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
    x = training_utils.standardize_input_data(
        x,
        feed_input_names,
        feed_input_shapes,
        check_batch_axis=False,  # Don't enforce the batch size.
        exception_prefix='input')

    if y is not None:
      if context.executing_eagerly():
        feed_output_names = self.output_names
        feed_output_shapes = None
        # Sample weighting not supported in this case.
        # TODO(fchollet): consider supporting it.
        feed_sample_weight_modes = [None for _ in self.outputs]
      elif not self._is_graph_network:
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
          if loss_fn is losses.sparse_categorical_crossentropy:
            if K.image_data_format() == 'channels_first':
              feed_output_shapes.append(
                  (output_shape[0], 1) + output_shape[2:])
            else:
              feed_output_shapes.append(output_shape[:-1] + (1,))
          elif (not hasattr(loss_fn, '__name__') or
                getattr(losses, loss_fn.__name__, None) is None):
            # If `loss_fn` is not a function (e.g. callable class)
            # or if it not in the `losses` module, then
            # it is a user-defined loss and we make no assumptions
            # about it.
            feed_output_shapes.append(None)
          else:
            feed_output_shapes.append(output_shape)

      # Standardize the outputs.
      y = training_utils.standardize_input_data(
          y,
          feed_output_names,
          feed_output_shapes,
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
      training_utils.check_array_lengths(x, y, sample_weights)
      if self._is_graph_network and not context.executing_eagerly():
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
    return x, y, sample_weights

  @checkpointable.no_automatic_dependency_tracking
  def _set_inputs(self, inputs, training=None):
    """Set model's input and output specs based on the input data received.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    Args:
      inputs: Single array, or list of arrays. The arrays could be placeholders,
        Numpy arrays, or data tensors.
        - if placeholders: the model is built on top of these placeholders,
          and we expect Numpy data to be fed for them when calling `fit`/etc.
        - if Numpy data: we create placeholders matching the shape of the Numpy
          arrays. We expect Numpy data to be fed for these placeholders
          when calling `fit`/etc.
        - if data tensors: the model is built on top of these tensors.
          We do not expect any Numpy data to be provided when calling `fit`/etc.
      training: Boolean or None. Only relevant in symbolic mode. Specifies
        whether to build the model's graph in inference mode (False), training
        mode (True), or using the Keras learning phase (None).
    """
    call_convention = getattr(
        self,
        '_call_convention',
        base_layer.CallConvention.EXPLICIT_INPUTS_ARGUMENT)
    if call_convention not in (
        base_layer.CallConvention.EXPLICIT_INPUTS_ARGUMENT,
        base_layer.CallConvention.SINGLE_POSITIONAL_ARGUMENT):
      raise NotImplementedError(
          'Subclassed Models without "inputs" (or single positional arguments) '
          'in their call() signatures do not yet support shape inference. File '
          'a feature request if this limitation bothers you.')
    if self.__class__.__name__ == 'Sequential':
      # Note: we can't test whether the model is `Sequential` via `isinstance`
      # since `Sequential` depends on `Model`.
      if isinstance(inputs, list):
        assert len(inputs) == 1
        inputs = inputs[0]

      if tensor_util.is_tensor(inputs):
        if context.executing_eagerly():
          input_shape = (None,) + tuple(inputs.get_shape().as_list()[1:])
          self.build(input_shape=input_shape)
        else:
          self.symbolic_set_inputs(inputs)
      else:
        input_shape = (None,) + inputs.shape[1:]
        self.build(input_shape=input_shape)
    elif context.executing_eagerly():
      self._eager_set_inputs(inputs)
    else:
      self._symbolic_set_inputs(inputs, training=training)

  @checkpointable.no_automatic_dependency_tracking
  def _eager_set_inputs(self, inputs):
    """Set model's input and output specs based on the input data received.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    We assume the number and ndim of outputs
    does not change over different calls.

    Args:
      inputs: Argument `x` (input data) passed by the user upon first model use.

    Raises:
      ValueError: If the model's inputs are already set.
    """
    assert context.executing_eagerly()
    if self.inputs:
      raise ValueError('Model inputs are already set.')
    # On-the-fly setting of model inputs/outputs as DeferredTensors,
    # to keep track of number of inputs and outputs and their ndim.
    if isinstance(inputs, (list, tuple)):
      if tensor_util.is_tensor(inputs[0]):
        dummy_output_values = self.call(
            training_utils.cast_if_floating_dtype(inputs))
      else:
        dummy_output_values = self.call(
            [ops.convert_to_tensor(v, dtype=K.floatx()) for v in inputs])
      dummy_input_values = list(inputs)
    else:
      if tensor_util.is_tensor(inputs):
        dummy_output_values = self.call(
            training_utils.cast_if_floating_dtype(inputs))
      else:
        dummy_output_values = self.call(
            ops.convert_to_tensor(inputs, dtype=K.floatx()))
      dummy_input_values = [inputs]
    if isinstance(dummy_output_values, (list, tuple)):
      dummy_output_values = list(dummy_output_values)
    else:
      dummy_output_values = [dummy_output_values]
    self.outputs = [
        base_layer.DeferredTensor(shape=(None for _ in v.shape),
                                  dtype=v.dtype) for v in dummy_output_values]
    self.inputs = [
        base_layer.DeferredTensor(shape=(None for _ in v.shape),
                                  dtype=v.dtype) for v in dummy_input_values]
    self.input_names = [
        'input_%d' % (i + 1) for i in range(len(dummy_input_values))]
    self.output_names = [
        'output_%d' % (i + 1) for i in range(len(dummy_output_values))]
    self.built = True

  @checkpointable.no_automatic_dependency_tracking
  def _symbolic_set_inputs(self, inputs, outputs=None, training=None):
    """Set model's inputs and output specs based.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    Args:
      inputs: Argument `x` (input data) passed by the user upon first model use.
      outputs: None, a data tensor, or a list of data tensors. If None, the
        outputs will be determined by invoking self.call(), otherwise the
        provided value will be used.
      training: Boolean or None. Only relevant in symbolic mode. Specifies
        whether to build the model's graph in inference mode (False), training
        mode (True), or using the Keras learning phase (None).

    Raises:
      ValueError: If the model's inputs are already set.
    """
    assert not context.executing_eagerly()
    if self.inputs:
      raise ValueError('Model inputs are already set.')

    # On-the-fly setting of symbolic model inputs (either by using the tensor
    # provided, or by creating a placeholder if Numpy data was provided).
    self.inputs = []
    self.input_names = []
    self._feed_inputs = []
    self._feed_input_names = []
    self._feed_input_shapes = []
    if isinstance(inputs, (list, tuple)):
      inputs = list(inputs)
    else:
      inputs = [inputs]

    for i, v in enumerate(inputs):
      name = 'input_%d' % (i + 1)
      self.input_names.append(name)
      if isinstance(v, list):
        v = np.asarray(v)
        if v.ndim == 1:
          v = np.expand_dims(v, 1)
      if isinstance(v, (np.ndarray)):
        # We fix the placeholder shape except the batch size.
        # This is suboptimal, but it is the best we can do with the info
        # we have. The user should call `model._set_inputs(placeholders)`
        # to specify custom placeholders if the need arises.
        shape = (None,) + v.shape[1:]
        placeholder = K.placeholder(shape=shape, name=name)
        self.inputs.append(placeholder)
        self._feed_inputs.append(placeholder)
        self._feed_input_names.append(name)
        self._feed_input_shapes.append(shape)
      else:
        # Assumed tensor - TODO(fchollet) additional type check?
        self.inputs.append(v)
        if K.is_placeholder(v):
          self._feed_inputs.append(v)
          self._feed_input_names.append(name)
          self._feed_input_shapes.append(K.int_shape(v))

    if outputs is None:
      # Obtain symbolic outputs by calling the model.
      if len(self.inputs) == 1:
        if self._expects_training_arg:
          outputs = self.call(self.inputs[0], training=training)
        else:
          outputs = self.call(self.inputs[0])
      else:
        if self._expects_training_arg:
          outputs = self.call(self.inputs, training=training)
        else:
          outputs = self.call(self.inputs)
    if isinstance(outputs, (list, tuple)):
      outputs = list(outputs)
    else:
      outputs = [outputs]
    self.outputs = outputs
    self.output_names = [
        'output_%d' % (i + 1) for i in range(len(self.outputs))]
    self.built = True

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
          - A `tf.data` dataset or a dataset iterator.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset or dataset
          iterator, `y` should not be specified
          (since targets will be obtained from the iterator).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, datasets, or dataset iterators
            (since they generate batches).
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
            See [callbacks](/callbacks).
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset or a dataset iterator.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset or a dataset iterator
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
            supported when `x` is a dataset or a dataset iterator.
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
        validation_steps: Only relevant if `steps_per_epoch`
            is specified. Total number of steps (batches of samples)
            to validate before stopping.
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
    # TODO(fchollet): this method may be creating reference cycles, which would
    # lead to accumulating garbage in memory when called in a loop. Investigate.

    # Backwards compatibility
    if batch_size is None and steps_per_epoch is None:
      batch_size = 32
    # Legacy support
    if 'nb_epoch' in kwargs:
      logging.warning(
          'The `nb_epoch` argument in `fit` '
          'has been renamed `epochs`.')
      epochs = kwargs.pop('nb_epoch')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps_per_epoch',
        steps=steps_per_epoch,
        validation_split=validation_split)

    # Prepare validation data.
    if validation_data:
      if (isinstance(validation_data, iterator_ops.Iterator) or
          isinstance(validation_data, iterator_ops.EagerIterator) or
          isinstance(validation_data, dataset_ops.Dataset)):
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

      # Validate and standardize validation data.
      val_x, val_y, val_sample_weights = self._standardize_user_data(
          val_x,
          val_y,
          sample_weight=val_sample_weight,
          batch_size=batch_size,
          steps=validation_steps)

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

    if context.executing_eagerly():
      return training_eager.fit_loop(
          self,
          inputs=x,
          targets=y,
          sample_weights=sample_weights,
          class_weight=class_weight,
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
          validation_steps=validation_steps)
    else:
      return training_arrays.fit_loop(
          self, x, y,
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
          validation_steps=validation_steps)

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None):
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
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely).
          If `x` is a dataset or a dataset iterator, `y` should not be specified
          (since targets will be obtained from the iterator/dataset).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` is your data is in the
            form of symbolic tensors, datasets, or dataset iterators
            (since they generate batches).
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
            supported when `x` is a dataset or a dataset iterator.
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    # Backwards compatibility.
    if batch_size is None and steps is None:
      batch_size = 32

    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        check_steps=True,
        steps_name='steps',
        steps=steps)

    if context.executing_eagerly():
      return training_eager.test_loop(
          self, inputs=x, targets=y, sample_weights=sample_weights,
          batch_size=batch_size, verbose=verbose, steps=steps)
    else:
      return training_arrays.test_loop(
          self, inputs=x, targets=y, sample_weights=sample_weights,
          batch_size=batch_size, verbose=verbose, steps=steps)

  def predict(self, x, batch_size=None, verbose=0, steps=None):
    """Generates output predictions for the input samples.

    Computation is done in batches.

    Arguments:
         x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset or a dataset iterator.
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` is your data is in the
            form of symbolic tensors, dataset, or dataset iterators
            (since they generate batches).
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    # Backwards compatibility.
    if batch_size is None and steps is None:
      batch_size = 32

    # Validate and standardize user data.
    x, _, _ = self._standardize_user_data(
        x, check_steps=True, steps_name='steps', steps=steps)

    if context.executing_eagerly():
      return training_eager.predict_loop(
          self, x, batch_size=batch_size, verbose=verbose, steps=steps)
    else:
      return training_arrays.predict_loop(
          self, x, batch_size=batch_size, verbose=verbose, steps=steps)

  def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None):
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
        class_weight: Optional dictionary mapping
            class indices (integers) to
            a weight (float) to apply to the model's loss for the samples
            from this class during training.
            This can be useful to tell the model to "pay more attention" to
            samples from an under-represented class.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
      ValueError: In case of invalid user-provided arguments.
    """
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, class_weight=class_weight)

    if context.executing_eagerly():
      outputs = training_eager.train_on_batch(
          self, x, y, sample_weights=sample_weights)
    else:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [1]
      else:
        ins = x + y + sample_weights

      self._make_train_function()
      outputs = self.train_function(ins)

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def test_on_batch(self, x, y=None, sample_weight=None):
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

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: In case of invalid user-provided arguments.
    """
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight)

    if context.executing_eagerly():
      outputs = training_eager.test_on_batch(
          self, x, y, sample_weights=sample_weights)
    else:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = x + y + sample_weights + [0]
      else:
        ins = x + y + sample_weights
      self._make_test_function()
      outputs = self.test_function(ins)

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
    # Validate and standardize user data.
    inputs, _, _ = self._standardize_user_data(x)
    if context.executing_eagerly():
      if (isinstance(x, iterator_ops.EagerIterator) or
          (isinstance(x, dataset_ops.Dataset) and context.executing_eagerly())):
        inputs = training_utils.cast_if_floating_dtype(inputs)
      else:
        inputs = [
            ops.convert_to_tensor(val, dtype=K.floatx()) for val in inputs
        ]
      return self(inputs)  # pylint: disable=not-callable

    if not context.executing_eagerly():
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        ins = inputs + [0]
      else:
        ins = inputs

      self._make_predict_function()
      outputs = self.predict_function(ins)

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
    if not self.built and not self._is_graph_network:
      raise NotImplementedError(
          '`fit_generator` is not yet enabled for unbuilt Model subclasses')

    return training_generator.fit_generator(
        self,
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch)

  def evaluate_generator(self,
                         generator,
                         steps=None,
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
            or an instance of Sequence (keras.utils.Sequence)
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
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
    if not self.built and not self._is_graph_network:
      raise NotImplementedError(
          '`evaluate_generator` is not yet enabled for '
          'unbuilt Model subclasses')

    return training_generator.evaluate_generator(
        self,
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose)

  def predict_generator(self,
                        generator,
                        steps=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: Generator yielding batches of input samples
            or an instance of Sequence (keras.utils.Sequence)
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
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
    if not self.built and not self._is_graph_network:
      raise NotImplementedError(
          '`predict_generator` is not yet enabled for unbuilt Model subclasses')

    return training_generator.predict_generator(
        self,
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose)
