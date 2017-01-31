# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A tf.learn implementation of tensor_forest (extremely random forests)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import trainable

from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import export

from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook


KEYS_NAME = 'keys'
LOSS_NAME = 'rf_training_loss'


def _assert_float32(tensors):
  """Assert all tensors are float32.

  Args:
    tensors: `Tensor` or `dict` of `Tensor` objects.

  Raises:
    TypeError: if any tensor is not float32.
  """
  if not isinstance(tensors, dict):
    tensors = [tensors]
  else:
    tensors = tensors.values()
  for tensor in tensors:
    if tensor.dtype.base_dtype != dtypes.float32:
      raise TypeError('Expected dtype=float32, %s.' % tensor)


class TensorForestLossHook(session_run_hook.SessionRunHook):
  """Monitor to request stop when loss stops decreasing."""

  def __init__(self, early_stopping_rounds):
    self.early_stopping_rounds = early_stopping_rounds
    self.min_loss = None
    self.last_step = -1
    # self.steps records the number of steps for which the loss has been
    # non-decreasing
    self.steps = 0

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        {'global_step': contrib_framework.get_global_step(),
         'current_loss': run_context.session.graph.get_operation_by_name(
             LOSS_NAME).outputs[0]})

  def after_run(self, run_context, run_values):
    current_loss = run_values.results['current_loss']
    current_step = run_values.results['global_step']
    self.steps += 1
    # Gaurd against the global step going backwards, which might happen
    # if we recover from something.
    if self.last_step == -1 or self.last_step > current_step:
      logging.info('TensorForestLossHook resetting last_step.')
      self.last_step = current_step
      self.steps = 0
      self.min_loss = None
      return

    self.last_step = current_step
    if self.min_loss is None or current_loss < self.min_loss:
      self.min_loss = current_loss
      self.steps = 0
    if self.steps > self.early_stopping_rounds:
      logging.info('TensorForestLossHook requesting stop.')
      run_context.request_stop()


def get_model_fn(params, graph_builder_class, device_assigner,
                 weights_name=None, keys_name=None, num_trainers=1,
                 trainer_id=0):
  """Return a model function given a way to construct a graph builder."""
  def _model_fn(features, labels, mode):
    """Function that returns predictions, training loss, and training op."""
    weights = None
    keys = None
    if weights_name and weights_name in features:
      weights = features.pop(weights_name)
    if keys_name and keys_name in features:
      keys = features.pop(keys_name)

    graph_builder = graph_builder_class(params, device_assigner=device_assigner)
    inference = {}
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.INFER):
      inference[eval_metrics.INFERENCE_PROB_NAME] = (
          graph_builder.inference_graph(features))

      if not params.regression:
        inference[eval_metrics.INFERENCE_PRED_NAME] = math_ops.argmax(
            inference[eval_metrics.INFERENCE_PROB_NAME], 1)
      if keys:
        inference[KEYS_NAME] = keys

    # labels might be None if we're doing prediction (which brings up the
    # question of why we force everything to adhere to a single model_fn).
    loss_deps = []
    training_graph = None
    if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
      training_graph = control_flow_ops.group(
          graph_builder.training_graph(
              features, labels, input_weights=weights,
              num_trainers=num_trainers,
              trainer_id=trainer_id),
          state_ops.assign_add(contrib_framework.get_global_step(), 1))
      loss_deps.append(training_graph)

    training_loss = None
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.TRAIN):
      with ops.control_dependencies(loss_deps):
        training_loss = graph_builder.training_loss(
            features, labels, name=LOSS_NAME)
    # Put weights back in
    if weights is not None:
      features[weights_name] = weights
    return (inference, training_loss, training_graph)
  return _model_fn


class TensorForestEstimator(evaluable.Evaluable, trainable.Trainable):
  """An estimator that can train and evaluate a random forest.

  Example:

  ```python
  params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
      num_classes=2, num_features=40, num_trees=10, max_nodes=1000)

  # Estimator using the default graph builder.
  estimator = TensorForestEstimator(params, model_dir=model_dir)

  # Or estimator using TrainingLossForest as the graph builder.
  estimator = TensorForestEstimator(
      params, graph_builder_class=tensor_forest.TrainingLossForest,
      model_dir=model_dir)

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```
  """

  def __init__(self, params, device_assigner=None, model_dir=None,
               graph_builder_class=tensor_forest.RandomForestGraphs,
               config=None, weights_name=None, keys_name=None,
               feature_engineering_fn=None, early_stopping_rounds=100,
               num_trainers=1, trainer_id=0):

    """Initializes a TensorForestEstimator instance.

    Args:
      params: ForestHParams object that holds random forest hyperparameters.
        These parameters will be passed into `model_fn`.
      device_assigner: An `object` instance that controls how trees get
        assigned to devices. If `None`, will use
        `tensor_forest.RandomForestDeviceAssigner`.
      model_dir: Directory to save model parameters, graph, etc. To continue
        training a previously saved model, load checkpoints saved to this
        directory into an estimator.
      graph_builder_class: An `object` instance that defines how TF graphs for
        random forest training and inference are built. By default will use
        `tensor_forest.RandomForestGraphs`.
      config: `RunConfig` object to configure the runtime settings.
      weights_name: A string defining feature column name representing
        weights. Will be multiplied by the loss of the example. Used to
        downweight or boost examples during training.
      keys_name: A string defining feature column name representing example
        keys. Used by `predict_with_keys` method.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      early_stopping_rounds: Allows training to terminate early if the forest is
        no longer growing. 100 by default.
      num_trainers: Number of training jobs, which will partition trees
        among them.
      trainer_id: Which trainer this instance is.

    Returns:
      A `TensorForestEstimator` instance.
    """
    self.params = params.fill()
    self.graph_builder_class = graph_builder_class
    self.early_stopping_rounds = early_stopping_rounds
    self.weights_name = weights_name
    self._estimator = estimator.Estimator(
        model_fn=get_model_fn(params, graph_builder_class, device_assigner,
                              weights_name=weights_name, keys_name=keys_name,
                              num_trainers=num_trainers, trainer_id=trainer_id),
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)
    self._skcompat = estimator.SKCompat(self._estimator)

  @property
  def model_dir(self):
    """See evaluable.Evaluable."""
    return self._estimator.model_dir

  def evaluate(self,
               x=None,
               y=None,
               input_fn=None,
               batch_size=None,
               steps=None,
               metrics=None,
               name=None,
               checkpoint_path=None,
               hooks=None):
    """See evaluable.Evaluable."""
    if x is not None and y is not None:
      return self._skcompat.score(x, y, batch_size=batch_size, steps=steps,
                                  metrics=metrics)
    elif input_fn is not None:
      return self._estimator.evaluate(
          input_fn=input_fn,
          steps=steps,
          metrics=metrics,
          name=name,
          checkpoint_path=checkpoint_path,
          hooks=hooks)
    else:
      raise ValueError(
          'evaluate: Must provide either both x and y or input_fn.')

  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    """See trainable.Trainable."""
    if not monitors:
      monitors = [TensorForestLossHook(self.early_stopping_rounds)]
    if x is not None and y is not None:
      self._skcompat.fit(x, y, batch_size=batch_size, steps=steps,
                         max_steps=max_steps, monitors=monitors)
    elif input is not None:
      self._estimator.fit(input_fn=input_fn, steps=steps, monitors=monitors,
                          max_steps=max_steps)
    else:
      raise ValueError('fit: Must provide either both x and y or input_fn.')

  def predict_proba(
      self, x=None, input_fn=None, batch_size=None):
    """Returns prediction probabilities for given features (classification).

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).

    Raises:
      ValueError: If both or neither of x and input_fn were given.
    """
    if x is not None:
      results = self._skcompat.predict(x, batch_size=batch_size)
      return results[eval_metrics.INFERENCE_PROB_NAME]
    else:
      results = self._estimator.predict(input_fn=input_fn, as_iterable=True)
      return (x[eval_metrics.INFERENCE_PROB_NAME] for x in results)

  def predict(
      self, x=None, input_fn=None, axis=None, batch_size=None):
    """Returns predictions for given features.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      axis: Axis on which to argmax (for classification).
            Last axis is used by default.
      batch_size: Override default batch size.

    Returns:
      Numpy array of predicted classes or regression values (or an iterable of
      predictions if as_iterable is True).
    """
    predict_name = (eval_metrics.INFERENCE_PROB_NAME if self.params.regression
                    else eval_metrics.INFERENCE_PRED_NAME)
    if x is not None:
      results = self._skcompat.predict(x, batch_size=batch_size)
      return results[predict_name]
    else:
      results = self._estimator.predict(input_fn=input_fn, as_iterable=True)
      return (x[predict_name] for x in results)

  def predict_with_keys(
      self, x=None, input_fn=None, axis=None, batch_size=None):
    """Same as predict but also returns the example keys."""
    predict_name = (eval_metrics.INFERENCE_PROB_NAME if self.params.regression
                    else eval_metrics.INFERENCE_PRED_NAME)
    if x is not None:
      results = self._skcompat.predict(x, batch_size=batch_size)
      return results[predict_name]
    else:
      results = self._estimator.predict(input_fn=input_fn, as_iterable=True)
      return ((x[predict_name], x.get(KEYS_NAME, None)) for x in results)

  def export(self,
             export_dir,
             input_fn,
             signature_fn=None,
             input_feature_key=None,
             default_batch_size=1):
    """See BaseEstimator.export."""
    # Reset model function with basic device assigner.
    # Servo doesn't support distributed inference
    # but it will try to respect device assignments if they're there.
    # pylint: disable=protected-access
    orig_model_fn = self._estimator._model_fn
    self._estimator._model_fn = get_model_fn(
        self.params, self.graph_builder_class,
        tensor_forest.RandomForestDeviceAssigner(),
        weights_name=self.weights_name)
    result = self._estimator.export(
        export_dir=export_dir,
        input_fn=input_fn,
        input_feature_key=input_feature_key,
        use_deprecated_input_fn=False,
        signature_fn=(signature_fn or
                      (export.regression_signature_fn
                       if self.params.regression else
                       export.classification_signature_fn_with_prob)),
        default_batch_size=default_batch_size,
        prediction_key=eval_metrics.INFERENCE_PROB_NAME)
    self._estimator._model_fn = orig_model_fn
    # pylint: enable=protected-access
    return result

  def export_savedmodel(self,
                        export_dir_base,
                        serving_input_fn,
                        default_output_alternative_key=None,
                        assets_extra=None,
                        as_text=False):
    return self._estimator.export_savedmodel(
        export_dir_base,
        serving_input_fn,
        default_output_alternative_key=default_output_alternative_key,
        assets_extra=assets_extra,
        as_text=as_text)
