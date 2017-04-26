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

from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
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


class EveryCheckpointPreSaveListener(
    basic_session_run_hooks.CheckpointSaverListener):
  """Runs a given op before each checkpoint save."""

  def __init__(self, op):
    """Initializes the object.

    Args:
      op: An op to run before each checkpoint save.
    """
    self._op = op

  def before_save(self, session, global_step_value):
    session.run(self._op)


def get_model_fn(params,
                 graph_builder_class,
                 device_assigner,
                 weights_name=None,
                 early_stopping_rounds=100,
                 num_trainers=1,
                 trainer_id=0,
                 report_feature_importances=False,
                 model_dir=None,
                 local_eval=False):
  """Return a model function given a way to construct a graph builder."""
  def _model_fn(features, labels, mode):
    """Function that returns predictions, training loss, and training op."""
    weights = None
    if weights_name and weights_name in features:
      weights = features.pop(weights_name)

    # If we're doing eval, optionally ignore device_assigner.
    # Also ignore device assigner if we're exporting (mode == INFER)
    dev_assn = device_assigner
    if (mode == model_fn_lib.ModeKeys.INFER or
        (local_eval and mode == model_fn_lib.ModeKeys.EVAL)):
      dev_assn = None

    graph_builder = graph_builder_class(params,
                                        device_assigner=dev_assn)
    inference = {}
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.INFER):
      inference[eval_metrics.INFERENCE_PROB_NAME] = (
          graph_builder.inference_graph(features))

      if not params.regression:
        inference[eval_metrics.INFERENCE_PRED_NAME] = math_ops.argmax(
            inference[eval_metrics.INFERENCE_PROB_NAME], 1)

      if report_feature_importances:
        inference[eval_metrics.FEATURE_IMPORTANCE_NAME] = (
            graph_builder.feature_importances())

    # labels might be None if we're doing prediction (which brings up the
    # question of why we force everything to adhere to a single model_fn).
    loss_deps = []
    training_graph = None
    training_hooks = []
    scaffold = None
    if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
      training_graph = control_flow_ops.group(
          graph_builder.training_graph(
              features, labels, input_weights=weights,
              num_trainers=num_trainers,
              trainer_id=trainer_id),
          state_ops.assign_add(contrib_framework.get_global_step(), 1))
      loss_deps.append(training_graph)
      if hasattr(graph_builder, 'finalize_training'):
        finalize_listener = EveryCheckpointPreSaveListener(
            graph_builder.finalize_training())
        scaffold = monitored_session.Scaffold()
        training_hooks.append(
            basic_session_run_hooks.CheckpointSaverHook(
                model_dir, save_secs=600, save_steps=None,
                scaffold=scaffold,
                listeners=[finalize_listener]))

    training_loss = None
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.TRAIN):
      with ops.control_dependencies(loss_deps):
        training_loss = graph_builder.training_loss(
            features, labels, name=LOSS_NAME)

    # Put weights back in
    if weights is not None:
      features[weights_name] = weights

    if early_stopping_rounds:
      training_hooks.append(TensorForestLossHook(early_stopping_rounds))

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=inference,
        loss=training_loss,
        train_op=training_graph,
        training_hooks=training_hooks,
        scaffold=scaffold)

  return _model_fn


class TensorForestEstimator(estimator.Estimator):
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

  # Predict returns an iterable of dicts.
  results = list(estimator.predict(x=x))
  prob0 = results[0][eval_metrics.INFERENCE_PROB_NAME]
  prediction0 = results[0][eval_metrics.INFERENCE_PRED_NAME]
  ```
  """

  def __init__(self, params, device_assigner=None, model_dir=None,
               graph_builder_class=tensor_forest.RandomForestGraphs,
               config=None, weights_name=None,
               feature_engineering_fn=None,
               early_stopping_rounds=100,
               num_trainers=1, trainer_id=0,
               report_feature_importances=False,
               local_eval=False):
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
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      early_stopping_rounds: Allows training to terminate early if the forest is
        no longer growing. 100 by default.  Set to a Falsy value to disable
        the default training hook.
      num_trainers: Number of training jobs, which will partition trees
        among them.
      trainer_id: Which trainer this instance is.
      report_feature_importances: If True, print out feature importances
        during evaluation.
      local_eval: If True, don't use a device assigner for eval. This is to
        support some common setups where eval is done on a single machine, even
        though training might be distributed.

    Returns:
      A `TensorForestEstimator` instance.
    """
    super(TensorForestEstimator, self).__init__(
        model_fn=get_model_fn(
            params.fill(),
            graph_builder_class,
            device_assigner,
            weights_name=weights_name,
            early_stopping_rounds=early_stopping_rounds,
            num_trainers=num_trainers,
            trainer_id=trainer_id,
            report_feature_importances=report_feature_importances,
            model_dir=model_dir,
            local_eval=local_eval),
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)

