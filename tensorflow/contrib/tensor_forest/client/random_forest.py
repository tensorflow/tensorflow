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
"""A tf.learn implementation of online extremely random forests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers

from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


KEYS_NAME = 'keys'
LOSS_NAME = 'rf_training_loss'
TREE_PATHS_PREDICTION_KEY = 'tree_paths'
VARIANCE_PREDICTION_KEY = 'regression_variance'

EPSILON = 0.000001


class TensorForestRunOpAtEndHook(session_run_hook.SessionRunHook):

  def __init__(self, op_dict):
    """Ops is a dict of {name: op} to run before the session is destroyed."""
    self._ops = op_dict

  def end(self, session):
    for name in sorted(self._ops.keys()):
      logging.info('{0}: {1}'.format(name, session.run(self._ops[name])))


class TensorForestLossHook(session_run_hook.SessionRunHook):
  """Monitor to request stop when loss stops decreasing."""

  def __init__(self,
               early_stopping_rounds,
               early_stopping_loss_threshold=None,
               loss_op=None):
    self.early_stopping_rounds = early_stopping_rounds
    self.early_stopping_loss_threshold = early_stopping_loss_threshold
    self.loss_op = loss_op
    self.min_loss = None
    self.last_step = -1
    # self.steps records the number of steps for which the loss has been
    # non-decreasing
    self.steps = 0

  def before_run(self, run_context):
    loss = (self.loss_op if self.loss_op is not None else
            run_context.session.graph.get_operation_by_name(
                LOSS_NAME).outputs[0])
    return session_run_hook.SessionRunArgs(
        {'global_step': training_util.get_global_step(),
         'current_loss': loss})

  def after_run(self, run_context, run_values):
    current_loss = run_values.results['current_loss']
    current_step = run_values.results['global_step']
    self.steps += 1
    # Guard against the global step going backwards, which might happen
    # if we recover from something.
    if self.last_step == -1 or self.last_step > current_step:
      logging.info('TensorForestLossHook resetting last_step.')
      self.last_step = current_step
      self.steps = 0
      self.min_loss = None
      return

    self.last_step = current_step
    if (self.min_loss is None or current_loss <
        (self.min_loss - self.min_loss * self.early_stopping_loss_threshold)):
      self.min_loss = current_loss
      self.steps = 0
    if self.steps > self.early_stopping_rounds:
      logging.info('TensorForestLossHook requesting stop.')
      run_context.request_stop()


def get_default_head(params, weights_name, name=None):
  if params.regression:
    return head_lib.regression_head(
        weight_column_name=weights_name,
        label_dimension=params.num_outputs,
        enable_centered_bias=False,
        head_name=name)
  else:
    return head_lib.multi_class_head(
        params.num_classes,
        weight_column_name=weights_name,
        enable_centered_bias=False,
        head_name=name)


def get_model_fn(params,
                 graph_builder_class,
                 device_assigner,
                 feature_columns=None,
                 weights_name=None,
                 model_head=None,
                 keys_name=None,
                 early_stopping_rounds=100,
                 early_stopping_loss_threshold=0.001,
                 num_trainers=1,
                 trainer_id=0,
                 report_feature_importances=False,
                 local_eval=False,
                 head_scope=None):
  """Return a model function given a way to construct a graph builder."""
  if model_head is None:
    model_head = get_default_head(params, weights_name)

  def _model_fn(features, labels, mode):
    """Function that returns predictions, training loss, and training op."""
    if (isinstance(features, ops.Tensor) or
        isinstance(features, sparse_tensor.SparseTensor)):
      features = {'features': features}
    if feature_columns:
      features = features.copy()
      features.update(layers.transform_features(features, feature_columns))

    weights = None
    if weights_name and weights_name in features:
      weights = features.pop(weights_name)

    keys = None
    if keys_name and keys_name in features:
      keys = features.pop(keys_name)

    # If we're doing eval, optionally ignore device_assigner.
    # Also ignore device assigner if we're exporting (mode == INFER)
    dev_assn = device_assigner
    if (mode == model_fn_lib.ModeKeys.INFER or
        (local_eval and mode == model_fn_lib.ModeKeys.EVAL)):
      dev_assn = None

    graph_builder = graph_builder_class(params,
                                        device_assigner=dev_assn)

    logits, tree_paths, regression_variance = graph_builder.inference_graph(
        features)

    summary.scalar('average_tree_size', graph_builder.average_size())
    # For binary classification problems, convert probabilities to logits.
    # Includes hack to get around the fact that a probability might be 0 or 1.
    if not params.regression and params.num_classes == 2:
      class_1_probs = array_ops.slice(logits, [0, 1], [-1, 1])
      logits = math_ops.log(
          math_ops.maximum(class_1_probs / math_ops.maximum(
              1.0 - class_1_probs, EPSILON), EPSILON))

    # labels might be None if we're doing prediction (which brings up the
    # question of why we force everything to adhere to a single model_fn).
    training_graph = None
    training_hooks = []
    if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
      with ops.control_dependencies([logits.op]):
        training_graph = control_flow_ops.group(
            graph_builder.training_graph(
                features, labels, input_weights=weights,
                num_trainers=num_trainers,
                trainer_id=trainer_id),
            state_ops.assign_add(training_util.get_global_step(), 1))

    # Put weights back in
    if weights is not None:
      features[weights_name] = weights

    # TensorForest's training graph isn't calculated directly from the loss
    # like many other models.
    def _train_fn(unused_loss):
      return training_graph

    model_ops = model_head.create_model_fn_ops(
        features=features,
        labels=labels,
        mode=mode,
        train_op_fn=_train_fn,
        logits=logits,
        scope=head_scope)

    # Ops are run in lexigraphical order of their keys. Run the resource
    # clean-up op last.
    all_handles = graph_builder.get_all_resource_handles()
    ops_at_end = {
        '9: clean up resources': control_flow_ops.group(
            *[resource_variable_ops.destroy_resource_op(handle)
              for handle in all_handles])}

    if report_feature_importances:
      ops_at_end['1: feature_importances'] = (
          graph_builder.feature_importances())

    training_hooks.append(TensorForestRunOpAtEndHook(ops_at_end))

    if early_stopping_rounds:
      training_hooks.append(
          TensorForestLossHook(
              early_stopping_rounds,
              early_stopping_loss_threshold=early_stopping_loss_threshold,
              loss_op=model_ops.loss))

    model_ops.training_hooks.extend(training_hooks)

    if keys is not None:
      model_ops.predictions[keys_name] = keys

    if params.inference_tree_paths:
      model_ops.predictions[TREE_PATHS_PREDICTION_KEY] = tree_paths

    model_ops.predictions[VARIANCE_PREDICTION_KEY] = regression_variance

    return model_ops

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

  def __init__(self,
               params,
               device_assigner=None,
               model_dir=None,
               feature_columns=None,
               graph_builder_class=tensor_forest.RandomForestGraphs,
               config=None,
               weight_column=None,
               keys_column=None,
               feature_engineering_fn=None,
               early_stopping_rounds=100,
               early_stopping_loss_threshold=0.001,
               num_trainers=1,
               trainer_id=0,
               report_feature_importances=False,
               local_eval=False,
               version=None,
               head=None):
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
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `_FeatureColumn`.
      graph_builder_class: An `object` instance that defines how TF graphs for
        random forest training and inference are built. By default will use
        `tensor_forest.RandomForestGraphs`. Can be overridden by version
        kwarg.
      config: `RunConfig` object to configure the runtime settings.
      weight_column: A string defining feature column name representing
        weights. Will be multiplied by the loss of the example. Used to
        downweight or boost examples during training.
      keys_column: A string naming one of the features to strip out and
        pass through into the inference/eval results dict.  Useful for
        associating specific examples with their prediction.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      early_stopping_rounds: Allows training to terminate early if the forest is
        no longer growing. 100 by default.  Set to a Falsy value to disable
        the default training hook.
      early_stopping_loss_threshold: Percentage (as fraction) that loss must
        improve by within early_stopping_rounds steps, otherwise training will
        terminate.
      num_trainers: Number of training jobs, which will partition trees
        among them.
      trainer_id: Which trainer this instance is.
      report_feature_importances: If True, print out feature importances
        during evaluation.
      local_eval: If True, don't use a device assigner for eval. This is to
        support some common setups where eval is done on a single machine, even
        though training might be distributed.
      version: Unused.
      head: A heads_lib.Head object that calculates losses and such. If None,
        one will be automatically created based on params.

    Returns:
      A `TensorForestEstimator` instance.
    """
    super(TensorForestEstimator, self).__init__(
        model_fn=get_model_fn(
            params.fill(),
            graph_builder_class,
            device_assigner,
            feature_columns=feature_columns,
            model_head=head,
            weights_name=weight_column,
            keys_name=keys_column,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_loss_threshold=early_stopping_loss_threshold,
            num_trainers=num_trainers,
            trainer_id=trainer_id,
            report_feature_importances=report_feature_importances,
            local_eval=local_eval),
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


def get_combined_model_fn(model_fns):
  """Get a combined model function given a list of other model fns.

  The model function returned will call the individual model functions and
  combine them appropriately.  For:

  training ops: tf.group them.
  loss: average them.
  predictions: concat probabilities such that predictions[*][0-C1] are the
    probablities for output 1 (where C1 is the number of classes in output 1),
    predictions[*][C1-(C1+C2)] are the probabilities for output 2 (where C2
    is the number of classes in output 2), etc.  Also stack predictions such
    that predictions[i][j] is the class prediction for example i and output j.

  This assumes that labels are 2-dimensional, with labels[i][j] being the
  label for example i and output j, where forest j is trained using only
  output j.

  Args:
    model_fns: A list of model functions obtained from get_model_fn.

  Returns:
    A ModelFnOps instance.
  """
  def _model_fn(features, labels, mode):
    """Function that returns predictions, training loss, and training op."""
    model_fn_ops = []
    for i in range(len(model_fns)):
      with variable_scope.variable_scope('label_{0}'.format(i)):
        sliced_labels = array_ops.slice(labels, [0, i], [-1, 1])
        model_fn_ops.append(
            model_fns[i](features, sliced_labels, mode))
    training_hooks = []
    for mops in model_fn_ops:
      training_hooks += mops.training_hooks
    predictions = {}
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.INFER):
      # Flatten the probabilities into one dimension.
      predictions[eval_metrics.INFERENCE_PROB_NAME] = array_ops.concat(
          [mops.predictions[eval_metrics.INFERENCE_PROB_NAME]
           for mops in model_fn_ops], axis=1)
      predictions[eval_metrics.INFERENCE_PRED_NAME] = array_ops.stack(
          [mops.predictions[eval_metrics.INFERENCE_PRED_NAME]
           for mops in model_fn_ops], axis=1)
    loss = None
    if (mode == model_fn_lib.ModeKeys.EVAL or
        mode == model_fn_lib.ModeKeys.TRAIN):
      loss = math_ops.reduce_sum(
          array_ops.stack(
              [mops.loss for mops in model_fn_ops])) / len(model_fn_ops)

    train_op = None
    if mode == model_fn_lib.ModeKeys.TRAIN:
      train_op = control_flow_ops.group(
          *[mops.train_op for mops in model_fn_ops])
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        scaffold=None,
        output_alternatives=None)

  return _model_fn


class MultiForestMultiHeadEstimator(estimator.Estimator):
  """An estimator that can train a forest for a multi-headed problems.

  This class essentially trains separate forests (each with their own
  ForestHParams) for each output.

  For multi-headed regression, a single-headed TensorForestEstimator can
  be used to train a single model that predicts all outputs.  This class can
  be used to train separate forests for each output.
  """

  def __init__(self,
               params_list,
               device_assigner=None,
               model_dir=None,
               feature_columns=None,
               graph_builder_class=tensor_forest.RandomForestGraphs,
               config=None,
               weight_column=None,
               keys_column=None,
               feature_engineering_fn=None,
               early_stopping_rounds=100,
               num_trainers=1,
               trainer_id=0,
               report_feature_importances=False,
               local_eval=False):
    """See TensorForestEstimator.__init__."""
    model_fns = []
    for i in range(len(params_list)):
      params = params_list[i].fill()
      model_fns.append(
          get_model_fn(
              params,
              graph_builder_class,
              device_assigner,
              model_head=get_default_head(
                  params, weight_column, name='head{0}'.format(i)),
              weights_name=weight_column,
              keys_name=keys_column,
              early_stopping_rounds=early_stopping_rounds,
              num_trainers=num_trainers,
              trainer_id=trainer_id,
              report_feature_importances=report_feature_importances,
              local_eval=local_eval,
              head_scope='output{0}'.format(i)))

    super(MultiForestMultiHeadEstimator, self).__init__(
        model_fn=get_combined_model_fn(model_fns),
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)
