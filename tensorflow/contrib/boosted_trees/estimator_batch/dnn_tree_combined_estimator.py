# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow estimators for combined DNN + GBDT training model.

The combined model trains a DNN first, then trains boosted trees to boost the
logits of the DNN. The input layer of the DNN (including the embeddings learned
over sparse features) can optionally be provided to the boosted trees as
an additional input feature.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from tensorflow.contrib import layers
from tensorflow.contrib.boosted_trees.estimator_batch import trainer_hooks
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.training.functions import gbdt_batch
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as contrib_model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util


_DNN_LEARNING_RATE = 0.001

_CORE_MODE_TO_CONTRIB_MODE_ = {
    model_fn_lib.ModeKeys.TRAIN: contrib_model_fn_lib.ModeKeys.TRAIN,
    model_fn_lib.ModeKeys.EVAL: contrib_model_fn_lib.ModeKeys.EVAL,
    model_fn_lib.ModeKeys.PREDICT: contrib_model_fn_lib.ModeKeys.INFER
}


def _core_mode_to_contrib_mode(mode):
  return _CORE_MODE_TO_CONTRIB_MODE_[mode]


def _export_outputs_to_output_alternatives(export_outputs):
  """Converts EstimatorSpec.export_outputs to output_alternatives.

  Args:
    export_outputs: export_outputs created by create_estimator_spec.
  Returns:
    converted output_alternatives.
  """
  output = dict()
  if export_outputs is not None:
    for key, value in export_outputs.items():
      if isinstance(value, export_output.ClassificationOutput):
        exported_predictions = {
            prediction_key.PredictionKey.SCORES: value.scores,
            prediction_key.PredictionKey.CLASSES: value.classes
        }
        output[key] = (constants.ProblemType.CLASSIFICATION,
                       exported_predictions)
    return output
  return None


def _estimator_spec_to_model_fn_ops(estimator_spec, is_regression):
  alternatives = []
  if not is_regression:
    _export_outputs_to_output_alternatives(estimator_spec.export_outputs)

  return model_fn.ModelFnOps(
      mode=_core_mode_to_contrib_mode(estimator_spec.mode),
      predictions=estimator_spec.predictions,
      loss=estimator_spec.loss,
      train_op=estimator_spec.train_op,
      eval_metric_ops=estimator_spec.eval_metric_ops,
      output_alternatives=alternatives)


def _get_optimizer(optimizer):
  if callable(optimizer):
    return optimizer()
  else:
    return optimizer


def _add_hidden_layer_summary(value, tag):
  summary.scalar("%s_fraction_of_zero_values" % tag, nn.zero_fraction(value))
  summary.histogram("%s_activation" % tag, value)


def _dnn_tree_combined_model_fn(features,
                                labels,
                                mode,
                                head,
                                dnn_hidden_units,
                                dnn_feature_columns,
                                tree_learner_config,
                                num_trees,
                                tree_examples_per_layer,
                                config=None,
                                dnn_optimizer="Adagrad",
                                dnn_activation_fn=nn.relu,
                                dnn_dropout=None,
                                dnn_input_layer_partitioner=None,
                                dnn_input_layer_to_tree=True,
                                dnn_steps_to_train=10000,
                                tree_feature_columns=None,
                                tree_center_bias=False,
                                use_core_versions=False,
                                is_regression=False):
  """DNN and GBDT combined model_fn.

  Args:
    features: `dict` of `Tensor` objects.
    labels: Labels used to train on.
    mode: Mode we are in. (TRAIN/EVAL/INFER)
    head: A `Head` instance.
    dnn_hidden_units: List of hidden units per layer.
    dnn_feature_columns: An iterable containing all the feature columns
      used by the model's DNN.
    tree_learner_config: A config for the tree learner.
    num_trees: Number of trees to grow model to after training DNN.
    tree_examples_per_layer: Number of examples to accumulate before
      growing the tree a layer. This value has a big impact on model
      quality and should be set equal to the number of examples in
      training dataset if possible. It can also be a function that computes
      the number of examples based on the depth of the layer that's
      being built.
    config: `RunConfig` of the estimator.
    dnn_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the DNN. If `None`, will use the Adagrad
      optimizer with default learning rate of 0.001.
    dnn_activation_fn: Activation function applied to each layer of the DNN.
      If `None`, will use `tf.nn.relu`.
    dnn_dropout: When not `None`, the probability to drop out a given
      unit in the DNN.
    dnn_input_layer_partitioner: Partitioner for input layer of the DNN.
      Defaults to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    dnn_input_layer_to_tree: Whether to provide the DNN's input layer
    as a feature to the tree.
    dnn_steps_to_train: Number of steps to train dnn for before switching
      to gbdt.
    tree_feature_columns: An iterable containing all the feature columns
      used by the model's boosted trees. If dnn_input_layer_to_tree is
      set to True, these features are in addition to dnn_feature_columns.
    tree_center_bias: Whether a separate tree should be created for
      first fitting the bias.
    use_core_versions: Whether feature columns and loss are from the core (as
      opposed to contrib) version of tensorflow.
    is_regression: Whether the problem is regression or not.

  Returns:
    A `ModelFnOps` object.
  Raises:
    ValueError: if inputs are not valid.
  """
  if not isinstance(features, dict):
    raise ValueError("features should be a dictionary of `Tensor`s. "
                     "Given type: {}".format(type(features)))

  if not dnn_feature_columns:
    raise ValueError("dnn_feature_columns must be specified")

  # Build DNN Logits.
  dnn_parent_scope = "dnn"
  dnn_partitioner = dnn_input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=config.num_ps_replicas,
          min_slice_size=64 << 20))

  with variable_scope.variable_scope(
      dnn_parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=dnn_partitioner):

    with variable_scope.variable_scope(
        "input_from_feature_columns",
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner) as input_layer_scope:
      if use_core_versions:
        input_layer = feature_column_lib.input_layer(
            features=features,
            feature_columns=dnn_feature_columns,
            weight_collections=[dnn_parent_scope])
      else:
        input_layer = layers.input_from_feature_columns(
            columns_to_tensors=features,
            feature_columns=dnn_feature_columns,
            weight_collections=[dnn_parent_scope],
            scope=input_layer_scope)
    previous_layer = input_layer
    for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
      with variable_scope.variable_scope(
          "hiddenlayer_%d" % layer_id,
          values=(previous_layer,)) as hidden_layer_scope:
        net = layers.fully_connected(
            previous_layer,
            num_hidden_units,
            activation_fn=dnn_activation_fn,
            variables_collections=[dnn_parent_scope],
            scope=hidden_layer_scope)
        if dnn_dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = layers.dropout(net, keep_prob=(1.0 - dnn_dropout))
      _add_hidden_layer_summary(net, hidden_layer_scope.name)
      previous_layer = net
    with variable_scope.variable_scope(
        "logits",
        values=(previous_layer,)) as logits_scope:
      dnn_logits = layers.fully_connected(
          previous_layer,
          head.logits_dimension,
          activation_fn=None,
          variables_collections=[dnn_parent_scope],
          scope=logits_scope)
    _add_hidden_layer_summary(dnn_logits, logits_scope.name)

    def _dnn_train_op_fn(loss):
      """Returns the op to optimize the loss."""
      return optimizers.optimize_loss(
          loss=loss,
          global_step=training_util.get_global_step(),
          learning_rate=_DNN_LEARNING_RATE,
          optimizer=_get_optimizer(dnn_optimizer),
          name=dnn_parent_scope,
          variables=ops.get_collection(
              ops.GraphKeys.TRAINABLE_VARIABLES,
              scope=dnn_parent_scope),
          # Empty summaries to prevent optimizers from logging training_loss.
          summaries=[])

  # Build Tree Logits.
  global_step = training_util.get_global_step()
  with ops.device(global_step.device):
    ensemble_handle = model_ops.tree_ensemble_variable(
        stamp_token=0,
        tree_ensemble_config="",  # Initialize an empty ensemble.
        name="ensemble_model")

  tree_features = features.copy()
  if dnn_input_layer_to_tree:
    tree_features["dnn_input_layer"] = input_layer
    tree_feature_columns.append(layers.real_valued_column("dnn_input_layer"))
  gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
      is_chief=config.is_chief,
      num_ps_replicas=config.num_ps_replicas,
      ensemble_handle=ensemble_handle,
      center_bias=tree_center_bias,
      examples_per_layer=tree_examples_per_layer,
      learner_config=tree_learner_config,
      feature_columns=tree_feature_columns,
      logits_dimension=head.logits_dimension,
      features=tree_features)

  with ops.name_scope("gbdt"):
    predictions_dict = gbdt_model.predict(mode)
    tree_logits = predictions_dict["predictions"]

    def _tree_train_op_fn(loss):
      """Returns the op to optimize the loss."""
      update_op = gbdt_model.train(loss, predictions_dict, labels)
      with ops.control_dependencies(
          [update_op]), (ops.colocate_with(global_step)):
        update_op = state_ops.assign_add(global_step, 1).op
        return update_op

  tree_train_logits = dnn_logits + tree_logits

  def _no_train_op_fn(loss):
    """Returns a no-op."""
    del loss
    return control_flow_ops.no_op()

  if use_core_versions:
    model_fn_ops = head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_no_train_op_fn,
        logits=tree_train_logits)
    dnn_train_op = head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_dnn_train_op_fn,
        logits=dnn_logits)
    dnn_train_op = _estimator_spec_to_model_fn_ops(dnn_train_op,
                                                   is_regression).train_op

    tree_train_op = head.create_estimator_spec(
        features=tree_features,
        mode=mode,
        labels=labels,
        train_op_fn=_tree_train_op_fn,
        logits=tree_train_logits)
    tree_train_op = _estimator_spec_to_model_fn_ops(tree_train_op,
                                                    is_regression).train_op

    model_fn_ops = _estimator_spec_to_model_fn_ops(model_fn_ops, is_regression)
  else:
    model_fn_ops = head.create_model_fn_ops(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_no_train_op_fn,
        logits=tree_train_logits)
    dnn_train_op = head.create_model_fn_ops(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_dnn_train_op_fn,
        logits=dnn_logits).train_op
    tree_train_op = head.create_model_fn_ops(
        features=tree_features,
        mode=mode,
        labels=labels,
        train_op_fn=_tree_train_op_fn,
        logits=tree_train_logits).train_op

  if tree_center_bias:
    num_trees += 1
  finalized_trees, attempted_trees = gbdt_model.get_number_of_trees_tensor()

  model_fn_ops.training_hooks.extend([
      trainer_hooks.SwitchTrainOp(
          dnn_train_op, dnn_steps_to_train, tree_train_op),
      trainer_hooks.StopAfterNTrees(
          num_trees, attempted_trees, finalized_trees)])

  return model_fn_ops


class DNNBoostedTreeCombinedClassifier(estimator.Estimator):
  """A classifier that uses a combined DNN/GBDT model."""

  def __init__(self,
               dnn_hidden_units,
               dnn_feature_columns,
               tree_learner_config,
               num_trees,
               tree_examples_per_layer,
               n_classes=2,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_name=None,
               label_keys=None,
               feature_engineering_fn=None,
               dnn_optimizer="Adagrad",
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               dnn_input_layer_partitioner=None,
               dnn_input_layer_to_tree=True,
               dnn_steps_to_train=10000,
               tree_feature_columns=None,
               tree_center_bias=False,
               use_core_versions=False):
    """Initializes a DNNBoostedTreeCombinedClassifier instance.

    Args:
      dnn_hidden_units: List of hidden units per layer for DNN.
      dnn_feature_columns: An iterable containing all the feature columns
        used by the model's DNN.
      tree_learner_config: A config for the tree learner.
      num_trees: Number of trees to grow model to after training DNN.
      tree_examples_per_layer: Number of examples to accumulate before
        growing the tree a layer. This value has a big impact on model
        quality and should be set equal to the number of examples in
        training dataset if possible. It can also be a function that computes
        the number of examples based on the depth of the layer that's
        being built.
      n_classes: The number of label classes.
      weight_column_name: The name of weight column.
      model_dir: Directory for model exports.
      config: `RunConfig` of the estimator.
      label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      dnn_optimizer: string, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN. If `None`, will use the Adagrad
        optimizer with default learning rate.
      dnn_activation_fn: Activation function applied to each layer of the DNN.
        If `None`, will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability to drop out a given
        unit in the DNN.
      dnn_input_layer_partitioner: Partitioner for input layer of the DNN.
        Defaults to `min_max_variable_partitioner` with `min_slice_size`
        64 << 20.
      dnn_input_layer_to_tree: Whether to provide the DNN's input layer
      as a feature to the tree.
      dnn_steps_to_train: Number of steps to train dnn for before switching
        to gbdt.
      tree_feature_columns: An iterable containing all the feature columns
        used by the model's boosted trees. If dnn_input_layer_to_tree is
        set to True, these features are in addition to dnn_feature_columns.
      tree_center_bias: Whether a separate tree should be created for
        first fitting the bias.
      use_core_versions: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
    """
    head = head_lib.multi_class_head(
        n_classes=n_classes,
        label_name=label_name,
        label_keys=label_keys,
        weight_column_name=weight_column_name,
        enable_centered_bias=False)

    def _model_fn(features, labels, mode, config):
      return _dnn_tree_combined_model_fn(
          features, labels, mode, head, dnn_hidden_units, dnn_feature_columns,
          tree_learner_config, num_trees, tree_examples_per_layer, config,
          dnn_optimizer, dnn_activation_fn, dnn_dropout,
          dnn_input_layer_partitioner, dnn_input_layer_to_tree,
          dnn_steps_to_train, tree_feature_columns, tree_center_bias,
          use_core_versions)

    super(DNNBoostedTreeCombinedClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir,
        config=config, feature_engineering_fn=feature_engineering_fn)


class DNNBoostedTreeCombinedRegressor(estimator.Estimator):
  """A regressor that uses a combined DNN/GBDT model."""

  def __init__(self,
               dnn_hidden_units,
               dnn_feature_columns,
               tree_learner_config,
               num_trees,
               tree_examples_per_layer,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_name=None,
               label_dimension=1,
               feature_engineering_fn=None,
               dnn_optimizer="Adagrad",
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               dnn_input_layer_partitioner=None,
               dnn_input_layer_to_tree=True,
               dnn_steps_to_train=10000,
               tree_feature_columns=None,
               tree_center_bias=False,
               use_core_versions=False):
    """Initializes a DNNBoostedTreeCombinedRegressor instance.

    Args:
      dnn_hidden_units: List of hidden units per layer for DNN.
      dnn_feature_columns: An iterable containing all the feature columns
        used by the model's DNN.
      tree_learner_config: A config for the tree learner.
      num_trees: Number of trees to grow model to after training DNN.
      tree_examples_per_layer: Number of examples to accumulate before
        growing the tree a layer. This value has a big impact on model
        quality and should be set equal to the number of examples in
        training dataset if possible. It can also be a function that computes
        the number of examples based on the depth of the layer that's
        being built.
      weight_column_name: The name of weight column.
      model_dir: Directory for model exports.
      config: `RunConfig` of the estimator.
      label_name: String, name of the key in label dict. Can be null if label
        is a tensor (single headed models).
      label_dimension: Number of regression labels per example. This is the size
        of the last dimension of the labels `Tensor` (typically, this has shape
        `[batch_size, label_dimension]`).
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      dnn_optimizer: string, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN. If `None`, will use the Adagrad
        optimizer with default learning rate.
      dnn_activation_fn: Activation function applied to each layer of the DNN.
        If `None`, will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability to drop out a given
        unit in the DNN.
      dnn_input_layer_partitioner: Partitioner for input layer of the DNN.
        Defaults to `min_max_variable_partitioner` with `min_slice_size`
        64 << 20.
      dnn_input_layer_to_tree: Whether to provide the DNN's input layer
      as a feature to the tree.
      dnn_steps_to_train: Number of steps to train dnn for before switching
        to gbdt.
      tree_feature_columns: An iterable containing all the feature columns
        used by the model's boosted trees. If dnn_input_layer_to_tree is
        set to True, these features are in addition to dnn_feature_columns.
      tree_center_bias: Whether a separate tree should be created for
        first fitting the bias.
      use_core_versions: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
    """
    head = head_lib.regression_head(
        label_name=label_name,
        label_dimension=label_dimension,
        weight_column_name=weight_column_name,
        enable_centered_bias=False)

    # num_classes needed for GradientBoostedDecisionTreeModel
    if label_dimension == 1:
      tree_learner_config.num_classes = 2
    else:
      tree_learner_config.num_classes = label_dimension

    def _model_fn(features, labels, mode, config):
      return _dnn_tree_combined_model_fn(
          features,
          labels,
          mode,
          head,
          dnn_hidden_units,
          dnn_feature_columns,
          tree_learner_config,
          num_trees,
          tree_examples_per_layer,
          config,
          dnn_optimizer,
          dnn_activation_fn,
          dnn_dropout,
          dnn_input_layer_partitioner,
          dnn_input_layer_to_tree,
          dnn_steps_to_train,
          tree_feature_columns,
          tree_center_bias,
          use_core_versions,
          is_regression=True)

    super(DNNBoostedTreeCombinedRegressor, self).__init__(
        model_fn=_model_fn, model_dir=model_dir,
        config=config, feature_engineering_fn=feature_engineering_fn)


class DNNBoostedTreeCombinedEstimator(estimator.Estimator):
  """An estimator that uses a combined DNN/GBDT model.

  Useful for training with user specified `Head`.
  """

  def __init__(self,
               dnn_hidden_units,
               dnn_feature_columns,
               tree_learner_config,
               num_trees,
               tree_examples_per_layer,
               head,
               model_dir=None,
               config=None,
               feature_engineering_fn=None,
               dnn_optimizer="Adagrad",
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               dnn_input_layer_partitioner=None,
               dnn_input_layer_to_tree=True,
               dnn_steps_to_train=10000,
               tree_feature_columns=None,
               tree_center_bias=False,
               use_core_versions=False):
    """Initializes a DNNBoostedTreeCombinedEstimator instance.

    Args:
      dnn_hidden_units: List of hidden units per layer for DNN.
      dnn_feature_columns: An iterable containing all the feature columns
        used by the model's DNN.
      tree_learner_config: A config for the tree learner.
      num_trees: Number of trees to grow model to after training DNN.
      tree_examples_per_layer: Number of examples to accumulate before
        growing the tree a layer. This value has a big impact on model
        quality and should be set equal to the number of examples in
        training dataset if possible. It can also be a function that computes
        the number of examples based on the depth of the layer that's
        being built.
      head: `Head` instance.
      model_dir: Directory for model exports.
      config: `RunConfig` of the estimator.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      dnn_optimizer: string, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN. If `None`, will use the Adagrad
        optimizer with default learning rate.
      dnn_activation_fn: Activation function applied to each layer of the DNN.
        If `None`, will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability to drop out a given
        unit in the DNN.
      dnn_input_layer_partitioner: Partitioner for input layer of the DNN.
        Defaults to `min_max_variable_partitioner` with `min_slice_size`
        64 << 20.
      dnn_input_layer_to_tree: Whether to provide the DNN's input layer
      as a feature to the tree.
      dnn_steps_to_train: Number of steps to train dnn for before switching
        to gbdt.
      tree_feature_columns: An iterable containing all the feature columns
        used by the model's boosted trees. If dnn_input_layer_to_tree is
        set to True, these features are in addition to dnn_feature_columns.
      tree_center_bias: Whether a separate tree should be created for
        first fitting the bias.
      use_core_versions: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
    """
    def _model_fn(features, labels, mode, config):
      return _dnn_tree_combined_model_fn(
          features, labels, mode, head, dnn_hidden_units, dnn_feature_columns,
          tree_learner_config, num_trees, tree_examples_per_layer, config,
          dnn_optimizer, dnn_activation_fn, dnn_dropout,
          dnn_input_layer_partitioner, dnn_input_layer_to_tree,
          dnn_steps_to_train, tree_feature_columns, tree_center_bias,
          use_core_versions)

    super(DNNBoostedTreeCombinedEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir,
        config=config, feature_engineering_fn=feature_engineering_fn)
