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
"""Training functions for Gradient boosted decision trees."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

from tensorflow.contrib import learn
from tensorflow.contrib.boosted_trees.lib.learner.batch import categorical_split_handler
from tensorflow.contrib.boosted_trees.lib.learner.batch import ordinal_split_handler
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.python.ops import batch_ops_utils
from tensorflow.contrib.boosted_trees.python.ops import gen_model_ops
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.ops import prediction_ops
from tensorflow.contrib.boosted_trees.python.ops import stats_accumulator_ops
from tensorflow.contrib.boosted_trees.python.ops import training_ops
from tensorflow.contrib.layers.python.layers import feature_column as feature_column_lib
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import device_setter


# Key names for prediction dict.
ENSEMBLE_STAMP = "ensemble_stamp"
PREDICTIONS = "predictions"
PARTITION_IDS = "partition_ids"
NUM_LAYERS_ATTEMPTED = "num_layers"
NUM_TREES_ATTEMPTED = "num_trees"
NUM_USED_HANDLERS = "num_used_handlers"
USED_HANDLERS_MASK = "used_handlers_mask"
LEAF_INDEX = "leaf_index"
_FEATURE_NAME_TEMPLATE = "%s_%d"

# Keys in Training state.
GBDTTrainingState = collections.namedtuple("GBDTTrainingState", [
    "num_layer_examples", "num_layer_steps", "num_layers", "active_tree",
    "active_layer", "continue_centering", "bias_stats_accumulator",
    "steps_accumulator", "handlers"
])


def _get_column_by_index(tensor, indices):
  """Returns columns from a 2-D tensor by index."""
  shape = array_ops.shape(tensor)
  p_flat = array_ops.reshape(tensor, [-1])
  i_flat = array_ops.reshape(
      array_ops.reshape(math_ops.range(0, shape[0]) * shape[1], [-1, 1]) +
      indices, [-1])
  return array_ops.reshape(array_ops.gather(p_flat, i_flat), [shape[0], -1])


def _make_predictions_dict(stamp,
                           logits,
                           partition_ids,
                           ensemble_stats,
                           used_handlers,
                           leaf_index=None):
  """Returns predictions for the given logits and n_classes.

  Args:
    stamp: The ensemble stamp.
    logits: A rank 2 `Tensor` with shape [batch_size, n_classes - 1]. that
      contains predictions when no dropout was applied.
    partition_ids: A rank 1 `Tensor` with shape [batch_size].
    ensemble_stats: A TreeEnsembleStatsOp result tuple.
    used_handlers: A TreeEnsembleUsedHandlerOp result tuple of an int and a
      boolean mask.
    leaf_index: A rank 2 `Tensor` with shape [batch_size, number of trees]. that
      contains leaf id for each example prediction.

  Returns:
    A dict of predictions.
  """
  result = {}
  result[ENSEMBLE_STAMP] = stamp
  result[PREDICTIONS] = logits
  result[PARTITION_IDS] = partition_ids
  result[NUM_LAYERS_ATTEMPTED] = ensemble_stats.attempted_layers
  result[NUM_TREES_ATTEMPTED] = ensemble_stats.attempted_trees
  result[NUM_USED_HANDLERS] = used_handlers.num_used_handlers
  result[USED_HANDLERS_MASK] = used_handlers.used_handlers_mask
  if leaf_index is not None:
    result[LEAF_INDEX] = leaf_index
  return result


class _OpRoundRobinStrategy(object):
  """Returns the next ps task index for placement via per-Op round-robin order.

  This strategy works slightly better for the GBDT graph because of using
  custom resources which vary significantly in compute cost.
  """

  def __init__(self, ps_ops, num_tasks):
    """Create a new `_RoundRobinStrategy`.

    Args:
      ps_ops: List of Op types to place on PS.
      num_tasks: Number of ps tasks to cycle among.
    """
    next_task = 0
    self._next_task_per_op = {}
    for op in ps_ops:
      self._next_task_per_op[op] = next_task
      next_task = (next_task + 1) % num_tasks if num_tasks else 0
    self._num_tasks = num_tasks

  def __call__(self, op):
    """Choose a ps task index for the given `Operation`.

    Args:
      op: An `Operation` to be placed on ps.

    Returns:
      The next ps task index to use for the `Operation`. Returns the next
      index, in the range `[offset, offset + num_tasks)`.

    Raises:
      ValueError: If attempting to place non-PS Op.
    """
    if op.type not in self._next_task_per_op:
      raise ValueError("Unknown op type '%s' for placement:" % op.type)
    task = self._next_task_per_op[op.type]
    self._next_task_per_op[op.type] = ((task + 1) % self._num_tasks
                                       if self._num_tasks else 0)
    return task


def extract_features(features, feature_columns, use_core_columns):
  """Extracts columns from a dictionary of features.

  Args:
    features: `dict` of `Tensor` objects.
    feature_columns: A list of feature_columns.

  Returns:
    Seven values:
      - A list of all feature column names.
      - A list of dense floats.
      - A list of sparse float feature indices.
      - A list of sparse float feature values.
      - A list of sparse float feature shapes.
      - A list of sparse int feature indices.
      - A list of sparse int feature values.
      - A list of sparse int feature shapes.
  Raises:
    ValueError: if features is not valid.
  """
  if not features:
    raise ValueError("Features dictionary must be specified.")

  # Make a shallow copy of features to ensure downstream usage
  # is unaffected by modifications in the model function.
  features = copy.copy(features)
  if feature_columns:
    scope = "gbdt"
    with variable_scope.variable_scope(scope):
      feature_columns = list(feature_columns)
      transformed_features = collections.OrderedDict()
      for fc in feature_columns:
        # pylint: disable=protected-access
        if use_core_columns:
          # pylint: disable=protected-access
          tensor = fc_core._transform_features(features, [fc])[fc]
          transformed_features[fc.name] = tensor
        elif isinstance(fc, feature_column_lib._EmbeddingColumn):
          # pylint: enable=protected-access
          transformed_features[fc.name] = fc_core.input_layer(
              features, [fc], weight_collections=[scope])
        else:
          result = feature_column_ops.transform_features(features, [fc])
          if len(result) > 1:
            raise ValueError("Unexpected number of output features")
          transformed_features[fc.name] = result[list(result.keys())[0]]
    features = transformed_features

  dense_float_names = []
  dense_floats = []
  sparse_float_names = []
  sparse_float_indices = []
  sparse_float_values = []
  sparse_float_shapes = []
  sparse_int_names = []
  sparse_int_indices = []
  sparse_int_values = []
  sparse_int_shapes = []
  for key in sorted(features.keys()):
    tensor = features[key]
    # TODO(nponomareva): consider iterating over feature columns instead.
    if isinstance(tensor, tuple):
      # Weighted categorical feature.
      categorical_tensor = tensor[0]
      weight_tensor = tensor[1]

      shape = categorical_tensor.dense_shape
      indices = array_ops.concat([
          array_ops.slice(categorical_tensor.indices, [0, 0], [-1, 1]),
          array_ops.expand_dims(
              math_ops.to_int64(categorical_tensor.values), -1)
      ], 1)
      tensor = sparse_tensor.SparseTensor(
          indices=indices, values=weight_tensor.values, dense_shape=shape)

    if isinstance(tensor, sparse_tensor.SparseTensor):
      if tensor.values.dtype == dtypes.float32:
        sparse_float_names.append(key)
        sparse_float_indices.append(tensor.indices)
        sparse_float_values.append(tensor.values)
        sparse_float_shapes.append(tensor.dense_shape)
      elif tensor.values.dtype == dtypes.int64:
        sparse_int_names.append(key)
        sparse_int_indices.append(tensor.indices)
        sparse_int_values.append(tensor.values)
        sparse_int_shapes.append(tensor.dense_shape)
      else:
        raise ValueError("Unsupported sparse feature %s with dtype %s." %
                         (tensor.indices.name, tensor.dtype))
    else:
      if tensor.dtype == dtypes.float32:
        if len(tensor.shape) > 1 and tensor.shape[1] > 1:
          unstacked = array_ops.unstack(tensor, axis=1)
          for i in range(len(unstacked)):
            dense_float_names.append(_FEATURE_NAME_TEMPLATE % (key, i))
            dense_floats.append(array_ops.reshape(unstacked[i], [-1, 1]))
        else:
          dense_float_names.append(key)
          dense_floats.append(tensor)
      else:
        raise ValueError("Unsupported dense feature %s with dtype %s." %
                         (tensor.name, tensor.dtype))
  # Feature columns are logically organized into incrementing slots starting
  # from dense floats, then sparse floats then sparse ints.
  fc_names = (dense_float_names + sparse_float_names + sparse_int_names)
  return (fc_names, dense_floats, sparse_float_indices, sparse_float_values,
          sparse_float_shapes, sparse_int_indices, sparse_int_values,
          sparse_int_shapes)


def _dropout_params(mode, ensemble_stats):
  """Returns parameters relevant for dropout.

  Args:
    mode: Train/Eval/Infer
    ensemble_stats: A TreeEnsembleStatsOp result tuple.

  Returns:
    Whether to apply dropout and a dropout seed.
  """
  if mode == learn.ModeKeys.TRAIN:
    # Do dropout only during training.
    apply_dropout = True
    seed = ensemble_stats.attempted_trees
  else:
    seed = -1
    apply_dropout = False
  return apply_dropout, seed


class GradientBoostedDecisionTreeModel(object):
  """A GBDT model function."""

  def __init__(self,
               is_chief,
               num_ps_replicas,
               ensemble_handle,
               center_bias,
               examples_per_layer,
               learner_config,
               features,
               logits_dimension,
               loss_reduction=losses.Reduction.SUM_OVER_NONZERO_WEIGHTS,
               feature_columns=None,
               use_core_columns=False,
               output_leaf_index=False,
               output_leaf_index_modes=None,
               num_quantiles=100):
    """Construct a new GradientBoostedDecisionTreeModel function.

    Args:
      is_chief: Whether to build the chief graph.
      num_ps_replicas: Number of parameter server replicas, can be 0.
      ensemble_handle: A handle to the ensemble variable.
      center_bias: Whether to center the bias before growing trees.
      examples_per_layer: Number of examples to accumulate before growing a tree
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      learner_config: A learner config.
      features: `dict` of `Tensor` objects.
      logits_dimension: An int, the dimension of logits.
      loss_reduction: Either `SUM_OVER_NONZERO_WEIGHTS` (mean) or `SUM`.
      feature_columns: A list of feature columns.
      use_core_columns: A boolean specifying whether core feature columns are
        used.
      output_leaf_index: A boolean variable indicating whether to output leaf
        index into predictions dictionary.
      output_leaf_index_modes: A list of modes from (TRAIN, EVAL, INFER) which
        dictates when leaf indices will be outputted. By default, leaf indices
        are only outputted in INFER mode.
      num_quantiles: Number of quantiles to build for numeric feature values.

    Raises:
      ValueError: if inputs are not valid.
    """
    if ensemble_handle is None:
      raise ValueError("ensemble_handle must be specified.")

    if learner_config is None:
      raise ValueError("learner_config must be specified.")

    if learner_config.num_classes < 2:
      raise ValueError("Number of classes must be >=2")

    self._logits_dimension = logits_dimension
    self._is_chief = is_chief
    self._num_ps_replicas = num_ps_replicas
    self._ensemble_handle = ensemble_handle
    self._center_bias = center_bias
    self._examples_per_layer = examples_per_layer

    # Check loss reduction value.
    if (loss_reduction != losses.Reduction.SUM and
        loss_reduction != losses.Reduction.SUM_OVER_NONZERO_WEIGHTS):
      raise ValueError(
          "Invalid loss reduction is provided: %s." % loss_reduction)
    self._loss_reduction = loss_reduction

    # Fill in the defaults.
    if (learner_config.multi_class_strategy ==
        learner_pb2.LearnerConfig.MULTI_CLASS_STRATEGY_UNSPECIFIED):
      if logits_dimension == 1:
        learner_config.multi_class_strategy = (
            learner_pb2.LearnerConfig.TREE_PER_CLASS)
      else:
        learner_config.multi_class_strategy = (
            learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)

    if logits_dimension == 1 or learner_config.multi_class_strategy == (
        learner_pb2.LearnerConfig.TREE_PER_CLASS):
      self._gradient_shape = tensor_shape.scalar()
      self._hessian_shape = tensor_shape.scalar()
    else:
      if center_bias:
        raise ValueError("Center bias should be False for multiclass.")

      self._gradient_shape = tensor_shape.TensorShape([logits_dimension])
      if (learner_config.multi_class_strategy ==
          learner_pb2.LearnerConfig.FULL_HESSIAN):
        self._hessian_shape = tensor_shape.TensorShape(
            ([logits_dimension, logits_dimension]))
      else:
        # Diagonal hessian strategy.
        self._hessian_shape = tensor_shape.TensorShape(([logits_dimension]))
    if (learner_config.growing_mode ==
        learner_pb2.LearnerConfig.GROWING_MODE_UNSPECIFIED):
      learner_config.growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER

    if (learner_config.weak_learner_type == learner_pb2.LearnerConfig
        .OBLIVIOUS_DECISION_TREE and learner_config.pruning_mode == learner_pb2
        .LearnerConfig.PRUNING_MODE_UNSPECIFIED):
      learner_config.pruning_mode = learner_pb2.LearnerConfig.PRE_PRUNE

    if (learner_config.pruning_mode ==
        learner_pb2.LearnerConfig.PRUNING_MODE_UNSPECIFIED):
      learner_config.pruning_mode = learner_pb2.LearnerConfig.POST_PRUNE

    if (learner_config.weak_learner_type == learner_pb2.LearnerConfig
        .OBLIVIOUS_DECISION_TREE and
        learner_config.pruning_mode == learner_pb2.LearnerConfig.POST_PRUNE):
      raise ValueError(
          "Post pruning is not implmented for oblivious decision trees.")

    if learner_config.constraints.max_tree_depth == 0:
      # Use 6 as the default maximum depth.
      learner_config.constraints.max_tree_depth = 6

    tuner = learner_config.learning_rate_tuner.WhichOneof("tuner")
    if not tuner:
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1

    self._learner_config = learner_config
    self._feature_columns = feature_columns
    self._learner_config_serialized = learner_config.SerializeToString()
    self._num_quantiles = num_quantiles
    self._max_tree_depth = variables.VariableV1(
        initial_value=self._learner_config.constraints.max_tree_depth)
    self._attempted_trees = variables.VariableV1(
        initial_value=array_ops.zeros([], dtypes.int64),
        trainable=False,
        name="attempted_trees")
    self._finalized_trees = variables.VariableV1(
        initial_value=array_ops.zeros([], dtypes.int64),
        trainable=False,
        name="finalized_trees")
    if not features:
      raise ValueError("Features dictionary must be specified.")
    (fc_names, dense_floats, sparse_float_indices, sparse_float_values,
     sparse_float_shapes, sparse_int_indices,
     sparse_int_values, sparse_int_shapes) = extract_features(
         features, self._feature_columns, use_core_columns)
    if (learner_config.weak_learner_type == learner_pb2.LearnerConfig
        .OBLIVIOUS_DECISION_TREE and sparse_float_indices):
      raise ValueError("Oblivious trees don't handle sparse float features yet."
                      )

    logging.info("Active Feature Columns: " + str(fc_names))
    logging.info("Learner config: " + str(learner_config))
    self._fc_names = fc_names
    self._dense_floats = dense_floats
    self._sparse_float_indices = sparse_float_indices
    self._sparse_float_values = sparse_float_values
    self._sparse_float_shapes = sparse_float_shapes
    self._sparse_int_indices = sparse_int_indices
    self._sparse_int_values = sparse_int_values
    self._sparse_int_shapes = sparse_int_shapes
    self._reduce_dim = (
        self._learner_config.multi_class_strategy ==
        learner_pb2.LearnerConfig.TREE_PER_CLASS and
        learner_config.num_classes == 2)

    if output_leaf_index_modes is None:
      output_leaf_index_modes = [learn.ModeKeys.INFER]
    elif not all(
        mode in (learn.ModeKeys.TRAIN, learn.ModeKeys.EVAL,
                 learn.ModeKeys.INFER) for mode in output_leaf_index_modes):
      raise ValueError("output_leaf_index_modes should only contain ModeKeys.")

    self._output_leaf_index = output_leaf_index
    self._output_leaf_index_modes = output_leaf_index_modes

  def _predict_and_return_dict(self, ensemble_handle, ensemble_stamp, mode):
    """Runs prediction and returns a dictionary of the prediction results.

    Args:
      ensemble_handle: ensemble resource handle.
      ensemble_stamp: stamp of ensemble resource.
      mode: learn.ModeKeys.TRAIN or EVAL or INFER.

    Returns:
      a dictionary of prediction results -
        ENSEMBLE_STAMP, PREDICTION, PARTITION_IDS,
        NUM_LAYER_ATTEMPTED, NUM_TREES_ATTEMPTED.
    """
    ensemble_stats = training_ops.tree_ensemble_stats(ensemble_handle,
                                                      ensemble_stamp)
    num_handlers = (
        len(self._dense_floats) + len(self._sparse_float_shapes) + len(
            self._sparse_int_shapes))
    # Used during feature selection.
    used_handlers = model_ops.tree_ensemble_used_handlers(
        ensemble_handle, ensemble_stamp, num_all_handlers=num_handlers)

    # We don't need dropout info - we can always restore it based on the
    # seed.
    apply_dropout, seed = _dropout_params(mode, ensemble_stats)
    # Make sure ensemble stats run. This will check that the ensemble has
    # the right stamp.
    with ops.control_dependencies(ensemble_stats):
      leaf_index = None
      if self._output_leaf_index and mode in self._output_leaf_index_modes:
        predictions, _, leaf_index = (
            prediction_ops).gradient_trees_prediction_verbose(
                ensemble_handle,
                seed,
                self._dense_floats,
                self._sparse_float_indices,
                self._sparse_float_values,
                self._sparse_float_shapes,
                self._sparse_int_indices,
                self._sparse_int_values,
                self._sparse_int_shapes,
                learner_config=self._learner_config_serialized,
                apply_dropout=apply_dropout,
                apply_averaging=mode != learn.ModeKeys.TRAIN,
                use_locking=True,
                center_bias=self._center_bias,
                reduce_dim=self._reduce_dim)
      else:
        leaf_index = None
        predictions, _ = prediction_ops.gradient_trees_prediction(
            ensemble_handle,
            seed,
            self._dense_floats,
            self._sparse_float_indices,
            self._sparse_float_values,
            self._sparse_float_shapes,
            self._sparse_int_indices,
            self._sparse_int_values,
            self._sparse_int_shapes,
            learner_config=self._learner_config_serialized,
            apply_dropout=apply_dropout,
            apply_averaging=mode != learn.ModeKeys.TRAIN,
            use_locking=True,
            center_bias=self._center_bias,
            reduce_dim=self._reduce_dim)
      partition_ids = prediction_ops.gradient_trees_partition_examples(
          ensemble_handle,
          self._dense_floats,
          self._sparse_float_indices,
          self._sparse_float_values,
          self._sparse_float_shapes,
          self._sparse_int_indices,
          self._sparse_int_values,
          self._sparse_int_shapes,
          use_locking=True)

    return _make_predictions_dict(ensemble_stamp, predictions, partition_ids,
                                  ensemble_stats, used_handlers, leaf_index)

  def predict(self, mode):
    """Returns predictions given the features and mode.

    Args:
      mode: Mode the graph is running in (train|predict|eval).

    Returns:
      A dict of predictions tensors.

    Raises:
      ValueError: if features is not valid.
    """

    # Use the current ensemble to predict on the current batch of input.
    # For faster prediction we check if the inputs are on the same device
    # as the model. If not, we create a copy of the model on the worker.
    input_deps = (
        self._dense_floats + self._sparse_float_indices +
        self._sparse_int_indices)
    if not input_deps:
      raise ValueError("No input tensors for prediction.")

    # Get most current model stamp.
    ensemble_stamp = model_ops.tree_ensemble_stamp_token(self._ensemble_handle)

    # Determine if ensemble is colocated with the inputs.
    if self._ensemble_handle.device != input_deps[0].device:
      # Create a local ensemble and get its local stamp.
      with ops.name_scope("local_ensemble", "TreeEnsembleVariable"):
        local_ensemble_handle = (
            gen_model_ops.decision_tree_ensemble_resource_handle_op(
                self._ensemble_handle.op.name + "/local_ensemble"))
        create_op = gen_model_ops.create_tree_ensemble_variable(
            local_ensemble_handle, stamp_token=-1, tree_ensemble_config="")
        with ops.control_dependencies([create_op]):
          local_stamp = model_ops.tree_ensemble_stamp_token(
              local_ensemble_handle)

      # Determine whether the local ensemble is stale and update it if needed.
      def _refresh_local_ensemble_fn():
        # Serialize the model from parameter server after reading the inputs.
        with ops.control_dependencies([input_deps[0]]):
          (ensemble_stamp, serialized_model) = (
              model_ops.tree_ensemble_serialize(self._ensemble_handle))

        # Update local ensemble with the serialized model from parameter server.
        with ops.control_dependencies([create_op]):
          return model_ops.tree_ensemble_deserialize(
              local_ensemble_handle,
              stamp_token=ensemble_stamp,
              tree_ensemble_config=serialized_model), ensemble_stamp

      refresh_local_ensemble, ensemble_stamp = control_flow_ops.cond(
          math_ops.not_equal(ensemble_stamp,
                             local_stamp), _refresh_local_ensemble_fn,
          lambda: (control_flow_ops.no_op(), ensemble_stamp))

      # Once updated, use the local model for prediction.
      with ops.control_dependencies([refresh_local_ensemble]):
        return self._predict_and_return_dict(local_ensemble_handle,
                                             ensemble_stamp, mode)
    else:
      # Use ensemble_handle directly, if colocated.
      with ops.device(self._ensemble_handle.device):
        return self._predict_and_return_dict(self._ensemble_handle,
                                             ensemble_stamp, mode)

  def _get_class_id(self, predictions_dict):
    # Handle different multiclass strategies.
    if (self._learner_config.multi_class_strategy ==
        learner_pb2.LearnerConfig.TREE_PER_CLASS and
        self._logits_dimension != 1):
      # Choose the class for which the tree is built (one vs rest).
      return math_ops.to_int32(
          predictions_dict[NUM_TREES_ATTEMPTED] % self._logits_dimension)
    return constant_op.constant(-1, dtype=dtypes.int32)

  def update_stats(self, loss, predictions_dict, gradients=None, hessians=None):
    """Update the accumulators with stats from this batch.

    Args:
      loss: A scalar tensor representing average loss of examples.
      predictions_dict: Dictionary of Rank 2 `Tensor` representing information
          about predictions per example.
      gradients: A tensor with the gradients with the respect to logits from
        predictions_dict. If not provided, tensorflow will do
        autodifferentiation.
      hessians: A tensor with the hessians with the respect to logits from
        predictions_dict. If not provided, tensorflow will do
        autodifferentiation.

    Returns:
      Three values:
        - An op that adds a new tree to the ensemble, and
        - An op that increments the stamp but removes all the trees and resets
            the handlers. This can be used to reset the state of the ensemble.
        - A dict containing the training state.

    Raises:
      ValueError: if inputs are not valid.
    """
    # Get the worker device from input dependencies.
    input_deps = (
        self._dense_floats + self._sparse_float_indices +
        self._sparse_int_indices)
    worker_device = input_deps[0].device

    # Get tensors relevant for training and form the loss.
    predictions = predictions_dict[PREDICTIONS]
    partition_ids = predictions_dict[PARTITION_IDS]
    ensemble_stamp = predictions_dict[ENSEMBLE_STAMP]
    if gradients is None:
      gradients = gradients_impl.gradients(
          loss,
          predictions,
          name="Gradients",
          colocate_gradients_with_ops=False,
          gate_gradients=0,
          aggregation_method=None)[0]
    strategy = self._learner_config.multi_class_strategy

    class_id = self._get_class_id(predictions_dict)
    # Handle different multiclass strategies.
    if strategy == learner_pb2.LearnerConfig.TREE_PER_CLASS:
      # We build one vs rest trees.
      if self._logits_dimension == 1:
        # We have only 1 score, gradients is of shape [batch, 1].
        if hessians is None:
          hessians = gradients_impl.gradients(
              gradients,
              predictions,
              name="Hessian",
              colocate_gradients_with_ops=False,
              gate_gradients=0,
              aggregation_method=None)[0]

        squeezed_gradients = array_ops.squeeze(gradients, axis=[1])
        squeezed_hessians = array_ops.squeeze(hessians, axis=[1])
      else:
        if hessians is not None:
          raise ValueError("Providing hessians is not yet supported here.")
        hessian_list = self._diagonal_hessian(gradients, predictions)
        # Assemble hessian list into a tensor.
        hessians = array_ops.stack(hessian_list, axis=1)
        # Use class id tensor to get the column with that index from gradients
        # and hessians.
        squeezed_gradients = array_ops.squeeze(
            _get_column_by_index(gradients, class_id))
        squeezed_hessians = array_ops.squeeze(
            _get_column_by_index(hessians, class_id))
    else:
      if hessians is not None:
        raise ValueError("Providing hessians is not yet supported here.")
      # Other multiclass strategies.
      if strategy == learner_pb2.LearnerConfig.FULL_HESSIAN:
        hessian_list = self._full_hessian(gradients, predictions)
      else:
        # Diagonal hessian strategy.
        hessian_list = self._diagonal_hessian(gradients, predictions)

      squeezed_gradients = gradients
      hessians = array_ops.stack(hessian_list, axis=1)
      squeezed_hessians = hessians

    # Get the weights for each example for quantiles calculation,
    weights = self._get_weights(self._hessian_shape, squeezed_hessians)

    # Create all handlers ensuring resources are evenly allocated across PS.
    fc_name_idx = 0
    handlers = []
    init_stamp_token = constant_op.constant(0, dtype=dtypes.int64)
    l1_regularization = constant_op.constant(
        self._learner_config.regularization.l1, dtypes.float32)
    l2_regularization = constant_op.constant(
        self._learner_config.regularization.l2, dtypes.float32)
    tree_complexity_regularization = constant_op.constant(
        self._learner_config.regularization.tree_complexity, dtypes.float32)
    min_node_weight = constant_op.constant(
        self._learner_config.constraints.min_node_weight, dtypes.float32)
    loss_uses_sum_reduction = self._loss_reduction == losses.Reduction.SUM
    loss_uses_sum_reduction = constant_op.constant(loss_uses_sum_reduction)
    weak_learner_type = constant_op.constant(
        self._learner_config.weak_learner_type)
    num_quantiles = self._num_quantiles
    epsilon = 1.0 / num_quantiles
    strategy_tensor = constant_op.constant(strategy)
    with ops.device(self._get_replica_device_setter(worker_device)):
      # Create handlers for dense float columns
      for dense_float_column_idx in range(len(self._dense_floats)):
        fc_name = self._fc_names[fc_name_idx]
        handlers.append(
            ordinal_split_handler.DenseSplitHandler(
                l1_regularization=l1_regularization,
                l2_regularization=l2_regularization,
                tree_complexity_regularization=tree_complexity_regularization,
                min_node_weight=min_node_weight,
                feature_column_group_id=constant_op.constant(
                    dense_float_column_idx),
                epsilon=epsilon,
                num_quantiles=num_quantiles,
                dense_float_column=self._dense_floats[dense_float_column_idx],
                name=fc_name,
                gradient_shape=self._gradient_shape,
                hessian_shape=self._hessian_shape,
                multiclass_strategy=strategy_tensor,
                init_stamp_token=init_stamp_token,
                loss_uses_sum_reduction=loss_uses_sum_reduction,
                weak_learner_type=weak_learner_type,
            ))
        fc_name_idx += 1

      # Create handlers for sparse float columns.
      for sparse_float_column_idx in range(len(self._sparse_float_indices)):
        fc_name = self._fc_names[fc_name_idx]
        handlers.append(
            ordinal_split_handler.SparseSplitHandler(
                l1_regularization=l1_regularization,
                l2_regularization=l2_regularization,
                tree_complexity_regularization=tree_complexity_regularization,
                min_node_weight=min_node_weight,
                feature_column_group_id=constant_op.constant(
                    sparse_float_column_idx),
                epsilon=epsilon,
                num_quantiles=num_quantiles,
                sparse_float_column=sparse_tensor.SparseTensor(
                    self._sparse_float_indices[sparse_float_column_idx],
                    self._sparse_float_values[sparse_float_column_idx],
                    self._sparse_float_shapes[sparse_float_column_idx]),
                name=fc_name,
                gradient_shape=self._gradient_shape,
                hessian_shape=self._hessian_shape,
                multiclass_strategy=strategy_tensor,
                init_stamp_token=init_stamp_token,
                loss_uses_sum_reduction=loss_uses_sum_reduction))
        fc_name_idx += 1

      # Create handlers for sparse int columns.
      for sparse_int_column_idx in range(len(self._sparse_int_indices)):
        fc_name = self._fc_names[fc_name_idx]
        handlers.append(
            categorical_split_handler.EqualitySplitHandler(
                l1_regularization=l1_regularization,
                l2_regularization=l2_regularization,
                tree_complexity_regularization=tree_complexity_regularization,
                min_node_weight=min_node_weight,
                feature_column_group_id=constant_op.constant(
                    sparse_int_column_idx),
                sparse_int_column=sparse_tensor.SparseTensor(
                    self._sparse_int_indices[sparse_int_column_idx],
                    self._sparse_int_values[sparse_int_column_idx],
                    self._sparse_int_shapes[sparse_int_column_idx]),
                name=fc_name,
                gradient_shape=self._gradient_shape,
                hessian_shape=self._hessian_shape,
                multiclass_strategy=strategy_tensor,
                init_stamp_token=init_stamp_token,
                loss_uses_sum_reduction=loss_uses_sum_reduction,
                weak_learner_type=weak_learner_type))
        fc_name_idx += 1

      # Create ensemble stats variables.
      num_layer_examples = variables.VariableV1(
          initial_value=array_ops.zeros([], dtypes.int64),
          name="num_layer_examples",
          trainable=False)
      num_layer_steps = variables.VariableV1(
          initial_value=array_ops.zeros([], dtypes.int64),
          name="num_layer_steps",
          trainable=False)
      num_layers = variables.VariableV1(
          initial_value=array_ops.zeros([], dtypes.int64),
          name="num_layers",
          trainable=False)
      active_tree = variables.VariableV1(
          initial_value=array_ops.zeros([], dtypes.int64),
          name="active_tree",
          trainable=False)
      active_layer = variables.VariableV1(
          initial_value=array_ops.zeros([], dtypes.int64),
          name="active_layer",
          trainable=False)
      # Variable that becomes false once bias centering is done.
      continue_centering = variables.VariableV1(
          initial_value=self._center_bias,
          name="continue_centering",
          trainable=False)
      # Create bias stats accumulator.
      bias_stats_accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=self._gradient_shape,
          hessian_shape=self._hessian_shape,
          name="BiasAccumulator")
      # Create steps accumulator.
      steps_accumulator = stats_accumulator_ops.StatsAccumulator(
          stamp_token=0,
          gradient_shape=tensor_shape.scalar(),
          hessian_shape=tensor_shape.scalar(),
          name="StepsAccumulator")
    # Create ensemble stats summaries.
    summary.scalar("layer_stats/num_examples", num_layer_examples)
    summary.scalar("layer_stats/num_steps", num_layer_steps)
    summary.scalar("ensemble_stats/active_tree", active_tree)
    summary.scalar("ensemble_stats/active_layer", active_layer)

    # Update bias stats.
    stats_update_ops = []

    stats_update_ops.append(
        control_flow_ops.cond(
            continue_centering,
            self._make_update_bias_stats_fn(ensemble_stamp, predictions,
                                            gradients, bias_stats_accumulator,
                                            hessians), control_flow_ops.no_op))

    # Update handler stats.
    handler_reads = collections.OrderedDict()
    for handler in handlers:
      handler_reads[handler] = handler.scheduled_reads()

    handler_results = batch_ops_utils.run_handler_scheduled_ops(
        handler_reads, ensemble_stamp, worker_device)
    per_handler_updates = collections.OrderedDict()
    # Two values per handler. First one is if the handler is active for the
    # current layer. The second one is if the handler is going to be active
    # for the next layer.
    subsampling_type = self._learner_config.WhichOneof("feature_fraction")
    if subsampling_type == "feature_fraction_per_level":
      seed = predictions_dict[NUM_LAYERS_ATTEMPTED]
      active_handlers_current_layer = stateless.stateless_random_uniform(
          shape=[len(handlers)], seed=[seed, 1])
      active_handlers_next_layer = stateless.stateless_random_uniform(
          shape=[len(handlers)], seed=[seed + 1, 1])
      active_handlers = array_ops.stack(
          [active_handlers_current_layer, active_handlers_next_layer], axis=1)
      active_handlers = (
          active_handlers < self._learner_config.feature_fraction_per_level)
    elif subsampling_type == "feature_fraction_per_tree":
      seed = predictions_dict[NUM_TREES_ATTEMPTED]
      active_handlers_current_layer = stateless.stateless_random_uniform(
          shape=[len(handlers)], seed=[seed, 2])
      active_handlers_current_layer = (
          active_handlers_current_layer <
          self._learner_config.feature_fraction_per_tree)
      active_handlers = array_ops.stack(
          [
              active_handlers_current_layer,
              array_ops.ones([len(handlers)], dtype=dtypes.bool)
          ],
          axis=1)
    else:
      active_handlers = array_ops.ones([len(handlers), 2], dtype=dtypes.bool)

    if self._learner_config.constraints.max_number_of_unique_feature_columns:
      target = (
          self._learner_config.constraints.max_number_of_unique_feature_columns)

      def _feature_selection_active_handlers():
        # The active list for current and the next iteration.
        used_handlers = array_ops.reshape(predictions_dict[USED_HANDLERS_MASK],
                                          [-1, 1])
        used_handlers = array_ops.concat([used_handlers, used_handlers], axis=1)
        return math_ops.logical_and(used_handlers, active_handlers)

      active_handlers = (
          control_flow_ops.cond(predictions_dict[NUM_USED_HANDLERS] >= target,
                                _feature_selection_active_handlers,
                                lambda: active_handlers))

    # Prepare empty gradients and hessians when handlers are not ready.
    empty_hess_shape = [1] + self._hessian_shape.as_list()
    empty_grad_shape = [1] + self._gradient_shape.as_list()

    empty_gradients = constant_op.constant_v1(
        [], dtype=dtypes.float32, shape=empty_grad_shape)
    empty_hessians = constant_op.constant_v1(
        [], dtype=dtypes.float32, shape=empty_hess_shape)

    active_handlers = array_ops.unstack(active_handlers, axis=0)
    for handler_idx in range(len(handlers)):
      handler = handlers[handler_idx]
      is_active = active_handlers[handler_idx]
      updates, scheduled_updates = handler.update_stats(
          ensemble_stamp, partition_ids, squeezed_gradients, squeezed_hessians,
          empty_gradients, empty_hessians, weights, is_active,
          handler_results[handler])
      stats_update_ops.append(updates)
      per_handler_updates[handler] = scheduled_updates

    update_results = batch_ops_utils.run_handler_scheduled_ops(
        per_handler_updates, ensemble_stamp, worker_device)
    for update in update_results.values():
      stats_update_ops += update

    training_state = GBDTTrainingState(
        num_layer_examples=num_layer_examples,
        num_layer_steps=num_layer_steps,
        num_layers=num_layers,
        active_tree=active_tree,
        active_layer=active_layer,
        continue_centering=continue_centering,
        bias_stats_accumulator=bias_stats_accumulator,
        steps_accumulator=steps_accumulator,
        handlers=handlers)

    reset_op = control_flow_ops.no_op()
    if self._is_chief:
      # Advance the ensemble stamp to throw away staggered workers.
      stamp_token, _ = model_ops.tree_ensemble_serialize(self._ensemble_handle)
      next_stamp_token = stamp_token + 1

      reset_ops = []
      for handler in handlers:
        reset_ops.append(handler.reset(stamp_token, next_stamp_token))
      if self._center_bias:
        reset_ops.append(
            bias_stats_accumulator.flush(stamp_token, next_stamp_token))
      reset_ops.append(steps_accumulator.flush(stamp_token, next_stamp_token))
      reset_ops.append(self._finalized_trees.assign(0).op)
      reset_ops.append(self._attempted_trees.assign(0).op)
      reset_ops.append(
          model_ops.tree_ensemble_deserialize(
              self._ensemble_handle,
              stamp_token=next_stamp_token,
              tree_ensemble_config="",
              name="reset_gbdt"))

      reset_op = control_flow_ops.group([reset_ops])

    return stats_update_ops, reset_op, training_state

  def increment_step_counter_and_maybe_update_ensemble(self, predictions_dict,
                                                       training_state):
    """Increments number of visited examples and grows the ensemble.

    If the number of visited examples reaches the target examples_per_layer,
    ensemble is updated.

    Args:
      predictions_dict: Dictionary of Rank 2 `Tensor` representing information
          about predictions per example.
      training_state: `dict` returned by update_stats.

    Returns:
      An op that updates the counters and potientially grows the ensemble.
    """
    batch_size = math_ops.cast(
        array_ops.shape(predictions_dict[PREDICTIONS])[0], dtypes.float32)
    ensemble_stamp = predictions_dict[ENSEMBLE_STAMP]
    # Accumulate a step after updating stats.

    steps_accumulator = training_state.steps_accumulator
    num_layer_examples = training_state.num_layer_examples
    num_layer_steps = training_state.num_layer_steps
    active_layer = training_state.active_layer
    add_step_op = steps_accumulator.add(
        ensemble_stamp, [0], [[0, 0]], [batch_size], [1.0])

    # After adding the step, decide if further processing is needed.
    ensemble_update_ops = [add_step_op]
    class_id = self._get_class_id(predictions_dict)

    with ops.control_dependencies([add_step_op]):
      if self._is_chief:
        dropout_seed = predictions_dict[NUM_TREES_ATTEMPTED]

        # Get accumulated steps and examples for the current layer.
        _, _, _, _, acc_examples, acc_steps = (
            steps_accumulator.saveable.serialize())
        acc_examples = math_ops.cast(acc_examples[0], dtypes.int64)
        acc_steps = math_ops.cast(acc_steps[0], dtypes.int64)
        ensemble_update_ops.append(
            num_layer_examples.assign(acc_examples))
        ensemble_update_ops.append(num_layer_steps.assign(acc_steps))
        # Determine whether we need to update tree ensemble.
        examples_per_layer = self._examples_per_layer
        if callable(examples_per_layer):
          examples_per_layer = examples_per_layer(active_layer)
        ensemble_update_ops.append(
            control_flow_ops.cond(
                acc_examples >= examples_per_layer,
                self.make_update_ensemble_fn(ensemble_stamp, training_state,
                                             dropout_seed, class_id),
                control_flow_ops.no_op))

    # Note, the loss is calculated from the prediction considering dropouts, so
    # that the value might look staggering over steps when the dropout ratio is
    # high. eval_loss might be referred instead in the aspect of convergence.
    return control_flow_ops.group(*ensemble_update_ops)

  def make_update_ensemble_fn(self, ensemble_stamp, training_state,
                              dropout_seed, class_id):
    """A method to create the function which updates the tree ensemble."""
    # Determine learning rate.
    learning_rate_tuner = self._learner_config.learning_rate_tuner.WhichOneof(
        "tuner")
    if learning_rate_tuner == "fixed" or learning_rate_tuner == "dropout":
      tuner = getattr(self._learner_config.learning_rate_tuner,
                      learning_rate_tuner)
      learning_rate = tuner.learning_rate
    else:
      # TODO(nponomareva, soroush) do the line search.
      raise ValueError("Line search learning rate is not yet supported.")

    def _update_ensemble():
      """A method to update the tree ensemble."""
      # Get next stamp token.
      next_ensemble_stamp = ensemble_stamp + 1
      # Finalize bias stats.
      _, _, _, bias_grads, bias_hess = (
          training_state.bias_stats_accumulator.flush(ensemble_stamp,
                                                      next_ensemble_stamp))

      # Finalize handler splits.
      are_splits_ready_list = []
      partition_ids_list = []
      gains_list = []
      split_info_list = []

      for handler in training_state.handlers:
        (are_splits_ready,
         partition_ids, gains, split_info) = handler.make_splits(
             ensemble_stamp, next_ensemble_stamp, class_id)
        are_splits_ready_list.append(are_splits_ready)
        partition_ids_list.append(partition_ids)
        gains_list.append(gains)
        split_info_list.append(split_info)
      # Stack all the inputs to one tensor per type.
      # This is a workaround for the slowness of graph building in tf.cond.
      # See (b/36554864).
      split_sizes = array_ops.reshape(
          array_ops.shape_n(partition_ids_list), [len(partition_ids_list)])
      partition_ids = array_ops.concat(partition_ids_list, axis=0)
      gains = array_ops.concat(gains_list, axis=0)
      split_infos = array_ops.concat(split_info_list, axis=0)

      # Determine if all splits are ready.
      are_all_splits_ready = math_ops.reduce_all(
          array_ops.stack(
              are_splits_ready_list, axis=0, name="stack_handler_readiness"))

      # Define bias centering update operation.
      def _center_bias_fn():
        # Center tree ensemble bias.
        delta_updates = array_ops.where(bias_hess > 0, -bias_grads / bias_hess,
                                        array_ops.zeros_like(bias_grads))
        center_bias = training_ops.center_tree_ensemble_bias(
            tree_ensemble_handle=self._ensemble_handle,
            stamp_token=ensemble_stamp,
            next_stamp_token=next_ensemble_stamp,
            delta_updates=delta_updates,
            learner_config=self._learner_config_serialized)
        return training_state.continue_centering.assign(center_bias)

      # Define ensemble growing operations.
      def _grow_ensemble_ready_fn():
        # Grow the ensemble given the current candidates.
        sizes = array_ops.unstack(split_sizes)
        partition_ids_list = list(array_ops.split(partition_ids, sizes, axis=0))
        # When using the oblivious decision tree as weak learner, it produces
        # one gain and one split per handler and not number of partitions.
        if self._learner_config.weak_learner_type == (
            learner_pb2.LearnerConfig.OBLIVIOUS_DECISION_TREE):
          sizes = len(training_state.handlers)

        gains_list = list(array_ops.split(gains, sizes, axis=0))
        split_info_list = list(array_ops.split(split_infos, sizes, axis=0))
        return training_ops.grow_tree_ensemble(
            tree_ensemble_handle=self._ensemble_handle,
            stamp_token=ensemble_stamp,
            next_stamp_token=next_ensemble_stamp,
            learning_rate=learning_rate,
            partition_ids=partition_ids_list,
            gains=gains_list,
            splits=split_info_list,
            learner_config=self._learner_config_serialized,
            dropout_seed=dropout_seed,
            center_bias=self._center_bias,
            max_tree_depth=self._max_tree_depth,
            weak_learner_type=self._learner_config.weak_learner_type)

      def _grow_ensemble_not_ready_fn():
        # Don't grow the ensemble, just update the stamp.
        return training_ops.grow_tree_ensemble(
            tree_ensemble_handle=self._ensemble_handle,
            stamp_token=ensemble_stamp,
            next_stamp_token=next_ensemble_stamp,
            learning_rate=0,
            partition_ids=[],
            gains=[],
            splits=[],
            learner_config=self._learner_config_serialized,
            dropout_seed=dropout_seed,
            center_bias=self._center_bias,
            max_tree_depth=self._max_tree_depth,
            weak_learner_type=self._learner_config.weak_learner_type)

      def _grow_ensemble_fn():
        # Conditionally grow an ensemble depending on whether the splits
        # from all the handlers are ready.
        return control_flow_ops.cond(are_all_splits_ready,
                                     _grow_ensemble_ready_fn,
                                     _grow_ensemble_not_ready_fn)

      # Update ensemble.
      update_ops = [are_all_splits_ready]
      if self._center_bias:
        update_model = control_flow_ops.cond(training_state.continue_centering,
                                             _center_bias_fn, _grow_ensemble_fn)
      else:
        update_model = _grow_ensemble_fn()
      update_ops.append(update_model)

      # Update ensemble stats.
      with ops.control_dependencies([update_model]):
        stats = training_ops.tree_ensemble_stats(
            self._ensemble_handle, stamp_token=next_ensemble_stamp)
        update_ops.append(self._finalized_trees.assign(stats.num_trees))
        update_ops.append(self._attempted_trees.assign(stats.attempted_trees))
        update_ops.append(training_state.num_layers.assign(stats.num_layers))
        update_ops.append(training_state.active_tree.assign(stats.active_tree))
        update_ops.append(
            training_state.active_layer.assign(stats.active_layer))

      # Flush step stats.
      update_ops.extend(
          training_state.steps_accumulator.flush(ensemble_stamp,
                                                 next_ensemble_stamp))
      return control_flow_ops.group(*update_ops, name="update_ensemble")

    return _update_ensemble

  def get_number_of_trees_tensor(self):
    return self._finalized_trees, self._attempted_trees

  def get_max_tree_depth(self):
    return self._max_tree_depth

  def train(self, loss, predictions_dict, labels, gradients=None,
            hessians=None):
    """Updates the accumalator stats and grows the ensemble.

    Args:
      loss: A scalar tensor representing average loss of examples.
      predictions_dict: Dictionary of Rank 2 `Tensor` representing information
          about predictions per example.
      labels: Rank 2 `Tensor` representing labels per example. Has no effect
          on the training and is only kept for backward compatibility.
      gradients: A tensor with the gradients with the respect to logits from
        predictions_dict. If not provided, tensorflow will do
        autodifferentiation.
      hessians: A tensor with the hessians with the respect to logits from
        predictions_dict. If not provided, tensorflow will do
        autodifferentiation.

    Returns:
      An op that adds a new tree to the ensemble.

    Raises:
      ValueError: if inputs are not valid.
    """
    del labels  # unused; kept for backward compatibility.
    update_op, _, training_state = self.update_stats(loss, predictions_dict,
                                                     gradients, hessians)
    with ops.control_dependencies(update_op):
      return self.increment_step_counter_and_maybe_update_ensemble(
          predictions_dict, training_state)

  def _get_weights(self, hessian_shape, hessians):
    """Derives weights to be used based on hessians and multiclass strategy."""
    if hessian_shape == tensor_shape.scalar():
      # This is tree per class.
      weights = hessians
    elif len(hessian_shape.dims) == 1:
      # This is diagonal hessian.
      weights = math_ops.reduce_sum(hessians, axis=1)
    else:
      # This is full hessian.
      weights = math_ops.trace(hessians)
    return weights

  def _full_hessian(self, grads, predictions):
    """Prepares hessians for full-hessian multiclass strategy."""
    # Because of
    # https://github.com/tensorflow/tensorflow/issues/675, we can't just
    # compute the full hessian with a single call to gradients, but instead
    # must compute it row-by-row.
    gradients_list = array_ops.unstack(
        grads, num=self._logits_dimension, axis=1)
    hessian_rows = []

    for row in range(self._logits_dimension):
      # If current row is i, K is number of classes,each row returns a tensor of
      # size batch_size x K representing for each example dx_i dx_1, dx_i dx_2
      # etc dx_i dx_K
      hessian_row = gradients_impl.gradients(
          gradients_list[row],
          predictions,
          name="Hessian_%d" % row,
          colocate_gradients_with_ops=False,
          gate_gradients=0,
          aggregation_method=None)

      # hessian_row is of dimension 1, batch_size, K, => trim first dimension
      # to get batch_size x K
      hessian_row = array_ops.squeeze(array_ops.unstack(hessian_row), [0])
      hessian_rows.append(hessian_row)
    return hessian_rows

  def _diagonal_hessian(self, grads, predictions):
    """Prepares hessians for diagonal-hessian multiclass mode."""
    diag_hessian_list = []

    gradients_list = array_ops.unstack(
        grads, num=self._logits_dimension, axis=1)

    for row, row_grads in enumerate(gradients_list):
      # If current row is i, K is number of classes,each row returns a tensor of
      # size batch_size x K representing for each example dx_i dx_1, dx_1 dx_2
      # etc dx_i dx_K
      hessian_row = gradients_impl.gradients(
          row_grads,
          predictions,
          name="Hessian_%d" % row,
          colocate_gradients_with_ops=False,
          gate_gradients=0,
          aggregation_method=None)

      # hessian_row is of dimension 1, batch_size, K, => trim first dimension
      # to get batch_size x K
      hessian_row = array_ops.squeeze(array_ops.unstack(hessian_row), [0])

      # Get dx_i^2 for the whole batch.
      elem = array_ops.transpose(hessian_row)[row]
      diag_hessian_list.append(elem)

    return diag_hessian_list

  def _get_replica_device_setter(self, worker_device):
    """Creates a replica device setter."""
    ps_tasks = self._num_ps_replicas
    ps_ops = list(device_setter.STANDARD_PS_OPS)
    ps_ops.extend([
        "DecisionTreeEnsembleResourceHandleOp",
        "StatsAccumulatorScalarResourceHandleOp",
        "StatsAccumulatorTensorResourceHandleOp",
    ])
    ps_strategy = _OpRoundRobinStrategy(ps_ops, ps_tasks)
    return device_setter.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=ps_tasks,
        merge_devices=True,
        ps_ops=ps_ops,
        ps_strategy=ps_strategy)

  def _make_update_bias_stats_fn(self,
                                 ensemble_stamp,
                                 predictions,
                                 gradients,
                                 bias_stats_accumulator,
                                 hessians=None):
    """A method to create the function which updates the bias stats."""

    def _update_bias_stats():
      """A method to update the bias stats."""
      # Get reduced gradients and hessians.
      grads_sum = math_ops.reduce_sum(gradients, 0)
      if hessians is not None:
        hess = hessians
      else:
        hess = gradients_impl.gradients(
            grads_sum,
            predictions,
            name="Hessians",
            colocate_gradients_with_ops=False,
            gate_gradients=0,
            aggregation_method=None)[0]
      hess_sum = math_ops.reduce_sum(hess, 0)

      # Accumulate gradients and hessians.
      partition_ids = math_ops.range(self._logits_dimension)
      feature_ids = array_ops.zeros(
          [self._logits_dimension, 2], dtype=dtypes.int64)

      add_stats_op = bias_stats_accumulator.add(
          ensemble_stamp, partition_ids, feature_ids, grads_sum, hess_sum)
      return control_flow_ops.group(*[add_stats_op], name="update_bias_stats")

    return _update_bias_stats
