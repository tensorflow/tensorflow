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
"""GTFlow Estimator definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.boosted_trees.estimator_batch import model
from tensorflow.contrib.boosted_trees.python.utils import losses
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.python.estimator.canned import head as core_head_lib
from tensorflow.python.estimator import estimator as core_estimator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import losses as core_losses
from tensorflow.contrib.boosted_trees.estimator_batch import custom_loss_head
from tensorflow.python.ops import array_ops

# ================== Old estimator interface===================================
# The estimators below were designed for old feature columns and old estimator
# interface. They can be used with new feature columns and losses by setting
# use_core_libs = True.


class GradientBoostedDecisionTreeClassifier(estimator.Estimator):
  """An estimator using gradient boosted decision trees."""

  def __init__(self,
               learner_config,
               examples_per_layer,
               n_classes=2,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_keys=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               use_core_libs=False,
               output_leaf_index=False,
               override_global_step_value=None,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeClassifier estimator instance.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      n_classes: Number of classes in the classification.
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      use_core_libs: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. It is a Tensor of rank 2 and its shape is
        [batch_size, num_trees].
        For example,
        result_iter = classifier.predict(...)
        for result_dict in result_iter:
          # access leaf index list by result_dict["leaf_index"]
          # which contains one leaf index per tree
      override_global_step_value: If after the training is done, global step
        value must be reset to this value. This should be used to reset global
        step to a number > number of steps used to train the current ensemble.
        For example, the usual way is to train a number of trees and set a very
        large number of training steps. When the training is done (number of
        trees were trained), this parameter can be used to set the global step
        to a large value, making it look like that number of training steps ran.
        If None, no override of global step will happen.
      num_quantiles: Number of quantiles to build for numeric feature values.

    Raises:
      ValueError: If learner_config is not valid.
    """
    if n_classes > 2:
      # For multi-class classification, use our loss implementation that
      # supports second order derivative.
      def loss_fn(labels, logits, weights=None):
        result = losses.per_example_maxent_loss(
            labels=labels,
            logits=logits,
            weights=weights,
            num_classes=n_classes)
        return math_ops.reduce_mean(result[0])
    else:
      loss_fn = None
    head = head_lib.multi_class_head(
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        enable_centered_bias=False,
        loss_fn=loss_fn,
        label_keys=label_keys)
    if learner_config.num_classes == 0:
      learner_config.num_classes = n_classes
    elif learner_config.num_classes != n_classes:
      raise ValueError("n_classes (%d) doesn't match learner_config (%d)." %
                       (learner_config.num_classes, n_classes))
    super(GradientBoostedDecisionTreeClassifier, self).__init__(
        model_fn=model.model_builder,
        params={
            'head': head,
            'feature_columns': feature_columns,
            'learner_config': learner_config,
            'num_trees': num_trees,
            'weight_column_name': weight_column_name,
            'examples_per_layer': examples_per_layer,
            'center_bias': center_bias,
            'logits_modifier_function': logits_modifier_function,
            'use_core_libs': use_core_libs,
            'output_leaf_index': output_leaf_index,
            'override_global_step_value': override_global_step_value,
            'num_quantiles': num_quantiles,
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


class GradientBoostedDecisionTreeRegressor(estimator.Estimator):
  """An estimator using gradient boosted decision trees."""

  def __init__(self,
               learner_config,
               examples_per_layer,
               label_dimension=1,
               num_trees=None,
               feature_columns=None,
               label_name=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               use_core_libs=False,
               output_leaf_index=False,
               override_global_step_value=None,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeRegressor estimator instance.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      label_dimension: Number of regression labels per example. This is the size
        of the last dimension of the labels `Tensor` (typically, this has shape
        `[batch_size, label_dimension]`).
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      label_name: String, name of the key in label dict. Can be null if label
          is a tensor (single headed models).
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      use_core_libs: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. For example,
        result_dict = classifier.predict(...)
        for example_prediction_result in result_dict:
          # access leaf index list by example_prediction_result["leaf_index"]
          # which contains one leaf index per tree
      override_global_step_value: If after the training is done, global step
        value must be reset to this value. This should be used to reset global
        step to a number > number of steps used to train the current ensemble.
        For example, the usual way is to train a number of trees and set a very
        large number of training steps. When the training is done (number of
        trees were trained), this parameter can be used to set the global step
        to a large value, making it look like that number of training steps ran.
        If None, no override of global step will happen.
      num_quantiles: Number of quantiles to build for numeric feature values.
    """
    head = head_lib.regression_head(
        label_name=label_name,
        label_dimension=label_dimension,
        weight_column_name=weight_column_name,
        enable_centered_bias=False)
    if label_dimension == 1:
      learner_config.num_classes = 2
    else:
      learner_config.num_classes = label_dimension
    super(GradientBoostedDecisionTreeRegressor, self).__init__(
        model_fn=model.model_builder,
        params={
            'head': head,
            'feature_columns': feature_columns,
            'learner_config': learner_config,
            'num_trees': num_trees,
            'weight_column_name': weight_column_name,
            'examples_per_layer': examples_per_layer,
            'logits_modifier_function': logits_modifier_function,
            'center_bias': center_bias,
            'use_core_libs': use_core_libs,
            'output_leaf_index': False,
            'override_global_step_value': override_global_step_value,
            'num_quantiles': num_quantiles,
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


class GradientBoostedDecisionTreeEstimator(estimator.Estimator):
  """An estimator using gradient boosted decision trees.

  Useful for training with user specified `Head`.
  """

  def __init__(self,
               learner_config,
               examples_per_layer,
               head,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               use_core_libs=False,
               output_leaf_index=False,
               override_global_step_value=None,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeEstimator estimator instance.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      head: `Head` instance.
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      use_core_libs: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. For example,
        result_dict = classifier.predict(...)
        for example_prediction_result in result_dict:
          # access leaf index list by example_prediction_result["leaf_index"]
          # which contains one leaf index per tree
      override_global_step_value: If after the training is done, global step
        value must be reset to this value. This should be used to reset global
        step to a number > number of steps used to train the current ensemble.
        For example, the usual way is to train a number of trees and set a very
        large number of training steps. When the training is done (number of
        trees were trained), this parameter can be used to set the global step
        to a large value, making it look like that number of training steps ran.
        If None, no override of global step will happen.
      num_quantiles: Number of quantiles to build for numeric feature values.
    """
    super(GradientBoostedDecisionTreeEstimator, self).__init__(
        model_fn=model.model_builder,
        params={
            'head': head,
            'feature_columns': feature_columns,
            'learner_config': learner_config,
            'num_trees': num_trees,
            'weight_column_name': weight_column_name,
            'examples_per_layer': examples_per_layer,
            'logits_modifier_function': logits_modifier_function,
            'center_bias': center_bias,
            'use_core_libs': use_core_libs,
            'output_leaf_index': False,
            'override_global_step_value': override_global_step_value,
            'num_quantiles': num_quantiles,
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


class GradientBoostedDecisionTreeRanker(estimator.Estimator):
  """A ranking estimator using gradient boosted decision trees."""

  def __init__(self,
               learner_config,
               examples_per_layer,
               head,
               ranking_model_pair_keys,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_keys=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=False,
               use_core_libs=False,
               output_leaf_index=False,
               override_global_step_value=None,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeRanker instance.

    This is an estimator that can be trained off the pairwise data and can be
    used for inference on non-paired data. This is essentially LambdaMart.
    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      head: `Head` instance.
      ranking_model_pair_keys: Keys to distinguish between features
        for left and right part of the training pairs for ranking. For example,
        for an Example with features "a.f1" and "b.f1", the keys would be
        ("a", "b").
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      use_core_libs: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. It is a Tensor of rank 2 and its shape is
        [batch_size, num_trees].
        For example,
        result_iter = classifier.predict(...)
        for result_dict in result_iter:
          # access leaf index list by result_dict["leaf_index"]
          # which contains one leaf index per tree
      override_global_step_value: If after the training is done, global step
        value must be reset to this value. This should be used to reset global
        step to a number > number of steps used to train the current ensemble.
        For example, the usual way is to train a number of trees and set a very
        large number of training steps. When the training is done (number of
        trees were trained), this parameter can be used to set the global step
        to a large value, making it look like that number of training steps ran.
        If None, no override of global step will happen.
      num_quantiles: Number of quantiles to build for numeric feature values.

    Raises:
      ValueError: If learner_config is not valid.
    """
    super(GradientBoostedDecisionTreeRanker, self).__init__(
        model_fn=model.ranking_model_builder,
        params={
            'head': head,
            'n_classes': 2,
            'feature_columns': feature_columns,
            'learner_config': learner_config,
            'num_trees': num_trees,
            'weight_column_name': weight_column_name,
            'examples_per_layer': examples_per_layer,
            'center_bias': center_bias,
            'logits_modifier_function': logits_modifier_function,
            'use_core_libs': use_core_libs,
            'output_leaf_index': output_leaf_index,
            'ranking_model_pair_keys': ranking_model_pair_keys,
            'override_global_step_value': override_global_step_value,
            'num_quantiles': num_quantiles,
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)

# When using this estimator, make sure to regularize the hessian (at least l2,
# min_node_weight)!
# TODO(nponomareva): extend to take multiple quantiles in one go.
class GradientBoostedDecisionTreeQuantileRegressor(estimator.Estimator):
  """An estimator that does quantile regression and returns quantile estimates.
  """

  def __init__(self,
               learner_config,
               examples_per_layer,
               quantiles,
               label_dimension=1,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               use_core_libs=False,
               output_leaf_index=False,
               override_global_step_value=None,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeQuantileRegressor instance.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      quantiles: a list of quantiles for the loss, each between 0 and 1.
      label_dimension: Dimension of regression label. This is the size
        of the last dimension of the labels `Tensor` (typically, this has shape
        `[batch_size, label_dimension]`). When label_dimension>1, it is
        recommended to use multiclass strategy diagonal hessian or full hessian.
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      use_core_libs: Whether feature columns and loss are from the core (as
        opposed to contrib) version of tensorflow.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. For example,
        result_dict = classifier.predict(...)
        for example_prediction_result in result_dict:
          # access leaf index list by example_prediction_result["leaf_index"]
          # which contains one leaf index per tree
      override_global_step_value: If after the training is done, global step
        value must be reset to this value. This should be used to reset global
        step to a number > number of steps used to train the current ensemble.
        For example, the usual way is to train a number of trees and set a very
        large number of training steps. When the training is done (number of
        trees were trained), this parameter can be used to set the global step
        to a large value, making it look like that number of training steps ran.
        If None, no override of global step will happen.
      num_quantiles: Number of quantiles to build for numeric feature values.
    """

    if len(quantiles) > 1:
      raise ValueError('For now, just one quantile per estimator is supported')

    def _quantile_regression_head(quantile):
      # Use quantile regression.
      head = custom_loss_head.CustomLossHead(
          loss_fn=functools.partial(
              losses.per_example_quantile_regression_loss, quantile=quantile),
          link_fn=array_ops.identity,
          logit_dimension=label_dimension)
      return head

    learner_config.num_classes = max(2, label_dimension)

    super(GradientBoostedDecisionTreeQuantileRegressor, self).__init__(
        model_fn=model.model_builder,
        params={
            'head': _quantile_regression_head(quantiles[0]),
            'feature_columns': feature_columns,
            'learner_config': learner_config,
            'num_trees': num_trees,
            'weight_column_name': weight_column_name,
            'examples_per_layer': examples_per_layer,
            'logits_modifier_function': logits_modifier_function,
            'center_bias': center_bias,
            'use_core_libs': use_core_libs,
            'output_leaf_index': False,
            'override_global_step_value': override_global_step_value,
            'num_quantiles': num_quantiles,
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)

# ================== New Estimator interface===================================
# The estimators below use new core Estimator interface and must be used with
# new feature columns and heads.


# For multiclass classification, use the following head since it uses loss
# that is twice differentiable.
def core_multiclass_head(
    n_classes,
    weight_column=None,
    loss_reduction=core_losses.Reduction.SUM_OVER_NONZERO_WEIGHTS):
  """Core head for multiclass problems."""

  def loss_fn(labels, logits):
    result = losses.per_example_maxent_loss(
        labels=labels,
        logits=logits,
        weights=weight_column,
        num_classes=n_classes)
    return result[0]

  # pylint:disable=protected-access
  head_fn = core_head_lib._multi_class_head_with_softmax_cross_entropy_loss(
      n_classes=n_classes,
      loss_fn=loss_fn,
      loss_reduction=loss_reduction,
      weight_column=weight_column)
  # pylint:enable=protected-access

  return head_fn


# For quantile regression, use this head with Core..Estimator, or use
# Core..QuantileRegressor directly,
def core_quantile_regression_head(
    quantiles,
    label_dimension=1,
    weight_column=None,
    loss_reduction=core_losses.Reduction.SUM_OVER_NONZERO_WEIGHTS):
  """Core head for quantile regression problems."""

  def loss_fn(labels, logits):
    result = losses.per_example_quantile_regression_loss(
        labels=labels,
        predictions=logits,
        weights=weight_column,
        quantile=quantiles)
    return result[0]

  # pylint:disable=protected-access
  head_fn = core_head_lib._regression_head(
      label_dimension=label_dimension,
      loss_fn=loss_fn,
      loss_reduction=loss_reduction,
      weight_column=weight_column)
  # pylint:enable=protected-access
  return head_fn


class CoreGradientBoostedDecisionTreeEstimator(core_estimator.Estimator):
  """An estimator using gradient boosted decision trees.

  Useful for training with user specified `Head`.
  """

  def __init__(self,
               learner_config,
               examples_per_layer,
               head,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_keys=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               output_leaf_index=False,
               num_quantiles=100):
    """Initializes a core version of GradientBoostedDecisionTreeEstimator.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      head: `Head` instance.
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. For example,
        result_dict = classifier.predict(...)
        for example_prediction_result in result_dict:
          # access leaf index list by example_prediction_result["leaf_index"]
          # which contains one leaf index per tree
      num_quantiles: Number of quantiles to build for numeric feature values.
    """

    def _model_fn(features, labels, mode, config):
      return model.model_builder(
          features=features,
          labels=labels,
          mode=mode,
          config=config,
          params={
              'head': head,
              'feature_columns': feature_columns,
              'learner_config': learner_config,
              'num_trees': num_trees,
              'weight_column_name': weight_column_name,
              'examples_per_layer': examples_per_layer,
              'center_bias': center_bias,
              'logits_modifier_function': logits_modifier_function,
              'use_core_libs': True,
              'output_leaf_index': output_leaf_index,
              'override_global_step_value': None,
              'num_quantiles': num_quantiles,
          },
          output_type=model.ModelBuilderOutputType.ESTIMATOR_SPEC)

    super(CoreGradientBoostedDecisionTreeEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


class CoreGradientBoostedDecisionTreeRanker(core_estimator.Estimator):
  """A ranking estimator using gradient boosted decision trees."""

  def __init__(self,
               learner_config,
               examples_per_layer,
               head,
               ranking_model_pair_keys,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_keys=None,
               logits_modifier_function=None,
               center_bias=False,
               output_leaf_index=False,
               num_quantiles=100):
    """Initializes a GradientBoostedDecisionTreeRanker instance.

    This is an estimator that can be trained off the pairwise data and can be
    used for inference on non-paired data. This is essentially LambdaMart.
    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      head: `Head` instance.
      ranking_model_pair_keys: Keys to distinguish between features
        for left and right part of the training pairs for ranking. For example,
        for an Example with features "a.f1" and "b.f1", the keys would be
        ("a", "b").
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. It is a Tensor of rank 2 and its shape is
        [batch_size, num_trees].
        For example,
        result_iter = classifier.predict(...)
        for result_dict in result_iter:
          # access leaf index list by result_dict["leaf_index"]
          # which contains one leaf index per tree
      num_quantiles: Number of quantiles to build for numeric feature values.

    Raises:
      ValueError: If learner_config is not valid.
    """

    def _model_fn(features, labels, mode, config):
      return model.ranking_model_builder(
          features=features,
          labels=labels,
          mode=mode,
          config=config,
          params={
              'head': head,
              'n_classes': 2,
              'feature_columns': feature_columns,
              'learner_config': learner_config,
              'num_trees': num_trees,
              'weight_column_name': weight_column_name,
              'examples_per_layer': examples_per_layer,
              'center_bias': center_bias,
              'logits_modifier_function': logits_modifier_function,
              'use_core_libs': True,
              'output_leaf_index': output_leaf_index,
              'ranking_model_pair_keys': ranking_model_pair_keys,
              'override_global_step_value': None,
              'num_quantiles': num_quantiles,
          },
          output_type=model.ModelBuilderOutputType.ESTIMATOR_SPEC)

    super(CoreGradientBoostedDecisionTreeRanker, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


# When using this estimator, make sure to regularize the hessian (at least l2,
# min_node_weight)!
# TODO(nponomareva): extend to take multiple quantiles in one go.
class CoreGradientBoostedDecisionTreeQuantileRegressor(
    core_estimator.Estimator):
  """An estimator that does quantile regression and returns quantile estimates.
  """

  def __init__(self,
               learner_config,
               examples_per_layer,
               quantiles,
               label_dimension=1,
               num_trees=None,
               feature_columns=None,
               weight_column_name=None,
               model_dir=None,
               config=None,
               label_keys=None,
               feature_engineering_fn=None,
               logits_modifier_function=None,
               center_bias=True,
               output_leaf_index=False,
               num_quantiles=100):
    """Initializes a core version of GradientBoostedDecisionTreeEstimator.

    Args:
      learner_config: A config for the learner.
      examples_per_layer: Number of examples to accumulate before growing a
        layer. It can also be a function that computes the number of examples
        based on the depth of the layer that's being built.
      quantiles: a list of quantiles for the loss, each between 0 and 1.
      label_dimension: Dimension of regression label. This is the size
        of the last dimension of the labels `Tensor` (typically, this has shape
        `[batch_size, label_dimension]`). When label_dimension>1, it is
        recommended to use multiclass strategy diagonal hessian or full hessian.
      num_trees: An int, number of trees to build.
      feature_columns: A list of feature columns.
      weight_column_name: Name of the column for weights, or None if not
        weighted.
      model_dir: Directory for model exports, etc.
      config: `RunConfig` object to configure the runtime settings.
      label_keys: Optional list of strings with size `[n_classes]` defining the
        label vocabulary. Only supported for `n_classes` > 2.
      feature_engineering_fn: Feature engineering function. Takes features and
        labels which are the output of `input_fn` and returns features and
        labels which will be fed into the model.
      logits_modifier_function: A modifier function for the logits.
      center_bias: Whether a separate tree should be created for first fitting
        the bias.
      output_leaf_index: whether to output leaf indices along with predictions
        during inference. The leaf node indexes are available in predictions
        dict by the key 'leaf_index'. For example,
        result_dict = classifier.predict(...)
        for example_prediction_result in result_dict:
          # access leaf index list by example_prediction_result["leaf_index"]
          # which contains one leaf index per tree
      num_quantiles: Number of quantiles to build for numeric feature values.
    """
    if len(quantiles) > 1:
      raise ValueError('For now, just one quantile per estimator is supported')

    def _model_fn(features, labels, mode, config):
      return model.model_builder(
          features=features,
          labels=labels,
          mode=mode,
          config=config,
          params={
              'head':
                  core_quantile_regression_head(
                      quantiles[0], label_dimension=label_dimension),
              'feature_columns':
                  feature_columns,
              'learner_config':
                  learner_config,
              'num_trees':
                  num_trees,
              'weight_column_name':
                  weight_column_name,
              'examples_per_layer':
                  examples_per_layer,
              'center_bias':
                  center_bias,
              'logits_modifier_function':
                  logits_modifier_function,
              'use_core_libs':
                  True,
              'output_leaf_index':
                  output_leaf_index,
              'override_global_step_value':
                  None,
              'num_quantiles':
                  num_quantiles,
          },
          output_type=model.ModelBuilderOutputType.ESTIMATOR_SPEC)

    super(CoreGradientBoostedDecisionTreeQuantileRegressor, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
