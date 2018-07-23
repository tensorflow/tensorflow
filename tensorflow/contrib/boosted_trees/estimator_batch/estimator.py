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

from tensorflow.contrib.boosted_trees.estimator_batch import model
from tensorflow.contrib.boosted_trees.python.utils import losses
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.python.ops import math_ops


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
               output_leaf_index=False):
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
               output_leaf_index=False):
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
               output_leaf_index=False):
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
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


class GradientBoostedDecisionTreeRanker(estimator.Estimator):
  """A ranking estimator using gradient boosted decision trees."""

  def __init__(
      self,
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
  ):
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
        },
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)
