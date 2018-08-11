# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Estimator classes for TensorForest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.ops import tensor_forest_ops

_ForestHParams = collections.namedtuple('TreeHParams', [
    'n_trees', 'max_nodes', 'num_splits_to_consider',
    'split_after_samples'
])
TREE_PATHS_PREDICTION_KEY = 'tree_paths'
VARIANCE_PREDICTION_KEY = 'prediction_variance'
EPSILON = 0.000001


def _ensure_logits(head, logits):
    # For binary classification problems, convert probabilities to logits.
    # Includes hack to get around the fact that a probability might be 0 or 1.
    if head.logits_dimension == 2:
        class_1_probs = array_ops.slice(logits, [0, 1], [-1, 1])
        logits = math_ops.log(
            math_ops.maximum(class_1_probs / math_ops.maximum(
                1.0 - class_1_probs, EPSILON), EPSILON))
    return logits


def _bt_model_fn(features, labels, mode, head, feature_columns, forest_hparams, config, name='tensor_forest'):
    """Tensor Forest model_fn.

    Args:
      features: dict of `Tensor`.
      labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
        dtype `int32` or `int64` in the range `[0, n_classes)`.
      mode: Defines whether this is training, evaluation or prediction.
        See `ModeKeys`.
      head: A `head_lib._Head` instance.
      feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
      tree_hparams: TODO. collections.namedtuple for hyper parameters.
      config: `RunConfig` object to configure the runtime settings.
      name: Name to use for the model.
    """
    graph_builder = tensor_forest_ops.RandomForestGraphs(
        forest_hparams, config)

    logits, tree_paths, regression_variance = graph_builder.inference_graph(
        features)

    logits = _ensure_logits(head, logits)

    summary.scalar('average_tree_size', graph_builder.average_size())

    training_graph = None
    if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
        with ops.control_dependencies([logits.op]):
            training_graph = control_flow_ops.group(
                graph_builder.training_graph(
                    features, labels),
                state_ops.assign_add(training_util.get_global_step(), 1), name=name)

    def _train_op_fn(unused_loss):
        return training_graph

    estimator_spec = head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)

    estimator_spec = estimator_spec._replace(
        predictions=estimator_spec.predictions.update({
            VARIANCE_PREDICTION_KEY: regression_variance,
            TREE_PATHS_PREDICTION_KEY: tree_paths
        }))
    return estimator_spec


@estimator_export('estimator.TensorForestClassifier')
class TensorForestClassifier(estimator.Estimator):
  """A Classifier for Tensorflow Tensor Forest models.

  """

  def __init__(self,
               feature_columns,
               model_dir=None,
               n_classes=2,
               label_vocabulary=None,
               head=None,
               n_trees=100,
               max_nodes=1000,
               num_splits_to_consider=10,
               split_after_samples=250,
               config=None):
    """Initializes a `TensorForestClassifier` instance.

    Example:

    ```python
    feature_1 = numeric_column('feature_1')
    feature_2 = numeric_column('feature_2')

    classifier = estimator.TensorForestClassifier(
        feature_columns=[feature_1, feature_2],
        n_trees=100,
        ... <some other params>
    )

    def input_fn_train():
      ...
      return dataset

    classifier.train(input_fn=input_fn_train)

    def input_fn_predict():
      ...
      return dataset

    classifier.predict(input_fn=input_fn_predict)

    def input_fn_eval():
      ...
      return dataset

    metrics = classifier.evaluate(input_fn=input_fn_eval)
    ```

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
	from FeatureColumn.
      n_classes: Defaults to 2. The number of classes in a classification problem.
      model_dir: Directory to save model parameters, graph and etc.
	This can also be used to load checkpoints from the directory
	into an estimator to continue training a previously saved model.
      label_vocabulary: A list of strings representing all possible label values.
	If provided, labels must be of string type and their values must be present
	in label_vocabulary list. If label_vocabulary is omitted, it is assumed that
	the labels are already encoded as integer values within {0, 1} for n_classes=2,
	or encoded as integer values in {0, 1,..., n_classes-1} for n_classes>2.
	If vocabulary is not provided and labels are of string,
	an error will be generated.
      head: A head_lib._Head instance, the loss would be calculated for metrics
	purpose and not being used for training. If not provided,
	one will be automatically created based on params
      n_trees: The number of trees to create. Defaults to 100.
	There usually isn't any accuracy gain from using higher
	values (assuming deep enough trees are built).
      max_nodes: Defaults to 10k. No tree is allowed to grow beyond max_nodes
	nodes, and training stops when all trees in the forest are this large.
      num_splits_to_consider: Defaults to sqrt(num_features). In the extremely
	randomized tree training algorithm, only this many potential splits
	are evaluated for each tree node.
      split_after_samples: Defaults to 250. In our online version of extremely
	randomized tree training, we pick a split for a node after it has
	accumulated this many training samples.
      config: RunConfig object to configure the runtime settings.

    Raises:
      ValueError: when wrong arguments are given or unsupported functionalities
         are requested.
    """
    if head is None:
        head = head_lib._binary_logistic_or_multi_class_head(
            n_classes=n_classes,
            weight_column=None,
            label_vocabulary=label_vocabulary,
            loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    forest_hparams = _ForestHParams(
        n_trees, max_nodes, num_splits_to_consider, split_after_samples)

    def _model_fn(features, labels, mode, config):
        return _bt_model_fn(
            features, labels, mode, head, feature_columns, forest_hparams, config)

    super(TensorForestClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
