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
"""Estimator classes for BoostedTrees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export

_TreeHParams = collections.namedtuple(
    'TreeHParams',
    ['n_trees', 'max_depth', 'learning_rate', 'l1', 'l2', 'tree_complexity'])

_HOLD_FOR_MULTI_CLASS_SUPPORT = object()
_HOLD_FOR_MULTI_DIM_SUPPORT = object()


def _get_transformed_features(features, feature_columns):
  """Gets the transformed features from features/feature_columns pair.

  Args:
    features: a dicionary of name to Tensor.
    feature_columns: a list/set of tf.feature_column.

  Returns:
    result_features: a list of the transformed features, sorted by the name.
    num_buckets: the maximum number of buckets across bucketized_columns.

  Raises:
    ValueError: when unsupported features/columns are tried.
  """
  num_buckets = 1
  # pylint:disable=protected-access
  for fc in feature_columns:
    if isinstance(fc, feature_column_lib._BucketizedColumn):
      # N boundaries creates (N+1) buckets.
      num_buckets = max(num_buckets, len(fc.boundaries) + 1)
    else:
      raise ValueError('For now, only bucketized_column is supported but '
                       'got: {}'.format(fc))
  transformed = feature_column_lib._transform_features(features,
                                                       feature_columns)
  # pylint:enable=protected-access
  result_features = []
  for column in sorted(transformed, key=lambda tc: tc.name):
    source_name = column.source_column.name
    squeezed_tensor = array_ops.squeeze(transformed[column], axis=1)
    if len(squeezed_tensor.shape) > 1:
      raise ValueError('For now, only supports features equivalent to rank 1 '
                       'but column `{}` got: {}'.format(
                           source_name, features[source_name].shape))
    result_features.append(squeezed_tensor)
  return result_features, num_buckets


def _keep_as_local_variable(tensor, name=None):
  """Stores a tensor as a local Variable for faster read."""
  return variable_scope.variable(
      initial_value=tensor,
      trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      validate_shape=False,
      name=name)


class _CacheTrainingStatesUsingHashTable(object):
  """Caching logits, etc. using MutableHashTable."""

  def __init__(self, example_ids, logits_dimension):
    """Creates a cache with the given configuration.

    It maintains a MutableDenseHashTable for all values.
    The API lookup() and insert() would have those specs,
      tree_ids: shape=[batch_size], dtype=int32
      node_ids: shape=[batch_size], dtype=int32
      logits: shape=[batch_size, logits_dimension], dtype=float32
    However in the MutableDenseHashTable, ids are bitcasted into float32 and
    all values are concatenated as a single tensor (of float32).

    Hence conversion happens internally before inserting to the HashTable and
    after lookup from it.

    Args:
      example_ids: a Rank 1 tensor to be used as a key of the cache.
      logits_dimension: a constant (int) for the dimension of logits.

    Raises:
      ValueError: if example_ids is other than int64 or string.
    """
    if dtypes.as_dtype(dtypes.int64).is_compatible_with(example_ids.dtype):
      empty_key = -1 << 62
    elif dtypes.as_dtype(dtypes.string).is_compatible_with(example_ids.dtype):
      empty_key = ''
    else:
      raise ValueError('Unsupported example_id_feature dtype %s.',
                       example_ids.dtype)
    # Cache holds latest <tree_id, node_id, logits> for each example.
    # tree_id and node_id are both int32 but logits is a float32.
    # To reduce the overhead, we store all of them together as float32 and
    # bitcast the ids to int32.
    self._table_ref = lookup_ops.mutable_dense_hash_table_v2(
        empty_key=empty_key, value_dtype=dtypes.float32, value_shape=[3])
    self._example_ids = example_ids
    self._logits_dimension = logits_dimension

  def lookup(self):
    """Returns cached_tree_ids, cached_node_ids, cached_logits."""
    cached_tree_ids, cached_node_ids, cached_logits = array_ops.split(
        lookup_ops.lookup_table_find_v2(
            self._table_ref, self._example_ids, default_value=[0.0, 0.0, 0.0]),
        [1, 1, self._logits_dimension],
        axis=1)
    cached_tree_ids = array_ops.squeeze(
        array_ops.bitcast(cached_tree_ids, dtypes.int32))
    cached_node_ids = array_ops.squeeze(
        array_ops.bitcast(cached_node_ids, dtypes.int32))
    return (cached_tree_ids, cached_node_ids, cached_logits)

  def insert(self, tree_ids, node_ids, logits):
    """Inserts values and returns the op."""
    insert_op = lookup_ops.lookup_table_insert_v2(
        self._table_ref, self._example_ids,
        array_ops.concat(
            [
                array_ops.expand_dims(
                    array_ops.bitcast(tree_ids, dtypes.float32), 1),
                array_ops.expand_dims(
                    array_ops.bitcast(node_ids, dtypes.float32), 1),
                logits,
            ],
            axis=1,
            name='value_concat_for_cache_insert'))
    return insert_op


class _CacheTrainingStatesUsingVariables(object):
  """Caching logits, etc. using Variables."""

  def __init__(self, batch_size, logits_dimension):
    """Creates a cache with the given configuration.

    It maintains three variables, tree_ids, node_ids, logits, for caching.
      tree_ids: shape=[batch_size], dtype=int32
      node_ids: shape=[batch_size], dtype=int32
      logits: shape=[batch_size, logits_dimension], dtype=float32

    Note, this can be used only with in-memory data setting.

    Args:
      batch_size: `int`, the size of the cache.
      logits_dimension: a constant (int) for the dimension of logits.
    """
    self._logits_dimension = logits_dimension
    self._tree_ids = _keep_as_local_variable(
        array_ops.zeros([batch_size], dtype=dtypes.int32),
        name='tree_ids_cache')
    self._node_ids = _keep_as_local_variable(
        array_ops.zeros([batch_size], dtype=dtypes.int32),
        name='node_ids_cache')
    self._logits = _keep_as_local_variable(
        array_ops.zeros([batch_size, logits_dimension], dtype=dtypes.float32),
        name='logits_cache')

  def lookup(self):
    """Returns cached_tree_ids, cached_node_ids, cached_logits."""
    return (self._tree_ids, self._node_ids, self._logits)

  def insert(self, tree_ids, node_ids, logits):
    """Inserts values and returns the op."""
    return control_flow_ops.group(
        [
            self._tree_ids.assign(tree_ids),
            self._node_ids.assign(node_ids),
            self._logits.assign(logits)
        ],
        name='cache_insert')


class _StopAtAttemptsHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at the number of attempts."""

  def __init__(self, num_finalized_trees_tensor, num_attempted_layers_tensor,
               max_trees, max_depth):
    self._num_finalized_trees_tensor = num_finalized_trees_tensor
    self._num_attempted_layers_tensor = num_attempted_layers_tensor
    self._max_trees = max_trees
    self._max_depth = max_depth

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        [self._num_finalized_trees_tensor, self._num_attempted_layers_tensor])

  def after_run(self, run_context, run_values):
    # num_* tensors should be retrieved by a separate session than the training
    # one, in order to read the values after growing.
    # So, if it's approaching to the limit, get the actual value by additional
    # session.
    num_finalized_trees, num_attempted_layers = run_values.results
    if (num_finalized_trees >= self._max_trees - 1 or
        num_attempted_layers > 2 * self._max_trees * self._max_depth - 1):
      num_finalized_trees, num_attempted_layers = run_context.session.run(
          [self._num_finalized_trees_tensor, self._num_attempted_layers_tensor])
    if (num_finalized_trees >= self._max_trees or
        num_attempted_layers > 2 * self._max_trees * self._max_depth):
      run_context.request_stop()


def _bt_model_fn(
    features,
    labels,
    mode,
    head,
    feature_columns,
    tree_hparams,
    n_batches_per_layer,
    config,
    closed_form_grad_and_hess_fn=None,
    example_id_column_name=None,
    # TODO(youngheek): replace this later using other options.
    train_in_memory=False,
    name='boosted_trees'):
  """Gradient Boosted Trees model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `head_lib._Head` instance.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    tree_hparams: TODO. collections.namedtuple for hyper parameters.
    n_batches_per_layer: A `Tensor` of `int64`. Each layer is built after at
      least n_batches_per_layer accumulations.
    config: `RunConfig` object to configure the runtime settings.
    closed_form_grad_and_hess_fn: a function that accepts logits and labels
      and returns gradients and hessians. By default, they are created by
      tf.gradients() from the loss.
    example_id_column_name: Name of the feature for a unique ID per example.
      Currently experimental -- not exposed to public API.
    train_in_memory: `bool`, when true, it assumes the dataset is in memory,
      i.e., input_fn should return the entire dataset as a single batch, and
      also n_batches_per_layer should be set as 1.
    name: Name to use for the model.

  Returns:
      An `EstimatorSpec` instance.

  Raises:
    ValueError: mode or params are invalid, or features has the wrong type.
  """
  is_single_machine = (config.num_worker_replicas <= 1)
  if train_in_memory:
    assert n_batches_per_layer == 1, (
        'When train_in_memory is enabled, input_fn should return the entire '
        'dataset as a single batch, and n_batches_per_layer should be set as '
        '1.')
  worker_device = control_flow_ops.no_op().device
  # maximum number of splits possible in the whole tree =2^(D-1)-1
  # TODO(youngheek): perhaps storage could be optimized by storing stats with
  # the dimension max_splits_per_layer, instead of max_splits (for the entire
  # tree).
  max_splits = (1 << tree_hparams.max_depth) - 1
  with ops.name_scope(name) as name:
    # Prepare.
    global_step = training_util.get_or_create_global_step()
    input_feature_list, num_buckets = _get_transformed_features(
        features, feature_columns)
    if train_in_memory and mode == model_fn.ModeKeys.TRAIN:
      input_feature_list = [
          _keep_as_local_variable(feature) for feature in input_feature_list
      ]
    num_features = len(input_feature_list)

    cache = None
    if mode == model_fn.ModeKeys.TRAIN:
      if train_in_memory and is_single_machine:  # maybe just train_in_memory?
        batch_size = array_ops.shape(input_feature_list[0])[0]
        cache = _CacheTrainingStatesUsingVariables(batch_size,
                                                   head.logits_dimension)
      elif example_id_column_name:
        example_ids = features[example_id_column_name]
        cache = _CacheTrainingStatesUsingHashTable(example_ids,
                                                   head.logits_dimension)

    # Create Ensemble resources.
    tree_ensemble = boosted_trees_ops.TreeEnsemble(name=name)
    # Create logits.
    if mode != model_fn.ModeKeys.TRAIN:
      logits = boosted_trees_ops.predict(
          # For non-TRAIN mode, ensemble doesn't change after initialization,
          # so no local copy is needed; using tree_ensemble directly.
          tree_ensemble_handle=tree_ensemble.resource_handle,
          bucketized_features=input_feature_list,
          logits_dimension=head.logits_dimension,
          max_depth=tree_hparams.max_depth)
    else:
      if is_single_machine:
        local_tree_ensemble = tree_ensemble
        ensemble_reload = control_flow_ops.no_op()
      else:
        # Have a local copy of ensemble for the distributed setting.
        with ops.device(worker_device):
          local_tree_ensemble = boosted_trees_ops.TreeEnsemble(
              name=name + '_local', is_local=True)
        # TODO(soroush): Do partial updates if this becomes a bottleneck.
        ensemble_reload = local_tree_ensemble.deserialize(
            *tree_ensemble.serialize())
      if cache:
        cached_tree_ids, cached_node_ids, cached_logits = cache.lookup()
      else:
        # Always start from the beginning when no cache is set up.
        batch_size = array_ops.shape(input_feature_list[0])[0]
        cached_tree_ids, cached_node_ids, cached_logits = (
            array_ops.zeros([batch_size], dtype=dtypes.int32),
            array_ops.zeros([batch_size], dtype=dtypes.int32),
            array_ops.zeros(
                [batch_size, head.logits_dimension], dtype=dtypes.float32))
      with ops.control_dependencies([ensemble_reload]):
        (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
         last_layer_nodes_range) = local_tree_ensemble.get_states()
        summary.scalar('ensemble/num_trees', num_trees)
        summary.scalar('ensemble/num_finalized_trees', num_finalized_trees)
        summary.scalar('ensemble/num_attempted_layers', num_attempted_layers)

        partial_logits, tree_ids, node_ids = boosted_trees_ops.training_predict(
            tree_ensemble_handle=local_tree_ensemble.resource_handle,
            cached_tree_ids=cached_tree_ids,
            cached_node_ids=cached_node_ids,
            bucketized_features=input_feature_list,
            logits_dimension=head.logits_dimension,
            max_depth=tree_hparams.max_depth)
      logits = cached_logits + partial_logits

    # Create training graph.
    def _train_op_fn(loss):
      """Run one training iteration."""
      train_op = []
      if cache:
        train_op.append(cache.insert(tree_ids, node_ids, logits))
      if closed_form_grad_and_hess_fn:
        gradients, hessians = closed_form_grad_and_hess_fn(logits, labels)
      else:
        gradients = gradients_impl.gradients(loss, logits, name='Gradients')[0]
        hessians = gradients_impl.gradients(
            gradients, logits, name='Hessians')[0]
      stats_summary_list = [
          array_ops.squeeze(
              boosted_trees_ops.make_stats_summary(
                  node_ids=node_ids,
                  gradients=gradients,
                  hessians=hessians,
                  bucketized_features_list=[input_feature_list[f]],
                  max_splits=max_splits,
                  num_buckets=num_buckets),
              axis=0) for f in range(num_features)
      ]

      def grow_tree_from_stats_summaries(stats_summary_list):
        """Updates ensemble based on the best gains from stats summaries."""
        (node_ids_per_feature, gains_list, thresholds_list,
         left_node_contribs_list, right_node_contribs_list) = (
             boosted_trees_ops.calculate_best_gains_per_feature(
                 node_id_range=last_layer_nodes_range,
                 stats_summary_list=stats_summary_list,
                 l1=tree_hparams.l1,
                 l2=tree_hparams.l2,
                 tree_complexity=tree_hparams.tree_complexity,
                 max_splits=max_splits))
        grow_op = boosted_trees_ops.update_ensemble(
            # Confirm if local_tree_ensemble or tree_ensemble should be used.
            tree_ensemble.resource_handle,
            feature_ids=math_ops.range(0, num_features, dtype=dtypes.int32),
            node_ids=node_ids_per_feature,
            gains=gains_list,
            thresholds=thresholds_list,
            left_node_contribs=left_node_contribs_list,
            right_node_contribs=right_node_contribs_list,
            learning_rate=tree_hparams.learning_rate,
            max_depth=tree_hparams.max_depth,
            pruning_mode=boosted_trees_ops.PruningMode.NO_PRUNING)
        return grow_op

      if train_in_memory and is_single_machine:
        train_op.append(distribute_lib.increment_var(global_step))
        train_op.append(grow_tree_from_stats_summaries(stats_summary_list))
      else:
        summary_accumulator = data_flow_ops.ConditionalAccumulator(
            dtype=dtypes.float32,
            # The stats consist of gradients and hessians (the last dimension).
            shape=[num_features, max_splits, num_buckets, 2],
            shared_name='stats_summary_accumulator')
        apply_grad = summary_accumulator.apply_grad(
            array_ops.stack(stats_summary_list, axis=0), stamp_token)

        def grow_tree_from_accumulated_summaries_fn():
          """Updates the tree with the best layer from accumulated summaries."""
          # Take out the accumulated summaries from the accumulator and grow.
          stats_summary_list = array_ops.unstack(
              summary_accumulator.take_grad(1), axis=0)
          grow_op = grow_tree_from_stats_summaries(stats_summary_list)
          return grow_op

        with ops.control_dependencies([apply_grad]):
          train_op.append(distribute_lib.increment_var(global_step))
          if config.is_chief:
            train_op.append(
                control_flow_ops.cond(
                    math_ops.greater_equal(
                        summary_accumulator.num_accumulated(),
                        n_batches_per_layer),
                    grow_tree_from_accumulated_summaries_fn,
                    control_flow_ops.no_op,
                    name='wait_until_n_batches_accumulated'))

      return control_flow_ops.group(train_op, name='train_op')

  estimator_spec = head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)
  if mode == model_fn.ModeKeys.TRAIN:
    # Add an early stop hook.
    estimator_spec = estimator_spec._replace(
        training_hooks=estimator_spec.training_hooks +
        (_StopAtAttemptsHook(num_finalized_trees, num_attempted_layers,
                             tree_hparams.n_trees, tree_hparams.max_depth),))
  return estimator_spec


def _create_classification_head(n_classes,
                                weight_column=None,
                                label_vocabulary=None):
  """Creates a classification head. Refer to canned.head for details on args."""
  # TODO(nponomareva): Support multi-class cases.
  if n_classes == 2:
    # pylint: disable=protected-access
    return head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    # pylint: enable=protected-access
  else:
    raise ValueError('For now only binary classification is supported.'
                     'n_classes given as {}'.format(n_classes))


def _create_classification_head_and_closed_form(n_classes, weight_column,
                                                label_vocabulary):
  """Creates a head for classifier and the closed form gradients/hessians."""
  head = _create_classification_head(n_classes, weight_column, label_vocabulary)
  if n_classes == 2 and weight_column is None and label_vocabulary is None:
    # Use the closed-form gradients/hessians for 2 class.
    def _grad_and_hess_for_logloss(logits, labels):
      # TODO(youngheek): add weights handling.
      predictions = math_ops.reciprocal(math_ops.exp(-logits) + 1.0)
      normalizer = math_ops.reciprocal(
          math_ops.cast(array_ops.size(predictions), dtypes.float32))
      gradients = (predictions - labels) * normalizer
      hessians = predictions * (1.0 - predictions) * normalizer
      return gradients, hessians

    closed_form = _grad_and_hess_for_logloss
  else:
    closed_form = None
  return (head, closed_form)


def _create_regression_head(label_dimension, weight_column=None):
  if label_dimension != 1:
    raise ValueError('For now only 1 dimension regression is supported.'
                     'label_dimension given as {}'.format(label_dimension))
  # pylint: disable=protected-access
  return head_lib._regression_head_with_mean_squared_error_loss(
      label_dimension=label_dimension,
      weight_column=weight_column,
      loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
  # pylint: enable=protected-access


@tf_export('estimator.BoostedTreesClassifier')
class BoostedTreesClassifier(estimator.Estimator):
  """A Classifier for Tensorflow Boosted Trees models."""

  def __init__(
      self,
      feature_columns,
      n_batches_per_layer,
      model_dir=None,
      n_classes=_HOLD_FOR_MULTI_CLASS_SUPPORT,
      weight_column=None,
      label_vocabulary=None,
      n_trees=100,
      max_depth=6,
      learning_rate=0.1,
      l1_regularization=0.,
      l2_regularization=0.,
      tree_complexity=0.,
      config=None):
    """Initializes a `BoostedTreesClassifier` instance.

    Example:

    ```python
    bucketized_feature_1 = bucketized_column(
      numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
    bucketized_feature_2 = bucketized_column(
      numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

    classifier = estimator.BoostedTreesClassifier(
        feature_columns=[bucketized_feature_1, bucketized_feature_2],
        n_trees=100,
        ... <some other params>
    )

    def input_fn_train():
      ...
      return dataset

    classifier.train(input_fn=input_fn_train)

    def input_fn_eval():
      ...
      return dataset

    metrics = classifier.evaluate(input_fn=input_fn_eval)
    ```

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      n_batches_per_layer: the number of batches to collect statistics per
        layer.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Multiclass support is not yet implemented.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to downweight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      n_trees: number trees to be created.
      max_depth: maximum depth of the tree to grow.
      learning_rate: shrinkage parameter to be used when a tree added to the
        model.
      l1_regularization: regularization multiplier applied to the absolute
        weights of the tree leafs.
      l2_regularization: regularization multiplier applied to the square weights
        of the tree leafs.
      tree_complexity: regularization factor to penalize trees with more leaves.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: when wrong arguments are given or unsupported functionalities
         are requested.
    """
    # TODO(nponomareva): Support multi-class cases.
    if n_classes == _HOLD_FOR_MULTI_CLASS_SUPPORT:
      n_classes = 2
    head, closed_form = _create_classification_head_and_closed_form(
        n_classes, weight_column, label_vocabulary=label_vocabulary)

    # HParams for the model.
    tree_hparams = _TreeHParams(
        n_trees, max_depth, learning_rate, l1_regularization, l2_regularization,
        tree_complexity)

    def _model_fn(features, labels, mode, config):
      return _bt_model_fn(  # pylint: disable=protected-access
          features,
          labels,
          mode,
          head,
          feature_columns,
          tree_hparams,
          n_batches_per_layer,
          config,
          closed_form_grad_and_hess_fn=closed_form)

    super(BoostedTreesClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


@tf_export('estimator.BoostedTreesRegressor')
class BoostedTreesRegressor(estimator.Estimator):
  """A Regressor for Tensorflow Boosted Trees models."""

  def __init__(
      self,
      feature_columns,
      n_batches_per_layer,
      model_dir=None,
      label_dimension=_HOLD_FOR_MULTI_DIM_SUPPORT,
      weight_column=None,
      n_trees=100,
      max_depth=6,
      learning_rate=0.1,
      l1_regularization=0.,
      l2_regularization=0.,
      tree_complexity=0.,
      config=None):
    """Initializes a `BoostedTreesRegressor` instance.

    Example:

    ```python
    bucketized_feature_1 = bucketized_column(
      numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
    bucketized_feature_2 = bucketized_column(
      numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

    regressor = estimator.BoostedTreesRegressor(
        feature_columns=[bucketized_feature_1, bucketized_feature_2],
        n_trees=100,
        ... <some other params>
    )

    def input_fn_train():
      ...
      return dataset

    regressor.train(input_fn=input_fn_train)

    def input_fn_eval():
      ...
      return dataset

    metrics = regressor.evaluate(input_fn=input_fn_eval)
    ```

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      n_batches_per_layer: the number of batches to collect statistics per
        layer.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      label_dimension: Number of regression targets per example.
        Multi-dimensional support is not yet implemented.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to downweight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      n_trees: number trees to be created.
      max_depth: maximum depth of the tree to grow.
      learning_rate: shrinkage parameter to be used when a tree added to the
        model.
      l1_regularization: regularization multiplier applied to the absolute
        weights of the tree leafs.
      l2_regularization: regularization multiplier applied to the square weights
        of the tree leafs.
      tree_complexity: regularization factor to penalize trees with more leaves.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: when wrong arguments are given or unsupported functionalities
         are requested.
    """
    # TODO(nponomareva): Extend it to multi-dimension cases.
    if label_dimension == _HOLD_FOR_MULTI_DIM_SUPPORT:
      label_dimension = 1
    head = _create_regression_head(label_dimension, weight_column)

    # HParams for the model.
    tree_hparams = _TreeHParams(
        n_trees, max_depth, learning_rate, l1_regularization, l2_regularization,
        tree_complexity)

    def _model_fn(features, labels, mode, config):
      return _bt_model_fn(  # pylint: disable=protected-access
          features, labels, mode, head, feature_columns, tree_hparams,
          n_batches_per_layer, config)

    super(BoostedTreesRegressor, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
