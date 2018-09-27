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
"""Deep Neural Network estimators with layer annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import pickle

from google.protobuf.any_pb2 import Any

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses
from tensorflow.python.saved_model import utils as saved_model_utils


class LayerAnnotationsCollectionNames(object):
  """Names for the collections containing the annotations."""

  UNPROCESSED_FEATURES = 'layer_annotations/unprocessed_features'
  PROCESSED_FEATURES = 'layer_annotatons/processed_features'
  FEATURE_COLUMNS = 'layer_annotations/feature_columns'

  @classmethod
  def keys(cls, collection_name):
    return '%s/keys' % collection_name

  @classmethod
  def values(cls, collection_name):
    return '%s/values' % collection_name


def serialize_feature_column(feature_column):
  if isinstance(feature_column, feature_column_lib._EmbeddingColumn):  # pylint: disable=protected-access
    # We can't pickle nested functions, and we don't need the value of
    # layer_creator in most cases anyway, so just discard its value.
    args = feature_column._asdict()
    args['layer_creator'] = None
    temp = type(feature_column)(**args)
    return pickle.dumps(temp)
  return pickle.dumps(feature_column)


def _to_any_wrapped_tensor_info(tensor):
  """Converts a `Tensor` to a `TensorInfo` wrapped in a proto `Any`."""
  any_buf = Any()
  tensor_info = saved_model_utils.build_tensor_info(tensor)
  any_buf.Pack(tensor_info)
  return any_buf


def make_input_layer_with_layer_annotations(original_input_layer):
  """Make an input_layer replacement function that adds layer annotations."""

  def input_layer_with_layer_annotations(features,
                                         feature_columns,
                                         weight_collections=None,
                                         trainable=True,
                                         cols_to_vars=None,
                                         scope=None,
                                         cols_to_output_tensors=None,
                                         from_template=False):
    """Returns a dense `Tensor` as input layer based on given `feature_columns`.

    Generally a single example in training data is described with
    FeatureColumns.
    At the first layer of the model, this column oriented data should be
    converted
    to a single `Tensor`.

    This is like tf.feature_column.input_layer, except with added
    Integrated-Gradient annotations.

    Args:
      features: A mapping from key to tensors. `_FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values can be a `SparseTensor` or a `Tensor` depends
        on corresponding `_FeatureColumn`.
      feature_columns: An iterable containing the FeatureColumns to use as
        inputs to your model. All items should be instances of classes derived
        from `_DenseColumn` such as `numeric_column`, `embedding_column`,
        `bucketized_column`, `indicator_column`. If you have categorical
        features, you can wrap them with an `embedding_column` or
        `indicator_column`.
      weight_collections: A list of collection names to which the Variable will
        be added. Note that variables will also be added to collections
        `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      cols_to_vars: If not `None`, must be a dictionary that will be filled with
        a mapping from `_FeatureColumn` to list of `Variable`s.  For example,
        after the call, we might have cols_to_vars = {_EmbeddingColumn(
        categorical_column=_HashedCategoricalColumn( key='sparse_feature',
        hash_bucket_size=5, dtype=tf.string), dimension=10): [<tf.Variable
        'some_variable:0' shape=(5, 10), <tf.Variable 'some_variable:1'
          shape=(5, 10)]} If a column creates no variables, its value will be an
          empty list.
      scope: A name or variable scope to use
      cols_to_output_tensors: If not `None`, must be a dictionary that will be
        filled with a mapping from '_FeatureColumn' to the associated output
        `Tensor`s.
      from_template: True if the method is being instantiated from a
        `make_template`.

    Returns:
      A `Tensor` which represents input layer of a model. Its shape
      is (batch_size, first_layer_dimension) and its dtype is `float32`.
      first_layer_dimension is determined based on given `feature_columns`.

    Raises:
      ValueError: features and feature_columns have different lengths.
    """

    local_cols_to_output_tensors = {}
    input_layer = original_input_layer(
        features=features,
        feature_columns=feature_columns,
        weight_collections=weight_collections,
        trainable=trainable,
        cols_to_vars=cols_to_vars,
        scope=scope,
        cols_to_output_tensors=local_cols_to_output_tensors,
        from_template=from_template)

    if cols_to_output_tensors is not None:
      cols_to_output_tensors = local_cols_to_output_tensors

    # Annotate features.
    # These are the parsed Tensors, before embedding.

    # Only annotate features used by FeatureColumns.
    # We figure which ones are used by FeatureColumns by creating a parsing
    # spec and looking at the keys.
    spec = feature_column_lib.make_parse_example_spec(feature_columns)
    for key in spec.keys():
      tensor = ops.convert_to_tensor(features[key])
      ops.add_to_collection(
          LayerAnnotationsCollectionNames.keys(
              LayerAnnotationsCollectionNames.UNPROCESSED_FEATURES), key)
      ops.add_to_collection(
          LayerAnnotationsCollectionNames.values(
              LayerAnnotationsCollectionNames.UNPROCESSED_FEATURES),
          _to_any_wrapped_tensor_info(tensor))

    # Annotate feature columns.
    for column in feature_columns:
      # TODO(cyfoo): Find a better way to serialize and deserialize
      # _FeatureColumn.
      ops.add_to_collection(LayerAnnotationsCollectionNames.FEATURE_COLUMNS,
                            serialize_feature_column(column))

    for column, tensor in local_cols_to_output_tensors.items():
      ops.add_to_collection(
          LayerAnnotationsCollectionNames.keys(
              LayerAnnotationsCollectionNames.PROCESSED_FEATURES), column.name)
      ops.add_to_collection(
          LayerAnnotationsCollectionNames.values(
              LayerAnnotationsCollectionNames.PROCESSED_FEATURES),
          _to_any_wrapped_tensor_info(tensor))

    return input_layer

  return input_layer_with_layer_annotations


@contextlib.contextmanager
def _monkey_patch(module, function, replacement):
  old_function = getattr(module, function)
  setattr(module, function, replacement)
  yield
  setattr(module, function, old_function)


def DNNClassifierWithLayerAnnotations(  # pylint: disable=invalid-name
    hidden_units,
    feature_columns,
    model_dir=None,
    n_classes=2,
    weight_column=None,
    label_vocabulary=None,
    optimizer='Adagrad',
    activation_fn=nn.relu,
    dropout=None,
    input_layer_partitioner=None,
    config=None,
    warm_start_from=None,
    loss_reduction=losses.Reduction.SUM):
  """A classifier for TensorFlow DNN models with layer annotations.

  This classifier is fuctionally identical to estimator.DNNClassifier as far as
  training and evaluating models is concerned. The key difference is that this
  classifier adds additional layer annotations, which can be used for computing
  Integrated Gradients.

  Integrated Gradients is a method for attributing a classifier's predictions
  to its input features (https://arxiv.org/pdf/1703.01365.pdf). Given an input
  instance, the method assigns attribution scores to individual features in
  proportion to the feature's importance to the classifier's prediction.

  See estimator.DNNClassifer for example code for training and evaluating models
  using this classifier.

  This classifier is checkpoint-compatible with estimator.DNNClassifier and
  therefore the following should work seamlessly:

  # Instantiate ordinary estimator as usual.
  estimator = tf.estimator.DNNClassifier(
    config, feature_columns, hidden_units, ...)

  # Train estimator, export checkpoint.
  tf.estimator.train_and_evaluate(estimator, ...)

  # Instantiate estimator with annotations with the same configuration as the
  # ordinary estimator.
  estimator_with_annotations = (
    tf.contrib.estimator.DNNClassifierWithLayerAnnotations(
      config, feature_columns, hidden_units, ...))

  # Call export_savedmodel with the same arguments as the ordinary estimator,
  # using the checkpoint produced for the ordinary estimator.
  estimator_with_annotations.export_saved_model(
    export_dir_base, serving_input_receiver, ...
    checkpoint_path='/path/to/ordinary/estimator/checkpoint/model.ckpt-1234')

  Args:
    hidden_units: Iterable of number hidden units per layer. All layers are
      fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second
      one has 32.
    feature_columns: An iterable containing all the feature columns used by the
      model. All items in the set should be instances of classes derived from
      `_FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    n_classes: Number of label classes. Defaults to 2, namely binary
      classification. Must be > 1.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary: A list of strings represents possible label values. If
      given, labels must be string type and have any value in
      `label_vocabulary`. If it is not given, that means labels are already
      encoded as integer or float within [0, 1] for `n_classes=2` and encoded as
      integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also there
      will be errors if vocabulary is not provided and labels are string.
    optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
      to Adagrad optimizer.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Optional. Partitioner for input layer. Defaults to
      `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.
    warm_start_from: A string filepath to a checkpoint to warm-start from, or a
      `WarmStartSettings` object to fully configure warm-starting.  If the
      string filepath is provided instead of a `WarmStartSettings`, then all
      weights are warm-started, and it is assumed that vocabularies and Tensor
      names are unchanged.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.

  Returns:
    DNNClassifier with layer annotations.
  """

  original = dnn.DNNClassifier(
      hidden_units=hidden_units,
      feature_columns=feature_columns,
      model_dir=model_dir,
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      optimizer=optimizer,
      activation_fn=activation_fn,
      dropout=dropout,
      input_layer_partitioner=input_layer_partitioner,
      config=config,
      warm_start_from=warm_start_from,
      loss_reduction=loss_reduction)

  def _model_fn(features, labels, mode, config):
    with _monkey_patch(
        feature_column_lib, '_internal_input_layer',
        make_input_layer_with_layer_annotations(
            feature_column_lib._internal_input_layer)):  # pylint: disable=protected-access
      return original.model_fn(features, labels, mode, config)

  return estimator.Estimator(
      model_fn=_model_fn,
      model_dir=model_dir,
      config=config,
      warm_start_from=warm_start_from)


def DNNRegressorWithLayerAnnotations(  # pylint: disable=invalid-name
    hidden_units,
    feature_columns,
    model_dir=None,
    label_dimension=1,
    weight_column=None,
    optimizer='Adagrad',
    activation_fn=nn.relu,
    dropout=None,
    input_layer_partitioner=None,
    config=None,
    warm_start_from=None,
    loss_reduction=losses.Reduction.SUM,
):
  """A regressor for TensorFlow DNN models with layer annotations.

  This regressor is fuctionally identical to estimator.DNNRegressor as far as
  training and evaluating models is concerned. The key difference is that this
  classifier adds additional layer annotations, which can be used for computing
  Integrated Gradients.

  Integrated Gradients is a method for attributing a classifier's predictions
  to its input features (https://arxiv.org/pdf/1703.01365.pdf). Given an input
  instance, the method assigns attribution scores to individual features in
  proportion to the feature's importance to the classifier's prediction.

  See estimator.DNNRegressor for example code for training and evaluating models
  using this regressor.

  This regressor is checkpoint-compatible with estimator.DNNRegressor and
  therefore the following should work seamlessly:

  # Instantiate ordinary estimator as usual.
  estimator = tf.estimator.DNNRegressor(
    config, feature_columns, hidden_units, ...)

  # Train estimator, export checkpoint.
  tf.estimator.train_and_evaluate(estimator, ...)

  # Instantiate estimator with annotations with the same configuration as the
  # ordinary estimator.
  estimator_with_annotations = (
    tf.contrib.estimator.DNNRegressorWithLayerAnnotations(
      config, feature_columns, hidden_units, ...))

  # Call export_savedmodel with the same arguments as the ordinary estimator,
  # using the checkpoint produced for the ordinary estimator.
  estimator_with_annotations.export_saved_model(
    export_dir_base, serving_input_receiver, ...
    checkpoint_path='/path/to/ordinary/estimator/checkpoint/model.ckpt-1234')

  Args:
    hidden_units: Iterable of number hidden units per layer. All layers are
      fully connected. Ex. `[64, 32]` means first layer has 64 nodes and second
      one has 32.
    feature_columns: An iterable containing all the feature columns used by the
      model. All items in the set should be instances of classes derived from
      `_FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    label_dimension: Number of regression targets per example. This is the size
      of the last dimension of the labels and logits `Tensor` objects
      (typically, these have shape `[batch_size, label_dimension]`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
      weight_column.normalizer_fn is applied on it to get weight tensor.
    optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
      to Adagrad optimizer.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Optional. Partitioner for input layer. Defaults to
      `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.
    warm_start_from: A string filepath to a checkpoint to warm-start from, or a
      `WarmStartSettings` object to fully configure warm-starting.  If the
      string filepath is provided instead of a `WarmStartSettings`, then all
      weights are warm-started, and it is assumed that vocabularies and Tensor
      names are unchanged.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.

  Returns:
    DNNRegressor with layer annotations.
  """

  original = dnn.DNNRegressor(
      hidden_units=hidden_units,
      feature_columns=feature_columns,
      model_dir=model_dir,
      label_dimension=label_dimension,
      weight_column=weight_column,
      optimizer=optimizer,
      activation_fn=activation_fn,
      dropout=dropout,
      input_layer_partitioner=input_layer_partitioner,
      config=config,
      warm_start_from=warm_start_from,
      loss_reduction=loss_reduction,
  )

  def _model_fn(features, labels, mode, config):
    with _monkey_patch(
        feature_column_lib, '_internal_input_layer',
        make_input_layer_with_layer_annotations(
            feature_column_lib._internal_input_layer)):  # pylint: disable=protected-access
      return original.model_fn(features, labels, mode, config)

  return estimator.Estimator(
      model_fn=_model_fn,
      model_dir=model_dir,
      config=config,
      warm_start_from=warm_start_from)
