# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to FeatureColumn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


def input_from_feature_columns(columns_to_tensors,
                               feature_columns,
                               weight_collections=None,
                               trainable=True,
                               scope=None):
  """A tf.contrib.layer style input layer builder based on FeatureColumns.

  Generally a single example in training data is described with feature columns.
  At the first layer of the model, this column oriented data should be converted
  to a single tensor. Each feature column needs a different kind of operation
  during this conversion. For example sparse features need a totally different
  handling than continuous features.

  An example usage of input_from_feature_columns is as follows:

    # Building model for training
    columns_to_tensor = tf.parse_example(...)
    first_layer = input_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns)
    second_layer = fully_connected(first_layer, ...)
    ...

    where feature_columns can be defined as follows:

    occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)
    occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                     combiner="sum")
    age = real_valued_column("age")
    age_buckets = bucketized_column(
        source_column=age,
        boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    occupation_x_age = crossed_column(columns=[occupation, age_buckets],
                                      hash_bucket_size=10000)

    feature_columns=[occupation_emb, occupation_x_age]

  Args:
    columns_to_tensors: A mapping from feature column to tensors. 'string' key
      means a base feature (not-transformed). It can have FeatureColumn as a
      key too. That means that FeatureColumn is already transformed by input
      pipeline. For example, `inflow` may have handled transformations.
    feature_columns: A set containing all the feature columns. All items in the
      set should be instances of classes derived by FeatureColumn.
    weight_collections: List of graph collections to which weights are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_op_scope.

  Returns:
    A Tensor which can be consumed by hidden layers in the neural network.

  Raises:
    ValueError: if FeatureColumn cannot be consumed by a neural network.
  """
  check_feature_columns(feature_columns)
  with variable_scope.variable_op_scope(columns_to_tensors.values(), scope,
                                        'input_from_feature_columns'):
    output_tensors = []
    transformer = _Transformer(columns_to_tensors)
    if weight_collections:
      weight_collections = list(set(list(weight_collections) +
                                    [ops.GraphKeys.VARIABLES]))

    for column in sorted(set(feature_columns), key=lambda x: x.key):
      try:
        transformed_tensor = transformer.transform(column)
        output_tensors.append(column.to_dnn_input_layer(
            transformed_tensor, weight_collections, trainable))
      except ValueError as e:
        raise ValueError('Error creating input layer for column: {}.\n'
                         '{}'.format(column.name, e))
    return array_ops.concat(1, output_tensors)


def weighted_sum_from_feature_columns(columns_to_tensors,
                                      feature_columns,
                                      num_outputs,
                                      weight_collections=None,
                                      trainable=True,
                                      scope=None):
  """A tf.contrib.layer style linear prediction builder based on FeatureColumns.

  Generally a single example in training data is described with feature columns.
  This function generates weighted sum for each num_outputs. Weighted sum refers
  to logits in classification problems. It refers to prediction itself for
  linear regression problems.

  An example usage of weighted_sum_from_feature_columns is as follows:

    # Building model for training
    columns_to_tensor = tf.parse_example(...)
    logits = weighted_sum_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns,
        num_outputs=1)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)

    where feature_columns can be defined as follows:

    occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)
    occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                     combiner="sum")
    age = real_valued_column("age")
    age_buckets = bucketized_column(
        source_column=age,
        boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    occupation_x_age = crossed_column(columns=[occupation, age_buckets],
                                      hash_bucket_size=10000)

    feature_columns=[occupation_emb, occupation_x_age]

  Args:
    columns_to_tensors: A mapping from feature column to tensors. 'string' key
      means a base feature (not-transformed). It can have FeatureColumn as a
      key too. That means that FeatureColumn is already transformed by input
      pipeline. For example, `inflow` may have handled transformations.
    feature_columns: A set containing all the feature columns. All items in the
      set should be instances of classes derived from FeatureColumn.
    num_outputs: An integer specifying number of outputs. Default value is 1.
    weight_collections: List of graph collections to which weights are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope fpor variable_op_scope.

  Returns:
    A tuple of followings:
      * A Tensor which represents predictions of a linear model.
      * A dictionary which maps feature_column to corresponding Variable.
      * A Variable which is used for bias.

  Raises:
    ValueError: if FeatureColumn cannot be used for linear predictions.
  """
  check_feature_columns(feature_columns)
  with variable_scope.variable_op_scope(columns_to_tensors.values(), scope,
                                        'weighted_sum_from_feature_columns'):
    output_tensors = []
    column_to_variable = dict()
    transformer = _Transformer(columns_to_tensors)
    for column in sorted(set(feature_columns), key=lambda x: x.key):
      try:
        transformed_tensor = transformer.transform(column)
        predictions, variable = column.to_weighted_sum(transformed_tensor,
                                                       num_outputs,
                                                       weight_collections,
                                                       trainable)
      except ValueError as e:
        raise ValueError('Error creating weighted sum for column: {}.\n'
                         '{}'.format(column.name, e))
      output_tensors.append(predictions)
      column_to_variable[column] = variable
      _log_variable(variable)

    predictions_no_bias = math_ops.add_n(output_tensors)
    bias = contrib_variables.model_variable(
        'bias_weight',
        shape=[num_outputs],
        initializer=init_ops.zeros_initializer,
        collections=fc._add_variable_collection(weight_collections))  # pylint: disable=protected-access
    _log_variable(bias)
    predictions = nn_ops.bias_add(predictions_no_bias, bias)

    return predictions, column_to_variable, bias


def parse_feature_columns_from_examples(serialized,
                                        feature_columns,
                                        name=None,
                                        example_names=None):
  """Parses tf.Examples to extract tensors for given feature_columns.

  This is a wrapper of 'tf.parse_example'. A typical usage is as follows:

  ```python
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=my_features)

  # Where my_features are:
  # Define features and transformations
  country = sparse_column_with_keys(column_name="native_country",
                                    keys=["US", "BRA", ...])
  country_emb = embedding_column(sparse_id_column=country, dimension=3,
                                 combiner="sum")
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")
  occupation_x_country = crossed_column(columns=[occupation, country],
                                        hash_bucket_size=10000)
  age = real_valued_column("age")
  age_buckets = bucketized_column(
      source_column=age,
      boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  my_features = [occupation_emb, age_buckets, country_emb]
  ```

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn.
    name: A name for this operation (optional).
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.

  Returns:
    A `dict` mapping FeatureColumn to `Tensor` and `SparseTensor` values.
  """
  check_feature_columns(feature_columns)
  columns_to_tensors = parsing_ops.parse_example(
      serialized=serialized,
      features=fc.create_feature_spec_for_parsing(feature_columns),
      name=name,
      example_names=example_names)

  transformer = _Transformer(columns_to_tensors)
  for column in sorted(set(feature_columns), key=lambda x: x.key):
    transformer.transform(column)
  return columns_to_tensors


def _log_variable(variable):
  if isinstance(variable, list):
    for var in variable:
      if isinstance(variable, variables.Variable):
        logging.info('Created variable %s, with device=%s', var.name,
                     var.device)
  elif isinstance(variable, variables.Variable):
    logging.info('Created variable %s, with device=%s', variable.name,
                 variable.device)


def _infer_real_valued_column_for_tensor(name, tensor):
  """Creates a real_valued_column for given tensor and name."""
  if isinstance(tensor, ops.SparseTensor):
    raise ValueError(
        'SparseTensor is not supported for auto detection. Please define '
        'corresponding FeatureColumn for tensor {} {}.', name, tensor)

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(
        'Non integer or non floating types are not supported for auto detection'
        '. Please define corresponding FeatureColumn for tensor {} {}.', name,
        tensor)

  shape = tensor.get_shape().as_list()
  dimension = 1
  for i in range(1, len(shape)):
    dimension *= shape[i]
  return fc.real_valued_column(name, dimension=dimension, dtype=tensor.dtype)


def infer_real_valued_columns(features):
  if not isinstance(features, dict):
    return [_infer_real_valued_column_for_tensor('', features)]

  feature_columns = []
  for key, value in features.items():
    feature_columns.append(_infer_real_valued_column_for_tensor(key, value))

  return feature_columns


def check_feature_columns(feature_columns):
  """Checks the validity of the set of FeatureColumns.

  Args:
    feature_columns: A set of instances or subclasses of FeatureColumn.

  Raises:
    ValueError: If there are duplicate feature column keys.
  """
  seen_keys = set()
  for f in feature_columns:
    key = f.key
    if key in seen_keys:
      raise ValueError('Duplicate feature column key found for column: {}. '
                       'This usually means that the column is almost identical '
                       'to another column, and one must be discarded.'.format(
                           f.name))
    seen_keys.add(key)


class _Transformer(object):
  """Handles all the transformations defined by FeatureColumn if needed.

  FeatureColumn specifies how to digest an input column to the network. Some
  feature columns require data transformations. This class handles those
  transformations if they are not handled already.

  Some features may be used in more than one places. For example one can use a
  bucketized feature by itself and a cross with it. In that case Transformer
  should create only one bucketization op instead of multiple ops for each
  feature column. To handle re-use of transformed columns, Transformer keeps all
  previously transformed columns.

  An example usage of Transformer is as follows:

    occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                                hash_bucket_size=1000)
    age = real_valued_column("age")
    age_buckets = bucketized_column(
        source_column=age,
        boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    occupation_x_age = crossed_column(columns=[occupation, age_buckets],
                                      hash_bucket_size=10000)

    columns_to_tensor = tf.parse_example(...)
    transformer = Transformer(columns_to_tensor)

    occupation_x_age_tensor = transformer.transform(occupation_x_age)
    occupation_tensor = transformer.transform(occupation)
    age_buckets_tensor = transformer.transform(age_buckets)
  """

  def __init__(self, columns_to_tensors):
    """Initializes transfomer.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have FeatureColumn as
        a key too. That means that FeatureColumn is already transformed by input
        pipeline. For example, `inflow` may have handled transformations.
        Transformed features are inserted in columns_to_tensors.
    """
    self._columns_to_tensors = columns_to_tensors

  def transform(self, feature_column):
    """Returns a Tensor which represents given feature_column.

    Args:
      feature_column: An instance of FeatureColumn.

    Returns:
      A Tensor which represents given feature_column. It may create a new Tensor
      or re-use an existing one.

    Raises:
      ValueError: if FeatureColumn cannot be handled by this Transformer.
    """
    logging.info('Transforming feature_column %s', feature_column)
    if feature_column in self._columns_to_tensors:
      # Feature_column is already transformed.
      return self._columns_to_tensors[feature_column]

    feature_column.insert_transformed_feature(self._columns_to_tensors)

    if feature_column not in self._columns_to_tensors:
      raise ValueError('Column {} is not supported.'.format(
          feature_column.name))

    return self._columns_to_tensors[feature_column]
