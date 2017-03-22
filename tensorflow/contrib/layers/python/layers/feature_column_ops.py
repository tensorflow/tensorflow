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

from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import embedding_ops
from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_py
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


def _embeddings_from_arguments(column,
                               args,
                               weight_collections,
                               trainable,
                               output_rank=2):
  """Returns embeddings for a column based on the computed arguments.

  Args:
   column: the column name.
   args: the _DeepEmbeddingLookupArguments for this column.
   weight_collections: collections to store weights in.
   trainable: whether these embeddings should be trainable.
   output_rank: the desired rank of the returned `Tensor`. Inner dimensions will
     be combined to produce the desired rank.

  Returns:
   the embeddings.

  Raises:
   ValueError: if not possible to create.
  """
  # pylint: disable=protected-access
  input_tensor = layers._inner_flatten(args.input_tensor, output_rank)
  weight_tensor = None
  if args.weight_tensor is not None:
    weight_tensor = layers._inner_flatten(args.weight_tensor, output_rank)
  # pylint: enable=protected-access

  # This option is only enabled for scattered_embedding_column.
  if args.hash_key:
    embeddings = contrib_variables.model_variable(
        name='weights',
        shape=[args.vocab_size],
        dtype=dtypes.float32,
        initializer=args.initializer,
        trainable=(trainable and args.trainable),
        collections=weight_collections)

    return embedding_ops.scattered_embedding_lookup_sparse(
        embeddings, input_tensor, args.dimension,
        hash_key=args.hash_key,
        combiner=args.combiner, name='lookup')

  if args.shared_embedding_name is not None:
    shared_embedding_collection_name = (
        'SHARED_EMBEDDING_COLLECTION_' + args.shared_embedding_name.upper())
    graph = ops.get_default_graph()
    shared_embedding_collection = (
        graph.get_collection_ref(shared_embedding_collection_name))
    shape = [args.vocab_size, args.dimension]
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError('Collection %s can only contain one '
                         '(partitioned) variable.'
                         % shared_embedding_collection_name)
      else:
        embeddings = shared_embedding_collection[0]
        if embeddings.get_shape() != shape:
          raise ValueError('The embedding variable with name {} already '
                           'exists, but its shape does not match required '
                           'embedding shape  here. Please make sure to use '
                           'different shared_embedding_name for different '
                           'shared embeddings.'.format(
                               args.shared_embedding_name))
    else:
      embeddings = contrib_variables.model_variable(
          name=args.shared_embedding_name,
          shape=shape,
          dtype=dtypes.float32,
          initializer=args.initializer,
          trainable=(trainable and args.trainable),
          collections=weight_collections)
      graph.add_to_collection(shared_embedding_collection_name, embeddings)
  else:
    embeddings = contrib_variables.model_variable(
        name='weights',
        shape=[args.vocab_size, args.dimension],
        dtype=dtypes.float32,
        initializer=args.initializer,
        trainable=(trainable and args.trainable),
        collections=weight_collections)

  if isinstance(embeddings, variables.Variable):
    embeddings = [embeddings]
  else:
    embeddings = embeddings._get_variable_list()  # pylint: disable=protected-access
  # pylint: disable=protected-access
  _maybe_restore_from_checkpoint(
      column._checkpoint_path(), embeddings)
  return embedding_ops.safe_embedding_lookup_sparse(
      embeddings,
      input_tensor,
      sparse_weights=weight_tensor,
      combiner=args.combiner,
      name=column.name + 'weights',
      max_norm=args.max_norm)


def _input_from_feature_columns(columns_to_tensors,
                                feature_columns,
                                weight_collections,
                                trainable,
                                scope,
                                output_rank,
                                default_name):
  """Implementation of `input_from(_sequence)_feature_columns`."""
  check_feature_columns(feature_columns)
  with variable_scope.variable_scope(scope,
                                     default_name=default_name,
                                     values=columns_to_tensors.values()):
    output_tensors = []
    transformer = _Transformer(columns_to_tensors)
    if weight_collections:
      weight_collections = list(set(list(weight_collections) +
                                    [ops.GraphKeys.GLOBAL_VARIABLES]))

    for column in sorted(set(feature_columns), key=lambda x: x.key):
      with variable_scope.variable_scope(None,
                                         default_name=column.name,
                                         values=columns_to_tensors.values()):
        transformed_tensor = transformer.transform(column)
        try:
          # pylint: disable=protected-access
          arguments = column._deep_embedding_lookup_arguments(
              transformed_tensor)
          output_tensors.append(_embeddings_from_arguments(
              column,
              arguments,
              weight_collections,
              trainable,
              output_rank=output_rank))

        except NotImplementedError as ee:
          try:
            # pylint: disable=protected-access
            output_tensors.append(column._to_dnn_input_layer(
                transformed_tensor,
                weight_collections,
                trainable,
                output_rank=output_rank))
          except ValueError as e:
            raise ValueError('Error creating input layer for column: {}.\n'
                             '{}, {}'.format(column.name, e, ee))
    return array_ops.concat(output_tensors, output_rank - 1)


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

  Example:

  ```python
    # Building model for training
    columns_to_tensor = tf.parse_example(...)
    first_layer = input_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns)
    second_layer = fully_connected(inputs=first_layer, ...)
    ...
  ```

  where feature_columns can be defined as follows:

  ```python
    sparse_feature = sparse_column_with_hash_bucket(
        column_name="sparse_col", ...)
    sparse_feature_emb = embedding_column(sparse_id_column=sparse_feature, ...)
    real_valued_feature = real_valued_column(...)
    real_valued_buckets = bucketized_column(
        source_column=real_valued_feature, ...)

    feature_columns=[sparse_feature_emb, real_valued_buckets]
  ```

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
    scope: Optional scope for variable_scope.

  Returns:
    A Tensor which can be consumed by hidden layers in the neural network.

  Raises:
    ValueError: if FeatureColumn cannot be consumed by a neural network.
  """
  return _input_from_feature_columns(columns_to_tensors,
                                     feature_columns,
                                     weight_collections,
                                     trainable,
                                     scope,
                                     output_rank=2,
                                     default_name='input_from_feature_columns')


@experimental
def sequence_input_from_feature_columns(columns_to_tensors,
                                        feature_columns,
                                        weight_collections=None,
                                        trainable=True,
                                        scope=None):
  """Builds inputs for sequence models from `FeatureColumn`s.

  See documentation for `input_from_feature_columns`. The following types of
  `FeatureColumn` are permitted in `feature_columns`: `_OneHotColumn`,
  `_EmbeddingColumn`, `_ScatteredEmbeddingColumn`, `_RealValuedColumn`,
  `_DataFrameColumn`. In addition, columns in `feature_columns` may not be
  constructed using any of the following: `ScatteredEmbeddingColumn`,
  `BucketizedColumn`, `CrossedColumn`.

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
    scope: Optional scope for variable_scope.

  Returns:
    A Tensor which can be consumed by hidden layers in the neural network.

  Raises:
    ValueError: if FeatureColumn cannot be consumed by a neural network.
  """
  _check_supported_sequence_columns(feature_columns)
  _check_forbidden_sequence_columns(feature_columns)

  return _input_from_feature_columns(
      columns_to_tensors,
      feature_columns,
      weight_collections,
      trainable,
      scope,
      output_rank=3,
      default_name='sequence_input_from_feature_columns')


def _create_embedding_lookup(column,
                             columns_to_tensors,
                             embedding_lookup_arguments,
                             num_outputs,
                             trainable,
                             weight_collections):
  """Creates variables and returns predictions for linear weights in a model.

  Args:
   column: the column we're working on.
   columns_to_tensors: a map from column name to tensors.
   embedding_lookup_arguments: arguments for embedding lookup.
   num_outputs: how many outputs.
   trainable: whether the variable we create is trainable.
   weight_collections: weights will be placed here.

  Returns:
  variables: the created embeddings.
  predictions: the computed predictions.
  """
  with variable_scope.variable_scope(
      None, default_name=column.name, values=columns_to_tensors.values()):
    variable = contrib_variables.model_variable(
        name='weights',
        shape=[embedding_lookup_arguments.vocab_size, num_outputs],
        dtype=dtypes.float32,
        initializer=embedding_lookup_arguments.initializer,
        trainable=trainable,
        collections=weight_collections)
    if isinstance(variable, variables.Variable):
      variable = [variable]
    else:
      variable = variable._get_variable_list()  # pylint: disable=protected-access
    predictions = embedding_ops.safe_embedding_lookup_sparse(
        variable,
        embedding_lookup_arguments.input_tensor,
        sparse_weights=embedding_lookup_arguments.weight_tensor,
        combiner=embedding_lookup_arguments.combiner,
        name=column.name + '_weights')
    return variable, predictions


def _maybe_restore_from_checkpoint(checkpoint_path, variable):
  if checkpoint_path is not None:
    path, tensor_name = checkpoint_path
    weights_to_restore = variable
    if len(variable) == 1:
      weights_to_restore = variable[0]
    checkpoint_utils.init_from_checkpoint(path,
                                          {tensor_name: weights_to_restore})


def _create_joint_embedding_lookup(columns_to_tensors,
                                   embedding_lookup_arguments,
                                   num_outputs,
                                   trainable,
                                   weight_collections):
  """Creates an embedding lookup for all columns sharing a single weight."""
  for arg in embedding_lookup_arguments:
    assert arg.weight_tensor is None, (
        'Joint sums for weighted sparse columns are not supported. '
        'Please use weighted_sum_from_feature_columns instead.')
    assert arg.combiner == 'sum', (
        'Combiners other than sum are not supported for joint sums. '
        'Please use weighted_sum_from_feature_columns instead.')
  assert len(embedding_lookup_arguments) >= 1, (
      'At least one column must be in the model.')
  prev_size = 0
  sparse_tensors = []
  for a in embedding_lookup_arguments:
    t = a.input_tensor
    values = t.values + prev_size
    prev_size += a.vocab_size
    sparse_tensors.append(
        sparse_tensor_py.SparseTensor(t.indices,
                                      values,
                                      t.dense_shape))
  sparse_tensor = sparse_ops.sparse_concat(1, sparse_tensors)
  with variable_scope.variable_scope(
      None, default_name='linear_weights', values=columns_to_tensors.values()):
    variable = contrib_variables.model_variable(
        name='weights',
        shape=[prev_size, num_outputs],
        dtype=dtypes.float32,
        initializer=init_ops.zeros_initializer(),
        trainable=trainable,
        collections=weight_collections)
    if isinstance(variable, variables.Variable):
      variable = [variable]
    else:
      variable = variable._get_variable_list()  # pylint: disable=protected-access
    predictions = embedding_ops.safe_embedding_lookup_sparse(
        variable,
        sparse_tensor,
        sparse_weights=None,
        combiner='sum',
        name='_weights')
    return variable, predictions


def joint_weighted_sum_from_feature_columns(columns_to_tensors,
                                            feature_columns,
                                            num_outputs,
                                            weight_collections=None,
                                            trainable=True,
                                            scope=None):
  """A restricted linear prediction builder based on FeatureColumns.

  As long as all feature columns are unweighted sparse columns this computes the
  prediction of a linear model which stores all weights in a single variable.

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
    scope: Optional scope for variable_scope.

  Returns:
    A tuple containing:

    * A Tensor which represents predictions of a linear model.
    * A list of Variables storing the weights.
    * A Variable which is used for bias.

  Raises:
    ValueError: if FeatureColumn cannot be used for linear predictions.

  """
  check_feature_columns(feature_columns)
  with variable_scope.variable_scope(
      scope,
      default_name='joint_weighted_sum_from_feature_columns',
      values=columns_to_tensors.values()):
    transformer = _Transformer(columns_to_tensors)
    embedding_lookup_arguments = []
    for column in sorted(set(feature_columns), key=lambda x: x.key):
      transformed_tensor = transformer.transform(column)
      try:
        embedding_lookup_arguments.append(
            column._wide_embedding_lookup_arguments(transformed_tensor))   # pylint: disable=protected-access
      except NotImplementedError:
        raise NotImplementedError('Real-valued columns are not supported. '
                                  'Use weighted_sum_from_feature_columns '
                                  'instead, or bucketize these columns.')

    variable, predictions_no_bias = _create_joint_embedding_lookup(
        columns_to_tensors,
        embedding_lookup_arguments,
        num_outputs,
        trainable,
        weight_collections)
    bias = contrib_variables.model_variable(
        'bias_weight',
        shape=[num_outputs],
        initializer=init_ops.zeros_initializer(),
        trainable=trainable,
        collections=_add_variable_collection(weight_collections))
    _log_variable(bias)
    predictions = nn_ops.bias_add(predictions_no_bias, bias)

    return predictions, variable, bias


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

  Example:

    ```
    # Building model for training
    feature_columns = (
        real_valued_column("my_feature1"),
        ...
    )
    columns_to_tensor = tf.parse_example(...)
    logits = weighted_sum_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns,
        num_outputs=1)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    ```

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
    scope: Optional scope for variable_scope.

  Returns:
    A tuple containing:

      * A Tensor which represents predictions of a linear model.
      * A dictionary which maps feature_column to corresponding Variable.
      * A Variable which is used for bias.

  Raises:
    ValueError: if FeatureColumn cannot be used for linear predictions.
  """
  check_feature_columns(feature_columns)
  with variable_scope.variable_scope(
      scope,
      default_name='weighted_sum_from_feature_columns',
      values=columns_to_tensors.values()):
    output_tensors = []
    column_to_variable = dict()
    transformer = _Transformer(columns_to_tensors)
    # pylint: disable=protected-access
    for column in sorted(set(feature_columns), key=lambda x: x.key):
      transformed_tensor = transformer.transform(column)
      try:
        embedding_lookup_arguments = column._wide_embedding_lookup_arguments(
            transformed_tensor)
        variable, predictions = _create_embedding_lookup(
            column,
            columns_to_tensors,
            embedding_lookup_arguments,
            num_outputs,
            trainable,
            weight_collections)
      except NotImplementedError:
        with variable_scope.variable_scope(
            None,
            default_name=column.name,
            values=columns_to_tensors.values()):
          tensor = column._to_dense_tensor(transformed_tensor)
          tensor = fc._reshape_real_valued_tensor(tensor, 2, column.name)
          variable = [
              contrib_variables.model_variable(
                  name='weight',
                  shape=[tensor.get_shape()[1], num_outputs],
                  initializer=init_ops.zeros_initializer(),
                  trainable=trainable,
                  collections=weight_collections)
          ]
          predictions = math_ops.matmul(tensor, variable[0], name='matmul')
      except ValueError as ee:
        raise ValueError('Error creating weighted sum for column: {}.\n'
                         '{}'.format(column.name, ee))
      output_tensors.append(array_ops.reshape(
          predictions, shape=(-1, num_outputs)))
      column_to_variable[column] = variable
      _log_variable(variable)
      _maybe_restore_from_checkpoint(column._checkpoint_path(), variable)
    # pylint: enable=protected-access
    predictions_no_bias = math_ops.add_n(output_tensors)
    bias = contrib_variables.model_variable(
        'bias_weight',
        shape=[num_outputs],
        initializer=init_ops.zeros_initializer(),
        trainable=trainable,
        collections=_add_variable_collection(weight_collections))
    _log_variable(bias)
    predictions = nn_ops.bias_add(predictions_no_bias, bias)

    return predictions, column_to_variable, bias


def parse_feature_columns_from_examples(serialized,
                                        feature_columns,
                                        name=None,
                                        example_names=None):
  """Parses tf.Examples to extract tensors for given feature_columns.

  This is a wrapper of 'tf.parse_example'.

  Example:

  ```python
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=my_features)

  # Where my_features are:
  # Define features and transformations
  sparse_feature_a = sparse_column_with_keys(
      column_name="sparse_feature_a", keys=["AB", "CD", ...])

  embedding_feature_a = embedding_column(
      sparse_id_column=sparse_feature_a, dimension=3, combiner="sum")

  sparse_feature_b = sparse_column_with_hash_bucket(
      column_name="sparse_feature_b", hash_bucket_size=1000)

  embedding_feature_b = embedding_column(
      sparse_id_column=sparse_feature_b, dimension=16, combiner="sum")

  crossed_feature_a_x_b = crossed_column(
      columns=[sparse_feature_a, sparse_feature_b], hash_bucket_size=10000)

  real_feature = real_valued_column("real_feature")
  real_feature_buckets = bucketized_column(
      source_column=real_feature, boundaries=[...])

  my_features = [embedding_feature_b, real_feature_buckets, embedding_feature_a]
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


def transform_features(features, feature_columns):
  """Returns transformed features based on features columns passed in.

  Example:

  ```python
  columns_to_tensor = transform_features(features=features,
                                         feature_columns=feature_columns)

  # Where my_features are:
  # Define features and transformations
  sparse_feature_a = sparse_column_with_keys(
      column_name="sparse_feature_a", keys=["AB", "CD", ...])

  embedding_feature_a = embedding_column(
      sparse_id_column=sparse_feature_a, dimension=3, combiner="sum")

  sparse_feature_b = sparse_column_with_hash_bucket(
      column_name="sparse_feature_b", hash_bucket_size=1000)

  embedding_feature_b = embedding_column(
      sparse_id_column=sparse_feature_b, dimension=16, combiner="sum")

  crossed_feature_a_x_b = crossed_column(
      columns=[sparse_feature_a, sparse_feature_b], hash_bucket_size=10000)

  real_feature = real_valued_column("real_feature")
  real_feature_buckets = bucketized_column(
      source_column=real_feature, boundaries=[...])

  feature_columns = [embedding_feature_b,
                     real_feature_buckets,
                     embedding_feature_a]
  ```

  Args:
    features: A dictionary of features.
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn.

  Returns:
    A `dict` mapping FeatureColumn to `Tensor` and `SparseTensor` values.
  """
  check_feature_columns(feature_columns)
  columns_to_tensor = features.copy()
  transformer = _Transformer(columns_to_tensor)
  for column in sorted(set(feature_columns), key=lambda x: x.key):
    transformer.transform(column)
  keys = list(columns_to_tensor.keys())
  for k in keys:
    if k not in feature_columns:
      columns_to_tensor.pop(k)
  return columns_to_tensor


def parse_feature_columns_from_sequence_examples(
    serialized,
    context_feature_columns,
    sequence_feature_columns,
    name=None,
    example_name=None):
  """Parses tf.SequenceExamples to extract tensors for given `FeatureColumn`s.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single serialized
      `SequenceExample` proto.
    context_feature_columns: An iterable containing the feature columns for
      context features. All items should be instances of classes derived from
      `_FeatureColumn`. Can be `None`.
    sequence_feature_columns: An iterable containing the feature columns for
      sequence features. All items should be instances of classes derived from
      `_FeatureColumn`. Can be `None`.
    name: A name for this operation (optional).
    example_name: A scalar (0-D Tensor) of type string (optional), the names of
      the serialized proto.

  Returns:
    A tuple consisting of:
    context_features: a dict mapping `FeatureColumns` from
      `context_feature_columns` to their parsed `Tensors`/`SparseTensor`s.
    sequence_features: a dict mapping `FeatureColumns` from
      `sequence_feature_columns` to their parsed `Tensors`/`SparseTensor`s.
  """
  # Sequence example parsing requires a single (scalar) example.
  try:
    serialized = array_ops.reshape(serialized, [])
  except ValueError as e:
    raise ValueError(
        'serialized must contain as single sequence example. Batching must be '
        'done after parsing for sequence examples. Error: {}'.format(e))

  if context_feature_columns is None:
    context_feature_columns = []
  if sequence_feature_columns is None:
    sequence_feature_columns = []

  check_feature_columns(context_feature_columns)
  context_feature_spec = fc.create_feature_spec_for_parsing(
      context_feature_columns)

  check_feature_columns(sequence_feature_columns)
  sequence_feature_spec = fc._create_sequence_feature_spec_for_parsing(  # pylint: disable=protected-access
      sequence_feature_columns, allow_missing_by_default=False)

  return parsing_ops.parse_single_sequence_example(serialized,
                                                   context_feature_spec,
                                                   sequence_feature_spec,
                                                   example_name,
                                                   name)


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
  if isinstance(tensor, sparse_tensor_py.SparseTensor):
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
    feature_columns: An iterable of instances or subclasses of FeatureColumn.

  Raises:
    ValueError: If `feature_columns` is a dict.
    ValueError: If there are duplicate feature column keys.
  """
  if isinstance(feature_columns, dict):
    raise ValueError('Expected feature_columns to be iterable, found dict.')
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

  Some features may be used in more than one place. For example, one can use a
  bucketized feature by itself and a cross with it. In that case Transformer
  should create only one bucketization op instead of multiple ops for each
  feature column. To handle re-use of transformed columns, Transformer keeps all
  previously transformed columns.

  Example:

  ```python
    sparse_feature = sparse_column_with_hash_bucket(...)
    real_valued_feature = real_valued_column(...)
    real_valued_buckets = bucketized_column(source_column=real_valued_feature,
                                            ...)
    sparse_x_real = crossed_column(
        columns=[sparse_feature, real_valued_buckets], hash_bucket_size=10000)

    columns_to_tensor = tf.parse_example(...)
    transformer = Transformer(columns_to_tensor)

    sparse_x_real_tensor = transformer.transform(sparse_x_real)
    sparse_tensor = transformer.transform(sparse_feature)
    real_buckets_tensor = transformer.transform(real_valued_buckets)
  ```
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
    logging.debug('Transforming feature_column %s', feature_column)
    if feature_column in self._columns_to_tensors:
      # Feature_column is already transformed.
      return self._columns_to_tensors[feature_column]

    feature_column.insert_transformed_feature(self._columns_to_tensors)

    if feature_column not in self._columns_to_tensors:
      raise ValueError('Column {} is not supported.'.format(
          feature_column.name))

    return self._columns_to_tensors[feature_column]


def _add_variable_collection(weight_collections):
  if weight_collections:
    weight_collections = list(
        set(list(weight_collections) + [ops.GraphKeys.GLOBAL_VARIABLES]))
  return weight_collections


# TODO(jamieas): remove the following logic once all FeatureColumn types are
# supported for sequences.
# pylint: disable=protected-access
_SUPPORTED_SEQUENCE_COLUMNS = (fc._OneHotColumn,
                               fc._EmbeddingColumn,
                               fc._RealValuedColumn)

_FORBIDDEN_SEQUENCE_COLUMNS = (fc._ScatteredEmbeddingColumn,
                               fc._BucketizedColumn,
                               fc._CrossedColumn)


def _check_supported_sequence_columns(feature_columns):
  """Asserts `feature_columns` are in `_SUPPORTED_SEQUENCE_COLUMNS`."""
  for col in feature_columns:
    if not isinstance(col, _SUPPORTED_SEQUENCE_COLUMNS):
      raise ValueError(
          'FeatureColumn type {} is not currently supported for sequence data.'.
          format(type(col).__name__))


def _get_parent_columns(feature_column):
  """Returns the tuple of `FeatureColumn`s that `feature_column` depends on."""
  if isinstance(feature_column, (fc._WeightedSparseColumn,
                                 fc._OneHotColumn,
                                 fc._EmbeddingColumn,)):
    return (feature_column.sparse_id_column,)
  if isinstance(feature_column, (fc._BucketizedColumn,)):
    return (feature_column.source_column,)
  if isinstance(feature_column, (fc._CrossedColumn)):
    return tuple(feature_column.columns)
  return tuple()


def _gather_feature_columns(feature_columns):
  """Returns a list of all ancestor `FeatureColumns` of `feature_columns`."""
  gathered = list(feature_columns)
  i = 0
  while i < len(gathered):
    for column in _get_parent_columns(gathered[i]):
      if column not in gathered:
        gathered.append(column)
    i += 1
  return gathered


def _check_forbidden_sequence_columns(feature_columns):
  """Recursively cecks `feature_columns` for `_FORBIDDEN_SEQUENCE_COLUMNS`."""
  all_feature_columns = _gather_feature_columns(feature_columns)
  for feature_column in all_feature_columns:
    if isinstance(feature_column, _FORBIDDEN_SEQUENCE_COLUMNS):
      raise ValueError(
          'Column {} is of type {}, which is not currently supported for '
          'sequences.'.format(feature_column.name,
                              type(feature_column).__name__))
