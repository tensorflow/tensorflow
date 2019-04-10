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
# ===================================================================
"""Tooling for support TPU embedding in TPUEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow.python.training import training

# pylint: disable=protected-access
_TPU_EMBEDDING_COLUMN_CLASSES = (tpu_fc._TPUEmbeddingColumn,
                                 tpu_fc._TPUSharedEmbeddingColumn)
_EMBEDDING_COLUMN_CLASSES = (core_fc._EmbeddingColumn,
                             core_fc_lib.EmbeddingColumn,
                             core_fc._SharedEmbeddingColumn)
_SUPPORTED_FEATURE_COLUMNS = (core_fc._NumericColumn, core_fc_lib.NumericColumn)
_SUPPORTED_OPTIMIZERS = (AdagradParameters, AdamParameters,
                         StochasticGradientDescentParameters)

# pylint: enable=protected-access

_TABLE_NAME_PREFIX = 'tbl_'
_LEN_TABLE_NAME_PREFIX = len(_TABLE_NAME_PREFIX)


def _get_table_name_from_embedding_var_name(embedding_var_name):
  return '{}{}'.format(_TABLE_NAME_PREFIX, embedding_var_name)


def _get_embedding_var_name_from_table_name(table_name):
  return table_name[_LEN_TABLE_NAME_PREFIX:]


def _get_embedding_variable_name(scope_name, var_name):
  return '{}/{}'.format(scope_name, var_name)


def _get_slot_variable_names(scope_name, var_name, optimization_parameters):
  """Return embedding variable names which are consistent with CPU runs."""
  if isinstance(optimization_parameters, tpu_embedding.AdagradParameters):
    return tpu_embedding.AdagradSlotVariableName(
        '{}/{}/Adagrad'.format(scope_name, var_name)
    )
  elif isinstance(optimization_parameters, tpu_embedding.AdamParameters):
    return tpu_embedding.AdamSlotVariableNames(
        '{}/{}/Adam/m'.format(scope_name, var_name),
        '{}/{}/Adam/v'.format(scope_name, var_name)
    )
  elif isinstance(optimization_parameters,
                  tpu_embedding.StochasticGradientDescentParameters):
    return None
  else:
    raise ValueError('Support to infer full variable name '
                     'for optimization_parameter {} has not been added.'
                     .format(optimization_parameters))


def get_full_variable_names(
    graph, table_to_config_dict, optimization_parameters=None):
  """Return embedding variable names and slot variables which are consistent with CPU runs."""
  collection = graph.get_collection_ref(tpu_fc._TPU_FC_TO_SCOPE)  # pylint: disable=protected-access
  if not collection:
    raise RuntimeError(
        'Embedding feature column did not capture any thing. Make sure the '
        'feature columns passed to TPUEstimator constructor is properly '
        'used in model_fn.')

  embedding_variable_name_by_table = {}
  slot_variable_names_by_table = {}
  for table_name in table_to_config_dict:
    embedding_var_name = _get_embedding_var_name_from_table_name(table_name)
    (scope_name, var_name) = collection[0][embedding_var_name]
    embedding_variable_name_by_table[table_name] = (
        _get_embedding_variable_name(scope_name, var_name))
    if optimization_parameters:
      slot_variable_names_by_table[table_name] = _get_slot_variable_names(
          scope_name, var_name, optimization_parameters)

  graph.clear_collection(tpu_fc._TPU_FC_TO_SCOPE)  # pylint: disable=protected-access
  return embedding_variable_name_by_table, slot_variable_names_by_table


def get_configs_from_feature_columns(feature_columns):
  """Create configs for TPUEmbedding etc from a list of feature columns.

  Args:
    feature_columns: a list of supported feature columns.

  Returns:
    A tuple of dicts, the first maps tables to their config, the second maps
    features to tables, and the third maps features to weight key names.
  """

  allowed = (tpu_fc._TPUEmbeddingColumn, tpu_fc._TPUSharedEmbeddingColumn)  # pylint: disable=protected-access

  for column in feature_columns:
    if not isinstance(column, allowed):
      raise TypeError(
          'Unsupported feature column {}. Supported types are {}.'.format(
              type(column), allowed))

  table_to_config = {}
  feature_to_table = {}
  feature_to_weight_key_name = {}
  for column in feature_columns:
    feature_name = column.get_feature_key_name()
    table_name = _get_table_name_from_embedding_var_name(
        column.get_embedding_var_name())
    if feature_name in feature_to_table:
      raise ValueError(
          'Feature column {} is used with multiple embeddings and this is '
          'not supported.'.format(feature_name))
    feature_to_table[feature_name] = table_name
    feature_to_weight_key_name[feature_name] = column.get_weight_key_name()
    vocabulary_size, dimension = column.get_embedding_table_size()
    table_to_config[table_name] = tpu_embedding.TableConfig(
        vocabulary_size=vocabulary_size,
        dimension=dimension,
        initializer=column.get_initializer(),
        combiner=column.get_combiner())

  return table_to_config, feature_to_table, feature_to_weight_key_name


class EmbeddingConfigSpec(
    collections.namedtuple('EmbeddingConfigSpec', [
        'feature_columns', 'optimization_parameters', 'clipping_limit',
        'pipeline_execution_with_tensor_core',
        'experimental_gradient_multiplier_fn'
    ])):
  """Class to keep track of embedding config specification."""

  def __new__(cls,
              feature_columns,
              optimization_parameters,
              clipping_limit=None,
              pipeline_execution_with_tensor_core=False,
              experimental_gradient_multiplier_fn=None):
    """Creates an EmbeddingConfigSpec instance.

    Args:
      feature_columns: All `FeatureColumn`s used by model.
      optimization_parameters: An instance of `AdagradParameters`,
        `AdamParameters` or `StochasticGradientDescentParameters`. This
        optimizer will be applied to all embedding variables specified by
        `feature_columns`.
      clipping_limit: (Optional) Clipping limit (absolute value).
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding IDs. Please see
        `tpu_embedding_configuration.proto` for details.
      experimental_gradient_multiplier_fn: (Optional) A Fn taking global step as
        input returning the current multiplier for all embedding gradients.

    Returns:
      An EmbeddingConfigSpec instance.

    Raises:
      ValueError: If the feature_columns are not specified.
      TypeError: If the feature columns are not of ths correct type (one of
        _SUPPORTED_FEATURE_COLUMNS, _TPU_EMBEDDING_COLUMN_CLASSES OR
        _EMBEDDING_COLUMN_CLASSES).
      ValueError: If `optimization_parameters` is not one of the required types.
    """
    if not feature_columns:
      raise ValueError('`feature_columns` cannot be `None` or empty.')

    # It is unknown at this moment, whether the TPUEstimator is running in CPU
    # or TPU mode. So allow non-TPU embedding columns also.
    supported_classes = tuple(
        list(_SUPPORTED_FEATURE_COLUMNS) + list(_TPU_EMBEDDING_COLUMN_CLASSES) +
        list(_EMBEDDING_COLUMN_CLASSES))

    for column in feature_columns:
      if not isinstance(column, supported_classes):
        raise TypeError(
            'All feature columns must be supported types in {}. Got {}'.format(
                supported_classes, type(column)))

    if not isinstance(optimization_parameters, _SUPPORTED_OPTIMIZERS):
      raise ValueError('optimization_parameters must be an instance of type '
                       '{}. Got {}.'.format(_SUPPORTED_OPTIMIZERS,
                                            type(optimization_parameters)))

    return super(EmbeddingConfigSpec, cls).__new__(
        cls,
        feature_columns=feature_columns,
        optimization_parameters=optimization_parameters,
        clipping_limit=clipping_limit,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
        experimental_gradient_multiplier_fn=experimental_gradient_multiplier_fn)


class EmbeddingConfig(object):
  """This is the internal immutable object for embedding config.

  `_EmbeddingConfig` is responsible to _translate_ user provided
  `EmbeddingConfigSpec` to internal data structures, mostly constructor
  arguments of `TPUEmbedding`.
  """

  def __init__(self, embedding_config_spec, train_batch_size, eval_batch_size,
               num_hosts, num_cores, run_config):
    if not embedding_config_spec:
      raise ValueError('embedding_config_spec cannot be None.')

    self._embedding_config_spec = embedding_config_spec
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._num_hosts = num_hosts
    self._num_cores = num_cores
    self._run_config = run_config

    (self._table_to_config_dict, self._feature_to_table_dict,
     self.feature_to_weight_key_name_dict) = (
         get_configs_from_feature_columns(
             embedding_config_spec.feature_columns))
    self._mode_to_tpu_embedding_dict = {}
    self.dummy_table_variables = None

    self._grad_multiplier_fn = (
        embedding_config_spec.experimental_gradient_multiplier_fn)

  def get_grad_multiplier(self):
    if self._grad_multiplier_fn:
      return ops.convert_to_tensor(
          self._grad_multiplier_fn(training.get_global_step()),
          dtype=dtypes.float32)

  def has_embedding_tables(self):
    return bool(self._table_to_config_dict)

  def _create_tpu_embedding(self, mode):
    """Create tpu_embedding.TPUEmbedding based on mode."""
    if mode == model_fn_lib.ModeKeys.TRAIN:
      batch_size = self._train_batch_size
    else:
      batch_size = self._eval_batch_size

    if mode == model_fn_lib.ModeKeys.TRAIN:
      tpu_embedding_mode = tpu_embedding.TRAINING
      optimization_parameters = (
          self._embedding_config_spec.optimization_parameters)
    elif (mode == model_fn_lib.ModeKeys.EVAL or
          mode == model_fn_lib.ModeKeys.PREDICT):
      tpu_embedding_mode = tpu_embedding.INFERENCE
      optimization_parameters = None
    else:
      raise ValueError('Mode {} is not supported.'.format(mode))

    if self._run_config.cluster:
      master = self._run_config.cluster.master()
      cluster_spec = self._run_config.cluster.cluster_spec()
      cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
    else:
      master = (
          self._run_config.evaluation_master
          if mode == model_fn_lib.ModeKeys.EVAL else self._run_config.master)
      cluster_def = None
    tpu_embedding_ = tpu_embedding.TPUEmbedding(
        self._table_to_config_dict,
        self._feature_to_table_dict,
        batch_size,
        tpu_embedding_mode,
        master,
        optimization_parameters,
        cluster_def,
        pipeline_execution_with_tensor_core=self._embedding_config_spec
        .pipeline_execution_with_tensor_core)
    return tpu_embedding_

  def get_tpu_embedding(self, mode):
    if mode not in self._mode_to_tpu_embedding_dict:
      self._mode_to_tpu_embedding_dict[mode] = (
          self._create_tpu_embedding(mode))
    return self._mode_to_tpu_embedding_dict[mode]


def split_inputs(ctx, features, labels):
  """Splits the dense and sparse tensors inside the features and labels."""
  enqueue_datas = collections.OrderedDict()
  if ctx.embedding_config:
    tpu_embedding_ = ctx.embedding_config.tpu_embedding
    feature_to_weight_key_name_dict = (
        ctx.embedding_config.feature_to_weight_key_name_dict)
    for feature_key in tpu_embedding_.feature_to_table_dict:
      sparse_feature = _get_sparse_feature_from_feature(feature_key, features)
      weight_key_name = feature_to_weight_key_name_dict[feature_key]
      if isinstance(sparse_feature, sparse_tensor.SparseTensor):
        weights = _get_weights_from_features(weight_key_name, features)
        enqueue_data = tpu_embedding.EnqueueData.from_sparse_tensor(
            sparse_feature, weights)
      else:
        if weight_key_name is not None:
          raise ValueError(
              'Found weights {} for weighted_categorical_column, which is not'
              'compatible with sparse feature {} enqueued as dense tensor.'
              .format(weight_key_name, feature_key))
        enqueue_data = tpu_embedding.EnqueueData(sparse_feature)
      enqueue_datas[feature_key] = enqueue_data

  return features, labels, enqueue_datas


def _get_sparse_feature_from_feature(feature_key, features):
  """Pop and return sparse feature."""
  sparse_feature = features.pop(feature_key)
  if not sparse_feature.dtype.is_integer:
    raise ValueError('SparseTensor with string as values are not supported. '
                     'If you are using vocabulary_file_categorical_column or '
                     'vocabulary_list_categorical_column, please call '
                     'your_column.categorical_column._transform_feature({{'
                     'your_column.key: features[your_column.key]}}) in'
                     'your input_fn() to convert string to int. '
                     'feature_key = {}.'.format(feature_key))
  return sparse_feature


def _get_weights_from_features(weight_key_name, features):
  """Pop and return feature for weights, possibly None."""
  weights = None
  if weight_key_name is not None:
    if weight_key_name in features:
      weights = features.pop(weight_key_name)
    else:
      raise ValueError(
          'Cannot find weights {} for weighted_categorical_column.'
          ' Please check if the weights are present in feature dict. Also'
          ' note weight-sharing among weighted_categorical_column is not '
          'supported on TPU.'.format(weight_key_name))
    if not isinstance(weights, sparse_tensor.SparseTensor):
      raise ValueError(
          'weighted_categorical_column with weight key name {} has dense '
          'weights. Dense weights are not supported on TPU. Please use '
          'sparse weights instead.'.format(weight_key_name))
    if weights.dtype is not dtypes.float32:
      weights = math_ops.to_float(weights)
  return weights
