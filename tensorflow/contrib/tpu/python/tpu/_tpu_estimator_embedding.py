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

from tensorflow.contrib.tpu.python.tpu import feature_column as tpu_fc
from tensorflow.contrib.tpu.python.tpu import tpu_embedding
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib

# pylint: disable=protected-access
_TPU_EMBEDDING_COLUMN_CLASSES = (tpu_fc._TPUEmbeddingColumn,
                                 tpu_fc._TPUSharedEmbeddingColumn)
_EMBEDDING_COLUMN_CLASSES = (core_fc._EmbeddingColumn,
                             core_fc_lib.EmbeddingColumn,
                             core_fc._SharedEmbeddingColumn)
_SUPPORTED_FEATURE_COLUMNS = (core_fc._NumericColumn, core_fc_lib.NumericColumn)

# pylint: enable=protected-access


def get_tpu_embedding_config_from_feature_columns(feature_columns):
  """Create configs for TPUEmbedding from a list of feature columns.

  This function will place one embedding tensor per table and the return is
  intended to be used as input to TPUEmbedding.

  Args:
    feature_columns: a list of supported feature columns.

  Returns:
    A pair of dicts, the first maps tables to their config, the second maps
    features to tables.
  """

  allowed = (tpu_fc._TPUEmbeddingColumn, tpu_fc._TPUSharedEmbeddingColumn)  # pylint: disable=protected-access

  for column in feature_columns:
    if not isinstance(column, allowed):
      raise TypeError(
          'Unsupported feature column {}. Supported types are {}.'.format(
              type(column), allowed))

  table_to_config = {}
  feature_to_table = {}
  for column in feature_columns:
    feature_name = column.get_feature_key_name()
    table_name = 'tbl_{}'.format(column.get_embedding_var_name())
    if feature_name in feature_to_table:
      raise ValueError(
          'Feature column {} is used with multiple embeddings and this is '
          'not supported.'.format(feature_name))
    feature_to_table[feature_name] = table_name
    vocabulary_size, dimension = column.get_embedding_table_size()
    table_to_config[table_name] = tpu_embedding.TableConfig(
        vocabulary_size=vocabulary_size,
        dimension=dimension,
        initializer=column.get_initializer(),
        combiner=column.get_combiner())

  return table_to_config, feature_to_table


def _get_tpu_embedding_optimization_parameters(embedding_config_spec):
  """Get tpu_embedding._OptimizationParameters from EmbeddingConfigSpec."""
  if embedding_config_spec.optimizer_type == 'adagrad':
    return tpu_embedding.AdagradParameters(
        embedding_config_spec.learning_rate,
        embedding_config_spec.adagrad_initial_accumulator,
        embedding_config_spec.use_gradient_accumulation)
  elif embedding_config_spec.optimizer_type == 'sgd':
    return tpu_embedding.StochasticGradientDescentParameters(
        embedding_config_spec.learning_rate,
        embedding_config_spec.use_gradient_accumulattion)
  elif embedding_config_spec.optimizer_type == 'adam':
    return tpu_embedding.AdamParameters(
        embedding_config_spec.learning_rate,
        embedding_config_spec.adam_parameters.beta1,
        embedding_config_spec.adam_parameters.beta2,
        embedding_config_spec.adam_parameters.epsilon,
        use_gradient_accumulation=embedding_config_spec
        .use_gradient_accumulation)
  else:
    raise ValueError('optimizer_type must be adagrad or sgd or adam for now.')


AdamParameters = collections.namedtuple('AdamParameters',
                                        ['beta1', 'beta2', 'epsilon'])


# TODO(shizhiw): Improve the API to support more optimizer parameters in API.
class EmbeddingConfigSpec(
    collections.namedtuple('EmbeddingConfigSpec', [
        'feature_columns', 'learning_rate', 'optimizer_type',
        'adagrad_initial_accumulator', 'clipping_limit',
        'use_gradient_accumulation', 'adam_parameters'
    ])):
  """Class to keep track of embedding config specification."""

  def __new__(cls,
              feature_columns,
              learning_rate,
              optimizer_type='adagrad',
              adagrad_initial_accumulator=None,
              clipping_limit=None,
              use_gradient_accumulation=False,
              adam_parameters=None):
    """Creates an EmbeddingConfigSpec instance.

    Args:
      feature_columns: All `FeatureColumn`s used by model.
      learning_rate: embedding optimizer learning rate.
      optimizer_type: (String) Name of the optimizer for embedding gradients
        updates. Must be either 'adagrad' ( `tf.train.AdagradOptimizer`, default
        value), 'sgd' (`tf.train.GradientDescentOptimizer`), or 'adam'
        (`tf.contrib.opt.LazyAdamOptimizer`) for lazy Adam. This optimizer will
        be applied to all embedding variables specified by `feature_columns`.
      adagrad_initial_accumulator: Initial accumulator for Adagrad. Used when
        optimizer_type is 'adagrad'. Default is `0.1`.
      clipping_limit: (Optional) Clipping limit (absolute value).
      use_gradient_accumulation: (Experimental) Whether to accumulate the
        gradients across TPU embedding mini-batches. Gradient accumulation does
        not affect SGD and therefore this is applicable only for Adagrad.
      adam_parameters: AdamParameters. Used when optimizer_type is 'adam'.
        Default is 0.9 for beta1, 0.999 for beta2 and 1e-8 for epsilon.

    Returns:
      An EmbeddingConfigSpec instance.

    Raises:
      ValueError: If the feature_columns are not specified.
      TypeError: If the feature columns are not of ths correct type (one of
        _SUPPORTED_FEATURE_COLUMNS, _TPU_EMBEDDING_COLUMN_CLASSES OR
        _EMBEDDING_COLUMN_CLASSES).
      ValueError: If use_gradient_accumulation is True for SGD.
      ValueError: If `optimizer_type` is not one of "adagrad" or "sgd" or
        "adam".
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

    if optimizer_type == 'adagrad':
      if adagrad_initial_accumulator is None:
        adagrad_initial_accumulator = 0.1
      if adagrad_initial_accumulator <= 0:
        raise ValueError('Adagrad initial_accumulator must be positive')
    elif optimizer_type == 'sgd':
      if use_gradient_accumulation:
        raise ValueError('Gradient accumulation makes sense for Adagrad only.')
    elif optimizer_type == 'adam':
      if adam_parameters is None:
        adam_parameters = AdamParameters(0.9, 0.999, 1e-8)
      if adam_parameters.beta1 < 0. or adam_parameters.beta1 >= 1.:
        raise ValueError('beta1 must be between 0. and 1; got {}.'.format(
            adam_parameters.beta1))
      if adam_parameters.beta2 < 0. or adam_parameters.beta2 >= 1.:
        raise ValueError('beta2 must be between 0. and 1; got {}.'.format(
            adam_parameters.beta2))
      if adam_parameters.epsilon <= 0.:
        raise ValueError('epsilon must be positive; got {}.'.format(
            adam_parameters.epsilon))
    else:
      raise ValueError('optimizer_type must be adagrad or sgd or adam for now.')

    return super(EmbeddingConfigSpec, cls).__new__(
        cls,
        feature_columns=feature_columns,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        adagrad_initial_accumulator=adagrad_initial_accumulator,
        clipping_limit=clipping_limit,
        use_gradient_accumulation=use_gradient_accumulation,
        adam_parameters=adam_parameters)


class EmbeddingConfig(object):
  """This is the internal immutable object for embedding config.

  `_EmbeddingConfig` is responsible to _translate_ user provided
  `EmbeddingConfigSpec` to internal data structures, mostly constructor
  arguments of `TPUEmbedding`.
  """

  def __init__(self, embedding_config_spec, train_batch_size, eval_batch_size,
               num_hosts, num_cores, master):
    self._embedding_config_spec = embedding_config_spec
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._num_hosts = num_hosts
    self._num_cores = num_cores
    self._master = master

    self._table_to_config_dict, self._feature_to_table_dict = (
        get_tpu_embedding_config_from_feature_columns(
            embedding_config_spec.feature_columns))
    self._optimization_parameters = _get_tpu_embedding_optimization_parameters(
        self._embedding_config_spec)
    self._mode_to_tpu_embedding_dict = {}

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
    elif (mode == model_fn_lib.ModeKeys.EVAL or
          mode == model_fn_lib.ModeKeys.PREDICT):
      tpu_embedding_mode = tpu_embedding.INFERENCE
    else:
      raise ValueError('Mode {} is not supported.'.format(mode))

    tpu_embedding_ = tpu_embedding.TPUEmbedding(
        self._table_to_config_dict,
        self._feature_to_table_dict,
        batch_size,
        tpu_embedding_mode,
        self._master,
        self._optimization_parameters,
    )
    return tpu_embedding_

  def get_tpu_embedding(self, mode):
    if mode not in self._mode_to_tpu_embedding_dict:
      self._mode_to_tpu_embedding_dict[mode] = (
          self._create_tpu_embedding(mode))
    return self._mode_to_tpu_embedding_dict[mode]


def split_inputs(ctx, features, labels):
  """Splits the dense and sparse tensors inside the features and labels."""
  sparse_features = collections.OrderedDict()
  if ctx.embedding_config:
    tpu_embedding_ = ctx.embedding_config.tpu_embedding
    for feature_key in tpu_embedding_.feature_to_table_dict:
      sparse_features[feature_key] = features.pop(feature_key)

  return features, labels, sparse_features
