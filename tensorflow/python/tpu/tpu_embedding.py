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
"""TPU embedding APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import math
import re

from typing import Optional

import six

from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export

TRAINING = elc.TPUEmbeddingConfiguration.TRAINING
INFERENCE = elc.TPUEmbeddingConfiguration.INFERENCE


# TODO(shizhiw): a more future-proof way is to have optimization_parameter such
#  as AdagradParameters etc instead of learning_rate.
class TableConfig(
    collections.namedtuple('TableConfig', [
        'vocabulary_size',
        'dimension',
        'initializer',
        'combiner',
        'hot_id_replication',
        'learning_rate',
        'learning_rate_fn',
        'optimization_parameters',
    ])):
  """Embedding table configuration."""

  def __new__(cls,
              vocabulary_size,
              dimension,
              initializer=None,
              combiner='mean',
              hot_id_replication=False,
              learning_rate=None,
              learning_rate_fn=None,
              optimization_parameters=None):
    """Embedding table configuration.

    Args:
      vocabulary_size: Number of vocabulary (/rows) in the table.
      dimension: The embedding dimension.
      initializer: A variable initializer function to be used in embedding
        variable initialization. If not specified, defaults to
        `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and standard
        deviation `1/sqrt(dimension)`.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn', 'sum' and None are
        supported, with 'mean' the default. 'sqrtn' often achieves good
        accuracy, in particular with bag-of-words columns. For more information,
        see `tf.nn.embedding_lookup_sparse`. None is only valid for dense rather
        than sparse tensors.
      hot_id_replication: If true, enables hot id replication, which can make
        embedding lookups faster if there are some hot rows in the table.
      learning_rate: float, static learning rate for this table. If
        learning_rate and learning_rate_fn are both `None`, static learning rate
        as specified in local `optimization_parameters` will be used. In case
        local `optimization_parameters` is `None`, global
        `optimization_parameters` in `TPUEmbedding` constructor will be used.
        `learning_rate_fn` must be `None` if `learning_rate` is not `None.
      learning_rate_fn: string, use dynamic learning rate given by the function.
        This function will be passed the current global step. If learning_rate
        and learning_rate_fn are both `None`, static learning rate as specified
        in `optimization_parameters` is used. `learning_rate` must be `None` if
        `learning_rate_fn` is not `None.
      optimization_parameters: `AdagradParameters`, `AdamParameters`,
        `Stochasticgradientdescentparameters`. Specifies table level optimizer.
        If it's `None` global optimizer in `TPUEmbedding` constructor is used.

    Returns:
      `TableConfig`.

    Raises:
      ValueError: if `vocabulary_size` is not positive integer.
      ValueError: if `dimension` is not positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
      ValueError: if `learning_rate` and `learning_rate_fn` are both not
        `None`.
    """
    if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
      raise ValueError('Invalid vocabulary_size {}.'.format(vocabulary_size))

    if not isinstance(dimension, int) or dimension < 1:
      raise ValueError('Invalid dimension {}.'.format(dimension))

    if (initializer is not None) and (not callable(initializer)):
      raise ValueError('initializer must be callable if specified.')
    if initializer is None:
      initializer = init_ops.truncated_normal_initializer(
          mean=0.0, stddev=1 / math.sqrt(dimension))

    if combiner not in ('mean', 'sum', 'sqrtn', None):
      raise ValueError('Invalid combiner {}'.format(combiner))

    if learning_rate is not None and learning_rate_fn is not None:
      raise ValueError('At most one of learning_rate and learning_rate_fn '
                       'can be None; got {} and {}'.format(
                           learning_rate, learning_rate_fn))

    if optimization_parameters is not None:
      if not isinstance(optimization_parameters, _OptimizationParameters):
        raise ValueError('`optimization_parameters` must inherit from '
                         '`_OptimizationParameters`. '
                         '`type(optimization_parameters)`={}'.format(
                             type(optimization_parameters)))

    return super(TableConfig,
                 cls).__new__(cls, vocabulary_size, dimension, initializer,
                              combiner, hot_id_replication, learning_rate,
                              learning_rate_fn, optimization_parameters)


class FeatureConfig(
    collections.namedtuple('FeatureConfig',
                           ['table_id', 'max_sequence_length', 'weight_key'])):
  """Feature configuration."""

  def __new__(cls, table_id, max_sequence_length=0, weight_key=None):
    """Feature configuration.

    Args:
      table_id: Which table the feature is uses for embedding lookups.
      max_sequence_length: If positive, the feature is a sequence feature with
        the corresponding maximum sequence length. If the sequence is longer
        than this, it will be truncated. If 0, the feature is not a sequence
        feature.
      weight_key: If using weights for the combiner, this key specifies which
        input feature contains the weights.

    Returns:
      `FeatureConfig`.

    Raises:
      ValueError: if `max_sequence_length` non-negative.
    """
    if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
      raise ValueError(
          'Invalid max_sequence_length {}.'.format(max_sequence_length))

    return super(FeatureConfig, cls).__new__(cls, table_id, max_sequence_length,
                                             weight_key)


class EnqueueData(
    collections.namedtuple(
        'EnqueueData',
        ['embedding_indices', 'sample_indices', 'aggregation_weights'])):
  """Data to be enqueued through generate_enqueue_ops()."""

  def __new__(cls,
              embedding_indices,
              sample_indices=None,
              aggregation_weights=None):
    """Data to be enqueued through generate_enqueue_ops().

    Args:
      embedding_indices: A rank 1 Tensor, indices into the embedding tables. It
        corresponds to sp_ids.values in embedding_lookup_sparse(). Both int32
        and int64 are allowed and will be converted to int32 internally.
      sample_indices: A rank 2 Tensor specifying the training example to which
        the corresponding embedding_indices and aggregation_weights values
        belong. It corresponds to sp_ids.indices in embedding_lookup_sparse().
        If it is None, we assume each embedding_indices belongs to a different
        sample. Both int32 and int64 are allowed and will be converted to int32
        internally.
      aggregation_weights: A rank 1 Tensor containing aggregation weights. It
        corresponds to sp_weights.values in embedding_lookup_sparse(). If it is
        None, we assume all weights are 1. Both float32 and float64 are allowed
        and will be converted to float32 internally.

    Returns:
      An EnqueueData tuple.

    """
    return super(EnqueueData, cls).__new__(cls, embedding_indices,
                                           sample_indices, aggregation_weights)

  @staticmethod
  def from_sparse_tensor(sp_tensor, weights=None):
    return EnqueueData(
        sp_tensor.values,
        sp_tensor.indices,
        aggregation_weights=weights.values if weights is not None else None)


class RaggedEnqueueData(
    collections.namedtuple(
        'RaggedEnqueueData',
        ['embedding_indices', 'sample_splits', 'aggregation_weights'])):
  """RaggedTensor Data to be enqueued through generate_enqueue_ops()."""

  def __new__(cls,
              embedding_indices,
              sample_splits=None,
              aggregation_weights=None):
    """Data to be enqueued through generate_enqueue_ops().

    Args:
      embedding_indices: A rank 1 Tensor, indices into the embedding tables. It
        corresponds to ids.values in embedding_lookup(), when ids is a
        RaggedTensor. Both int32 and int64 are allowed and will be converted to
        int32 internally.
      sample_splits: A rank 1 Tensor specifying the break points for splitting
        embedding_indices and aggregation_weights into rows. It corresponds to
        ids.row_splits in embedding_lookup(), when ids is a RaggedTensor. Both
        int32 and int64 are allowed and will be converted to int32 internally.
      aggregation_weights: A rank 1 Tensor containing per training example
        aggregation weights. It corresponds to the values field of a
        RaggedTensor with the same row_splits as ids in embedding_lookup(), when
        ids is a RaggedTensor.

    Returns:
      An RaggedEnqueueData tuple.

    """
    return super(RaggedEnqueueData,
                 cls).__new__(cls, embedding_indices, sample_splits,
                              aggregation_weights)

  @staticmethod
  def from_ragged_tensor(rg_tensor, weights=None):
    return RaggedEnqueueData(
        rg_tensor.values,
        rg_tensor.row_splits,
        aggregation_weights=weights.values if weights is not None else None)


def get_enqueue_datas_list_from_sparse_tensors_list(sp_tensors_list):
  """Convenient function for generate_enqueue_ops().

  Args:
    sp_tensors_list: a list of dictionary mapping from string of feature names
      to SparseTensor. Each dictionary is for one TPU core. Dictionaries for the
      same host should be contiguous on the list.

  Returns:
    enqueue_datas_list: a list of dictionary mapping from string
      of feature names to EnqueueData. Each dictionary is for one
      TPU core. Dictionaries for the same host should be contiguous
      on the list.

  """
  enqueue_datas_list = []
  for sp_tensors in sp_tensors_list:
    enqueue_datas = collections.OrderedDict(
        (k, EnqueueData.from_sparse_tensor(v))
        for k, v in six.iteritems(sp_tensors))
    enqueue_datas_list.append(enqueue_datas)
  return enqueue_datas_list


def get_enqueue_datas_list_from_ragged_tensors_list(rg_tensors_list):
  """Convenient function for generate_enqueue_ops().

  Args:
    rg_tensors_list: a list of dictionary mapping from string of feature names
      to RaggedTensor. Each dictionary is for one TPU core. Dictionaries for the
      same host should be contiguous on the list.

  Returns:
    enqueue_datas_list: a list of dictionary mapping from string
      of feature names to RaggedEnqueueData. Each dictionary is for one
      TPU core. Dictionaries for the same host should be contiguous
      on the list.

  """
  enqueue_datas_list = []
  for rg_tensors in rg_tensors_list:
    enqueue_datas = collections.OrderedDict(
        (k, RaggedEnqueueData.from_ragged_tensor(v))
        for k, v in six.iteritems(rg_tensors))
    enqueue_datas_list.append(enqueue_datas)
  return enqueue_datas_list


AdamSlotVariableNames = collections.namedtuple('AdamSlotVariableNames',
                                               ['m', 'v'])

AdagradSlotVariableName = collections.namedtuple('AdagradSlotVariableName',
                                                 ['accumulator'])

MomentumSlotVariableName = collections.namedtuple('MomentumSlotVariableName',
                                                  ['momenta'])

RMSPropSlotVariableNames = collections.namedtuple('RMSPropSlotVariableNames',
                                                  ['ms', 'mom'])

ProximalAdagradSlotVariableName = collections.namedtuple(
    'ProximalAdagradSlotVariableName', ['accumulator'])

FtrlSlotVariableName = collections.namedtuple('FtrlSlotVariableName',
                                              ['accumulator', 'linear'])

ProximalYogiSlotVariableNames = collections.namedtuple(
    'ProximalYogiSlotVariableNames', ['v', 'm'])

FrequencyEstimatorSlotVariableName = collections.namedtuple(
    'FrequencyEstimatorSlotVariableName', ['last_hit_step'])

AdamSlotVariables = collections.namedtuple('AdamSlotVariables', ['m', 'v'])

MomentumSlotVariable = collections.namedtuple('MomentumSlotVariable',
                                              ['momenta'])

RMSPropSlotVariables = collections.namedtuple('RMSPropSlotVariables',
                                              ['ms', 'mom'])

AdagradSlotVariable = collections.namedtuple('AdagradSlotVariable',
                                             ['accumulator'])

ProximalAdagradSlotVariable = collections.namedtuple(
    'ProximalAdagradSlotVariable', ['accumulator'])

FtrlSlotVariable = collections.namedtuple('FtrlSlotVariable',
                                          ['accumulator', 'linear'])

ProximalYogiSlotVariables = collections.namedtuple('ProximalYogiSlotVariables',
                                                   ['v', 'm'])

FrequencyEstimatorSlotVariables = collections.namedtuple(
    'FrequencyEstimatorSlotVariables', ['last_hit_step'])

VariablesAndOps = collections.namedtuple('VariablesAndOps', [
    'embedding_variables_by_table', 'slot_variables_by_table', 'load_ops',
    'retrieve_ops'
])


class _OptimizationParameters(object):
  """Parameters common to all optimizations."""

  def __init__(
      self,
      learning_rate: float,
      use_gradient_accumulation: bool,
      clip_weight_min: Optional[float],
      clip_weight_max: Optional[float],
      weight_decay_factor: Optional[float],
      multiply_weight_decay_factor_by_learning_rate: Optional[bool],
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    self.learning_rate = learning_rate
    self.use_gradient_accumulation = use_gradient_accumulation
    self.clip_weight_min = clip_weight_min
    self.clip_weight_max = clip_weight_max
    self.weight_decay_factor = weight_decay_factor
    self.multiply_weight_decay_factor_by_learning_rate = (
        multiply_weight_decay_factor_by_learning_rate)
    self.clip_gradient_min = clip_gradient_min
    self.clip_gradient_max = clip_gradient_max

    if not use_gradient_accumulation and (clip_gradient_min is not None or
                                          clip_gradient_max is not None):
      ValueError('When using gradient clipping limits, gradient accumulation '
                 'must be enabled.')


@tf_export(v1=['tpu.experimental.AdagradParameters'])
class AdagradParameters(_OptimizationParameters):
  """Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdagradParameters(0.1),
          ...))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      initial_accumulator: float = 0.1,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: used for updating embedding table.
      initial_accumulator: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(AdagradParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    if initial_accumulator <= 0:
      raise ValueError('Adagrad initial_accumulator must be positive')
    self.initial_accumulator = initial_accumulator


class ProximalAdagradParameters(_OptimizationParameters):
  """Optimization parameters for ProximalAdagrad with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.
  """

  def __init__(
      self,
      learning_rate: float,
      initial_accumulator: float = 0.1,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: used for updating embedding table.
      initial_accumulator: initial accumulator for Adagrad.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details. for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(ProximalAdagradParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    if initial_accumulator <= 0:
      raise ValueError('Adagrad initial_accumulator must be positive')
    if l1_regularization_strength < 0.:
      raise ValueError('l1_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l1_regularization_strength))

    if l2_regularization_strength < 0.:
      raise ValueError('l2_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l2_regularization_strength))

    self.initial_accumulator = initial_accumulator
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength


@tf_export(v1=['tpu.experimental.AdamParameters'])
class AdamParameters(_OptimizationParameters):
  """Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdamParameters(0.1),
          ...))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      beta1: float = 0.9,
      beta2: float = 0.999,
      epsilon: float = 1e-08,
      lazy_adam: bool = True,
      sum_inside_sqrt: bool = True,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for Adam.

    Args:
      learning_rate: a floating point value. The learning rate.
      beta1: A float value. The exponential decay rate for the 1st moment
        estimates.
      beta2: A float value. The exponential decay rate for the 2nd moment
        estimates.
      epsilon: A small constant for numerical stability.
      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster. See
        `optimization_parameters.proto` for details.
      sum_inside_sqrt: This improves training speed. Please see
        `optimization_parameters.proto` for details.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(AdamParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    if beta1 < 0. or beta1 >= 1.:
      raise ValueError('beta1 must be between 0. and 1; got {}.'.format(beta1))
    if beta2 < 0. or beta2 >= 1.:
      raise ValueError('beta2 must be between 0. and 1; got {}.'.format(beta2))
    if epsilon <= 0.:
      raise ValueError('epsilon must be positive; got {}.'.format(epsilon))
    if not use_gradient_accumulation and not lazy_adam:
      raise ValueError(
          'When disabling Lazy Adam, gradient accumulation must be used.')

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.lazy_adam = lazy_adam
    self.sum_inside_sqrt = sum_inside_sqrt


@tf_export(v1=['tpu.experimental.FtrlParameters'])
class FtrlParameters(_OptimizationParameters):
  """Optimization parameters for Ftrl with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.FtrlParameters(0.1),
          ...))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      learning_rate_power: float = -0.5,
      initial_accumulator_value: float = 0.1,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      multiply_linear_by_learning_rate: bool = False,
      beta: float = 0,
      allow_zero_accumulator: bool = False,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for Ftrl.

    Implements FTRL as described in the following [paper](
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

    Args:
      learning_rate: a floating point value. The learning rate.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for a
        fixed learning rate. See section 3.1 in the
        [paper](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details. for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      multiply_linear_by_learning_rate: When true, multiplies the usages of the
        linear slot in the weight update by the learning rate. This is useful
        when ramping up learning rate from 0 (which would normally produce
        NaNs).
      beta: The beta parameter for FTRL.
      allow_zero_accumulator: Changes the implementation of the square root to
        allow for the case of initial_accumulator_value being zero. This will
        cause a slight performance drop.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(FtrlParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    if learning_rate_power > 0.:
      raise ValueError('learning_rate_power must be less than or equal to 0. '
                       'got {}.'.format(learning_rate_power))

    if initial_accumulator_value < 0.:
      raise ValueError('initial_accumulator_value must be greater than or equal'
                       ' to 0. got {}.'.format(initial_accumulator_value))

    if l1_regularization_strength < 0.:
      raise ValueError('l1_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l1_regularization_strength))

    if l2_regularization_strength < 0.:
      raise ValueError('l2_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l2_regularization_strength))

    self.learning_rate_power = learning_rate_power
    self.initial_accumulator_value = initial_accumulator_value
    self.initial_linear_value = 0.0
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength
    self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate
    self.beta = beta
    self.allow_zero_accumulator = allow_zero_accumulator


class ProximalYogiParameters(_OptimizationParameters):
  # pylint: disable=line-too-long
  """Optimization parameters for Proximal Yogi with TPU embeddings.

  Implements the Yogi optimizer as described in
  [Adaptive Methods for Nonconvex
  Optimization](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization).

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.
  """

  # pylint: enable=line-too-long

  def __init__(
      self,
      learning_rate: float = 0.01,
      beta1: float = 0.9,
      beta2: float = 0.999,
      epsilon: float = 1e-3,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      initial_accumulator_value: float = 1e-6,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for Proximal Yogi.

    Args:
      learning_rate: a floating point value. The learning rate.
      beta1: A float value. The exponential decay rate for the 1st moment
        estimates.
      beta2: A float value. The exponential decay rate for the 2nd moment
        estimates.
      epsilon: A small constant for numerical stability.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details. for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(ProximalYogiParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    if beta1 < 0. or beta1 >= 1.:
      raise ValueError('beta1 must be between 0. and 1; got {}.'.format(beta1))
    if beta2 < 0. or beta2 >= 1.:
      raise ValueError('beta2 must be between 0. and 1; got {}.'.format(beta2))
    if epsilon <= 0.:
      raise ValueError('epsilon must be positive; got {}.'.format(epsilon))
    if l1_regularization_strength < 0.:
      raise ValueError('l1_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l1_regularization_strength))
    if l2_regularization_strength < 0.:
      raise ValueError('l2_regularization_strength must be greater than or '
                       'equal to 0. got {}.'.format(l2_regularization_strength))

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength
    self.initial_accumulator_value = initial_accumulator_value


class MomentumParameters(_OptimizationParameters):
  """Optimization parameters for Momentum with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.MomentumParameters(0.1),
          ...))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      momentum: float,
      use_nesterov: bool = False,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for momentum.

    Args:
      learning_rate: a floating point value. The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_nesterov: If `True` use Nesterov Momentum. See (Sutskever et al.,
        2013). This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.
        This implementation is an approximation of the original formula, valid
        for high values of momentum. It will compute the "adjusted gradient" in
        NAG by assuming that the new gradient will be estimated by the current
        average gradient plus the product of momentum and the change in the
        average gradient.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(MomentumParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    self.momentum = momentum
    self.use_nesterov = use_nesterov


class RMSPropParameters(_OptimizationParameters):
  """Optimization parameters for RMSProp with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.MomentumParameters(0.1),
          ...))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      rho: float,
      momentum: float,
      epsilon: float,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for RMS prop.

    Args:
      learning_rate: a floating point value. The learning rate.
      rho: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details. for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
    super(RMSPropParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )
    self.rho = rho
    self.momentum = momentum
    self.epsilon = epsilon


@tf_export(v1=['tpu.experimental.StochasticGradientDescentParameters'])
class StochasticGradientDescentParameters(_OptimizationParameters):
  """Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=(
              tf.tpu.experimental.StochasticGradientDescentParameters(0.1))))
  ```

  """

  def __init__(
      self,
      learning_rate: float,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: Optional[bool] = None,
      clip_gradient_min: Optional[float] = None,
      clip_gradient_max: Optional[float] = None,
  ):
    """Optimization parameters for stochastic gradient descent.

    Args:
      learning_rate: a floating point value. The learning rate.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
    """
    # Gradient accumulation is generally a no-op for SGD, but if gradient
    # clipping is enabled, then we must also enable gradient accumulation.
    # In the other optimizers this up to the user, but we don't give the user
    # the option to turn gradient accumulation on or off for SGD.
    use_gradient_accumulation = False
    if (clip_gradient_min is not None or clip_gradient_max is not None):
      use_gradient_accumulation = True
    super(StochasticGradientDescentParameters, self).__init__(
        learning_rate=learning_rate,
        use_gradient_accumulation=use_gradient_accumulation,
        clip_weight_min=clip_weight_min,
        clip_weight_max=clip_weight_max,
        weight_decay_factor=weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            multiply_weight_decay_factor_by_learning_rate),
        clip_gradient_min=clip_gradient_min,
        clip_gradient_max=clip_gradient_max,
    )


class FrequencyEstimatorParameters(_OptimizationParameters):
  """Optimization parameters for Frequency Estimator TPU embeddings.

  This is a non-standard optimizer, which returns the estimated frequency of
  lookup for the feature passed to it. It should only be used on a table of
  width 1. The gradient fed back to the TPU embedding should always be zero.
  This can be acomplished via using `tf.stop_gradients` on the feature before
  using it.

  You must use the dynamic learning rate mechanism to set the 'learning rate'
  for this table to be the a float32 cast of the global training step counter.

  See `tensorflow/core/protobuf/tpu/optimization_parameters.proto` for more
  details on this optimizer.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=FrequencyEstimatorParameters(0.1),
          ...))
  ```

  """

  def __init__(self, tau: float, max_delta: float, outlier_threshold: float,
               weight_exponent: float):
    """Optimization parameters for frequency estimator.

    Args:
      tau: Learning rate between (0, 1) that is used to update the array.
      max_delta: Maximum value of delta, the difference between the current
        global step and the last global step at which the row was sampled.
      outlier_threshold: Threshold used to determine whether the current update
        is an outlier.
      weight_exponent: The weight exponent used to transform the estimated delta
        into weights.
    """
    super(FrequencyEstimatorParameters, self).__init__(
        learning_rate=1.0,
        use_gradient_accumulation=True,
        clip_weight_min=None,
        clip_weight_max=None,
        weight_decay_factor=None,
        multiply_weight_decay_factor_by_learning_rate=None,
    )
    self.tau = tau
    self.max_delta = max_delta
    self.outlier_threshold = outlier_threshold
    self.weight_exponent = weight_exponent


DeviceConfig = collections.namedtuple('DeviceConfig',
                                      ['num_hosts', 'num_cores', 'job_name'])


class TPUEmbedding(object):
  """API for using TPU for embedding.

    Example:
    ```
    table_config_user = tpu_embedding.TableConfig(
        vocabulary_size=4, dimension=2,
        initializer=initializer, combiner='mean')
    table_to_config_dict = {'video': table_config_video,
                          'user': table_config_user}
    feature_to_config_dict = {'watched': tpu_embedding.FeatureConfig('video'),
                              'favorited': tpu_embedding.FeatureConfig('video'),
                              'friends': tpu_embedding.FeatureConfig('user')}
    batch_size = 4
    num_hosts = 1
    optimization_parameters = tpu_embedding.AdagradParameters(1., 1.)
    mode = tpu_embedding.TRAINING
    embedding = tpu_embedding.TPUEmbedding(
        table_to_config_dict, feature_to_config_dict,
        batch_size, num_hosts, mode, optimization_parameters)

    batch_size_per_core = embedding.batch_size_per_core
    sparse_features_list = []
    for host in hosts:
      with ops.device(host):
        for _ in range(embedding.num_cores_per_host):
          sparse_features = {}
          sparse_features['watched'] = sparse_tensor.SparseTensor(...)
          sparse_features['favorited'] = sparse_tensor.SparseTensor(...)
          sparse_features['friends'] = sparse_tensor.SparseTensor(...)
          sparse_features_list.append(sparse_features)

    enqueue_ops = embedding.generate_enqueue_ops(sparse_features_list)
    embedding_variables_and_ops = embedding.create_variables_and_ops()

    def computation():
      activations = embedding.get_activations()
      loss = compute_loss(activations)

      base_optimizer = gradient_descent.GradientDescentOptimizer(
          learning_rate=1)
      cross_shard_optimizer = tpu_optimizer.CrossShardOptimizer(
          base_optimizer)

      train_op = cross_shard_optimizer.minimize(loss)
      gradients = (
          tpu_embedding_gradient.get_gradients_through_compute_gradients(
              cross_shard_optimizer, loss, activations)
      send_gradients_op = embedding.generate_send_gradients_op(gradients)
      with ops.control_dependencies([train_op, send_gradients_op]):
        loss = array_ops.identity(loss)

    loss = tpu.shard(computation,
                     num_shards=embedding.num_cores)

    with self.test_session() as sess:
      sess.run(tpu.initialize_system(embedding_config=
                                     embedding.config_proto))
      sess.run(variables.global_variables_initializer())
      sess.run(embedding_variables_and_ops.load_ops())
      sess.run(enqueue_ops)
      loss_val = sess.run(loss)
    ```

  Example with weight decay:

  >>> def learning_rate_fn(global_step):
  ...   return tf.compat.v1.train.polynomial_decay(
  ...     learning_rate=5e-5,
  ...     global_step=global_step,
  ...     decay_steps=100000,
  ...     end_learning_rate=0.0)
  >>> wordpiece_table_config = TableConfig(
  ...   vocabulary_size=119547,
  ...   dimension=256,
  ...   learning_rate_fn=learning_rate_fn)
  >>> wordpiece_feature_config = FeatureConfig(
  ...   table_id='bert/embeddings/word_embeddings',
  ...   max_sequence_length=512)
  >>> optimization_parameters = AdamParameters(
  ...   learning_rate=5e-5,
  ...   epsilon=1e-6,
  ...   weight_decay_factor=0.01,
  ...   multiply_weight_decay_factor_by_learning_rate=True)
  >>> tpu_embedding = TPUEmbedding(
  ...  table_to_config_dict={
  ...    'bert/embeddings/word_embeddings': wordpiece_table_config,
  ...  },
  ...  feature_to_config_dict={'input_ids': wordpiece_feature_config},
  ...  batch_size=128,
  ...  mode=TRAINING,
  ...  optimization_parameters=optimization_parameters,
  ...  master='')
  >>> with tf.Graph().as_default():
  ...   init_tpu_op = tf.compat.v1.tpu.initialize_system(
  ...     embedding_config=tpu_embedding.config_proto)
  ...   tf.compat.v1.Session().run(init_tpu_op)
  """

  # TODO(shizhiw): Consider adding a field to FeatureConfig that indicates that
  # the feature should not be used to update embedding table (cr/204852758,
  # cr/204940540). Also, this can support different combiners for different
  # features within the same table.
  # TODO(shizhiw, b/118512626): Remove `batch_size` from `__init__` and move it
  # to `FeatureConfig`?

  # TODO(shizhiw): will it be cleaner to make `table_to_config_dict` and
  # `feature_to_config_dict` lists of `TableSpec` and `FeatureSpec`
  # respectively?

  # TODO(shizhiw): Consider adding `input_fn` as an option to remove boilerplate
  # for-loops around construction of inputs.

  # `optimization_parameter` applies to all tables. If the need arises,
  # we can add `optimization_parameters` to `TableConfig` to override this
  # global setting.
  def __init__(self,
               table_to_config_dict,
               feature_to_config_dict,
               batch_size,
               mode,
               master=None,
               optimization_parameters=None,
               cluster_def=None,
               pipeline_execution_with_tensor_core=False,
               partition_strategy='div',
               profile_data_directory=None,
               device_config=None,
               master_job_name=None):
    """API for using TPU for embedding lookups.

    Args:
      table_to_config_dict: A dictionary mapping from string of table name to
        `TableConfig`. Table refers to an embedding table, e.g. `params`
        argument to `tf.nn.embedding_lookup_sparse()`.
      feature_to_config_dict: A dictionary mapping from string of feature name
        to `FeatureConfig`. Feature refers to ids to lookup in embedding table,
        e.g. `sp_ids` argument to `tf.nn.embedding_lookup_sparse()`.
      batch_size: An `int` representing the global batch size.
      mode: `TRAINING` or `INFERENCE`.
      master: A `string` representing the TensorFlow master to use.
      optimization_parameters: `AdagradParameters`, `AdamParameters`,
        `Stochasticgradientdescentparameters`. Must be set in training unless
        all tables specify their own optimizers. And it must be `None` in
        inference.
      cluster_def: A ClusterDef object describing the TPU cluster.
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding IDs. Please see
        `tpu_embedding_configuration.proto` for details.
      partition_strategy: A string, either 'mod' or 'div', specifying how to map
        the lookup id to the embedding tensor. For more information see
        `tf.nn.embedding_lookup_sparse`.
      profile_data_directory: Directory where embedding lookup statistics are
        stored. These statistics summarize information about the inputs to the
        embedding lookup operation, in particular, the average number of
        embedding IDs per example and how well the embedding IDs are load
        balanced across the system. The lookup statistics are used during TPU
        initialization for embedding table partitioning. Collection of lookup
        statistics is done at runtime by  profiling the embedding inputs: only
        3% of input samples are profiled to minimize host CPU overhead. Once
        a suitable number of samples are profiled, the lookup statistics are
        saved to table-specific files in the profile data directory generally
        at the end of a TPU training loop. The filename corresponding to each
        table is obtained by hashing table specific parameters (e.g., table
        name and number of features) and global configuration parameters (e.g.,
        sharding strategy and task count). The same profile data directory can
        be shared among several models to reuse embedding lookup statistics.
      device_config: A DeviceConfig instance, used when `master` and
        `cluster_def` are both `None`.
      master_job_name: if set, overrides the master job name used to schedule
        embedding ops.

    Raises:
      ValueError: if any input is invalid.
    """
    if partition_strategy not in ('div', 'mod'):
      raise ValueError(
          'Invalid partition_strategy {}'.format(partition_strategy))
    self._partition_strategy = partition_strategy

    self._profile_data_directory = profile_data_directory

    _validate_table_to_config_dict(table_to_config_dict)
    # Avoid nondeterminism from `Dict` iteration order by using `OrderedDict`.
    self._table_to_config_dict = _create_ordered_dict(table_to_config_dict)

    _validate_feature_to_config_dict(table_to_config_dict,
                                     feature_to_config_dict)
    self._feature_to_config_dict = _create_ordered_dict(feature_to_config_dict)
    self._table_to_features_dict, self._table_to_num_features_dict = (
        _create_table_to_features_and_num_features_dicts(
            self._feature_to_config_dict))
    self._combiners = _create_combiners(self._table_to_config_dict,
                                        self._table_to_features_dict)

    self._batch_size = batch_size

    if master is None and cluster_def is None:
      if device_config is None:
        raise ValueError('When master and cluster_def are both None,'
                         'device_config must be set but is not.')
      if device_config.num_cores % device_config.num_hosts:
        raise ValueError('num_hosts ({}) should divide num_cores ({}) '
                         'but does not.'.format(device_config.num_cores,
                                                device_config.num_hosts))
      self._num_hosts = device_config.num_hosts
      self._num_cores = device_config.num_cores
      self._num_cores_per_host = self._num_cores // self._num_hosts
      self._hosts = [
          '{}/replica:0/task:{}/device:CPU:0'.format(device_config.job_name, i)
          for i in range(self._num_hosts)
      ]
    else:
      tpu_system_metadata = (
          tpu_system_metadata_lib._query_tpu_system_metadata(  # pylint: disable=protected-access
              master,
              cluster_def=cluster_def))
      if tpu_system_metadata.num_cores == 0:
        raise ValueError('TPUEmbedding needs TPUs, but master {} does not have '
                         'TPUs.'.format(master))
      self._num_hosts = tpu_system_metadata.num_hosts
      if master_job_name is None:
        try:
          master_job_name = tpu_system_metadata_lib.master_job(
              master, cluster_def)
        except ValueError as e:
          raise ValueError(str(e) + ' Please specify a master_job_name.')
      self._hosts = []
      for device in tpu_system_metadata.devices:
        if 'device:CPU:' in device.name and (master_job_name is None or
                                             master_job_name in device.name):
          self._hosts.append(device.name)
      self._num_cores_per_host = tpu_system_metadata.num_of_cores_per_host
      self._num_cores = tpu_system_metadata.num_cores

    _validate_batch_size(self._batch_size, self._num_cores)
    self._batch_size_per_core = self._batch_size // self._num_cores

    # TODO(shizhiw): remove `mode`?
    if mode == TRAINING:
      _validate_optimization_parameters(optimization_parameters,
                                        self._table_to_config_dict)
      self._optimization_parameters = optimization_parameters
    elif mode == INFERENCE:
      if optimization_parameters is not None:
        raise ValueError('`optimization_parameters` should be `None` '
                         'for inference mode.')
      self._optimization_parameters = (StochasticGradientDescentParameters(1.))
    else:
      raise ValueError('`mode` only supports {} and {}; got {}.'.format(
          TRAINING, INFERENCE, mode))
    self._mode = mode

    # TODO(shizhiw): move `optimization_parameters` into `_optimizer_handler`
    # and create special handler for inference that inherits from
    # StochasticGradientDescentHandler with more user-friendly error message
    # on get_slot().
    self._optimizer_handler_dict = self._get_optimizer_handler_by_table()

    self._pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core)
    self._learning_rate_fn = list(
        set(c.learning_rate_fn
            for c in self._table_to_config_dict.values()
            if c.learning_rate_fn is not None))
    self._learning_rate_fn_to_tag = {
        fn: id for id, fn in enumerate(self._learning_rate_fn)
    }

    self._config_proto = self._create_config_proto()

  @property
  def hosts(self):
    """A list of device names for CPU hosts.

    Returns:
      A list of device names for CPU hosts.
    """
    return copy.copy(self._hosts)

  # TODO(shizhiw): change to num_tensor_cores_per_host to be more explicit and
  # to be consistent with `tpu_embedding_configuration.proto`.
  @property
  def num_cores_per_host(self):
    """Number of TPU cores on a CPU host.

    Returns:
      Number of TPU cores on a CPU host.
    """
    return self._num_cores_per_host

  @property
  def num_cores(self):
    """Total number of TPU cores on all hosts.

    Returns:
      Total number of TPU cores on all hosts.
    """
    return self._num_cores

  @property
  def batch_size_per_core(self):
    """Batch size for each TPU core.

    The sparse tensors in `sparse_features_list` to `generate_enqueue_ops`
       must have batch dimension equal to this.

    Returns:
      Batch size for each TPU core.
    """
    return self._batch_size_per_core

  @property
  def config_proto(self):
    """Create embedding config proto for `tpu.initialize_system()`.

    Returns:
      an `TPUEmbeddingConfiguration` proto describing the desired
         configuration of the hardware embedding lookup tables, which
         is passed to `tpu.initialize_system()`.
    """
    return self._config_proto

  @property
  def table_to_config_dict(self):
    return copy.copy(self._table_to_config_dict)

  @property
  def feature_to_config_dict(self):
    return copy.copy(self._feature_to_config_dict)

  @property
  def table_to_features_dict(self):
    return copy.copy(self._table_to_features_dict)

  @property
  def optimization_parameters(self):
    return self._optimization_parameters

  def _create_config_proto(self):
    """Create `TPUEmbeddingConfiguration`."""
    config_proto = elc.TPUEmbeddingConfiguration()
    for table in self._table_to_config_dict:
      table_descriptor = config_proto.table_descriptor.add()
      table_descriptor.name = table

      table_config = self._table_to_config_dict[table]
      # For small tables, we pad to the number of hosts so that at least one
      # id will be assigned to each host.
      table_descriptor.vocabulary_size = max(table_config.vocabulary_size,
                                             len(self.hosts))
      table_descriptor.dimension = table_config.dimension

      table_descriptor.num_features = self._table_to_num_features_dict[table]

      optimization_parameters = (
          self._optimizer_handler_dict[table].get_optimization_parameters())

      parameters = table_descriptor.optimization_parameters
      if table_config.learning_rate:
        parameters.learning_rate.constant = table_config.learning_rate
      elif table_config.learning_rate_fn:
        parameters.learning_rate.dynamic.tag = (
            self._learning_rate_fn_to_tag[table_config.learning_rate_fn])
      else:
        parameters.learning_rate.constant = (
            optimization_parameters.learning_rate)
      parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.ENABLED
          if optimization_parameters.use_gradient_accumulation else
          optimization_parameters_pb2.GradientAccumulationStatus.DISABLED)

      if optimization_parameters.clip_gradient_min is not None:
        parameters.gradient_clipping_limits.lower.value = (
            optimization_parameters.clip_gradient_min)
      if optimization_parameters.clip_gradient_max is not None:
        parameters.gradient_clipping_limits.upper.value = (
            optimization_parameters.clip_gradient_max)

      if optimization_parameters.clip_weight_min is not None:
        parameters.clipping_limits.lower.value = (
            optimization_parameters.clip_weight_min)
      if optimization_parameters.clip_weight_max is not None:
        parameters.clipping_limits.upper.value = (
            optimization_parameters.clip_weight_max)
      if optimization_parameters.weight_decay_factor:
        parameters.weight_decay_factor = (
            optimization_parameters.weight_decay_factor)
        if (optimization_parameters
            .multiply_weight_decay_factor_by_learning_rate):
          parameters.multiply_weight_decay_factor_by_learning_rate = True
      if table_config.hot_id_replication:
        parameters.hot_id_replication_configuration.status = (
            optimization_parameters_pb2.HotIdReplicationConfiguration.ENABLED)
      optimizer_handler = self._optimizer_handler_dict[table]
      optimizer_handler.set_optimization_parameters(table_descriptor)

    config_proto.mode = self._mode
    config_proto.batch_size_per_tensor_core = self._batch_size_per_core
    config_proto.num_hosts = self._num_hosts
    config_proto.num_tensor_cores = self._num_cores
    config_proto.sharding_strategy = (
        elc.TPUEmbeddingConfiguration.DIV_DEFAULT
        if self._partition_strategy == 'div' else
        elc.TPUEmbeddingConfiguration.MOD)
    config_proto.pipeline_execution_with_tensor_core = (
        self._pipeline_execution_with_tensor_core)
    if self._profile_data_directory:
      config_proto.profile_data_directory = self._profile_data_directory

    return config_proto

  def create_variables_and_ops(self,
                               embedding_variable_name_by_table=None,
                               slot_variable_names_by_table=None):
    """Create embedding and slot variables, with ops to load and retrieve them.

    N.B.: the retrieve embedding variables (including slot variables) ops are
    returned as lambda fn, as the call side might want to impose control
    dependencies between the TPU computation and retrieving actions. For
    example, the following code snippet ensures the TPU computation finishes
    first, and then we pull the variables back from TPU to CPU.

    ```
    updates_ops = []
    with ops.control_dependencies([loss]):
      for op_fn in retrieve_parameters_op_fns:
        update_ops.append(op_fn())
    ```

    Args:
      embedding_variable_name_by_table: A dictionary mapping from string of
        table name to string of embedding variable name. If `None`, defaults
        from `get_default_slot_variable_names()` will be used.
      slot_variable_names_by_table: A dictionary mapping from string of table
        name to `AdamSlotVariableNames`, `AdagradSlotVariableNames` etc. If
        `None`, defaults from `get_default_slot_variable_names()` will be used.

    Returns:
      `tpu_embedding.VariablesAndOps` with:
        A dictionary mapping from string of table name to embedding variables,
        A dictionary mapping from string of table name to AdagradSlotVariable,
         AdamSlotVariables etc with slot variables,
        A function which returns a list of ops to load embedding and slot
         variables from CPU to TPU.
        A function which returns a list of ops to retrieve embedding and slot
         variables from TPU to CPU.
    """
    embedding_variables_by_table = {}
    slot_variables_by_table = {}
    load_op_fns = []
    retrieve_op_fns = []

    for i, table in enumerate(self._table_to_config_dict):
      if embedding_variable_name_by_table:
        embedding_variable_name = embedding_variable_name_by_table[table]
      else:
        embedding_variable_name = table
      if slot_variable_names_by_table:
        slot_variable_names = slot_variable_names_by_table[table]
      else:
        optimizer_handler = self._optimizer_handler_dict[table]
        slot_variable_names = (
            optimizer_handler.get_default_slot_variable_names(table))

      # TODO(b/139144091): Multi-host support for mid-level API in
      #  eager context (TF 2.0)
      # Workaround below allows single-host use case in TF 2.0
      if context.executing_eagerly():
        device = ''
      else:
        device = _create_device_fn(self._hosts)

      with ops.device(device):
        table_variables = _create_partitioned_variables(
            name=embedding_variable_name,
            num_hosts=self._num_hosts,
            vocabulary_size=self._table_to_config_dict[table].vocabulary_size,
            embedding_dimension=self._table_to_config_dict[table].dimension,
            initializer=self._table_to_config_dict[table].initializer,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES])
        embedding_variables_by_table[table] = table_variables

        # Only loads embedding config to load/retrieve nodes for the first table
        # on the first host, other nodes would use config from the first node.
        config = None if i else self.config_proto.SerializeToString()
        slot_variables_for_table, load_ops_fn, retrieve_ops_fn = (
            self._optimizer_handler_dict[table].create_variables_and_ops(
                table, slot_variable_names, self._num_hosts,
                self._table_to_config_dict[table], table_variables, config))
        slot_variables_by_table[table] = slot_variables_for_table
        load_op_fns.append(load_ops_fn)
        retrieve_op_fns.append(retrieve_ops_fn)

    def load_ops():
      """Calls and returns the load ops for each embedding table.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_ops_list = []
      for load_op_fn in load_op_fns:
        load_ops_list.extend(load_op_fn())
      return load_ops_list

    def retrieve_ops():
      """Calls and returns the retrieve ops for each embedding table.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_ops_list = []
      for retrieve_op_fn in retrieve_op_fns:
        retrieve_ops_list.extend(retrieve_op_fn())
      return retrieve_ops_list

    return VariablesAndOps(embedding_variables_by_table,
                           slot_variables_by_table, load_ops, retrieve_ops)

  def generate_enqueue_ops(
      self,
      enqueue_datas_list,
      mode_override=None,
      ragged=False,
  ):
    """Generate enqueue ops.

    Args:
      enqueue_datas_list: a list of dictionary mapping from string of feature
        names to EnqueueData. Each dictionary is for one TPU core. Dictionaries
        for the same host should be contiguous in the list.
      mode_override: A string input that overrides the mode specified in the
        TPUEmbeddingConfiguration. Supported values are {'unspecified',
        'inference', 'training', 'backward_pass_only'}. When set to
        'unspecified', the mode set in TPUEmbeddingConfiguration is used,
        otherwise mode_override is used (optional).
      ragged: If True, creates RaggedTensor enqueue ops rather than
        SparseTensor.

    Returns:
      Ops to enqueue to TPU for embedding.
    """
    self._validate_generate_enqueue_ops_enqueue_datas_list(enqueue_datas_list)
    return [
        self._generate_enqueue_op(  # pylint: disable=g-complex-comprehension
            enqueue_datas,
            device_ordinal=i % self._num_cores_per_host,
            mode_override=mode_override,
            ragged=ragged,
        ) for i, enqueue_datas in enumerate(enqueue_datas_list)
    ]

  def _validate_generate_enqueue_ops_enqueue_datas_list(self,
                                                        enqueue_datas_list):
    """Validate `enqueue_datas_list`."""

    def _check_agreement(data, name, feature, enqueue_data):
      """Helper function to check device agreement."""
      if (data is not None and
          data.device != enqueue_data.embedding_indices.device):
        raise ValueError('Device of {0} does not agree with that of'
                         'embedding_indices for feature {1}.'.format(
                             name, feature))

    feature_set = set(self._feature_to_config_dict.keys())
    contiguous_device = None
    for i, enqueue_datas in enumerate(enqueue_datas_list):
      used_feature_set = set(enqueue_datas.keys())

      # Check features are valid.
      missing_feature_set = feature_set - used_feature_set
      if missing_feature_set:
        raise ValueError('`enqueue_datas_list[{}]` misses a feature that is '
                         'in `feature_to_config_dict`: {}.'.format(
                             i, missing_feature_set))

      extra_feature_set = used_feature_set - feature_set
      if extra_feature_set:
        raise ValueError('`enqueue_datas_list[{}]` has a feature that is not '
                         'in `feature_to_config_dict`: {}.'.format(
                             i, extra_feature_set))

      device = None
      device_feature = None
      for feature, enqueue_data in six.iteritems(enqueue_datas):
        combiner = self._table_to_config_dict[
            self._feature_to_config_dict[feature].table_id].combiner

        if isinstance(enqueue_data, EnqueueData):
          if enqueue_data.sample_indices is None and combiner:
            logging.warn(
                'No sample indices set for features %f table %f but '
                'combiner is set to %s.', feature,
                self._feature_to_config_dict[feature].table_id, combiner)
          _check_agreement(enqueue_data.sample_indices, 'sample_indices',
                           feature, enqueue_data)
          _check_agreement(enqueue_data.aggregation_weights,
                           'aggregation_weights', feature, enqueue_data)

        elif isinstance(enqueue_data, RaggedEnqueueData):
          if enqueue_data.sample_splits is None and combiner:
            logging.warn(
                'No sample splits set for features %f table %f but '
                'combiner is set to %s.', feature,
                self._feature_to_config_dict[feature].table_id, combiner)
          _check_agreement(enqueue_data.sample_splits, 'sample_splits', feature,
                           enqueue_data)
          _check_agreement(enqueue_data.aggregation_weights,
                           'aggregation_weights', feature, enqueue_data)
        else:
          raise ValueError(
              '`enqueue_datas_list[{}]` has a feature that is not mapped to '
              '`EnqueueData` or `RaggedEnqueueData`. `feature`: {}'.format(
                  i, feature))
        # Check all features are on the same device.
        if device is None:
          device = enqueue_data.embedding_indices.device
          device_feature = feature
        else:
          if device != enqueue_data.embedding_indices.device:
            raise ValueError('Devices are different between features in '
                             '`enqueue_datas_list[{}]`; '
                             'devices: {}, {}; features: {}, {}.'.format(
                                 i, device,
                                 enqueue_data.embedding_indices.device, feature,
                                 device_feature))

      if i % self._num_cores_per_host:
        if device != contiguous_device:
          raise ValueError('We expect the `enqueue_datas` which are on the '
                           'same host to be contiguous in '
                           '`enqueue_datas_list`, '
                           '`enqueue_datas_list[{}]` is on device {}, '
                           'but is expected to be on device {}.'.format(
                               i, device, contiguous_device))
      else:
        contiguous_device = device

  def _generate_enqueue_op(self,
                           enqueue_datas,
                           device_ordinal,
                           mode_override=None,
                           ragged=False):
    """Creates op for enqueuing batch to TPU."""
    enqueue_data0 = list(enqueue_datas.values())[0]
    with ops.colocate_with(enqueue_data0.embedding_indices):
      if ragged:
        # note that this is currently identical in behavior
        return tpu_ops.enqueue_tpu_embedding_ragged_tensor_batch(
            device_ordinal=device_ordinal,
            combiners=self._combiners,
            mode_override=mode_override,
            **self._format_for_tpu_embedding_ragged_tensor_batch(enqueue_datas))
      else:
        return tpu_ops.enqueue_tpu_embedding_sparse_tensor_batch(
            device_ordinal=device_ordinal,
            combiners=self._combiners,
            mode_override=mode_override,
            **self._format_for_tpu_embedding_sparse_tensor_batch(enqueue_datas))

  def _format_for_tpu_embedding_ragged_tensor_batch(self, enqueue_datas):
    """Format sparse features for `enqueue_tpu_embedding_ragged_tensor_batch()`.

    Args:
      enqueue_datas: a `Dict` of `RaggedEnqueueData` objects for embedding.

    Returns:
      Dict of arguments for `enqueue_tpu_embedding_ragged_tensor_batch()`.
    """

    kwargs = {
        'sample_splits': [],
        'embedding_indices': [],
        'aggregation_weights': [],
        'table_ids': [],
        'max_sequence_lengths': [],
    }
    int_zeros = array_ops.zeros((0,), dtype=dtypes.int64)
    float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)
    for table_id, table in enumerate(self._table_to_features_dict):
      features = self._table_to_features_dict[table]
      for feature in features:
        enqueue_data = enqueue_datas[feature]

        kwargs['sample_splits'].append(
            enqueue_data.sample_splits
            if enqueue_data.sample_splits is not None else int_zeros)

        kwargs['aggregation_weights'].append(
            enqueue_data.aggregation_weights
            if enqueue_data.aggregation_weights is not None else float_zeros)

        kwargs['embedding_indices'].append(enqueue_data.embedding_indices)

        kwargs['table_ids'].append(table_id)
        kwargs['max_sequence_lengths'].append(
            self._feature_to_config_dict[feature].max_sequence_length)

    return kwargs

  def _format_for_tpu_embedding_sparse_tensor_batch(self, enqueue_datas):
    """Format sparse features for `enqueue_tpu_embedding_sparse_tensor_batch()`.

    Args:
      enqueue_datas: a `Dict` of `EnqueueData` objects for embedding.

    Returns:
      Dict of arguments for `enqueue_tpu_embedding_sparse_tensor_batch()`.
    """
    kwargs = {
        'sample_indices': [],
        'embedding_indices': [],
        'aggregation_weights': [],
        'table_ids': [],
        'max_sequence_lengths': [],
    }
    int_zeros = array_ops.zeros((0,), dtype=dtypes.int64)
    float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)
    for table_id, table in enumerate(self._table_to_features_dict):
      features = self._table_to_features_dict[table]
      for feature in features:
        enqueue_data = enqueue_datas[feature]

        kwargs['sample_indices'].append(
            enqueue_data.sample_indices
            if enqueue_data.sample_indices is not None else int_zeros)

        kwargs['aggregation_weights'].append(
            enqueue_data.aggregation_weights
            if enqueue_data.aggregation_weights is not None else float_zeros)

        kwargs['embedding_indices'].append(enqueue_data.embedding_indices)

        kwargs['table_ids'].append(table_id)
        kwargs['max_sequence_lengths'].append(
            self._feature_to_config_dict[feature].max_sequence_length)

    return kwargs

  def get_activations(self):
    """Get activations for features.

    This should be called within `computation` that is passed to
      `tpu.replicate` and friends.

    Returns:
      A dictionary mapping from `String` of feature name to `Tensor`
        of activation.
    """
    recv_activations = tpu_ops.recv_tpu_embedding_activations(
        num_outputs=len(self._table_to_config_dict),
        config=self._config_proto.SerializeToString())

    activations = collections.OrderedDict()
    for table_id, table in enumerate(self._table_to_features_dict):
      features = self._table_to_features_dict[table]
      num_features = self._table_to_num_features_dict[table]
      feature_index = 0
      table_activations = array_ops.reshape(
          recv_activations[table_id],
          [self.batch_size_per_core, num_features, -1])
      for feature in features:
        seq_length = self._feature_to_config_dict[feature].max_sequence_length
        if not seq_length:
          activations[feature] = table_activations[:, feature_index, :]
          feature_index = feature_index + 1
        else:
          activations[feature] = (
              table_activations[:,
                                feature_index:(feature_index + seq_length), :])
          feature_index = feature_index + seq_length

    return activations

  def generate_send_gradients_op(self, feature_to_gradient_dict, step=None):
    """Send gradient to TPU embedding.

    Args:
      feature_to_gradient_dict: dict mapping feature names to gradient wrt
        activations.
      step: the current global step, used for dynamic learning rate.

    Returns:
      SendTPUEmbeddingGradients Op.

    Raises:
      RuntimeError: If `mode` is not `TRAINING`.
    """
    if self._mode != TRAINING:
      raise RuntimeError('Only in training mode gradients need to '
                         'be sent to TPU embedding; got mode {}.'.format(
                             self._mode))
    if step is None and self._learning_rate_fn:
      raise ValueError('There are dynamic learning rates but step is None.')

    gradients = []
    for table in self._table_to_features_dict:
      features = self._table_to_features_dict[table]
      table_gradients = []
      for feature in features:
        gradient = feature_to_gradient_dict[feature]
        # Expand dims for non-sequence feature to match sequence features.
        if gradient.shape.ndims == 2:
          gradient = array_ops.expand_dims(gradient, 1)
        table_gradients.append(gradient)
      interleaved_table_grads = array_ops.reshape(
          array_ops.concat(table_gradients, axis=1),
          [-1, array_ops.shape(table_gradients[0])[-1]])
      gradients.append(interleaved_table_grads)

    return tpu_ops.send_tpu_embedding_gradients(
        inputs=gradients,
        learning_rates=[
            math_ops.cast(fn(step), dtype=dtypes.float32)
            for fn in self._learning_rate_fn
        ],
        config=self.config_proto.SerializeToString())

  def _get_optimizer_handler_by_table(self):
    optimizer_handlers = {}
    for table, table_config in self.table_to_config_dict.items():
      if table_config.optimization_parameters is not None:
        optimizer = table_config.optimization_parameters
      else:
        optimizer = self._optimization_parameters
      optimizer_handlers[table] = _get_optimization_handler(optimizer)

    return optimizer_handlers


def _validate_table_to_config_dict(table_to_config_dict):
  """Validate `table_to_config_dict`."""
  for k, v in six.iteritems(table_to_config_dict):
    if not isinstance(v, TableConfig):
      raise ValueError('Value of `table_to_config_dict` must be of type '
                       '`TableConfig`, got {} for {}.'.format(type(v), k))


def _validate_feature_to_config_dict(table_to_config_dict,
                                     feature_to_config_dict):
  """Validate `feature_to_config_dict`."""
  used_table_set = set(
      [feature.table_id for feature in feature_to_config_dict.values()])
  table_set = set(table_to_config_dict.keys())

  unused_table_set = table_set - used_table_set
  if unused_table_set:
    raise ValueError(
        '`table_to_config_dict` specifies table that is not '
        'used in `feature_to_config_dict`: {}.'.format(unused_table_set))

  extra_table_set = used_table_set - table_set
  if extra_table_set:
    raise ValueError(
        '`feature_to_config_dict` refers to a table that is not '
        'specified in `table_to_config_dict`: {}.'.format(extra_table_set))


def _validate_batch_size(batch_size, num_cores):
  if batch_size % num_cores:
    raise ValueError('`batch_size` is not a multiple of number of '
                     'cores. `batch_size`={}, `_num_cores`={}.'.format(
                         batch_size, num_cores))


def _validate_optimization_parameters(optimization_parameters,
                                      table_to_config_dict):
  """Validate global optimization_parameters and per table optimizers.

  If global optimizer is `None`, all table optimizers should be non `None`.

  Args:
      optimization_parameters: global optimizer provided in `TPUEmbedding`
        constructor.
      table_to_config_dict: A dictionary mapping from string of table name to
        `TableConfig`.
  """
  tbl_optimizer_missing = False
  for _, table_config in table_to_config_dict.items():
    if table_config.optimization_parameters is None:
      tbl_optimizer_missing = True
      break

  if optimization_parameters:
    if not isinstance(optimization_parameters, _OptimizationParameters):
      raise ValueError('`optimization_parameters` must inherit from '
                       '`_OptimizationParameters`. '
                       '`type(optimization_parameters)`={}'.format(
                           type(optimization_parameters)))
  else:
    # Missing global optimization_parameters.
    if tbl_optimizer_missing:
      raise ValueError('`optimization_parameters` is missing.')


class _OptimizerHandler(object):
  """Interface class for handling optimizer specific logic."""

  def __init__(self, optimization_parameters):
    self._optimization_parameters = optimization_parameters

  def get_optimization_parameters(self):
    return self._optimization_parameters

  def set_optimization_parameters(self, table_descriptor):
    raise NotImplementedError()

  def get_default_slot_variable_names(self, table):
    raise NotImplementedError()

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    raise NotImplementedError()


class _AdagradHandler(_OptimizerHandler):
  """Handles Adagrad specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.adagrad.SetInParent()

  def get_default_slot_variable_names(self, table):
    return AdagradSlotVariableName('{}/{}'.format(table, 'Adagrad'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    accumulator_initializer = init_ops.constant_initializer(
        self._optimization_parameters.initial_accumulator)
    accumulator_variables = _create_partitioned_variables(
        name=slot_variable_names.accumulator,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=accumulator_initializer)
    slot_variables = AdagradSlotVariable(accumulator_variables)

    def load_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      config = config_proto
      load_op_list = []
      for host_id, table_variable, accumulator_variable in zip(
          range(num_hosts), table_variables, accumulator_variables):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_adagrad_parameters(
                  parameters=table_variable,
                  accumulators=accumulator_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      config = config_proto
      retrieve_op_list = []
      for host_id, table_variable, accumulator_variable in (zip(
          range(num_hosts), table_variables, accumulator_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_accumulator = (
              tpu_ops.retrieve_tpu_embedding_adagrad_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(accumulator_variable, retrieved_accumulator))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _ProximalAdagradHandler(_OptimizerHandler):
  """Handles ProximalAdagrad specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.proximal_adagrad.SetInParent()
    table_descriptor.optimization_parameters.proximal_adagrad.l1 = (
        self._optimization_parameters.l1_regularization_strength)
    table_descriptor.optimization_parameters.proximal_adagrad.l2 = (
        self._optimization_parameters.l2_regularization_strength)

  def get_default_slot_variable_names(self, table):
    return ProximalAdagradSlotVariableName('{}/{}'.format(
        table, 'ProximalAdagrad'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    accumulator_initializer = init_ops.constant_initializer(
        self._optimization_parameters.initial_accumulator)
    accumulator_variables = _create_partitioned_variables(
        name=slot_variable_names.accumulator,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=accumulator_initializer)
    slot_variables = ProximalAdagradSlotVariable(accumulator_variables)

    def load_ops_fn():
      """Returns the retrieve ops for Proximal AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      config = config_proto
      load_op_list = []
      for host_id, table_variable, accumulator_variable in zip(
          range(num_hosts), table_variables, accumulator_variables):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_proximal_adagrad_parameters(
                  parameters=table_variable,
                  accumulators=accumulator_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Proximal AdaGrad embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      config = config_proto
      retrieve_op_list = []
      for host_id, table_variable, accumulator_variable in (zip(
          range(num_hosts), table_variables, accumulator_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_accumulator = (
              tpu_ops.retrieve_tpu_embedding_proximal_adagrad_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(accumulator_variable, retrieved_accumulator))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _AdamHandler(_OptimizerHandler):
  """Handles Adam specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.adam.beta1 = (
        self._optimization_parameters.beta1)
    table_descriptor.optimization_parameters.adam.beta2 = (
        self._optimization_parameters.beta2)
    table_descriptor.optimization_parameters.adam.epsilon = (
        self._optimization_parameters.epsilon)
    table_descriptor.optimization_parameters.adam.use_non_lazy_adam = (
        not self._optimization_parameters.lazy_adam)
    table_descriptor.optimization_parameters.adam.use_sum_inside_sqrt = (
        self._optimization_parameters.sum_inside_sqrt)

  def get_default_slot_variable_names(self, table):
    return AdamSlotVariableNames('{}/{}/m'.format(table, 'Adam'),
                                 '{}/{}/v'.format(table, 'Adam'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    m_initializer = init_ops.zeros_initializer()
    m_variables = _create_partitioned_variables(
        name=slot_variable_names.m,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=m_initializer)
    v_initializer = init_ops.zeros_initializer()
    v_variables = _create_partitioned_variables(
        name=slot_variable_names.v,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=v_initializer)
    slot_variables = AdamSlotVariables(m_variables, v_variables)

    def load_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable, m_variable, v_variable in (zip(
          range(num_hosts), table_variables, m_variables, v_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_adam_parameters(
                  parameters=table_variable,
                  momenta=m_variable,
                  velocities=v_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        # Set config to None to enforce that config is only loaded to the first
        # table.
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Adam embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable, m_variable, v_variable in (zip(
          range(num_hosts), table_variables, m_variables, v_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_m, retrieved_v = (
              tpu_ops.retrieve_tpu_embedding_adam_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(m_variable, retrieved_m),
              state_ops.assign(v_variable, retrieved_v))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _FtrlHandler(_OptimizerHandler):
  """Handles Ftrl specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.ftrl.lr_power = (
        self._optimization_parameters.learning_rate_power)
    table_descriptor.optimization_parameters.ftrl.l1 = (
        self._optimization_parameters.l1_regularization_strength)
    table_descriptor.optimization_parameters.ftrl.l2 = (
        self._optimization_parameters.l2_regularization_strength)
    table_descriptor.optimization_parameters.ftrl.multiply_linear_by_lr = (
        self._optimization_parameters.multiply_linear_by_learning_rate)
    table_descriptor.optimization_parameters.ftrl.beta = (
        self._optimization_parameters.beta)
    table_descriptor.optimization_parameters.ftrl.allow_zero_accumulator = (
        self._optimization_parameters.allow_zero_accumulator)

  def get_default_slot_variable_names(self, table):
    # These match the default slot variable names created by
    # tf.train.FtrlOptimizer.
    return FtrlSlotVariableName(
        '{}/{}'.format(table, 'Ftrl'),  # accumulator
        '{}/{}'.format(table, 'Ftrl_1'))  # linear

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    accumulator_initializer = init_ops.constant_initializer(
        self._optimization_parameters.initial_accumulator_value)
    accumulator_variables = _create_partitioned_variables(
        name=slot_variable_names.accumulator,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=accumulator_initializer)
    linear_initializer = init_ops.constant_initializer(
        self._optimization_parameters.initial_linear_value)
    linear_variables = _create_partitioned_variables(
        name=slot_variable_names.linear,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=linear_initializer)
    slot_variables = FtrlSlotVariable(accumulator_variables, linear_variables)

    def load_ops_fn():
      """Returns the retrieve ops for Ftrl embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      config = config_proto
      load_op_list = []
      for host_id, table_variable, accumulator_variable, linear_variable in zip(
          range(num_hosts), table_variables, accumulator_variables,
          linear_variables):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_ftrl_parameters(
                  parameters=table_variable,
                  accumulators=accumulator_variable,
                  linears=linear_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Ftrl embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      config = config_proto
      retrieve_op_list = []
      for host_id, table_variable, accumulator_variable, linear_variable in zip(
          range(num_hosts), table_variables, accumulator_variables,
          linear_variables):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_accumulator, retrieved_linear = (
              tpu_ops.retrieve_tpu_embedding_ftrl_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(accumulator_variable, retrieved_accumulator),
              state_ops.assign(linear_variable, retrieved_linear))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _ProximalYogiHandler(_OptimizerHandler):
  """Handles Proximal Yogi specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.proximal_yogi.SetInParent()
    table_descriptor.optimization_parameters.proximal_yogi.beta1 = (
        self._optimization_parameters.beta1)
    table_descriptor.optimization_parameters.proximal_yogi.beta2 = (
        self._optimization_parameters.beta2)
    table_descriptor.optimization_parameters.proximal_yogi.epsilon = (
        self._optimization_parameters.epsilon)
    table_descriptor.optimization_parameters.proximal_yogi.l1 = (
        self._optimization_parameters.l1_regularization_strength)
    table_descriptor.optimization_parameters.proximal_yogi.l2 = (
        self._optimization_parameters.l2_regularization_strength)

  def get_default_slot_variable_names(self, table):
    return ProximalYogiSlotVariableNames(
        '{}/{}'.format(table, 'ProximalYogi'),  # v
        '{}/{}_1'.format(table, 'ProximalYogi'))  # m

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    v_initializer = init_ops.constant_initializer(
        self._optimization_parameters.initial_accumulator_value)
    v_variables = _create_partitioned_variables(
        name=slot_variable_names.v,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=v_initializer)
    m_initializer = init_ops.zeros_initializer()
    m_variables = _create_partitioned_variables(
        name=slot_variable_names.m,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=m_initializer)
    slot_variables = ProximalYogiSlotVariables(v_variables, m_variables)

    def load_ops_fn():
      """Returns the load ops for Proximal Yogi embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable, v_variable, m_variable in (zip(
          range(num_hosts), table_variables, v_variables, m_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_proximal_yogi_parameters(
                  parameters=table_variable,
                  v=v_variable,
                  m=m_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        # Set config to None to enforce that config is only loaded to the first
        # table.
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Proximal Yogi embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable, v_variable, m_variable in (zip(
          range(num_hosts), table_variables, v_variables, m_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_v, retrieved_m = (
              tpu_ops.retrieve_tpu_embedding_proximal_yogi_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(v_variable, retrieved_v),
              state_ops.assign(m_variable, retrieved_m))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _MomentumHandler(_OptimizerHandler):
  """Handles Momentum specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    (table_descriptor.optimization_parameters.momentum.SetInParent())
    table_descriptor.optimization_parameters.momentum.momentum = (
        self._optimization_parameters.momentum)
    table_descriptor.optimization_parameters.momentum.use_nesterov = (
        self._optimization_parameters.use_nesterov)

  def get_default_slot_variable_names(self, table):
    return MomentumSlotVariableName('{}/{}'.format(table, 'Momentum'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):

    momenta_initializer = init_ops.zeros_initializer()
    momenta_variables = _create_partitioned_variables(
        name=slot_variable_names.momenta,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=momenta_initializer)
    slot_variables = MomentumSlotVariable(momenta_variables)

    def load_ops_fn():
      """Returns the retrieve ops for Momentum embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable, momenta_variable in (zip(
          range(num_hosts), table_variables, momenta_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = tpu_ops.load_tpu_embedding_momentum_parameters(
              parameters=table_variable,
              momenta=momenta_variable,
              table_name=table,
              num_shards=num_hosts,
              shard_id=host_id,
              config=config,
          )
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Momentum embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable, momenta_variable in (zip(
          range(num_hosts), table_variables, momenta_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_momenta = (
              tpu_ops.retrieve_tpu_embedding_momentum_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config,
              ))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(momenta_variable, retrieved_momenta))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _RMSPropHandler(_OptimizerHandler):
  """Handles RMS prop specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    (table_descriptor.optimization_parameters.rms_prop.SetInParent())
    table_descriptor.optimization_parameters.rms_prop.rho = (
        self._optimization_parameters.rho)
    table_descriptor.optimization_parameters.rms_prop.epsilon = (
        self._optimization_parameters.epsilon)
    table_descriptor.optimization_parameters.rms_prop.momentum = (
        self._optimization_parameters.momentum)

  def get_default_slot_variable_names(self, table):
    return RMSPropSlotVariableNames('{}/{}/ms'.format(table, 'RMSProp'),
                                    '{}/{}/mom'.format(table, 'RMSProp'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):

    ms_variables = _create_partitioned_variables(
        name=slot_variable_names.ms,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=init_ops.zeros_initializer(),
    )
    mom_variables = _create_partitioned_variables(
        name=slot_variable_names.mom,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=init_ops.zeros_initializer(),
    )
    slot_variables = RMSPropSlotVariables(ms_variables, mom_variables)

    def load_ops_fn():
      """Returns the retrieve ops for RMS Prop embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable, ms_variable, mom_variable in (zip(
          range(num_hosts), table_variables, ms_variables, mom_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = tpu_ops.load_tpu_embedding_rms_prop_parameters(
              parameters=table_variable,
              ms=ms_variable,
              mom=mom_variable,
              table_name=table,
              num_shards=num_hosts,
              shard_id=host_id,
              config=config,
          )
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for RMS Prop embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable, ms_variable, mom_variable in (zip(
          range(num_hosts), table_variables, ms_variables, mom_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_ms, retrieved_mom = (
              tpu_ops.retrieve_tpu_embedding_rms_prop_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config,
              ))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(ms_variable, retrieved_ms),
              state_ops.assign(mom_variable, retrieved_mom))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _FrequencyEstimatorHandler(_OptimizerHandler):
  """Handles frequency estimator specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.frequency_estimator.SetInParent()
    freq = table_descriptor.optimization_parameters.frequency_estimator
    freq.tau = self._optimization_parameters.tau
    freq.max_delta = self._optimization_parameters.max_delta
    freq.outlier_threshold = self._optimization_parameters.outlier_threshold
    freq.weight_exponent = self._optimization_parameters.weight_exponent

  def get_default_slot_variable_names(self, table):
    return FrequencyEstimatorSlotVariableName(
        '{}/FrequencyEstimator'.format(table))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    if table_config.dimension != 1:
      raise ValueError('FrequencyEstimator tables should only have a dimension '
                       'of 1. Received dimension {}'.format(
                           table_config.dimension))

    last_hit_step_variables = _create_partitioned_variables(
        name=slot_variable_names.last_hit_step,
        num_hosts=num_hosts,
        vocabulary_size=table_config.vocabulary_size,
        embedding_dimension=table_config.dimension,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        initializer=init_ops.zeros_initializer(),
    )
    slot_variables = FrequencyEstimatorSlotVariables(last_hit_step_variables)

    def load_ops_fn():
      """Returns the retrieve ops for Frequency Estimator embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable, last_hit_step_variable in (zip(
          range(num_hosts), table_variables, last_hit_step_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_frequency_estimator_parameters(
                  parameters=table_variable,
                  last_hit_step=last_hit_step_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Frequency Estimator embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable, last_hit_step_variable in (zip(
          range(num_hosts), table_variables, last_hit_step_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_last_hit_step = (
              tpu_ops.retrieve_tpu_embedding_frequency_estimator_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config,
              ))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(last_hit_step_variable, retrieved_last_hit_step))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _StochasticGradientDescentHandler(_OptimizerHandler):
  """Handles stochastic gradient descent specific logic."""

  def set_optimization_parameters(self, table_descriptor):
    (table_descriptor.optimization_parameters.stochastic_gradient_descent
     .SetInParent())

  def get_default_slot_variable_names(self, table):
    return None

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables, config_proto):
    del table_config

    def load_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      config = config_proto
      for host_id, table_variable in enumerate(table_variables):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_stochastic_gradient_descent_parameters(
                  parameters=table_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
        config = None
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for SGD embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      config = config_proto
      for host_id, table_variable in enumerate(table_variables):
        with ops.colocate_with(table_variable):
          retrieved_table = (
              tpu_ops
              .retrieve_tpu_embedding_stochastic_gradient_descent_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id,
                  config=config))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return None, load_ops_fn, retrieve_ops_fn


def _get_optimization_handler(optimization_parameters):
  """Gets the optimization handler given the parameter type."""
  if isinstance(optimization_parameters, AdagradParameters):
    return _AdagradHandler(optimization_parameters)
  elif isinstance(optimization_parameters, ProximalAdagradParameters):
    return _ProximalAdagradHandler(optimization_parameters)
  elif isinstance(optimization_parameters, AdamParameters):
    return _AdamHandler(optimization_parameters)
  elif isinstance(optimization_parameters, FtrlParameters):
    return _FtrlHandler(optimization_parameters)
  elif isinstance(optimization_parameters, ProximalYogiParameters):
    return _ProximalYogiHandler(optimization_parameters)
  elif isinstance(optimization_parameters, StochasticGradientDescentParameters):
    return _StochasticGradientDescentHandler(optimization_parameters)
  elif isinstance(optimization_parameters, MomentumParameters):
    return _MomentumHandler(optimization_parameters)
  elif isinstance(optimization_parameters, RMSPropParameters):
    return _RMSPropHandler(optimization_parameters)
  elif isinstance(optimization_parameters, FrequencyEstimatorParameters):
    return _FrequencyEstimatorHandler(optimization_parameters)
  return NotImplementedError()


def _create_ordered_dict(d):
  """Create an OrderedDict from Dict."""
  return collections.OrderedDict((k, d[k]) for k in sorted(d))


def _create_combiners(table_to_config_dict, table_to_features_dict):
  """Create a per feature list of combiners, ordered by table."""
  combiners = []
  for table in table_to_config_dict:
    combiner = table_to_config_dict[table].combiner or 'sum'
    combiners.extend([combiner] * len(table_to_features_dict[table]))
  return combiners


def _create_table_to_features_and_num_features_dicts(feature_to_config_dict):
  """Create mapping from table to a list of its features."""
  table_to_features_dict_tmp = {}
  table_to_num_features_dict_tmp = {}
  for feature, feature_config in six.iteritems(feature_to_config_dict):
    if feature_config.table_id in table_to_features_dict_tmp:
      table_to_features_dict_tmp[feature_config.table_id].append(feature)
    else:
      table_to_features_dict_tmp[feature_config.table_id] = [feature]
      table_to_num_features_dict_tmp[feature_config.table_id] = 0
    if feature_config.max_sequence_length == 0:
      table_to_num_features_dict_tmp[feature_config.table_id] = (
          table_to_num_features_dict_tmp[feature_config.table_id] + 1)
    else:
      table_to_num_features_dict_tmp[feature_config.table_id] = (
          table_to_num_features_dict_tmp[feature_config.table_id] +
          feature_config.max_sequence_length)

  table_to_features_dict = collections.OrderedDict()
  table_to_num_features_dict = collections.OrderedDict()
  for table in sorted(table_to_features_dict_tmp):
    table_to_features_dict[table] = sorted(table_to_features_dict_tmp[table])
    table_to_num_features_dict[table] = table_to_num_features_dict_tmp[table]
  return table_to_features_dict, table_to_num_features_dict


def _create_device_fn(hosts):
  """Create device_fn() to use with _create_partitioned_variables()."""

  def device_fn(op):
    """Returns the `device` for `op`."""
    part_match = re.match(r'.*/part_(\d+)(/|$)', op.name)
    dummy_match = re.match(r'.*dummy_(\d+).*', op.name)
    if not part_match and not dummy_match:
      raise RuntimeError(
          'Internal Error: Expected {} to contain /part_* or dummy_*'.format(
              op.name))

    if part_match:
      idx = int(part_match.group(1))
    else:
      idx = int(dummy_match.group(1))  # pytype: disable=attribute-error

    device = hosts[idx]
    logging.debug('assigning {} to {}.', op, device)
    return device

  return device_fn


def _create_partitioned_variables(name,
                                  num_hosts,
                                  vocabulary_size,
                                  embedding_dimension,
                                  initializer,
                                  collections=None):  # pylint: disable=redefined-outer-name
  """Creates PartitionedVariables based on `num_hosts` for `table`."""

  num_slices = min(vocabulary_size, num_hosts)

  var_list = list(
      variable_scope.get_variable(
          name,
          shape=(vocabulary_size, embedding_dimension),
          partitioner=partitioned_variables.fixed_size_partitioner(num_slices),
          dtype=dtypes.float32,
          initializer=initializer,
          collections=collections,
          trainable=False))

  if vocabulary_size >= num_hosts:
    return var_list

  # For padded part, define the dummy variable to be loaded into TPU system.
  for idx in range(num_hosts - vocabulary_size):
    var_list.append(
        variable_scope.get_variable(
            'dummy_{}_{}'.format(vocabulary_size + idx, name),
            shape=(1, embedding_dimension),
            dtype=dtypes.float32,
            initializer=initializer,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            trainable=False))

  return var_list
