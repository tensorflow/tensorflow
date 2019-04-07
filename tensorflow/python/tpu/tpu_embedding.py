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
import six

from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops

TRAINING = elc.TPUEmbeddingConfiguration.TRAINING
INFERENCE = elc.TPUEmbeddingConfiguration.INFERENCE


class TableConfig(
    collections.namedtuple(
        'TableConfig',
        ['vocabulary_size', 'dimension', 'initializer', 'combiner'])):
  """Embedding table configuration."""

  def __new__(cls,
              vocabulary_size,
              dimension,
              initializer=None,
              combiner='mean'):
    """Embedding table configuration.

    Args:
      vocabulary_size: Number of vocabulary (/rows) in the table.
      dimension: The embedding dimension.
      initializer: A variable initializer function to be used in embedding
        variable initialization. If not specified, defaults to
        `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
        `1/sqrt(dimension)`.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn', 'sum' and None are
        supported, with 'mean' the default. 'sqrtn' often achieves good
        accuracy, in particular with bag-of-words columns. For more information,
        see `tf.nn.embedding_lookup_sparse`. None is only valid for dense rather
        than sparse tensors.

    Returns:
      `TableConfig`.

    Raises:
      ValueError: if `vocabulary_size` is not positive integer.
      ValueError: if `dimension` is not positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
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

    return super(TableConfig, cls).__new__(cls, vocabulary_size, dimension,
                                           initializer, combiner)


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
      embedding_indices: A rank 1 Tensors, indices into the embedding tables. It
        corresponds to sp_ids.values in embedding_lookup_sparse(). Both int32
        and int64 are allowed and will be converted to int32 internally.
      sample_indices: A rank 2 Tensors specifying the training example to which
        the corresponding embedding_indices and aggregation_weights values
        belong. It corresponds to sp_ids.indices in embedding_lookup_sparse().
        If it is None, we assume each embedding_indices belongs to a different
        sample. Both int32 and int64 are allowed and will be converted to int32
        internally.
      aggregation_weights: A rank 1 Tensors containing per training example
        aggregation weights. It corresponds to sp_weights.values in
        embedding_lookup_sparse(). If it is None, we assume all weights are 1.
        Both float32 and float64 are allowed and will be converted to float32
        internally.

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


AdamSlotVariableNames = collections.namedtuple(
    'AdamSlotVariableNames', ['m', 'v'])

AdagradSlotVariableName = collections.namedtuple(
    'AdagradSlotVariableName', ['accumulator'])

AdamSlotVariables = collections.namedtuple(
    'AdamSlotVariables', ['m', 'v'])

AdagradSlotVariable = collections.namedtuple(
    'AdagradSlotVariable', ['accumulator'])

VariablesAndOps = collections.namedtuple(
    'VariablesAndOps',
    ['embedding_variables_by_table', 'slot_variables_by_table',
     'load_ops', 'retrieve_ops']
)


class _OptimizationParameters(object):
  """Parameters common to all optimizations."""

  def __init__(self, learning_rate, use_gradient_accumulation):
    self.learning_rate = learning_rate
    self.use_gradient_accumulation = use_gradient_accumulation


class AdagradParameters(_OptimizationParameters):
  """Optimization parameters for Adagrad."""

  def __init__(self, learning_rate, initial_accumulator=0.1,
               use_gradient_accumulation=True):
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: used for updating embedding table.
      initial_accumulator: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
        for details.
    """
    super(AdagradParameters, self).__init__(learning_rate,
                                            use_gradient_accumulation)
    if initial_accumulator <= 0:
      raise ValueError('Adagrad initial_accumulator must be positive')
    self.initial_accumulator = initial_accumulator


class AdamParameters(_OptimizationParameters):
  """Optimization parameters for Adam."""

  def __init__(self, learning_rate,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-08,
               lazy_adam=True,
               sum_inside_sqrt=True,
               use_gradient_accumulation=True):
    """Optimization parameters for Adam.

    Args:
      learning_rate: a floating point value. The learning rate.
      beta1: A float value.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      lazy_adam: Use lazy Adam instead of Adam. Lazy Adam trains faster.
        Please see `optimization_parameters.proto` for details.
      sum_inside_sqrt: This improves training speed. Please see
        `optimization_parameters.proto` for details.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
        for details.
    """
    super(AdamParameters, self).__init__(learning_rate,
                                         use_gradient_accumulation)
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


class StochasticGradientDescentParameters(_OptimizationParameters):
  """Optimization parameters for stochastic gradient descent.

  Args:
    learning_rate: a floating point value. The learning rate.
  """

  def __init__(self, learning_rate):
    super(StochasticGradientDescentParameters, self).__init__(
        learning_rate, False)


class TPUEmbedding(object):
  """API for using TPU for embedding.

    Example:
    ```
    table_config_user = tpu_embedding.TableConfig(
        vocabulary_size=4, dimension=2,
        initializer=initializer, combiner='mean')
    table_to_config_dict = {'video': table_config_video,
                          'user': table_config_user}
    feature_to_table_dict = {'watched': 'video',
                             'favorited': 'video',
                             'friends': 'user'}
    batch_size = 4
    num_hosts = 1
    optimization_parameters = tpu_embedding.AdagradParameters(1., 1.)
    mode = tpu_embedding.TRAINING
    embedding = tpu_embedding.TPUEmbedding(
        table_to_config_dict, feature_to_table_dict,
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
  """

  # TODO(shizhiw): Instead of `feature_to_table_dict` which maps to table
  # name, consider `feature_to_config_dict` which maps to `FeatureConfig`.
  # `FeatureConfig` could have fields other than table name. For example, it
  # could have a field to indicate that the feature should not be used to
  # update embedding table (cr/204852758, cr/204940540). Also, this can support
  # different combiners for different features within the same table.
  # TODO(shizhiw, b/118512626): Remove `batch_size` from `__init__` and move it
  # to `FeatureConfig`?

  # TODO(shizhiw): will it be cleaner to make `table_to_config_dict` and
  # `feature_to_table_dict` lists of `TableSpec` and `FeatureSpec` respectively?

  # TODO(shizhiw): Consider adding `input_fn` as an option to remove boilerplate
  # for-loops around construction of inputs.

  # `optimization_parameter` applies to all tables. If the need arises,
  # we can add `optimization_parameters` to `TableConfig` to override this
  # global setting.
  def __init__(self,
               table_to_config_dict,
               feature_to_table_dict,
               batch_size,
               mode,
               master,
               optimization_parameters=None,
               cluster_def=None,
               pipeline_execution_with_tensor_core=False):
    """API for using TPU for embedding lookups.

    Args:
      table_to_config_dict: A dictionary mapping from string of table name to
        `TableConfig`. Table refers to an embedding table, e.g. `params`
        argument to `tf.nn.embedding_lookup_sparse()`.
      feature_to_table_dict: A dictionary mapping from string of feature name
        to string of table name. Feature refers to ids to lookup in embedding
        table, e.g. `sp_ids` argument to `tf.nn.embedding_lookup_sparse()`.
      batch_size: An `int` representing the global batch size.
      mode: `TRAINING` or `INFERENCE`.
      master: A `string` representing the TensorFlow master to use.
      optimization_parameters: `AdagradParameters`, `AdamParameters`,
        `Stochasticgradientdescentparameters`. Must be set in training and must
        be `None` in inference.
      cluster_def: A ClusterDef object describing the TPU cluster.
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding IDs. Please see
        `tpu_embedding_configuration.proto` for details.

    Raises:
      ValueError: if any input is invalid.
    """
    _validate_table_to_config_dict(table_to_config_dict)
    # Avoid nondeterminism from `Dict` iteration order by using `OrderedDict`.
    self._table_to_config_dict = _create_ordered_dict(table_to_config_dict)

    _validate_feature_to_table_dict(table_to_config_dict, feature_to_table_dict)
    self._feature_to_table_dict = _create_ordered_dict(feature_to_table_dict)
    self._table_to_features_dict = _create_table_to_features_dict(
        self._feature_to_table_dict)
    self._combiners = _create_combiners(self._table_to_config_dict,
                                        self._table_to_features_dict)

    self._batch_size = batch_size

    self._master = master
    self._cluster_def = cluster_def
    self._tpu_system_metadata = (
        tpu_system_metadata_lib._query_tpu_system_metadata(  # pylint: disable=protected-access
            self._master, cluster_def=self._cluster_def))
    if self._tpu_system_metadata.num_cores == 0:
      raise ValueError('TPUEmbedding needs TPUs, but master {} does not have '
                       'TPUs.'.format(self._master))
    self._num_hosts = self._tpu_system_metadata.num_hosts
    master_job_name = tpu_system_metadata_lib.master_job(self._master,
                                                         self._cluster_def)
    self._hosts = sorted([
        device.name for device in self._tpu_system_metadata.devices
        if 'device:CPU:' in device.name and (master_job_name is None or
                                             master_job_name in device.name)])
    self._num_cores_per_host = self._tpu_system_metadata.num_of_cores_per_host
    self._num_cores = self._tpu_system_metadata.num_cores

    _validate_batch_size(self._batch_size, self._num_cores)
    self._batch_size_per_core = self._batch_size // self._num_cores

    # TODO(shizhiw): remove `mode`?
    if mode == TRAINING:
      _validate_optimization_parameters(optimization_parameters)
      self._optimization_parameters = optimization_parameters
    elif mode == INFERENCE:
      if optimization_parameters is not None:
        raise ValueError('`optimization_parameters` should be `None` '
                         'for inference mode.')
      self._optimization_parameters = (
          StochasticGradientDescentParameters(1.))
    else:
      raise ValueError('`mode` only supports {} and {}; got {}.'
                       .format(TRAINING, INFERENCE, mode))
    self._mode = mode

    # TODO(shizhiw): move `optimization_parameters` into `_optimizer_handler`
    # and create special handler for inference that inherits from
    # StochasticGradientDescentHandler with more user-friendly error message
    # on get_slot().
    self._optimizer_handler = _get_optimization_handler(
        self._optimization_parameters)
    self._pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core)

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
  def feature_to_table_dict(self):
    return copy.copy(self._feature_to_table_dict)

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
      table_descriptor.vocabulary_size = table_config.vocabulary_size
      table_descriptor.dimension = table_config.dimension

      features_for_table = self._table_to_features_dict[table]
      table_descriptor.num_features = len(features_for_table)

      table_descriptor.optimization_parameters.learning_rate.constant = (
          self._optimization_parameters.learning_rate)
      table_descriptor.optimization_parameters.gradient_accumulation_status = (
          optimization_parameters_pb2.GradientAccumulationStatus.ENABLED
          if self._optimization_parameters.use_gradient_accumulation else
          optimization_parameters_pb2.GradientAccumulationStatus.DISABLED)
      self._optimizer_handler.set_optimization_parameters(table_descriptor)

    config_proto.mode = self._mode
    config_proto.batch_size_per_tensor_core = self._batch_size_per_core
    config_proto.num_hosts = self._num_hosts
    config_proto.num_tensor_cores = self._num_cores
    config_proto.sharding_strategy = elc.TPUEmbeddingConfiguration.DIV_DEFAULT
    config_proto.pipeline_execution_with_tensor_core = (
        self._pipeline_execution_with_tensor_core)

    return config_proto

  def create_variables_and_ops(self, embedding_variable_name_by_table=None,
                               slot_variable_names_by_table=None):
    """Create embedding and slot variables, with ops to load and retrieve them.

    Args:
      embedding_variable_name_by_table: A dictionary mapping from string of
        table name to string of embedding variable name. If `None`,
        defaults from `get_default_slot_variable_names()` will be used.
      slot_variable_names_by_table: A dictionary mapping from string of table
        name to `AdamSlotVariableNames`, `AdagradSlotVariableNames` etc. If
        `None`, defaults from `get_default_slot_variable_names()` will be used.

    Returns:
      `tpu_embedding.VariablesAndOps` with:
        A dictionary mapping from string of table name to embedding variables,
        A dictionary mapping from string of table name to AdagradSlotVariable,
         AdamSlotVariables etc with slot variables,
        A function which returns a list of ops to load embedding and slot
         variables from TPU to CPU.
        A function which returns a list of ops to retrieve embedding and slot
         variables from TPU to CPU.
    """
    embedding_variables_by_table = {}
    slot_variables_by_table = {}
    load_op_fns = []
    retrieve_op_fns = []
    for table in self._table_to_config_dict:
      if embedding_variable_name_by_table:
        embedding_variable_name = embedding_variable_name_by_table[table]
      else:
        embedding_variable_name = table
      if slot_variable_names_by_table:
        slot_variable_names = slot_variable_names_by_table[table]
      else:
        slot_variable_names = (
            self._optimizer_handler.get_default_slot_variable_names(table))

      device_fn = _create_device_fn(self._hosts)
      with ops.device(device_fn):
        table_variables = _create_partitioned_variables(
            name=embedding_variable_name,
            num_hosts=self._num_hosts,
            vocabulary_size=self._table_to_config_dict[table].vocabulary_size,
            embedding_dimension=self._table_to_config_dict[table].dimension,
            initializer=self._table_to_config_dict[table].initializer,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES])
        embedding_variables_by_table[table] = table_variables

        slot_variables_for_table, load_ops_fn, retrieve_ops_fn = (
            self._optimizer_handler.create_variables_and_ops(
                table, slot_variable_names, self._num_hosts,
                self._table_to_config_dict[table], table_variables)
        )
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
                           slot_variables_by_table,
                           load_ops, retrieve_ops)

  def generate_enqueue_ops(self, enqueue_datas_list):
    """Generate enqueue ops.

    Args:
      enqueue_datas_list: a list of dictionary mapping from string
        of feature names to EnqueueData. Each dictionary is for one
        TPU core. Dictionaries for the same host should be contiguous
        on the list.

    Returns:
      Ops to enqueue to TPU for embedding.
    """
    self._validate_generate_enqueue_ops_enqueue_datas_list(enqueue_datas_list)
    return [
        self._generate_enqueue_op(
            enqueue_datas, device_ordinal=i % self._num_cores_per_host)
        for i, enqueue_datas in enumerate(enqueue_datas_list)
    ]

  def _validate_generate_enqueue_ops_enqueue_datas_list(self,
                                                        enqueue_datas_list):
    """Validate `enqueue_datas_list`."""
    feature_set = set(self._feature_to_table_dict.keys())
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
            self._feature_to_table_dict[feature]].combiner
        if not isinstance(enqueue_data, EnqueueData):
          raise ValueError('`enqueue_datas_list[{}]` has a feature that is '
                           'not mapped to `EnqueueData`. `feature`: {}'.format(
                               i, feature))

        if enqueue_data.sample_indices is None and combiner:
          raise ValueError('`enqueue_datas_list[{}]` has a feature that has '
                           'neither `EnqueueData` or `combiner`.'
                           '`feature`: {}, combiner: {}.'.format(
                               i, feature, combiner))

        if (enqueue_data.sample_indices is not None and
            enqueue_data.sample_indices.op.device !=
            enqueue_data.embedding_indices.op.device):
          raise ValueError(
              'Device of sample_indices does not agree with '
              'that of emebdding_indices for feature {}.'.format(feature))
        if (enqueue_data.aggregation_weights is not None and
            enqueue_data.aggregation_weights.op.device !=
            enqueue_data.embedding_indices.op.device):
          raise ValueError(
              'Device of aggregation_weights does not agree with '
              'that of emebdding_indices for feature {}.'.format(feature))
        # Check all features are on the same device.
        if device is None:
          device = enqueue_data.embedding_indices.op.device
          device_feature = feature
        else:
          if device != enqueue_data.embedding_indices.op.device:
            raise ValueError('Devices are different between features in '
                             '`enqueue_datas_list[{}]`; '
                             'devices: {}, {}; features: {}, {}.'.format(
                                 i, device,
                                 enqueue_data.embedding_indices.op.device,
                                 feature, device_feature))

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

  def _generate_enqueue_op(self, enqueue_datas, device_ordinal):
    enqueue_data0 = list(enqueue_datas.values())[0]
    with ops.colocate_with(enqueue_data0.embedding_indices):
      (sample_indices_list, embedding_indices_list, aggregation_weights_list,
       table_ids) = (
           self._format_for_tpu_embedding_sparse_tensor_batch(enqueue_datas))
      return tpu_ops.enqueue_tpu_embedding_sparse_tensor_batch(
          sample_indices_list,
          embedding_indices_list,
          aggregation_weights_list,
          table_ids,
          device_ordinal=device_ordinal,
          combiners=self._combiners)

  def _format_for_tpu_embedding_sparse_tensor_batch(self, enqueue_datas):
    """Format sparse features for `enqueue_tpu_embedding_sparse_tensor_batch()`.

    Args:
      enqueue_datas: a `Dict` of tensors for embedding. Can be sparse or
      dense.

    Returns:
      Arguments for `enqueue_tpu_embedding_sparse_tensor_batch()`.
    """

    (sample_indices_list, embedding_indices_list, aggregation_weights_list,
     table_ids) = [], [], [], []
    for table_id, table in enumerate(self._table_to_features_dict):
      features = self._table_to_features_dict[table]
      for feature in features:
        enqueue_data = enqueue_datas[feature]

        sample_indices = (
            enqueue_data.sample_indices
            if enqueue_data.sample_indices is not None else array_ops.zeros(
                (0,), dtype=dtypes.int32))
        sample_indices_list.append(sample_indices)

        aggregation_weights = (
            enqueue_data.aggregation_weights if
            enqueue_data.aggregation_weights is not None else array_ops.zeros(
                (0,), dtype=dtypes.float32))
        aggregation_weights_list.append(aggregation_weights)

        embedding_indices_list.append(enqueue_data.embedding_indices)

        table_ids.append(table_id)

    return (sample_indices_list, embedding_indices_list,
            aggregation_weights_list, table_ids)

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
      for lookup_id, feature in enumerate(features):
        stride = len(self._table_to_features_dict[table])
        activations[feature] = recv_activations[table_id][lookup_id::stride, :]
    return activations

  def generate_send_gradients_op(self, feature_to_gradient_dict):
    """Send gradient to TPU embedding.

    Args:
      feature_to_gradient_dict: dict mapping feature names to gradient wrt
        activations.

    Returns:
      SendTPUEmbeddingGradients Op.

    Raises:
      RuntimeError: If `mode` is not `TRAINING`.
    """
    if self._mode != TRAINING:
      raise RuntimeError('Only in training mode gradients need to '
                         'be sent to TPU embedding; got mode {}.'
                         .format(self._mode))
    gradients = []
    for table in self._table_to_features_dict:
      features = self._table_to_features_dict[table]
      table_gradients = [
          feature_to_gradient_dict[feature] for feature in features
      ]
      interleaved_table_grads = array_ops.reshape(
          array_ops.stack(table_gradients, axis=1),
          [-1, table_gradients[0].shape[1]])
      gradients.append(interleaved_table_grads)
    return tpu_ops.send_tpu_embedding_gradients(
        inputs=gradients, config=self.config_proto.SerializeToString())


def _validate_table_to_config_dict(table_to_config_dict):
  """Validate `table_to_config_dict`."""
  for k, v in six.iteritems(table_to_config_dict):
    if not isinstance(v, TableConfig):
      raise ValueError('Value of `table_to_config_dict` must be of type '
                       '`TableConfig`, got {} for {}.'.format(type(v), k))


def _validate_feature_to_table_dict(table_to_config_dict,
                                    feature_to_table_dict):
  """Validate `feature_to_table_dict`."""
  used_table_set = set(feature_to_table_dict.values())
  table_set = set(table_to_config_dict.keys())

  unused_table_set = table_set - used_table_set
  if unused_table_set:
    raise ValueError('`table_to_config_dict` specifies table that is not '
                     'used in `feature_to_table_dict`: {}.'
                     .format(unused_table_set))

  extra_table_set = used_table_set - table_set
  if extra_table_set:
    raise ValueError('`feature_to_table_dict` refers to a table that is not '
                     'specified in `table_to_config_dict`: {}.'
                     .format(extra_table_set))


def _validate_batch_size(batch_size, num_cores):
  if batch_size % num_cores:
    raise ValueError('`batch_size` is not a multiple of number of '
                     'cores. `batch_size`={}, `_num_cores`={}.'.format(
                         batch_size, num_cores))


def _validate_optimization_parameters(optimization_parameters):
  if not isinstance(optimization_parameters, _OptimizationParameters):
    raise ValueError('`optimization_parameters` must inherit from '
                     '`_OptimizationPramaters`. '
                     '`type(optimization_parameters)`={}'.format(
                         type(optimization_parameters)))


class _OptimizerHandler(object):
  """Interface class for handling optimizer specific logic."""

  def __init__(self, optimization_parameters):
    self._optimization_parameters = optimization_parameters

  def set_optimization_parameters(self, table_descriptor):
    raise NotImplementedError()

  def get_default_slot_variable_names(self, table):
    raise NotImplementedError()

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables):
    raise NotImplementedError()


class _AdagradHandler(_OptimizerHandler):
  """Handles Adagrad specific logic."""

  def __init__(self, optimization_parameters):
    super(_AdagradHandler, self).__init__(optimization_parameters)
    self._table_to_accumulator_variables_dict = {}

  def set_optimization_parameters(self, table_descriptor):
    table_descriptor.optimization_parameters.adagrad.SetInParent()

  def get_default_slot_variable_names(self, table):
    return AdagradSlotVariableName('{}/{}'.format(table, 'Adagrad'))

  def create_variables_and_ops(self, table, slot_variable_names, num_hosts,
                               table_config, table_variables):
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
      load_op_list = []
      for host_id, table_variable, accumulator_variable in (zip(
          range(num_hosts), table_variables, accumulator_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_adagrad_parameters(
                  parameters=table_variable,
                  accumulators=accumulator_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
      retrieve_op_list = []
      for host_id, table_variable, accumulator_variable in (zip(
          range(num_hosts), table_variables, accumulator_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_accumulator = (
              tpu_ops.retrieve_tpu_embedding_adagrad_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(accumulator_variable, retrieved_accumulator))
        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return slot_variables, load_ops_fn, retrieve_ops_fn


class _AdamHandler(_OptimizerHandler):
  """Handles Adam specific logic."""

  def __init__(self, optimization_parameters):
    super(_AdamHandler, self).__init__(optimization_parameters)
    self._table_to_m_variables_dict = {}
    self._table_to_v_variables_dict = {}

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
                               table_config, table_variables):
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
      for host_id, table_variable, m_variable, v_variable in (zip(
          range(num_hosts), table_variables,
          m_variables, v_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops.load_tpu_embedding_adam_parameters(
                  parameters=table_variable,
                  momenta=m_variable,
                  velocities=v_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))
        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for Adam embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """

      retrieve_op_list = []
      for host_id, table_variable, m_variable, v_variable in (zip(
          range(num_hosts), table_variables,
          m_variables, v_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table, retrieved_m, retrieved_v = (
              tpu_ops.retrieve_tpu_embedding_adam_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table),
              state_ops.assign(m_variable, retrieved_m),
              state_ops.assign(v_variable, retrieved_v))

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
                               table_config, table_variables):
    del table_config

    def load_ops_fn():
      """Returns the retrieve ops for AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
      load_op_list = []
      for host_id, table_variable in (zip(
          range(num_hosts), table_variables)):
        with ops.colocate_with(table_variable):
          load_parameters_op = (
              tpu_ops
              .load_tpu_embedding_stochastic_gradient_descent_parameters(
                  parameters=table_variable,
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))

        load_op_list.append(load_parameters_op)
      return load_op_list

    def retrieve_ops_fn():
      """Returns the retrieve ops for SGD embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """

      retrieve_op_list = []
      for host_id, table_variable in (zip(
          range(num_hosts), table_variables)):
        with ops.colocate_with(table_variable):
          retrieved_table = (
              tpu_ops
              .retrieve_tpu_embedding_stochastic_gradient_descent_parameters(
                  table_name=table,
                  num_shards=num_hosts,
                  shard_id=host_id))
          retrieve_parameters_op = control_flow_ops.group(
              state_ops.assign(table_variable, retrieved_table))

        retrieve_op_list.append(retrieve_parameters_op)
      return retrieve_op_list

    return None, load_ops_fn, retrieve_ops_fn


def _get_optimization_handler(optimization_parameters):
  if isinstance(optimization_parameters, AdagradParameters):
    return _AdagradHandler(optimization_parameters)
  elif isinstance(optimization_parameters, AdamParameters):
    return _AdamHandler(optimization_parameters)
  elif isinstance(optimization_parameters, StochasticGradientDescentParameters):
    return _StochasticGradientDescentHandler(optimization_parameters)
  else:
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


def _create_table_to_features_dict(feature_to_table_dict):
  """Create mapping from table to a list of its features."""
  table_to_features_dict_tmp = {}
  for feature, table in six.iteritems(feature_to_table_dict):
    if table in table_to_features_dict_tmp:
      table_to_features_dict_tmp[table].append(feature)
    else:
      table_to_features_dict_tmp[table] = [feature]

  table_to_features_dict = collections.OrderedDict()
  for table in sorted(table_to_features_dict_tmp):
    table_to_features_dict[table] = sorted(table_to_features_dict_tmp[table])
  return table_to_features_dict


def _create_device_fn(hosts):
  """Create device_fn() to use with _create_partitioned_variables()."""

  def device_fn(op):
    """Returns the `device` for `op`."""
    part_match = re.match(r'.*/part_(\d+)(/|$)', op.name)

    if part_match:
      idx = int(part_match.group(1))
    else:
      raise RuntimeError('Internal Error: '
                         'Expected %s to contain /part_*.' % op.name)

    device = hosts[idx]
    return device

  return device_fn


def _create_partitioned_variables(name,
                                  num_hosts,
                                  vocabulary_size,
                                  embedding_dimension,
                                  initializer,
                                  collections=None):  # pylint: disable=redefined-outer-name
  """Creates ParitionedVariables based on `num_hosts` for `table`."""
  # TODO(shizhiw): automatically place embedding lookup elsewhere?
  if vocabulary_size < num_hosts:
    raise ValueError('`vocabulary_size`({}) is smaller than `num_hosts`({}). '
                     'As TPU embedding is not optimized for small tables, '
                     'please consider other ways for this embedding lookup.')

  return list(variable_scope.get_variable(
      name,
      shape=(vocabulary_size, embedding_dimension),
      partitioner=partitioned_variables.fixed_size_partitioner(num_hosts),
      dtype=dtypes.float32,
      initializer=initializer,
      collections=collections,
      trainable=False))
