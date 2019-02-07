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

from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.tpu.ops import gen_tpu_ops
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

TRAINING = elc.TPUEmbeddingConfiguration.TRAINING
INFERENCE = elc.TPUEmbeddingConfiguration.INFERENCE


class TableConfig(
    collections.namedtuple(
        'TableConfig',
        ['vocabulary_size', 'dimension', 'initializer', 'combiner'])):
  """Embedding table configuration."""

  @experimental
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
        in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
        'mean' the default. 'sqrtn' often achieves good accuracy, in particular
        with bag-of-words columns. For more information, see
        `tf.nn.embedding_lookup_sparse`.

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

    if combiner not in ('mean', 'sum', 'sqrtn'):
      raise ValueError('Invalid combiner {}'.format(combiner))

    return super(TableConfig, cls).__new__(cls, vocabulary_size, dimension,
                                           initializer, combiner)


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


# TODO(shizhiw): Factor `use_gradient_accumulation` and
# `pipeline_execution_with_tensor_core` out of `_OptimizationParameters`.
class _OptimizationParameters(object):
  """Parameters common to all optimizations."""

  def __init__(self, learning_rate, use_gradient_accumulation,
               pipeline_execution_with_tensor_core):
    self.learning_rate = learning_rate
    self.use_gradient_accumulation = use_gradient_accumulation
    self.pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core)


class AdagradParameters(_OptimizationParameters):
  """Optimization parameters for Adagrad."""

  def __init__(self, learning_rate, initial_accumulator,
               use_gradient_accumulation=False,
               pipeline_execution_with_tensor_core=True):
    """Optimization parameters for Adagrad.

    Args:
      learning_rate: used for updating embedding table.
      initial_accumulator: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `True` makes embedding
         gradients calculation more accurate but slower. Please see
         `optimization_parameters.proto` for details.
         for details.
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding ID. Please see
        `tpu_embedding_configuration.proto` for details.
    """
    super(AdagradParameters, self).__init__(learning_rate,
                                            use_gradient_accumulation,
                                            pipeline_execution_with_tensor_core)
    self.initial_accumulator = initial_accumulator


class AdamParameters(_OptimizationParameters):
  """Optimization parameters for Adam."""

  def __init__(self, learning_rate,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-08,
               lazy_adam=True,
               sum_inside_sqrt=True,
               use_gradient_accumulation=False,
               pipeline_execution_with_tensor_core=True):
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
      use_gradient_accumulation: setting this to `True` makes embedding
        gradients calculation more accurate but slower. Please see
        `optimization_parameters.proto` for details.
        for details.
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding ID. Please see
        `tpu_embedding_configuration.proto` for details.
    """
    super(AdamParameters, self).__init__(learning_rate,
                                         use_gradient_accumulation,
                                         pipeline_execution_with_tensor_core)
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.lazy_adam = lazy_adam
    self.sum_inside_sqrt = sum_inside_sqrt


class StochasticGradientDescentParameters(_OptimizationParameters):
  """Optimization parameters for stochastic gradient descent.

  Args:
    learning_rate: a floating point value. The learning rate.
    use_gradient_accumulation: setting this to `True` makes embedding
      gradients calculation more accurate but slower. Please see
         `optimization_parameters.proto` for details.
    pipeline_execution_with_tensor_core: setting this to `True` makes training
      faster, but trained model will be different if step N and step N+1
      involve the same set of embedding ID. Please see
      `tpu_embedding_configuration.proto` for details.
    """

  def __init__(self, learning_rate, use_gradient_accumulation=False,
               pipeline_execution_with_tensor_core=True):
    super(StochasticGradientDescentParameters, self).__init__(
        learning_rate, use_gradient_accumulation,
        pipeline_execution_with_tensor_core)


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
      # `train_op` and `send_gradients_op` must happen in order.
      with ops.control_dependencies([train_op]):
        send_gradients_op = embedding.generate_send_gradients_op()
      with ops.control_dependencies([send_gradients_op]):
        loss = array_ops.identity(loss)

    loss = tpu.shard(computation,
                     num_shards=embedding.num_cores)

    with self.test_session() as sess:
      sess.run(tpu.initialize_system(embedding_config=
                                     embedding.config_proto))
      sess.run(variables.global_variables_initializer())
      sess.run(embedding.init_ops)
      sess.run(embedding_variables_and_ops.load_ops)
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
  @experimental
  def __init__(self,
               table_to_config_dict,
               feature_to_table_dict,
               batch_size,
               mode,
               master,
               optimization_parameters=None):
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

    Raises:
      ValueError: if any input is invalid.
    """
    _validate_table_to_config_dict(table_to_config_dict)
    # Avoid nondeterminism from `Dict` iteration order by using `OrderedDict`.
    self._table_to_config_dict = _create_ordered_dict(table_to_config_dict)
    self._combiners = _create_combiners(self._table_to_config_dict)

    _validate_feature_to_table_dict(table_to_config_dict, feature_to_table_dict)
    self._feature_to_table_dict = _create_ordered_dict(feature_to_table_dict)
    self._table_to_features_dict = _create_table_to_features_dict(
        self._feature_to_table_dict)

    self._batch_size = batch_size

    self._master = master
    self._tpu_system_metadata = (
        tpu_system_metadata_lib._query_tpu_system_metadata(self._master))  # pylint: disable=protected-access
    if self._tpu_system_metadata.num_cores == 0:
      raise ValueError('TPUEmbedding needs TPUs, but master {} does not have '
                       'TPUs.'.format(self._master))
    self._num_hosts = self._tpu_system_metadata.num_hosts
    self._hosts = [device.name for device in self._tpu_system_metadata.devices
                   if 'device:CPU:' in device.name]
    self._num_cores_per_host = self._tpu_system_metadata.num_of_cores_per_host
    self._num_cores = self._tpu_system_metadata.num_cores

    _validate_batch_size(self._batch_size, self._num_cores)
    self._batch_size_per_core = self._batch_size // self._num_cores

    self._init_ops = []

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

    dummy_table_variables_init_op = self._create_dummy_table_variables()
    self._init_ops.append(dummy_table_variables_init_op)

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
  def init_ops(self):
    """Initialization ops for TPU embedding.

    It must be called after all global variables have been initialized,
    i.e. after `global_variables_initializer()`, as it loads embedding
    tables into TPU.

    Returns:
      A list of ops.
    """
    return self._init_ops

  @property
  def feature_to_table_dict(self):
    return copy.copy(self._feature_to_table_dict)

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
      # For compatibility with old TPU workers.
      table_descriptor.optimization_parameters.use_gradient_accumulation = (
          self._optimization_parameters.use_gradient_accumulation)
      self._optimizer_handler.set_optimization_parameters(table_descriptor)

    config_proto.mode = self._mode
    config_proto.batch_size_per_tensor_core = self._batch_size_per_core
    config_proto.num_hosts = self._num_hosts
    config_proto.num_tensor_cores = self._num_cores
    config_proto.sharding_strategy = elc.TPUEmbeddingConfiguration.DIV_DEFAULT
    config_proto.pipeline_execution_with_tensor_core = (
        self._optimization_parameters.pipeline_execution_with_tensor_core)

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
        A list of ops to load embedding and slot variables on CPU to TPU,
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
    """
    embedding_variables_by_table = {}
    slot_variables_by_table = {}
    load_ops = []
    retrieve_ops = []
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

        slot_variables_for_table, load_ops_for_table, retrieve_ops_for_table = (
            self._optimizer_handler.create_variables_and_ops(
                table, slot_variable_names, self._num_hosts,
                self._table_to_config_dict[table], table_variables)
        )
        slot_variables_by_table[table] = slot_variables_for_table
        load_ops.extend(load_ops_for_table)
        retrieve_ops.extend(retrieve_ops_for_table)
    return VariablesAndOps(embedding_variables_by_table,
                           slot_variables_by_table,
                           load_ops, retrieve_ops)

  def _create_dummy_table_variables(self):
    """Create dummy embedding table variables.

    The sole purpose of these dummy variables are to trigger gradient
    calcuation wrt them so that the gradients wrt activation can be captured
    and later sent to TPU embedding.

    Returns:
      Initializer for these variables.

    Raises:
      RuntimeError: if collection to store gradients already exists and is not
      empty.
    """
    self._dummy_table_variables = []
    # TODO(shizhiw): remove table id.
    for table_id, table in enumerate(self._table_to_features_dict):
      self._dummy_table_variables.append(
          variable_scope.get_variable(
              'tpu_embedding_dummy_table_variable_%s' % table,
              dtype=dtypes.float32,
              shape=[1],
              use_resource=True,
              trainable=True,
              # TODO(shizhiw): Remove these dummy variables as
              # tensorflow optimizer creates slot variable for them which
              # is undesirable.
              # e.g. tpu_embedding_dummy_table_variable_mlp_user/Adam{_1}.
              # Explicitly specifying collections prevents this variable from
              # being added to the GLOBAL_VARIABLES collection, so that Saver()
              # ignores it.
              collections=['tpu_embedding_dummy_table_variables']))

      g = ops.get_default_graph()
      table_gradients = g.get_collection_ref(
          'tpu_embedding_gradients_table_%d' % table_id)
      if table_gradients:
        raise RuntimeError(
            'tpu_embedding_gradients_table_%d is not empty.' % table_id)
      table_gradients.extend([None] * len(self._table_to_features_dict[table]))

    return variables.variables_initializer(
        self._dummy_table_variables,
        name='tpu_embedding_dummy_table_variables_init')

  def generate_enqueue_ops(self, sparse_features_list):
    """Generate enqueue ops.

    Args:
      sparse_features_list: a list of dictionary mapping from string
        of feature names to sparse tensor. Each dictionary is for one
        TPU core. Dictionaries for the same core should be contiguous
        on the list.

    Returns:
      Ops to enqueue to TPU for embedding.
    """
    self._validate_generate_enqueue_ops_sparse_features_list(
        sparse_features_list)
    return [
        self._generate_enqueue_op(
            sparse_features, device_ordinal=i % self._num_cores_per_host)
        for i, sparse_features in enumerate(sparse_features_list)
    ]

  def _validate_generate_enqueue_ops_sparse_features_list(
      self, sparse_features_list):
    """Validate `sparse_features_list`."""
    if len(sparse_features_list) != self._num_cores:
      raise ValueError('Length of `sparse_features_list` should match the '
                       'number of cores; '
                       '`len(sparse_features_list)` is {}, '
                       'number of cores is {}.'.format(
                           len(sparse_features_list), self._num_cores))

    feature_set = set(self._feature_to_table_dict.keys())
    contiguous_device = None
    for i, sparse_features in enumerate(sparse_features_list):
      used_feature_set = set(sparse_features.keys())

      # Check features are valid.
      missing_feature_set = feature_set - used_feature_set
      if missing_feature_set:
        raise ValueError('`sparse_features_list[{}]` misses a feature that is '
                         'in `feature_to_config_dict`: {}.'.format(
                             i, missing_feature_set))

      extra_feature_set = used_feature_set - feature_set
      if extra_feature_set:
        raise ValueError('`sparse_features_list[{}]` has a feature that is not '
                         'in `feature_to_config_dict`: {}.'.format(
                             i, extra_feature_set))

      device = None
      device_feature = None
      for feature, tensor in six.iteritems(sparse_features):
        if not isinstance(tensor, sparse_tensor.SparseTensor):
          raise ValueError('`sparse_features_list[{}]` has a feature that is '
                           'not mapped to `SparseTensor`. '
                           '`feature`: {}, type: {}'.format(
                               i, feature, type(tensor)))

        # Check all features are on the same device.
        if device is None:
          device = tensor.op.device
          device_feature = feature
        else:
          if device != tensor.op.device:
            raise ValueError('Devices are different between features in '
                             '`sparse_features_list[{}]`; '
                             'devices: {}, {}; features: {}, {}.'.format(
                                 i, device, tensor.op.device, feature,
                                 device_feature))

      if i % self._num_cores_per_host:
        if device != contiguous_device:
          raise ValueError('We expect the `sparse_features` which are on the '
                           'same host to be contiguous in '
                           '`sparse_features_list`, '
                           '`sparse_features_list[{}]` is on device {}, '
                           'but is expected to be on device {}.'.format(
                               i, device, contiguous_device))
      else:
        contiguous_device = device

  def _generate_enqueue_op(self, sparse_features, device_ordinal):
    with ops.colocate_with(list(sparse_features.values())[0]):
      sample_idcs, embedding_idcs, aggregation_weights = (
          self._format_for_tpu_embedding_sparse_batch(sparse_features))
      return tpu_ops.enqueue_tpu_embedding_sparse_batch(
          sample_idcs,
          embedding_idcs,
          aggregation_weights,
          combiners=self._combiners,
          device_ordinal=device_ordinal)

  def _format_for_tpu_embedding_sparse_batch(self, sparse_features):
    """Format sparse features for `enqueue_tpu_embedding_sparse_batch()`.

    Args:
      sparse_features: a `Dict` of `SparseTensor`s for embedding.

    Returns:
      Arguments for `enqueue_tpu_embedding_sparse_batch()`.
    """

    sample_idcs, embedding_idcs, aggregation_weights = list(), list(), list()
    for table in self._table_to_features_dict:
      sample_t, indices_t, weights_t = list(), list(), list()

      features = self._table_to_features_dict[table]
      for i, feature in enumerate(features):
        tensor = sparse_features[feature]
        sample_indices = tensor.indices[:, 0]
        embedding_indices = tensor.values
        weights = array_ops.ones_like(embedding_indices)
        sample_t.append(i * self._batch_size_per_core + sample_indices)
        indices_t.append(embedding_indices)
        weights_t.append(weights)

      sample_idcs.append(
          math_ops.cast(array_ops.concat(sample_t, axis=0), dtype=dtypes.int32))
      embedding_idcs.append(
          math_ops.cast(
              array_ops.concat(indices_t, axis=0), dtype=dtypes.int32))
      aggregation_weights.append(
          math_ops.cast(
              array_ops.concat(weights_t, axis=0), dtype=dtypes.float32))

    return sample_idcs, embedding_idcs, aggregation_weights

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
        start_row = lookup_id * self._batch_size_per_core
        end_row = start_row + self._batch_size_per_core
        activations[feature] = gen_tpu_ops.tpu_embedding_activations(
            self._dummy_table_variables[table_id],
            recv_activations[table_id][start_row:end_row, :],
            table_id=table_id,
            lookup_id=lookup_id)
    return activations

  # TODO(shizhiw): Make `gradient_multiplier` per feature. Setting it to 0 would
  # have the effect of `tf.stop_gradients()`.
  # TODO(shizhiw): Consider alternative ways to capture gradients wrt embedding
  # layer outputs to remove `_dummy_table_variables`,
  # `_embedding_activation_grad` and `tpu_embedding_gradients_table_%d'.
  def generate_send_gradients_op(self, gradient_multipliers=None):
    """Retrieve gradients from collections and send them to TPU embedding.

    Args:
      gradient_multipliers: None, or dict mapping table names to gradient
        multiplier Tensors.

    Returns:
      SendTPUEmbeddingGradients Op.

    Raises:
      ValueError: If required gradients have not been defined.
      RuntimeError: If `mode` is not `TRAINING`.
    """
    if self._mode != TRAINING:
      raise RuntimeError('Only in training mode gradients need to '
                         'be sent to TPU embedding; got mode {}.'
                         .format(self._mode))

    g = ops.get_default_graph()
    gradients = list()
    for table_id, table in enumerate(self._table_to_config_dict):
      table_gradients = g.get_collection(
          'tpu_embedding_gradients_table_%d' % table_id)
      if any(gradient is None for gradient in table_gradients):
        raise ValueError(
            'Table {}/{} has undefined gradients: this is probably because the '
            'model asked TPUEmbedding to compute activations that were not '
            'used.'.format(table_id, table))
      concat_table_grads = array_ops.concat(table_gradients, axis=0)
      if gradient_multipliers is not None:
        concat_table_grads *= gradient_multipliers[table.name]
      gradients.append(concat_table_grads)

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

    load_ops = []
    retrieve_ops = []
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
        retrieved_table, retrieved_accumulator = (
            tpu_ops.retrieve_tpu_embedding_adagrad_parameters(
                table_name=table,
                num_shards=num_hosts,
                shard_id=host_id))
        retrieve_parameters_op = control_flow_ops.group(
            state_ops.assign(table_variable, retrieved_table),
            state_ops.assign(accumulator_variable, retrieved_accumulator))

      load_ops.append(load_parameters_op)
      retrieve_ops.append(retrieve_parameters_op)
    return slot_variables, load_ops, retrieve_ops


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

    load_ops = []
    retrieve_ops = []
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
        retrieved_table, retrieved_m, retrieved_v = (
            tpu_ops.retrieve_tpu_embedding_adam_parameters(
                table_name=table,
                num_shards=num_hosts,
                shard_id=host_id))
        retrieve_parameters_op = control_flow_ops.group(
            state_ops.assign(table_variable, retrieved_table),
            state_ops.assign(m_variable, retrieved_m),
            state_ops.assign(v_variable, retrieved_v))

      load_ops.append(load_parameters_op)
      retrieve_ops.append(retrieve_parameters_op)
    return slot_variables, load_ops, retrieve_ops


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

    load_ops = []
    retrieve_ops = []
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
        retrieved_table = (
            tpu_ops
            .retrieve_tpu_embedding_stochastic_gradient_descent_parameters(
                table_name=table,
                num_shards=num_hosts,
                shard_id=host_id))
        retrieve_parameters_op = control_flow_ops.group(
            state_ops.assign(table_variable, retrieved_table))

      load_ops.append(load_parameters_op)
      retrieve_ops.append(retrieve_parameters_op)
    return None, load_ops, retrieve_ops


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


def _create_combiners(table_to_config_dict):
  return [table_to_config_dict[t].combiner for t in table_to_config_dict]


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
