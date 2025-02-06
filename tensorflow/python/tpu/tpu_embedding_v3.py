# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Mid level API for TPU Embeddings With V2 Embedding Accelerator."""

import collections
import copy
import dataclasses
import functools
import hashlib
import operator
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from absl import logging

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.tpu.kernels import sparse_core_layout_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import _pywrap_sparse_core_layout
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_embedding_v3_checkpoint_adapter
from tensorflow.python.tpu import tpu_embedding_v3_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import gen_xla_ops as xla_ops
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


_PIPELINE_ATTRIBUTE = "_embedding_pipelining"
_PIPELINE_MODE_FORWARD = "forward"
_PIPELINE_MODE_BACKWARD = "backward"


TableConfig = tpu_embedding_v2_utils.TableConfig
FeatureConfig = tpu_embedding_v2_utils.TableConfig
QuantizationConfig = tpu_embedding_v2_utils.QuantizationConfig


@tf_export("tpu.experimental.embedding.SparseCoreEmbeddingConfig")
@dataclasses.dataclass(frozen=True)
class SparseCoreEmbeddingConfig:
  """Config for sparsecore embedding."""

  disable_table_stacking: bool = False
  max_ids_per_chip_per_sample: int = 64
  max_ids_per_table: Optional[Dict[str, int]] = None
  max_unique_ids_per_table: Optional[Dict[str, int]] = None
  allow_id_dropping: bool = False
  initialize_tables_on_host: bool = True
  enable_fast_table_initialization: bool = False


class EmbeddingPipeliningContext(control_flow_ops.ControlFlowContext):
  """Sets the _embedding_pipelining attribute on all ops created in the scope."""

  def __init__(self, mode: str, enable: bool):
    super().__init__()
    self._name = "EmbeddingPipelinigContext"
    self._mode = attr_value_pb2.AttrValue(s=compat.as_bytes(mode))
    recording_summaries = summary_ops_v2.is_recording_summaries()
    if enable and recording_summaries:
      logging.info(
          "Embedding pipelining requested but summaries are being recorded:"
          " Disabling embedding pipelining."
      )
      self._enable = False
    else:
      self._enable = enable

  def to_control_flow_context_def(
      self, context_def: Any, export_scope: Any = None
  ):
    # pylint: disable=useless-super-delegation
    # The method is required by `ControlFlowContext`.
    super().to_control_flow_context_def(context_def, export_scope)

  def AddOp(self, op: ops.Operation):
    # pylint: disable=protected-access
    if self._enable:
      op._set_attr(_PIPELINE_ATTRIBUTE, self._mode)
    if self._outer_context:
      self._outer_context.AddOp(op)


class TPUEmbeddingShardedSaveable(saveable_object.SaveableObject):
  """Defines how to save and restore a shard of TPUEmbedding sharded variable."""

  def __init__(
      self,
      variable: tf_variables.Variable,
      shard_id: int,
      num_shards: int,
      shard_dim: int,
      name: str,
  ):
    """Init TPUEmbeddingShardedSaveable."""
    self._shard_id = shard_id
    self._variable = variable

    var_offset = [0] * len(variable.shape)
    # NOTE: always assume even sharding
    var_offset[shard_dim] = shard_id * variable.shape[shard_dim]
    fullshape = variable.shape.as_list()
    fullshape[shard_dim] = num_shards * fullshape[shard_dim]
    save_slice_info = tf_variables.Variable.SaveSliceInfo(
        full_name=name,
        full_shape=fullshape,
        var_offset=var_offset,
        var_shape=variable.shape.as_list(),
    )

    spec = saveable_object.SaveSpec(
        tensor=variable.read_value,
        slice_spec=save_slice_info.spec,
        name=name,
        dtype=variable.dtype,
        device=variable.device,
    )
    super().__init__(variable.read_value, [spec], name)

  def restore(
      self,
      restored_tensors: List[tensor.Tensor],
      restored_shapes: List[tensor_shape.TensorShape],
  ) -> Any:
    del restored_shapes
    restored_tensor = restored_tensors[0]

    return values_util.assign_on_device(
        self._variable.device, self._variable, restored_tensor
    )


def _fielddict():
  return dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TableStacking:
  """Information about how we stack tables."""

  # Indexed by stacked table name:
  stacked_table_to_tables: Dict[str, TableConfig] = _fielddict()
  quantization_configs: Dict[str, QuantizationConfig] = _fielddict()

  # Indexed by table name:
  table_name_to_table: Dict[str, TableConfig] = _fielddict()
  table_to_padding_rows: Dict[str, int] = _fielddict()
  table_to_padding_columns: Dict[str, int] = _fielddict()
  table_to_sample_count: Dict[str, int] = _fielddict()
  table_to_layout: Dict[str, sparse_core_layout_pb2.SparseCoreTableLayout] = (
      _fielddict()
  )
  # Maps table name to (stacked table, row offset, shard rotation)
  table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]] = _fielddict()

  # Indexed by feature_path the key of flat_features:
  feature_to_sample_offset: Dict[str, int] = _fielddict()


@saveable_compat.legacy_saveable_name("")
class TPUEmbeddingShardedVariable(
    tpu_values.TPUVariableMixin, values.DistributedVariable
):
  """A ShardedVariable class for Embedding tables on TPU."""

  def _is_mirrored(self) -> bool:
    return False

  # Only support sharding on the first dimension.
  @property
  def shard_dim(self) -> int:
    return 0

  @property
  def shape(self) -> tensor_shape.TensorShape:
    """Returns the shape of the embedding variable for the current context."""
    local_shape = self._values[0].shape
    global_shape = local_shape.as_list()
    global_shape[self.shard_dim] = global_shape[self.shard_dim] * len(
        self.values
    )
    return tensor_shape.TensorShape(global_shape)

  def _write_object_proto(self, proto, options):
    super()._write_object_proto(proto, options)
    # TODO(b/305882915): Reset the saved model shape to the local shape
    # for backward compatibility of users that directly access the full
    # variable shape as the shape of values.
    proto.variable.shape.CopyFrom(self._values[0].shape.as_proto())

  def _gather_saveables_for_checkpoint(self) -> Dict[str, Callable[..., Any]]:
    """Overrides Trackable method.

    Returns:
      A dictionary mapping attribute names to `SaveableObject` factories.
    """

    def _saveable_factory(name=self._common_name):
      saveables = []
      num_shards = len(self.values)
      for shard_id in range(num_shards):
        saveables.append(
            TPUEmbeddingShardedSaveable(
                self.values[shard_id],
                shard_id,
                num_shards,
                self.shard_dim,
                name,
            )
        )
      return saveables

    return {base.VARIABLE_VALUE_KEY: _saveable_factory}

  def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
    """Converts a variable to a tensor."""
    # pylint: disable=protected-access
    if tpu_util.enclosing_tpu_context() is None:
      return self._values[0].read_value()
    else:
      return self._read_variable_op()

  def read_value(self) -> Any:
    if tpu_util.enclosing_tpu_context() is None:
      raise NotImplementedError(
          "Reading in cross replica mode is not yet supported"
          "for TPUEmbeddingShardedVariable."
      )
    else:
      return self._read_variable_op()

  def assign(
      self,
      value: Any,
      use_locking: bool = False,
      name: Optional[Any] = None,
      read_value: bool = True,
  ) -> Any:
    if tpu_util.enclosing_tpu_context() is None:
      # Running in a host context
      for device in self.distribute_strategy.extended.worker_devices:
        with ops.device(device):
          self.assign_on_device(device, value)
    return tpu_util.make_raw_assign_fn(
        gen_resource_variable_ops.assign_variable_op
    )(
        self,
        value=value,
        use_locking=use_locking,
        name=name,
        read_value=read_value,
    )

  def assign_on_device(self, device, value):
    if self._packed_var is None:
      raise NotImplementedError("Required packed variable support")
    with ops.device(device):
      gen_resource_variable_ops.assign_variable_op(
          resource=self._packed_var.handle, value=value
      )

  def read_from_device(self, device):
    if self._packed_var is None:
      raise NotImplementedError("Required packed variable support")
    with ops.device(device):
      return gen_resource_variable_ops.read_variable_op(
          resource=self._packed_var.handle, dtype=self.dtype
      )


# TODO(pineapplejuice233): Add debug string representation of the class.
PartitionedCsrFormatTensor = collections.namedtuple(
    "PartitionedCsrFormatTensor",
    [
        "row_pointers",
        "sorted_sample_ids",
        "sorted_token_ids",
        "sorted_gains",
        "sample_count",
        "num_minibatches_per_physical_sparse_core",
    ],
)


def _clone_feature_config(feature_config):
  old_to_new_table = {}
  new_features = []

  for old_feature in nest.flatten(feature_config):
    feature = copy.copy(old_feature)
    if feature.table not in old_to_new_table:
      old_to_new_table[feature.table] = copy.copy(feature.table)
    feature.table = old_to_new_table[feature.table]
    new_features.append(feature)

  return nest.pack_sequence_as(feature_config, new_features)


def _stack_tables_with_same_table_dim_and_optimizer(
    table_config: Sequence[TableConfig],
    flat_features: Sequence[Tuple[Any, FeatureConfig]],
    num_partitions: int,
    num_sc_per_partition: int,
    sparse_core_embedding_config: Optional[SparseCoreEmbeddingConfig] = None,
) -> TableStacking:
  """Stack tables with the same table dim and optimizer."""
  logging.info("Number of tables before stacking is %d", len(table_config))
  disable_table_stacking = False
  if sparse_core_embedding_config:
    disable_table_stacking = sparse_core_embedding_config.disable_table_stacking

  if disable_table_stacking:
    logging.warn("Table stacking is disabled.")

  stacker = _pywrap_sparse_core_layout.SparseCoreLayoutStacker(
      num_partitions=num_partitions,
      sparse_cores_per_partition=num_sc_per_partition,
      disable_table_stacking=disable_table_stacking,
  )
  s = TableStacking()
  s.table_name_to_table = {table.name: table for table in table_config}
  table_to_num_samples = {table.name: 0 for table in table_config}
  for _, feature in flat_features:
    table_to_num_samples[feature.table.name] += functools.reduce(
        operator.mul, feature.output_shape
    )
    # First generate stacking for any tables our caller didn't stack for us.
    # Note that we process the tables sorted by name so the ordering is
    # deterministic.
    sorted_tables = sorted(table_config, key=lambda t: t.name)
    for table in sorted_tables:
      if not table.layout:
        # All tables in a stack have to have the same hyperparemeters; this key
        # contains everything we care about. The key is an arbitrary string
        # whose value is not particularly meaningful except that it has to be
        # different if the tables cannot be stacked together.
        #
        # Note that later we rewrite the stack name based on the tables in that
        # stack; this is just a temporary initial name.
        #
        # The key does not need to include the embedding width; that is handled
        # separately.
        key_tuple = (
            # Optimizers don't have a repr but do support hash.
            hash(table.optimizer),
            # Quantization configs don't have a hash but do support repr.
            repr(table.quantization_config),
        )
        key_str = hashlib.sha1(
            repr(key_tuple).encode(),
            usedforsecurity=False,
        ).hexdigest()
        key = "_xxtpuv3internal_" + key_str
        stacker.AddTable(
            table_name=table.name,
            table_height=table.vocabulary_size,
            table_width=table.dim,
            group=key,
            output_samples=table_to_num_samples[table.name],
        )
    # First generate stacking for any tables our caller didn't stack for us.
    # Note that we process the tables sorted by name so the ordering is
    # deterministic.
    # Put the layout information we just computed back into the tables, so we
    # can treat tables whose layouts were given by the caller and tables whose
    # layouts we computed the same.
    for layout in stacker.GetLayouts().tables:
      table = s.table_name_to_table[layout.table_name]
      assert not table.layout  # It's a bug if it was already set.
      table.layout = layout

    # Collect all the layout information from all the tables, whether we just
    # computed it above, or whether the caller passed it as part of the
    # TableConfig:
    tables_by_stack = collections.defaultdict(list)
    for table in sorted_tables:
      layout = table.layout
      assert layout.table_name == table.name
      s.table_to_layout[table.name] = layout
      tables_by_stack[layout.stacked_table_name].append(table)

    for stack_name, tables in tables_by_stack.items():
      s.quantization_configs[stack_name] = tables[0].quantization_config
      s.stacked_table_to_tables[stack_name] = tables

      logging.vlog(1, "Stacked table name: %s", stack_name)
      for table in tables:
        layout = table.layout
        logging.vlog(
            1,
            "  Table %s: offset %d, rotation %d",
            table.name,
            layout.sparse_core_shard_row_offset,
            layout.sparse_core_shard_rotation,
        )
        s.table_to_stacked_table_offset[table.name] = (
            stack_name,
            layout.sparse_core_shard_row_offset
            * num_partitions
            * num_sc_per_partition,
            layout.sparse_core_shard_rotation,
        )
        # Update dimensions in the table to the padded dimensions.
        table.vocabulary_size = layout.unsharded_padded_shape[0]
        table.dim = layout.unsharded_padded_shape[1]
        s.table_to_padding_rows[table.name] = (
            layout.unsharded_padded_shape[0] - layout.unsharded_shape[0]
        )
        s.table_to_padding_columns[table.name] = (
            layout.unsharded_padded_shape[1] - layout.unsharded_shape[1]
        )

    logging.info(
        "Number of tables after stacking is %d.",
        len(s.stacked_table_to_tables),
    )

    s.table_to_sample_count = {
        table_name: 0 for table_name in s.stacked_table_to_tables
    }
    for feature_path, feature in flat_features:
      stacked_table_name = s.table_to_stacked_table_offset[feature.table.name][
          0
      ]
      s.feature_to_sample_offset[feature_path] = s.table_to_sample_count[
          stacked_table_name
      ]
      s.table_to_sample_count[stacked_table_name] += functools.reduce(
          operator.mul, feature.output_shape
      )

    return s


# TODO(b/233952762): Add tests of this version of the mid-level API.
@tf_export("tpu.experimental.embedding.TPUEmbeddingV2")
class TPUEmbeddingV2(tpu_embedding_base.TPUEmbeddingBase):
  """The TPUEmbedding mid level API running on TPU with sparse core accelerator."""

  DEFAULT_MAX_IDS_PER_TABLE = 256
  DEFAULT_MAX_UNIQUE_IDS_PER_TABLE = 256

  def __init__(
      self,
      feature_config: Union[tpu_embedding_v2_utils.FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = None,  # pylint:disable=protected-access
      pipeline_execution_with_tensor_core: bool = False,
      sparse_core_embedding_config: Optional[SparseCoreEmbeddingConfig] = None,
  ):
    """Creates the TPUEmbeddingV2 mid level API object.

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. When not created under TPUStrategy
        may be set to None to avoid the creation of the optimizer slot
        variables, useful for optimizing memory consumption when exporting the
        model for serving where slot variables aren't needed.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old). Set to True for improved performance.
      sparse_core_embedding_config: Configs for sparse core embedding including
        settings for table stacking, input feature static buffer size etc.

    Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad) or None when created under a TPUStrategy.
      RuntimeError: If not created under TPUStrategy.
    """
    # We do a clone on the feature_config here as we will alter settings in it
    # and we don't want the user to see these. We can't just use clone here
    # as we need to maintain some object relationships.
    super().__init__(_clone_feature_config(feature_config), optimizer)
    self._strategy = distribute_lib.get_strategy()
    if not isinstance(
        self._strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2)
    ):
      raise RuntimeError(
          "TPUEmbeddingV2 should be created under TPUStrategy but found {}."
          .format(self._strategy)
      )

    # TODO(pineapplejuice233): Remove this once weight decay is supported.
    for table in self._table_config:
      if (
          table.optimizer.weight_decay_factor is not None
          or table.optimizer.multiply_weight_decay_factor_by_learning_rate
          is not None
      ):
        raise NotImplementedError(
            "weight_decay_factor and"
            " multiply_weight_decay_factor_by_learning_rate are not supported"
            f" yet. But found in table {table.name} setting."
        )

    self._num_sc_per_chip = (
        self._strategy.extended.tpu_hardware_feature.num_embedding_devices_per_chip
    )
    if self._num_sc_per_chip == 0:
      logging.warning(
          "No embedding devices per chip info is found. Using 4 as the default"
          " value for SparseCore."
      )
      self._num_sc_per_chip = 4

    self._num_sc_shards = (
        self._strategy.num_replicas_in_sync * self._num_sc_per_chip
    )

    # We need this in multiple places, so avoid flattening multiple times.
    # This order will also be used when stacking features.
    self._flat_features = nest.flatten_with_joined_string_paths(
        self._feature_config
    )

    if sparse_core_embedding_config is None:
      self._sparse_core_embedding_config = SparseCoreEmbeddingConfig()
      logging.warning(
          "SparseCoreEmbeddingConfig is not provided. Using default values %s",
          self._sparse_core_embedding_config,
      )
    else:
      self._sparse_core_embedding_config = sparse_core_embedding_config

    self._s = _stack_tables_with_same_table_dim_and_optimizer(
        self._table_config,
        self._flat_features,
        self._strategy.num_replicas_in_sync,
        self._num_sc_per_chip,
        self._sparse_core_embedding_config,
    )

    self._table_name_to_table = self._s.table_name_to_table
    self._stacked_table_to_tables = self._s.stacked_table_to_tables
    self._table_to_padding_columns = self._s.table_to_padding_columns
    self._table_to_padding_rows = self._s.table_to_padding_rows
    self._table_to_stacked_table_offset = self._s.table_to_stacked_table_offset
    self._table_to_sample_count = self._s.table_to_sample_count
    self._feature_to_sample_offset = self._s.feature_to_sample_offset
    self._quantization_configs = self._s.quantization_configs

    # These hyperparameters will be provided by the FDO. Currently hardcode
    # here just for testing.
    self.max_ids_per_chip_per_sample = (
        self._sparse_core_embedding_config.max_ids_per_chip_per_sample
    )
    self.max_minibatches_per_sc = 64

    self._table_to_max_ids_per_sparse_core = {}
    self._table_to_max_unique_ids_per_sparse_core = {}

    self._update_sparse_core_buffer_size_after_table_stacking()

    self._pipelining = pipeline_execution_with_tensor_core

  def _compute_sc_shard_info(
      self,
      table: TableConfig,
      partition_shape: tuple[int, int],
      partition_offset: List[int],
      total_vocab_size: int,
      sc_idx: int,
  ) -> base.ShardInfo:
    # Scale the partition to get sizes for the current table,
    # then select this sc shard.
    sc_shard_size = (
        table.vocabulary_size
        * partition_shape[0]
        // total_vocab_size
        // self._num_sc_per_chip
    )
    sc_shard_offset = (
        table.vocabulary_size
        * partition_offset[0]
        // total_vocab_size
    ) + sc_idx * sc_shard_size

    return base.ShardInfo([sc_shard_size, table.dim], [sc_shard_offset, 0])

  def _compute_sc_shard_idx_and_offset(
      self,
      table_name: str,
      shard_info: base.ShardInfo
  ) -> tuple[int, int]:
    tpu_devices = self._strategy.extended._tpu_devices  # pylint:disable=protected-access
    num_replicas, num_cores_per_replica = tpu_devices.shape
    num_devices = num_replicas * num_cores_per_replica

    shift = self._s.table_to_stacked_table_offset[table_name][2]
    shard_index = shard_info.offset[0] // shard_info.shape[0]
    # Rotate the shards.
    shard_index = (shard_index - shift) % self._num_sc_shards
    num_sc = num_devices * self._num_sc_per_chip

    return shard_index, num_sc

  def _host_table_initializer(
      self,
      stacked_tables: List[TableConfig],
      total_vocab_size: int,
      partition_shape: tuple[int, int],
      dtype: dtypes.DType,
  ) -> Dict[int, List[Dict[str, tensor.Tensor]]]:
    cpu_table_tensors = {}

    tpu_devices = self._strategy.extended._tpu_devices  # pylint:disable=protected-access
    num_replicas, num_cores_per_replica = tpu_devices.shape
    num_devices = num_replicas * num_cores_per_replica

    partition_shape = (partition_shape[0] // num_devices, partition_shape[1])
    partition_offset = [0] * len(partition_shape)

    for rid in range(num_replicas):
      for cid in range(num_cores_per_replica):
        device_cpu = (
            tf_device.DeviceSpec.from_string(tpu_devices[rid][cid])
            .replace(device_type="CPU", device_index=0)
            .to_string()
        )

        shard_dim_offset = (
            (rid * num_cores_per_replica) + cid
        ) * partition_shape[0]
        cpu_table_tensors[shard_dim_offset] = []

        for i in range(self._num_sc_per_chip):
          # Each underlying table has column lookups rotated by 1 to avoid hot
          # spots on core 0 for id=0. We shift the initializer as well to help
          # with comparisons against CPU.
          full_tables = {}
          cpu_table_tensors[shard_dim_offset].append({})
          for table in stacked_tables:
            arg_spec = tf_inspect.getfullargspec(table.initializer)
            sharding_aware = (
                "shard_info" in arg_spec.args
                or "shard_info" in arg_spec.kwonlyargs
            )

            if (
                self._sparse_core_embedding_config.initialize_tables_on_host
                and not sharding_aware
            ):
              # When the user-initializer is not sharding aware but includes
              # shard info, we pre-construct the full initial table on the host
              # and then slice out the individual shards.
              partition_offset[0] = shard_dim_offset
              sc_shard_info = self._compute_sc_shard_info(
                  table,
                  partition_shape,
                  partition_offset,
                  total_vocab_size,
                  i,
              )
              shard_index, shard_offset = self._compute_sc_shard_idx_and_offset(
                  table.name, sc_shard_info
              )

              with ops.device(device_cpu):
                if table.name not in full_tables:
                  full_tables[table.name] = table.initializer(
                      shape=(table.vocabulary_size, table.dim),
                      dtype=dtype,
                  )
                sc_shard = full_tables[table.name][shard_index::shard_offset, :]
                cpu_table_tensors[shard_dim_offset][i][table.name] = sc_shard

    return cpu_table_tensors

  def _update_sparse_core_buffer_size_after_table_stacking(self):
    """Update the sparse core buffer size after table stacking."""
    for table_name in self._stacked_table_to_tables:
      if (
          self._sparse_core_embedding_config.max_ids_per_table is None
          or table_name
          not in self._sparse_core_embedding_config.max_ids_per_table
      ):
        logging.warning(
            "Table %s is not found in max_ids_per_table provided by"
            " SparseCoreEmbeddingConfig. Using default value 256.",
            table_name,
        )
        self._table_to_max_ids_per_sparse_core[table_name] = (
            self.DEFAULT_MAX_IDS_PER_TABLE
        )
      else:
        self._table_to_max_ids_per_sparse_core[table_name] = (
            self._sparse_core_embedding_config.max_ids_per_table[table_name]
        )
      if (
          self._sparse_core_embedding_config.max_unique_ids_per_table is None
          or table_name
          not in self._sparse_core_embedding_config.max_unique_ids_per_table
      ):
        logging.warning(
            (
                "Table %s is not found in max_unique_ids_per_table provided by"
                " SparseCoreEmbeddingConfig. Using default value 256."
            ),
            table_name,
        )
        self._table_to_max_unique_ids_per_sparse_core[table_name] = (
            self.DEFAULT_MAX_UNIQUE_IDS_PER_TABLE
        )
      else:
        self._table_to_max_unique_ids_per_sparse_core[table_name] = (
            self._sparse_core_embedding_config.max_unique_ids_per_table[
                table_name
            ]
        )

  @property
  def embedding_tables(
      self,
  ) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    self._maybe_build()
    # Only return the tables and not the slot variables.
    return {
        stacked_table_name: self._variables[stacked_table_name]["parameters"]
        for stacked_table_name in self._stacked_table_to_tables
    }

  @property
  def embedding_table_shards(
      self,
  ) -> Dict[tpu_embedding_v2_utils.TableConfig, List[tf_variables.Variable]]:
    """Returns a dict of embedding tables, keyed by `TableConfig`."""
    self._maybe_build()

    # This reflects the device assignment used by the TPU Strategy.
    ordered_devices = []
    for devices in self._strategy.extended._tpu_devices:  # pylint: disable=protected-access
      ordered_devices.extend(devices)

    table_shards = {
        name: [
            (device, var.read_from_device(device)) for device in ordered_devices
        ]
        for name, var in self.embedding_tables.items()
    }

    return table_shards

  @property
  def embedding_layouts(
      self,
  ) -> Dict[str, sparse_core_layout_pb2.SparseCoreTableLayout]:
    """Returns how the tables are laid out in the variables.

    The SparseCoreTableLayout describes how a table is stored in its internal
    state. You need this only if you need to pull apart the internal state.
    """
    return self._s.table_to_layout

  @property
  def variables(
      self,
  ) -> Dict[
      tpu_embedding_v2_utils.TableConfig, Dict[str, tf_variables.Variable]
  ]:
    """Returns a dict of variables, keyed by `TableConfig`, then by slot name."""
    self._maybe_build()
    return self._variables

  def _create_variables(
      self,
      stacked_tables: List[tpu_embedding_v2_utils.TableConfig],
      stacked_table_name: str,
  ) -> Dict[str, tf_variables.Variable]:
    """Create all variables including table variables and slot variables."""
    total_vocab_size = sum([table.vocabulary_size for table in stacked_tables])
    table_dim = stacked_tables[0].dim
    variable_shape = (total_vocab_size, table_dim)
    variable_dtype = dtypes.float32
    optimizer = stacked_tables[0].optimizer

    # Compute those table shards early on host that might otherwise saturate
    # device HBM from needing to initialize full embedding tables.
    host_table_tensors = self._host_table_initializer(
        stacked_tables, total_vocab_size, variable_shape, variable_dtype,
    )

    def table_initialize_fn(shape, dtype, shard_info=None):
      # If enable fast table initialization, we will initialize the table
      # directly on the device and use the initializer from the first table.
      if self._sparse_core_embedding_config.enable_fast_table_initialization:
        return stacked_tables[0].initializer(
            shape=(shard_info.shape[0], stacked_tables[0].dim),
            dtype=dtype,
        )

      # Concat all the tables along the first axis.
      concat_tensors = []

      # Temporary patch, we need to initialize tables with the SC level
      # sharding. Note that we need to ensure that the vocab size is divisible
      # by the global number of SC.
      for i in range(self._num_sc_per_chip):
        # Each underlying table has column lookups rotated by 1 to avoid hot
        # spots on core 0 for id=0. We shift the initializer as well to help
        # with comparisons against CPU.
        for table in stacked_tables:
          arg_spec = tf_inspect.getfullargspec(table.initializer)
          sharding_aware = (
              "shard_info" in arg_spec.args
              or "shard_info" in arg_spec.kwonlyargs
          )

          if shard_info:
            sc_shard_info = self._compute_sc_shard_info(
                table,
                shard_info.shape,
                shard_info.offset,
                total_vocab_size,
                i,
            )
            if not sharding_aware:
              if (
                  host_table_tensors
                  and table.name in host_table_tensors[shard_info.offset[0]][i]
              ):
                sc_shard = host_table_tensors[shard_info.offset[0]][i][
                    table.name
                ]
              else:
                shard_index, shard_offset = (
                    self._compute_sc_shard_idx_and_offset(
                        table.name, sc_shard_info
                    )
                )
                sc_shard = table.initializer(
                    shape=(table.vocabulary_size, table.dim), dtype=dtype
                )[shard_index::shard_offset, :]

            else:
              sc_shard = table.initializer(
                  shape=(table.vocabulary_size, table.dim),
                  dtype=dtype,
                  shard_info=sc_shard_info,
              )
          else:
            sc_shard = table.initializer(
                shape=(
                    (table.vocabulary_size * shape[0])
                    // total_vocab_size
                    // self._num_sc_per_chip,
                    table.dim,
                ),
                dtype=dtype,
            )
          concat_tensors.append(sc_shard)
      return array_ops.concat(concat_tensors, axis=0)

    def getter(name, shape, dtype, initializer, trainable):
      del shape
      initial_value = functools.partial(
          initializer, shape=variable_shape, dtype=dtype
      )
      # _add_variable_with_custom_getter clears the shape sometimes, so we
      # take the global shape from outside the getter.
      return tf_variables.Variable(
          name=name,
          initial_value=initial_value,
          shape=variable_shape,
          dtype=dtype,
          trainable=trainable,
      )

    def variable_creator(name, initializer):
      # Use add_variable_with_custom_getter here so that we take advantage of
      # the checkpoint loading to allow restore before the variables get
      # created which avoids double initialization.
      return self._add_variable_with_custom_getter(
          name=name,
          initializer=initializer,
          shape=variable_shape,
          dtype=variable_dtype,
          getter=getter,
          trainable=False,
      )

    with variable_scope.variable_creator_scope(
        make_sharded_variable_creator(self._strategy)
    ):
      parameters = variable_creator(stacked_table_name, table_initialize_fn)

    def slot_creator(name, initializer):
      return variable_creator(stacked_table_name + "/" + name, initializer)

    if optimizer is not None:
      with variable_scope.variable_creator_scope(
          make_sharded_variable_creator(self._strategy)
      ):
        slot_vars = optimizer._create_slots(parameters, slot_creator)  # pylint: disable=protected-access
    else:
      slot_vars = {}
    slot_vars["parameters"] = parameters
    return slot_vars

  def _create_variables_and_slots(
      self,
  ) -> Dict[str, Dict[str, tf_variables.Variable]]:
    """Create variables for TPU embeddings.

    Returns:
      A dict of dicts. The outer dict is keyed by the table names and the inner
      dicts are keyed by 'parameters' and the slot variable names.
    """
    variables = {}
    for stacked_table_name, tables in self._stacked_table_to_tables.items():
      variables[stacked_table_name] = self._create_variables(
          tables, stacked_table_name=stacked_table_name
      )
    return variables

  def _track_restore_info_for_cpu(self) -> None:
    layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
    layouts.tables.extend(self.embedding_layouts.values())
    logging.info(
        "Saving sparse core layouts for %s tables", len(layouts.tables)
    )
    with ops.device("/cpu:0"):
      self._track_trackable(
          tpu_embedding_v3_utils.SparseCoreLayoutsTrackable(
              constant_op.constant(
                  layouts.SerializeToString(), dtype=dtypes.string
              )
          ),
          tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY,
      )

  def _checkpoint_adapter(self, path):
    # The TPUEmbedding may need to reshard checkpoint values during restore.
    return tpu_embedding_v3_checkpoint_adapter.TpuEmbeddingV3CheckpointAdapter.create_from_checkpoint(
        path
    )

  def _maybe_build(self):
    if not self._built:
      # This can be called while tracing a function, so we wrap the
      # initialization code with init_scope so it runs eagerly, this means that
      # it will not be included in the function graph generated by tracing so
      # that we can be sure that we only initialize the TPU for embeddings
      # exactly once.
      with ops.init_scope():
        self.build()

  def build(self):
    """Create variables and slots variables for TPU embeddings."""
    if self._built:
      return
    self._variables = self._create_variables_and_slots()
    self._track_restore_info_for_cpu()
    self._built = True

  def apply_gradients(
      self,
      gradients: Any,
      preserved_outputs: Dict[str, PartitionedCsrFormatTensor],
  ):
    """Applies the gradient update to the embedding tables.

    If a gradient of `None` is passed in any position of the nested structure,
    then a gradient update with a zero gradient is applied for that feature.
    For optimizers like SGD or Adagrad, this is the same as applying no update
    at all. For lazy Adam and other sparsely applied optimizers with decay,
    ensure you understand the effect of applying a zero gradient.

    Args:
      gradients: A nested structure of gradients, with structure matching the
        `feature_config` passed to this object.
      preserved_outputs: A dicts of PartitionedCsrFormatTensor, coming from the
        second output of the embedding lookup call.

    Raises:
      RuntimeError: if not built.
      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a
        `tf.Tensor` of the incorrect shape is passed in. Also if
        the size of any sequence in `gradients` does not match corresponding
        sequence in `feature_config`.
      TypeError: If the type of any sequence in `gradients` does not match
        corresponding sequence in `feature_config`.
    """
    if not self._built:
      raise RuntimeError(
          "apply_gradients called on unbuilt TPUEmbeddingV2 object. Please"
          " either call the embedding lookup method first or manually call the"
          " build method."
      )
    nest.assert_same_structure(self._feature_config, gradients)

    # Note that stacking gradients is placed on the core of the trianing step
    # to reduce the number of input/output arguments of the training loop during
    # pipelining.
    gradients = self._stack_gradients(gradients)

    context = EmbeddingPipeliningContext(
        _PIPELINE_MODE_BACKWARD, self._pipelining
    )
    context.Enter()

    def _wrap_param(param, dtype=dtypes.float32):
      if callable(param):
        param = math_ops.cast(param(), dtype=dtype)
      return ops.convert_to_tensor(param, dtype=dtype)

    # Take num_minibatches_per_physical_sparse_core from any table as
    # they are the same across tables.
    num_minibatches_per_physical_sparse_core = list(preserved_outputs.values())[
        0
    ].num_minibatches_per_physical_sparse_core

    for table_name in self._stacked_table_to_tables:
      gradient = gradients[table_name]
      partitioned_tensor = preserved_outputs[table_name]

      table = self.variables[table_name]["parameters"]
      optimizer = self._stacked_table_to_tables[table_name][0].optimizer
      if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
        updated_embedding_table = xla_ops.xla_sparse_dense_matmul_grad_with_sgd_and_static_buffer_size(
            row_pointers=partitioned_tensor.row_pointers,
            sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
            sorted_token_ids=partitioned_tensor.sorted_token_ids,
            sorted_gains=partitioned_tensor.sorted_gains,
            activation_gradients=gradient,
            learning_rate=_wrap_param(optimizer.learning_rate),
            embedding_table=table.read_value(),
            num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
            max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                table_name
            ],
            max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                table_name
            ],
            table_name=table_name,
        )
        table.assign(updated_embedding_table)
      elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
        accumulators = self.variables[table_name]["accumulators"]
        updated_embedding_table, updated_accumulator = (
            xla_ops.xla_sparse_dense_matmul_grad_with_adagrad_and_static_buffer_size(
                row_pointers=partitioned_tensor.row_pointers,
                sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
                sorted_token_ids=partitioned_tensor.sorted_token_ids,
                sorted_gains=partitioned_tensor.sorted_gains,
                activation_gradients=gradient,
                learning_rate=_wrap_param(optimizer.learning_rate),
                embedding_table=table.read_value(),
                accumulator=accumulators.read_value(),
                num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                    table_name
                ],
                max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                    table_name
                ],
                table_name=table_name,
            )
        )
        accumulators.assign(updated_accumulator)
        table.assign(updated_embedding_table)
      elif isinstance(optimizer, tpu_embedding_v2_utils.AdagradMomentum):
        accumulators = self.variables[table_name]["accumulators"]
        momenta = self.variables[table_name]["momenta"]
        updated_embedding_table, updated_accumulator, updated_momenta = (
            xla_ops.xla_sparse_dense_matmul_grad_with_adagrad_momentum_and_static_buffer_size(
                row_pointers=partitioned_tensor.row_pointers,
                sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
                sorted_token_ids=partitioned_tensor.sorted_token_ids,
                sorted_gains=partitioned_tensor.sorted_gains,
                activation_gradients=gradient,
                learning_rate=_wrap_param(optimizer.learning_rate),
                embedding_table=table.read_value(),
                accumulator=accumulators.read_value(),
                momenta=momenta.read_value(),
                num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                use_nesterov=optimizer.use_nesterov,
                exponent=optimizer.exponent,
                beta1=optimizer.momentum,
                beta2=optimizer.beta2,
                epsilon=optimizer.epsilon,
                max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                    table_name
                ],
                max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                    table_name
                ],
                table_name=table_name,
            )
        )
        momenta.assign(updated_momenta)
        accumulators.assign(updated_accumulator)
        table.assign(updated_embedding_table)
      elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
        momenta = self.variables[table_name]["momenta"]
        velocity = self.variables[table_name]["velocities"]
        updated_embedding_table, updated_momenta, updated_velocity = (
            xla_ops.xla_sparse_dense_matmul_grad_with_adam_and_static_buffer_size(
                row_pointers=partitioned_tensor.row_pointers,
                sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
                sorted_token_ids=partitioned_tensor.sorted_token_ids,
                sorted_gains=partitioned_tensor.sorted_gains,
                activation_gradients=gradient,
                learning_rate=_wrap_param(optimizer.learning_rate),
                embedding_table=table.read_value(),
                momenta=momenta.read_value(),
                velocity=velocity.read_value(),
                num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                use_sum_inside_sqrt=optimizer.sum_inside_sqrt,
                beta1=optimizer.beta_1,
                beta2=optimizer.beta_2,
                epsilon=optimizer.epsilon,
                max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                    table_name
                ],
                max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                    table_name
                ],
                table_name=table_name,
            )
        )
        velocity.assign(updated_velocity)
        momenta.assign(updated_momenta)
        table.assign(updated_embedding_table)
      elif isinstance(optimizer, tpu_embedding_v2_utils.FTRL):
        accumulators = self.variables[table_name]["accumulators"]
        linears = self.variables[table_name]["linears"]
        (updated_table_tensor, updated_accum_tensor, updated_linear_tensor) = (
            xla_ops.xla_sparse_dense_matmul_grad_with_ftrl_and_static_buffer_size(
                row_pointers=partitioned_tensor.row_pointers,
                sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
                sorted_token_ids=partitioned_tensor.sorted_token_ids,
                sorted_gains=partitioned_tensor.sorted_gains,
                activation_gradients=gradient,
                learning_rate=_wrap_param(optimizer.learning_rate),
                embedding_table=table.read_value(),
                accumulator=accumulators.read_value(),
                linear=linears.read_value(),
                num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
                multiply_linear_by_learning_rate=optimizer.multiply_linear_by_learning_rate,
                beta=optimizer.beta,
                learning_rate_power=optimizer.learning_rate_power,
                l1_regularization_strength=optimizer.l1_regularization_strength,
                l2_regularization_strength=optimizer.l2_regularization_strength,
                max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                    table_name
                ],
                max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                    table_name
                ],
                table_name=table_name,
            )
        )
        linears.assign(updated_linear_tensor)
        accumulators.assign(updated_accum_tensor)
        table.assign(updated_table_tensor)
      else:
        raise ValueError("Unsupported optimizer in minibatching mode.")

    context.Exit()

  def __call__(
      self, features: Any, weights: Optional[Any] = None
  ) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
    """Call the mid level api to do embedding lookup."""
    return self.embedding_lookup(features, weights)

  # TODO(pineapplejuice233): Duplicated helper function from tpu_embedding_v2.py. Remove
  # this once this file is open souced.
  def _raise_error_for_incorrect_control_flow_context(self):
    """Raises an error if we are not in the TPUReplicateContext."""
    # Do not allow any XLA control flow (i.e. control flow in between a
    # TPUStrategy's run call and the call to this function), as we can't
    # extract the enqueue from the head when in XLA control flow.
    graph = ops.get_default_graph()
    in_tpu_ctx = False
    while graph is not None:
      ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
      while ctx is not None:
        if isinstance(ctx, tpu_replication.TPUReplicateContext):
          in_tpu_ctx = True
          break
        ctx = ctx.outer_context
      if in_tpu_ctx:
        break
      graph = getattr(graph, "outer_graph", None)
    if graph != ops.get_default_graph() and in_tpu_ctx:
      raise RuntimeError(
          "Current graph {} does not match graph which contains "
          "TPUReplicateContext {}. This is most likely due to the fact that "
          "enqueueing embedding data is called inside control flow or a "
          "tf.function inside `strategy.run`. This is not supported because "
          "outside compilation fails to extract the enqueue ops as the head of "
          "a computation.".format(ops.get_default_graph(), graph)
      )
    return in_tpu_ctx

  @classmethod
  def compute_sparse_core_stats(
      cls,
      features: Any,
      feature_config: Union[FeatureConfig, Iterable],  # pylint:disable=g-bare-generic
      num_tpu_chips: int,
      num_sc_per_chip: int = 4,
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = None,  # pylint:disable=protected-access
      sparse_core_embedding_config: Optional[SparseCoreEmbeddingConfig] = None,
  ) -> Tuple[Any, Any]:
    """Computes the max_ids/unique ids settings from the input features."""
    copy_feature_config = _clone_feature_config(feature_config)
    table_config = []
    for feature in nest.flatten(copy_feature_config):
      table_config.append(feature.table)

    for table in table_config:
      if table.optimizer is None:
        table.optimizer = optimizer

    flat_features = nest.flatten_with_joined_string_paths(copy_feature_config)

    s = _stack_tables_with_same_table_dim_and_optimizer(
        table_config,
        flat_features,
        num_tpu_chips,
        num_sc_per_chip,
        sparse_core_embedding_config,
    )

    flat_inputs = nest.flatten(features)

    # First process them to be COO tensors.
    table_to_list_of_coos = (
        TPUEmbeddingV2._preprocess_inputs_and_weights_to_list_of_coo_tensors(
            flat_inputs=flat_inputs,
            flat_weights=[None] * len(flat_inputs),
            flat_features=flat_features,
            stacked_table_to_tables=s.stacked_table_to_tables,
            table_to_stacked_table_offset=s.table_to_stacked_table_offset,
            feature_to_sample_offset=s.feature_to_sample_offset,
            num_sc_per_chip=num_sc_per_chip,
            stacked_table_to_sample_count=s.table_to_sample_count,
            num_sc_shards=num_sc_per_chip * num_tpu_chips,
        )
    )

    table_to_max_ids_per_sparse_core = {
        table_name: 0 for table_name in s.stacked_table_to_tables
    }
    table_to_max_unique_ids_per_sparse_core = {
        table_name: 0 for table_name in s.stacked_table_to_tables
    }

    for table_name in s.stacked_table_to_tables:
      feature_width = s.stacked_table_to_tables[table_name][0].dim

      total_vocab_size = sum([
          table.vocabulary_size
          for table in s.stacked_table_to_tables[table_name]
      ])
      for i in range(num_sc_per_chip):
        row_ids_list = table_to_list_of_coos[table_name][0][i]
        col_ids_list = table_to_list_of_coos[table_name][1][i]
        gains_list = table_to_list_of_coos[table_name][2][i]
        sample_count_list = table_to_list_of_coos[table_name][3]
        col_offset_list = table_to_list_of_coos[table_name][4]

        (
            max_ids_per_sparse_core,
            max_unique_ids_per_sparse_core,
        ) = xla_ops.get_stats_from_list_of_sparse_core_coo_tensors(
            row_ids_list=row_ids_list,
            col_ids_list=col_ids_list,
            gains_list=gains_list,
            sample_count_list=sample_count_list,
            col_offset_list=col_offset_list,
            num_replica=num_tpu_chips,
            table_vocab_size=total_vocab_size,
            feature_width=feature_width,
            num_sc_per_chip=num_sc_per_chip,
            table_name=table_name,
        )

        table_to_max_ids_per_sparse_core[table_name] = math_ops.maximum(
            table_to_max_ids_per_sparse_core[table_name],
            max_ids_per_sparse_core,
        )
        table_to_max_unique_ids_per_sparse_core[table_name] = math_ops.maximum(
            table_to_max_unique_ids_per_sparse_core[table_name],
            max_unique_ids_per_sparse_core,
        )
    return (
        table_to_max_ids_per_sparse_core,
        table_to_max_unique_ids_per_sparse_core,
    )

  def enqueue(
      self,
      features: Any,
      weights: Optional[Any] = None,
      device: Optional[str] = None,
  ) -> Any:
    """Preprocessing the features on host."""
    nest.assert_same_structure(self._feature_config, features)

    flat_inputs = nest.flatten(features)
    flat_weights = [None] * len(flat_inputs)
    if weights is not None:
      nest.assert_same_structure(self._feature_config, weights)
      flat_weights = nest.flatten(weights)

    in_tpu_context = self._raise_error_for_incorrect_control_flow_context()

    if in_tpu_context:
      # Automatically apply outside compilation if we are in tpu context.
      return tpu_replication.outside_compilation(
          self._preprocess_features,
          num_replicas_in_sync=self._strategy.num_replicas_in_sync,
          max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample,
          max_minibatches_per_sc=self.max_minibatches_per_sc,
          num_sc_per_chip=self._num_sc_per_chip,
          num_sc_shards=self._num_sc_shards,
          stacked_table_to_tables=self._stacked_table_to_tables,
          table_to_stacked_table_offset=self._table_to_stacked_table_offset,
          table_to_sample_count=self._table_to_sample_count,
          feature_to_sample_offset=self._feature_to_sample_offset,
          flat_features=self._flat_features,
          flat_inputs=flat_inputs,
          flat_weights=flat_weights,
      )
    elif device is None:
      # This is used by keras function tracing. Use any of the TPU devices
      # and trace once for a single device.
      tpu_devices = self._strategy.extended._tpu_devices  # pylint:disable=protected-access

      with ops.device(device_util.get_host_for_device(tpu_devices[0][0])):
        return self._preprocess_features(
            num_replicas_in_sync=self._strategy.num_replicas_in_sync,
            max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample,
            max_minibatches_per_sc=self.max_minibatches_per_sc,
            num_sc_per_chip=self._num_sc_per_chip,
            num_sc_shards=self._num_sc_shards,
            stacked_table_to_tables=self._stacked_table_to_tables,
            table_to_stacked_table_offset=self._table_to_stacked_table_offset,
            table_to_sample_count=self._table_to_sample_count,
            feature_to_sample_offset=self._feature_to_sample_offset,
            flat_features=self._flat_features,
            flat_inputs=flat_inputs,
            flat_weights=flat_weights,
        )
    else:
      device_spec = tf_device.DeviceSpec.from_string(device)
      if device_spec.device_type != "TPU":
        raise ValueError("Non-TPU device {} passed to enqueue.".format(device))

      with ops.device(device_util.get_host_for_device(device)):
        return self._preprocess_features(
            num_replicas_in_sync=self._strategy.num_replicas_in_sync,
            max_ids_per_chip_per_sample=self.max_ids_per_chip_per_sample,
            max_minibatches_per_sc=self.max_minibatches_per_sc,
            num_sc_per_chip=self._num_sc_per_chip,
            num_sc_shards=self._num_sc_shards,
            stacked_table_to_tables=self._stacked_table_to_tables,
            table_to_stacked_table_offset=self._table_to_stacked_table_offset,
            table_to_sample_count=self._table_to_sample_count,
            feature_to_sample_offset=self._feature_to_sample_offset,
            flat_features=self._flat_features,
            flat_inputs=flat_inputs,
            flat_weights=flat_weights,
        )

  def _copy_tensors_to_device(
      self,
      partitioned_tensors: Dict[str, Any],
  ) -> Any:
    """Copy tensors to device."""
    partitioned_device_tensors = {}
    for table_name in partitioned_tensors:
      partitioned_tensor = partitioned_tensors[table_name][0]
      row_pointers_unpadded_size = partitioned_tensors[table_name][1]
      ids_unpadded_size = partitioned_tensors[table_name][2]

      row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains = (
          xla_ops.tpu_copy_with_dynamic_shape(
              [
                  partitioned_tensor.row_pointers,
                  partitioned_tensor.sorted_sample_ids,
                  partitioned_tensor.sorted_token_ids,
                  partitioned_tensor.sorted_gains,
              ],
              [
                  row_pointers_unpadded_size,
                  ids_unpadded_size,
                  ids_unpadded_size,
                  ids_unpadded_size,
              ],
          )
      )

      # Placeholder Op for pipelining.
      row_pointers, sorted_sample_ids, sorted_token_ids, sorted_gains = (
          xla_ops.tpu_annotate_tensors_with_dynamic_shape([
              row_pointers,
              sorted_sample_ids,
              sorted_token_ids,
              sorted_gains,
          ])
      )

      partitioned_device_tensors[table_name] = PartitionedCsrFormatTensor(
          row_pointers=row_pointers,
          sorted_sample_ids=sorted_sample_ids,
          sorted_token_ids=sorted_token_ids,
          sorted_gains=sorted_gains,
          sample_count=partitioned_tensor.sample_count,
          num_minibatches_per_physical_sparse_core=(
              partitioned_tensor.num_minibatches_per_physical_sparse_core
          ),
      )
    return partitioned_device_tensors

  def dequeue(
      self,
      partitioned_tensors: Tuple[
          Dict[str, PartitionedCsrFormatTensor], int, int
      ],
  ) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
    """Perform embedding lookup."""
    # We expect this dequeue function will always run inside tpu context.
    context = EmbeddingPipeliningContext(
        _PIPELINE_MODE_FORWARD, self._pipelining
    )
    context.Enter()
    partitioned_tensors = tpu_replication.outside_compilation(
        self._copy_tensors_to_device,
        partitioned_tensors=partitioned_tensors,
    )

    activations = {}
    # Take num_minibatches_per_physical_sparse_core from any table as
    # they are the same across tables.
    num_minibatches_per_physical_sparse_core = list(
        partitioned_tensors.values()
    )[0].num_minibatches_per_physical_sparse_core

    for table_name in self._stacked_table_to_tables:
      partitioned_tensor = partitioned_tensors[table_name]

      table = self.variables[table_name]["parameters"]
      quantization_config = self._quantization_configs[table_name]
      if not isinstance(partitioned_tensor, PartitionedCsrFormatTensor):
        raise ValueError(
            "Expect PartitionedCsrFormatTensor but get"
            f" {type(partitioned_tensor)}."
        )
      activation = xla_ops.xla_sparse_dense_matmul_with_static_buffer_size(
          row_pointers=partitioned_tensor.row_pointers,
          sorted_sample_ids=partitioned_tensor.sorted_sample_ids,
          sorted_token_ids=partitioned_tensor.sorted_token_ids,
          sorted_gains=partitioned_tensor.sorted_gains,
          input_size=self._table_to_sample_count[table_name],
          embedding_table=table,
          num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
          quantization_config_low=(
              quantization_config.lower if quantization_config else 0
          ),
          quantization_config_high=(
              quantization_config.upper if quantization_config else 0
          ),
          quantization_config_num_buckets=(
              quantization_config.num_buckets if quantization_config else 0
          ),
          max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
              table_name
          ],
          max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
              table_name
          ],
          table_name=table_name,
      )

      activations[table_name] = activation

    context.Exit()
    # Note that unstacking gradients is placed on the core of the trianing step
    # to reduce the number of input/output arguments of the training loop during
    # pipelining.
    activations = self._unstack_activations(activations)

    return (activations, partitioned_tensors)

  def embedding_lookup(
      self, features: Any, weights: Optional[Any] = None
  ) -> Tuple[Any, Dict[str, PartitionedCsrFormatTensor]]:
    """Perform embedding lookup on the input feature.

    Args:
      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or
        `tf.RaggedTensor`s, with the same structure as `feature_config`. Inputs
        will be downcast to `tf.int32`. Only one type out of `tf.SparseTensor`
        or `tf.RaggedTensor` is supported per call.
      weights: If not `None`, a nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except
        that the tensors should be of float type (and they will be downcast to
        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the
        same for the parallel entries from `features` and similarly for
        `tf.RaggedTensor`s we assume the row_splits are the same.

    Raises:
      ValueError: If the input feature is not one of the Tensor, SparseTensor or
        RaggedTensor type.
      TypeError: If the type of any sequence in `features` does not match
        corresponding sequence in `feature_config`. Similarly for `weights`, if
        not `None`.

    Returns:
      packed_activations: Embedding lookup results packed as the same sequence
        of the input feature.
      packed_output: A dict of PartitionedCsrFormatTensors.
    """
    if not self._built:
      self._maybe_build()

    context = EmbeddingPipeliningContext(
        _PIPELINE_MODE_FORWARD, self._pipelining
    )
    context.Enter()

    partitioned_tensors = self.enqueue(features, weights)

    context.Exit()

    result = self.dequeue(partitioned_tensors)

    return result

  def _preprocess_features(
      self,
      num_replicas_in_sync: int,
      max_ids_per_chip_per_sample: int,
      max_minibatches_per_sc: int,
      num_sc_per_chip: int,
      num_sc_shards: int,
      stacked_table_to_tables: Dict[str, Any],
      table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]],
      table_to_sample_count: Dict[str, int],
      feature_to_sample_offset: Dict[str, int],
      flat_features: Any,
      flat_inputs: Any,
      flat_weights: Optional[Any] = None,
  ) -> Any:
    """Function to preprocess features."""
    # Preprocess the inputs into list of COO tensor.
    table_to_list_of_coos = (
        TPUEmbeddingV2._preprocess_inputs_and_weights_to_list_of_coo_tensors(
            flat_inputs,
            flat_weights,
            flat_features,
            stacked_table_to_tables,
            table_to_stacked_table_offset,
            feature_to_sample_offset,
            num_sc_per_chip,
            table_to_sample_count,
            num_sc_shards,
        )
    )

    # Sort the COO tensors.
    table_to_sorted_coo_tensor = self._sort_list_of_coo_tensors(
        num_replicas_in_sync,
        table_to_list_of_coos,
        stacked_table_to_tables,
        num_sc_per_chip,
    )

    table_to_csr_format_tensor = (
        self._get_csr_wrapped_coo_from_sorted_coo_tensor(
            num_replicas_in_sync,
            max_ids_per_chip_per_sample,
            max_minibatches_per_sc,
            table_to_sorted_coo_tensor,
            stacked_table_to_tables,
            table_to_sample_count,
            num_sc_per_chip,
        )
    )

    return table_to_csr_format_tensor

  @classmethod
  def _convert_input_feature_to_list_of_coo_tensors(
      cls,
      input_feature: Union[
          tensor.Tensor, sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor
      ],
      weight: Optional[tensor.Tensor],
      feature_config: tpu_embedding_v2_utils.FeatureConfig,
      row_offset: int,
      col_offset: int,
      col_shift: int,
      vocab_size: int,
      num_sc_per_chip: int,
      num_sc_shards: int,
      stacked_table_sample_count: int,
  ) -> Any:
    """Convert any of the expected input types to a COO format."""
    sample_count = functools.reduce(operator.mul, feature_config.output_shape)
    if isinstance(input_feature, tensor.Tensor):
      input_feature = array_ops.reshape(input_feature, [-1])
      if weight is None:
        weight = array_ops.ones_like(input_feature, dtype=dtypes.float32)
      elif isinstance(weight, tensor.Tensor):
        weight = array_ops.reshape(weight, [-1])
      else:
        raise ValueError(
            f"Expect weight to be Tensor type but got {type(weight)}"
        )
      row_ids_list, col_ids_list, gains_list = (
          xla_ops.convert_to_list_of_sparse_core_coo_tensors(
              indices_or_row_splits=array_ops.zeros((0,), dtype=dtypes.int32),
              values=math_ops.cast(input_feature, dtype=dtypes.int32),
              weights=math_ops.cast(weight, dtypes.float32),
              sample_count=sample_count,
              combiner=feature_config.table.combiner,
              num_sc_per_chip=num_sc_per_chip,
              row_offset=row_offset,
              col_offset=col_offset,
              col_shift=col_shift,
              num_sc_shards=num_sc_shards,
              stacked_table_sample_count=stacked_table_sample_count,
          )
      )
    elif isinstance(input_feature, sparse_tensor.SparseTensor):
      if weight is None:
        weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
      elif isinstance(weight, sparse_tensor.SparseTensor):
        weight = weight.values
      else:
        raise ValueError(
            f"Expect weight to be SparseTensor type but got {type(weight)}"
        )
      row_ids_list, col_ids_list, gains_list = (
          xla_ops.convert_to_list_of_sparse_core_coo_tensors(
              indices_or_row_splits=math_ops.cast(
                  input_feature.indices, dtype=dtypes.int32
              ),
              values=math_ops.cast(input_feature.values, dtype=dtypes.int32),
              weights=math_ops.cast(weight, dtypes.float32),
              sample_count=sample_count,
              combiner=feature_config.table.combiner,
              num_sc_per_chip=num_sc_per_chip,
              row_offset=row_offset,
              col_offset=col_offset,
              col_shift=col_shift,
              num_sc_shards=num_sc_shards,
              stacked_table_sample_count=stacked_table_sample_count,
          )
      )
    elif isinstance(input_feature, ragged_tensor.RaggedTensor):
      if not weight:
        weight = array_ops.ones_like(input_feature.values, dtype=dtypes.float32)
      elif isinstance(weight, ragged_tensor.RaggedTensor):
        weight = weight.values
      else:
        raise ValueError(
            f"Expect weight to be RaggedTensor type but got {type(weight)}"
        )
      row_ids_list, col_ids_list, gains_list = (
          xla_ops.convert_to_list_of_sparse_core_coo_tensors(
              indices_or_row_splits=math_ops.cast(
                  input_feature.row_splits, dtype=dtypes.int32
              ),
              values=math_ops.cast(input_feature.values, dtype=dtypes.int32),
              weights=math_ops.cast(weight, dtypes.float32),
              sample_count=sample_count,
              combiner=feature_config.table.combiner,
              num_sc_per_chip=num_sc_per_chip,
              row_offset=row_offset,
              col_offset=col_offset,
              col_shift=col_shift,
              num_sc_shards=num_sc_shards,
              stacked_table_sample_count=stacked_table_sample_count,
          )
      )
    else:
      raise ValueError(
          f"Input of unknown type {type(input_feature)}. Please only pass "
          "Tensor, SparseTensor or RaggedTensor as input to embedding "
          "lookup."
      )
    return row_ids_list, col_ids_list, gains_list, sample_count

  @classmethod
  def _preprocess_inputs_and_weights_to_list_of_coo_tensors(
      cls,
      flat_inputs: Any,
      flat_weights: Any,
      flat_features: Any,
      stacked_table_to_tables: Dict[str, Any],
      table_to_stacked_table_offset: Dict[str, Tuple[str, int, int]],
      feature_to_sample_offset: Dict[str, int],
      num_sc_per_chip: int,
      stacked_table_to_sample_count: Dict[str, int],
      num_sc_shards: int,
  ) -> Dict[str, Any]:
    """Convert the raw inputs into list of coo tensors."""
    table_to_list_of_coos = {  # pylint: disable=g-complex-comprehension
        table_name: (
            [[], [], [], []],
            [[], [], [], []],
            [[], [], [], []],
            [],
            [],
        )
        for table_name in stacked_table_to_tables
    }
    for inp, weight, (feature_path, feature) in zip(
        flat_inputs, flat_weights, flat_features
    ):
      table_name, col_offset, col_shift = table_to_stacked_table_offset[
          feature.table.name
      ]
      stacked_table_sample_count = stacked_table_to_sample_count[table_name]
      row_offset = feature_to_sample_offset[feature_path]
      # Consider making this into one op per table rather than per feature?
      row_ids_list, col_ids_list, gains_list, sample_count = (
          TPUEmbeddingV2._convert_input_feature_to_list_of_coo_tensors(
              inp,
              weight,
              feature,
              row_offset,
              col_offset,
              col_shift,
              feature.table.vocabulary_size,
              num_sc_per_chip,
              num_sc_shards,
              stacked_table_sample_count,
          )
      )
      for i in range(num_sc_per_chip):
        table_to_list_of_coos[table_name][0][i].append(row_ids_list[i])
        table_to_list_of_coos[table_name][1][i].append(col_ids_list[i])
        table_to_list_of_coos[table_name][2][i].append(gains_list[i])
      table_to_list_of_coos[table_name][3].append(
          sample_count // num_sc_per_chip
      )
      table_to_list_of_coos[table_name][4].append(col_offset)
    return table_to_list_of_coos

  def _sort_list_of_coo_tensors(
      self,
      num_replicas_in_sync: int,
      table_to_list_of_coos: Dict[str, Any],
      stacked_table_to_tables: Dict[str, Any],
      num_sc_per_chip: int,
  ) -> Tuple[Dict[str, Any], List[tensor.Tensor]]:
    """Sort the coo tensors by replica."""
    table_to_sorted_coo_tensor = {
        table_name: ([], [], [], []) for table_name in stacked_table_to_tables
    }
    for table_name in stacked_table_to_tables:
      # Feature width are the same across stacked tables.
      feature_width = stacked_table_to_tables[table_name][0].dim

      total_vocab_size = sum([
          table.vocabulary_size for table in stacked_table_to_tables[table_name]
      ])
      for i in range(num_sc_per_chip):
        row_ids_list = table_to_list_of_coos[table_name][0][i]
        col_ids_list = table_to_list_of_coos[table_name][1][i]
        gains_list = table_to_list_of_coos[table_name][2][i]
        sample_count_list = table_to_list_of_coos[table_name][3]
        col_offset_list = table_to_list_of_coos[table_name][4]

        (
            sorted_row_ids,
            sorted_col_ids,
            sorted_gains,
            id_counts,
        ) = xla_ops.sort_list_of_sparse_core_coo_tensors(
            row_ids_list=row_ids_list,
            col_ids_list=col_ids_list,
            gains_list=gains_list,
            sample_count_list=sample_count_list,
            col_offset_list=col_offset_list,
            num_replica=num_replicas_in_sync,
            table_vocab_size=total_vocab_size,
            feature_width=feature_width,
            num_sc_per_chip=num_sc_per_chip,
            max_ids_per_sparse_core=self._table_to_max_ids_per_sparse_core[
                table_name
            ],
            max_unique_ids_per_sparse_core=self._table_to_max_unique_ids_per_sparse_core[
                table_name
            ],
            table_name=table_name,
        )

        table_to_sorted_coo_tensor[table_name][0].append(sorted_row_ids)
        table_to_sorted_coo_tensor[table_name][1].append(sorted_col_ids)
        table_to_sorted_coo_tensor[table_name][2].append(sorted_gains)
        table_to_sorted_coo_tensor[table_name][3].append(id_counts)

    return table_to_sorted_coo_tensor

  def _get_csr_wrapped_coo_from_sorted_coo_tensor(
      self,
      num_replicas_in_sync: int,
      max_ids_per_chip_per_sample: int,
      max_minibatches_per_sc: int,
      table_to_sorted_coo_tensor: Dict[str, Any],
      stacked_table_to_tables: Dict[str, Any],
      table_to_sample_count: Dict[str, int],
      num_sc_per_chip: int,
  ) -> Any:
    """Get csr wrapped coo tensor from the sorted coo tensor."""
    table_to_csr_format_tensor = {}
    for table_name in stacked_table_to_tables:
      (
          sorted_row_ids_list,
          sorted_col_ids_list,
          sorted_gains_list,
          id_counts_list,
      ) = table_to_sorted_coo_tensor[table_name]

      # Feature width are the same across stacked tables.
      feature_width = stacked_table_to_tables[table_name][0].dim

      total_vocab_size = sum([
          table.vocabulary_size for table in stacked_table_to_tables[table_name]
      ])
      (
          row_pointers,
          sorted_sample_ids,
          sorted_token_ids,
          sorted_gains,
          row_pointers_unpadded_size,
          ids_unpadded_size,
          num_minibatches_per_physical_sparse_core,
      ) = xla_ops.convert_to_sparse_core_csr_wrapped_coo_tensor(
          sorted_row_ids_list=sorted_row_ids_list,
          sorted_col_ids_list=sorted_col_ids_list,
          sorted_gains_list=sorted_gains_list,
          id_counts_list=id_counts_list,
          splits=constant_op.constant(
              0, dtype=dtypes.int64
          ),  # no splits are needed.
          sample_count_per_sc=table_to_sample_count[table_name]
          // num_sc_per_chip,
          num_replica=num_replicas_in_sync,
          max_minibatches_per_sc=max_minibatches_per_sc,
          max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
          table_vocab_size=total_vocab_size,
          feature_width=feature_width,
          table_name=table_name,
          allow_id_dropping=self._sparse_core_embedding_config.allow_id_dropping,
      )
      table_to_csr_format_tensor[table_name] = (
          PartitionedCsrFormatTensor(
              row_pointers=row_pointers,
              sorted_sample_ids=sorted_sample_ids,
              sorted_token_ids=sorted_token_ids,
              sorted_gains=sorted_gains,
              sample_count=table_to_sample_count[table_name],
              num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
          ),
          row_pointers_unpadded_size,
          ids_unpadded_size,
      )
    return table_to_csr_format_tensor

  def _unstack_activations(self, activations: Dict[str, tensor.Tensor]):
    """Untack the incoming per table activations into per feature."""

    # Activations are stacked in a particular order. That order is the order
    # features appear in the self._flat_features.
    flattened_activations = []
    table_to_current_offset = {
        table_name: 0 for table_name in self._stacked_table_to_tables
    }
    for table_name in self._stacked_table_to_tables:
      activation_shape = activations[table_name].shape
      activations[table_name] = array_ops.reshape(
          activations[table_name],
          [self._num_sc_per_chip, -1, activation_shape[-1]],
      )
    for _, feature in self._flat_features:
      sample_count = functools.reduce(operator.mul, feature.output_shape)
      table_name = self._table_to_stacked_table_offset[feature.table.name][0]
      extra_cols = self._table_to_padding_columns[feature.table.name]
      activation = array_ops.slice(
          activations[table_name],
          [0, table_to_current_offset[table_name], 0],
          [
              self._num_sc_per_chip,
              sample_count // self._num_sc_per_chip,
              feature.table.dim - extra_cols,
          ],
      )
      # Reshape to follow the user's requested output shape.
      activation = array_ops.reshape(
          activation,
          list(feature.output_shape) + [feature.table.dim - extra_cols],
      )
      flattened_activations.append(activation)
      table_to_current_offset[table_name] += (
          sample_count // self._num_sc_per_chip
      )

    return nest.pack_sequence_as(self._feature_config, flattened_activations)

  def _stack_gradients(self, gradients):
    """Stack the incoming gradients to per table gradients."""

    # Gradients are stacked in a particular order. That order is the order
    # features appear in the self._flat_features.
    table_to_gradient_list = {
        table_name: [[], [], [], []]
        for table_name in self._stacked_table_to_tables
    }
    flattend_gradients = nest.flatten(gradients)
    for gradient, (path, feature) in zip(
        flattend_gradients, self._flat_features
    ):
      sample_count = functools.reduce(operator.mul, feature.output_shape)
      if gradient is not None and not isinstance(gradient, tensor.Tensor):
        raise ValueError(
            f"found non-tensor type: {type(gradient)} at path {path}."
        )
      if gradient is None:
        # TODO(bfontain): In the case that an entire table's gradient is gone
        # then maybe we can just omit the update all together?
        logging.warning(
            (
                "No gradient passed for feature %s, sending zero "
                "gradient. This may not be correct behavior for certain "
                "optimizers like Adam."
            ),
            path,
        )
        gradient = array_ops.zeros(
            (sample_count, feature.table.dim), dtype=dtypes.float32
        )
      table_name = self._table_to_stacked_table_offset[feature.table.name][0]
      extra_cols = self._table_to_padding_columns[feature.table.name]
      gradient = array_ops.reshape(
          gradient, [-1, feature.table.dim - extra_cols]
      )
      if extra_cols != 0:
        gradient = array_ops.pad(gradient, [[0, 0], [0, extra_cols]])
        # Ensure static shape after padding.
        gradient.set_shape([sample_count, feature.table.dim])
      per_sc_sample_count = sample_count // self._num_sc_per_chip
      for i in range(self._num_sc_per_chip):
        table_to_gradient_list[table_name][i].append(
            array_ops.slice(
                gradient,
                [i * per_sc_sample_count, 0],
                [
                    per_sc_sample_count,
                    feature.table.dim,
                ],
            )
        )
    for table_name in table_to_gradient_list:
      table_to_gradient_list[table_name] = array_ops.concat(
          [
              array_ops.concat(table_to_gradient_list[table_name][i], axis=0)
              for i in range(self._num_sc_per_chip)
          ],
          axis=0,
      )
    return table_to_gradient_list


# TODO(pineapplejuice233): Merge this function with the one in tpu_embeding_v2.py once
# this file is OSSed.
def extract_variable_info(
    kwargs: Any,
) -> Tuple[
    str, Tuple[int, ...], dtypes.DType, Callable[[], Any], Optional[int]
]:
  """Extracts the variable creation attributes from the kwargs.

  Args:
    kwargs: a dict of keyword arguments that were passed to a variable creator
      scope.

  Returns:
    A tuple of variable name, shape, dtype, initialization function,
    restore_uid.
  """

  def get_restore_uid(initial_value: Callable[..., Any]) -> int | None:
    return getattr(initial_value, "restore_uid", None)

  if isinstance(kwargs["initial_value"], functools.partial) and (
      "shape" in kwargs["initial_value"].keywords
      or kwargs["initial_value"].args
  ):
    # Sometimes shape is passed positionally, sometimes it's passed as a kwarg.
    if "shape" in kwargs["initial_value"].keywords:
      shape = kwargs["initial_value"].keywords["shape"]
    else:
      shape = kwargs["initial_value"].args[0]
    return (
        kwargs["name"],
        shape,
        kwargs["initial_value"].keywords.get("dtype", kwargs["dtype"]),
        kwargs["initial_value"].func,
        get_restore_uid(kwargs["initial_value"].func),
    )
  elif (
      "shape" not in kwargs
      or kwargs["shape"] is None
      or not callable(kwargs["initial_value"])
  ):
    raise ValueError(
        "Unable to extract initializer function and shape from {}. Please "
        "either pass a function that expects a shape and dtype as the "
        "initial value for your variable or functools.partial object with "
        "the shape and dtype kwargs set. This is needed so that we can "
        "initialize the shards of the ShardedVariable locally.".format(
            kwargs["initial_value"]
        )
    )
  else:
    return (
        kwargs["name"],
        kwargs["shape"],
        kwargs["dtype"],
        kwargs["initial_value"],
        get_restore_uid(kwargs["initial_value"]),
    )


def is_checkpoint_initial_value(initial_value: Any) -> bool:
  """Whether the initial value is from checkpoint."""
  return (
      isinstance(initial_value, base.CheckpointInitialValue)
      or isinstance(initial_value, base.CheckpointInitialValueCallable)
      or (
          isinstance(initial_value, functools.partial)
          and isinstance(
              initial_value.func, base.CheckpointInitialValueCallable
          )
      )
  )


def make_sharded_variable_creator(
    strategy: distribute_lib.Strategy,
) -> Callable[..., Any]:
  """Create a variable creator which shards across all the tpu device.

  Args:
    strategy: a TPUStrategy object.

  Returns:
    The sharded variable creator.
  """
  tpu_devices = strategy.extended._tpu_devices  # pylint:disable=protected-access

  def _create_sharded_variable(next_creator, *args, **kwargs):
    """Create a TPUEmbeddingShardedVariable."""
    # Avoid the default mirror variable creator.
    kwargs["skip_mirrored_creator"] = True

    # Only support sharding on the first dimension.
    shard_dim = 0

    num_replicas, num_cores_per_replica = tpu_devices.shape

    is_ckpt_init_value = is_checkpoint_initial_value(kwargs["initial_value"])

    arg_spec = tf_inspect.getfullargspec(kwargs["initial_value"])
    if (
        is_ckpt_init_value
        and "shard_info" not in arg_spec.args
        and "shard_info" not in arg_spec.kwonlyargs
    ):
      raise ValueError(
          "When a sharded variable is initialized from a checkpoint, "
          "shard_info must be in arguments of the init function."
      )

    name, shape, dtype, unwrapped_initial_value, restore_uid = (
        extract_variable_info(kwargs)
    )

    shape = ops.tensor_shape.TensorShape(shape)
    num_devices = num_replicas * num_cores_per_replica
    # NOTE: only support sharding variables evenly across devices.
    if shape[shard_dim] % num_devices != 0:
      raise ValueError(
          "Only evenly sharding across devices is currently supported. "
          "Got shape {} and {} devices".format(shape, num_devices)
      )

    partition_shape = shape.as_list()
    partition_shape[shard_dim] = partition_shape[shard_dim] // num_devices

    unwrapped_arg_spec = tf_inspect.getargspec(unwrapped_initial_value)
    sharding_aware = "shard_info" in unwrapped_arg_spec.args

    variables = []
    # Keep track of offset for sharding aware initializers.
    partition_offset = [0] * len(shape)
    for replica_id in range(num_replicas):
      for logic_core_id in range(num_cores_per_replica):
        with ops.device(tpu_devices[replica_id][logic_core_id]):
          kwargs["name"] = f"{name}/{replica_id}"
          kwargs["shape"] = partition_shape
          if sharding_aware:
            # TODO(pineapplejuice233): Change this to use MOD sharding logic.
            shard_info = base.ShardInfo(
                tensor_shape.as_shape(partition_shape),
                copy.deepcopy(partition_offset),
            )
            kwargs["initial_value"] = functools.partial(
                kwargs["initial_value"], shard_info=shard_info
            )
            partition_offset[shard_dim] += partition_shape[shard_dim]
          else:
            kwargs["initial_value"] = functools.partial(
                unwrapped_initial_value, shape=partition_shape, dtype=dtype
            )
          variables.append(next_creator(*args, **kwargs))

    result = TPUEmbeddingShardedVariable(
        strategy, variables, tf_variables.VariableAggregation.NONE, None
    )
    if restore_uid is not None:
      result._maybe_initialize_trackable()  # pylint: disable=protected-access
      result._update_uid = restore_uid  # pylint: disable=protected-access
    return result

  return _create_sharded_variable
