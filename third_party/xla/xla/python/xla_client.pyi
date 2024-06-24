# Copyright 2021 The OpenXLA Authors.
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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import enum
from typing import Any, Union

import numpy

from . import xla_extension as _xla
from .xla_extension import ArrayImpl as ArrayImpl
from .xla_extension import Client as Client
from .xla_extension import CompileOptions as CompileOptions
from .xla_extension import Device as Device
from .xla_extension import DeviceAssignment as DeviceAssignment
from .xla_extension import DeviceList as DeviceList
from .xla_extension import DeviceTopology as DeviceTopology
from .xla_extension import DistributedRuntimeClient as DistributedRuntimeClient
from .xla_extension import FftType as FftType
from .xla_extension import Frame as Frame
from .xla_extension import GSPMDSharding as GSPMDSharding
from .xla_extension import HloSharding as HloSharding
from .xla_extension import HostBufferSemantics as HostBufferSemantics
from .xla_extension import ifrt_programs as ifrt_programs
from .xla_extension import Layout as Layout
from .xla_extension import LoadedExecutable as LoadedExecutable
from .xla_extension import Memory as Memory
from .xla_extension import NamedSharding as NamedSharding
from .xla_extension import ops as ops
from .xla_extension import OpSharding as OpSharding
from .xla_extension import PjRtLayout as PjRtLayout
from .xla_extension import PmapSharding as PmapSharding
from .xla_extension import PrimitiveType as PrimitiveType
from .xla_extension import profiler as profiler
from .xla_extension import Shape as Shape
from .xla_extension import Sharding as Sharding
from .xla_extension import SingleDeviceSharding as SingleDeviceSharding
from .xla_extension import Traceback as Traceback
from .xla_extension import XlaBuilder as XlaBuilder
from .xla_extension import XlaComputation as XlaComputation
from .xla_extension import XlaOp as XlaOp

_version: int

mlir_api_version: int

bfloat16: type[numpy.generic]
float8_e4m3fn: type[numpy.generic]
float8_e4m3b11fnuz: type[numpy.generic]
float8_e4m3fnuz: type[numpy.generic]
float8_e5m2: type[numpy.generic]
float8_e5m2fnuz: type[numpy.generic]
XLA_ELEMENT_TYPE_TO_DTYPE: dict[PrimitiveType, numpy.dtype]

_NameValueMapping = Mapping[str, Union[str, int, list[int], float, bool]]

def dtype_to_etype(dtype: numpy.dtype) -> PrimitiveType:
  ...

def execute_with_python_values(executable: LoadedExecutable, arguments: Sequence[Any],
                               backend: Client) -> Sequence[numpy.ndarray]: ...

def execute_with_python_values_replicated(
    executable: LoadedExecutable, arguments: Sequence[Sequence[Any]],
    backend: Client) -> Sequence[Sequence[numpy.ndarray]]: ...

def shape_from_pyval(pyval: Any, layout: Sequence[int] | None = None) -> Any: ...

def heap_profile(client: Client) -> bytes:
  ...

XlaRuntimeError = _xla.XlaRuntimeError

def make_cpu_client(
    asynchronous: bool = ...,
    distributed_client: DistributedRuntimeClient | None = ...,
    node_id: int = ...,
    num_nodes: int = ...,
    collectives: _xla.CpuCollectives | None = ...,
) -> Client:
  ...

def make_gpu_client(
    distributed_client: DistributedRuntimeClient | None = ...,
    node_id: int = ...,
    num_nodes: int = ...,
    platform_name: str | None = ...,
    allowed_devices: set[int] | None = ...,
    mock: bool | None = ...,
) -> Client:
  ...

def make_tfrt_tpu_c_api_client(options: _NameValueMapping | None = None) -> Client:
  ...

def make_tfrt_tpu_c_api_device_topology(
    topology_name: str | None = None, **kwargs
) -> DeviceTopology:
  ...

def make_c_api_device_topology(c_api: Any, topology_name: str = '', **kwargs) -> DeviceTopology:
  ...

def get_topology_for_devices(devices: list[Device]) -> DeviceTopology:
  ...

def make_tpu_client(
    library_path: str | None, options: _NameValueMapping | None = None
) -> Client:
  ...

def make_c_api_client(
    plugin_name: str,
    options: _NameValueMapping | None = None,
    distributed_client: DistributedRuntimeClient | None = None,
) -> Client:
  ...

def pjrt_plugin_loaded(plugin_name: str) -> bool:
  ...

def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> Any:
  ...

def load_pjrt_plugin_with_c_api(plugin_name: str, c_api: Any) -> None:
  ...

def pjrt_plugin_initialized(plugin_name: str) -> bool:
  ...

def initialize_pjrt_plugin(plugin_name: str) -> None:
  ...

def generate_pjrt_gpu_plugin_options() -> _NameValueMapping:
  ...

class OpMetadata:

  def __init__(
      self,
      op_type: str | None = ...,
      op_name: str | None = ...,
      source_file: str | None = ...,
      source_line: int | None = ...,
  ):
    ...
  op_type: str | None
  op_name: str | None
  source_file: str | None
  source_line: int | None

class PaddingConfigDimension:
  edge_padding_low: int
  edge_padding_high: int
  interior_padding: int

class PaddingConfig:
  dimensions: list[PaddingConfigDimension]

def make_padding_config(
    padding_config: Union[PaddingConfig, Sequence[tuple[int, int, int]]],
) -> PaddingConfig:
  ...

class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2

class DotDimensionNumbers:
  lhs_contracting_dimensions: list[int]
  rhs_contracting_dimensions: list[int]
  lhs_batch_dimensions: list[int]
  rhs_batch_dimensions: list[int]

def make_dot_dimension_numbers(
    dimension_numbers: Union[
        DotDimensionNumbers,
        tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]],
    ],
) -> DotDimensionNumbers:
  ...

class ConvolutionDimensionNumbers:
  input_batch_dimension: int
  input_feature_dimension: int
  input_spatial_dimensions: list[int]
  kernel_input_feature_dimension: int
  kernel_output_feature_dimension: int
  kernel_spatial_dimensions: list[int]
  output_batch_dimension: int
  output_feature_dimension: int
  output_spatial_dimensions: list[int]

def make_convolution_dimension_numbers(
    dimension_numbers: Union[
        None, ConvolutionDimensionNumbers, tuple[str, str, str]
    ],
    num_spatial_dimensions: int,
) -> ConvolutionDimensionNumbers:
  ...

class PrecisionConfig:
  Precision = _xla.PrecisionConfig_Precision
  operand_precision: list[_xla.PrecisionConfig_Precision]

class GatherDimensionNumbers:
  offset_dims: list[int]
  collapsed_slice_dims: list[int]
  start_index_map: list[int]
  index_vector_dim: int

class ScatterDimensionNumbers:
  update_window_dims: list[int]
  inserted_window_dims: list[int]
  scatter_dims_to_operand_dims: list[int]
  index_vector_dim: int

class ReplicaGroup:
  replica_ids: list[int]

def make_replica_groups(
    replica_groups: Sequence[Sequence[int]] | None,
) -> list[ReplicaGroup]:
  ...

def weakref_lru_cache(cache_context_fn: Callable, call: Callable, maxsize=...):
  ...

def batched_copy_array_to_devices_with_sharding(
    arrays: Sequence[ArrayImpl],
    devices: Sequence[list[Device]],
    sharding: Sequence[Any],
) -> Sequence[ArrayImpl]: ...

def batched_device_put(
    aval: Any,
    sharding: Any,
    shards: Sequence[Any],
    devices: list[Device],
    committed: bool = ...,
    force_copy: bool = ...,
    host_buffer_semantics: Any = ...,
) -> ArrayImpl: ...

def batched_block_until_ready(x: Sequence[ArrayImpl]) -> None: ...

def check_and_canonicalize_memory_kind(
    memory_kind: str | None, device_list: DeviceList
) -> str | None: ...

def array_result_handler(
               aval: Any,
               sharding: Any,
               committed: bool,
               _skip_checks: bool = ...) -> Callable:
  ...

class CustomCallTargetTraits(enum.IntFlag):
  DEFAULT = 0
  COMMAND_BUFFER_COMPATIBLE = 1

def register_custom_call_target(
    name: str,
    fn: Any,
    platform: str = ...,
    api_version: int = ...,
    traits: CustomCallTargetTraits = ...,
) -> None: ...

def register_custom_call_handler(
    xla_platform_name: str, handler: Any
) -> None: ...

def custom_call_targets(platform: str) -> dict[str, Any]: ...

def encode_inspect_sharding_callback(handler: Any) -> bytes: ...
