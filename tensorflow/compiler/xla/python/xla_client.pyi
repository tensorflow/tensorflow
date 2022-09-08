# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy

from . import xla_extension as _xla
from .xla_extension import Shape as Shape
from .xla_extension import Layout as Layout
from .xla_extension import ops as ops
from .xla_extension import profiler as profiler

from .xla_extension import Buffer as Buffer
from .xla_extension import ShardedBuffer as ShardedBuffer
from .xla_extension import Array as Array
from .xla_extension import Client as Client
from .xla_extension import CompileOptions as CompileOptions
from .xla_extension import Device as Device
from .xla_extension import DeviceArrayBase as DeviceArrayBase
from .xla_extension import DeviceAssignment as DeviceAssignment
from .xla_extension import DistributedRuntimeClient as DistributedRuntimeClient
from .xla_extension import Executable as Executable
from .xla_extension import FftType as FftType
from .xla_extension import Frame as Frame
from .xla_extension import HostBufferSemantics as HostBufferSemantics
from .xla_extension import OpSharding as OpSharding
from .xla_extension import HloSharding as HloSharding
from .xla_extension import PrimitiveType as PrimitiveType
from .xla_extension import Traceback as Traceback
from .xla_extension import XlaBuilder as XlaBuilder
from .xla_extension import XlaComputation as XlaComputation
from .xla_extension import XlaOp as XlaOp

_version: int

mlir_api_version: int

bfloat16: numpy.dtype
XLA_ELEMENT_TYPE_TO_DTYPE: Dict[PrimitiveType, numpy.dtype]


def dtype_to_etype(dtype: numpy.dtype) -> PrimitiveType:
  ...

def execute_with_python_values(executable: Executable, arguments: Sequence[Any],
                               backend: Client) -> Sequence[numpy.ndarray]: ...

def execute_with_python_values_replicated(
    executable: Executable, arguments: Sequence[Sequence[Any]],
    backend: Client) -> Sequence[Sequence[numpy.ndarray]]: ...

def shape_from_pyval(pyval: Any) -> Any: ...

def heap_profile(client: Client) -> bytes:
  ...


def make_cpu_client(*, use_tfrt: bool = ...) -> Client:
  ...


def make_gpu_client(
    distributed_client: Optional[DistributedRuntimeClient] = ...,
    node_id: int = ...,
    platform_name: Optional[str] = ...,
    allowed_devices: Optional[Set[int]] = ...) -> Client:
  ...


def make_interpreter_client() -> Client:
  ...


def make_tfrt_tpu_c_api_client() -> Client:
  ...


def make_tpu_client() -> Client:
  ...


class OpMetadata:

  def __init__(self,
               op_type: Optional[str] = ...,
               op_name: Optional[str] = ...,
               source_file: Optional[str] = ...,
               source_line: Optional[int] = ...):
    ...

  op_type: Optional[str]
  op_name: Optional[str]
  source_file: Optional[str]
  source_line: Optional[int]


class PaddingConfigDimension:
  edge_padding_low: int
  edge_padding_high: int
  interior_padding: int


class PaddingConfig:
  dimensions: List[PaddingConfigDimension]


def make_padding_config(
    padding_config: Union[PaddingConfig, Sequence[Tuple[int, int, int]]]
) -> PaddingConfig:
  ...

class PaddingType(enum.Enum):
  VALID = 1
  SAME = 2


class DotDimensionNumbers:
  lhs_contracting_dimensions: List[int]
  rhs_contracting_dimensions: List[int]
  lhs_batch_dimensions: List[int]
  rhs_batch_dimensions: List[int]


def make_dot_dimension_numbers(
    dimension_numbers: Union[DotDimensionNumbers,
                             Tuple[Tuple[List[int], List[int]],
                                   Tuple[List[int], List[int]]]]
) -> DotDimensionNumbers:
  ...


class ConvolutionDimensionNumbers:
  input_batch_dimension: int
  input_feature_dimension: int
  input_spatial_dimensions: List[int]
  kernel_input_feature_dimension: int
  kernel_output_feature_dimension: int
  kernel_spatial_dimensions: List[int]
  output_batch_dimension: int
  output_feature_dimension: int
  output_spatial_dimensions: List[int]


def make_convolution_dimension_numbers(
    dimension_numbers: Union[None, ConvolutionDimensionNumbers, Tuple[str, str,
                                                                      str]],
    num_spatial_dimensions: int) -> ConvolutionDimensionNumbers:
  ...


class PrecisionConfig:
  Precision = _xla.PrecisionConfig_Precision
  operand_precision: List[_xla.PrecisionConfig_Precision]


class GatherDimensionNumbers:
  offset_dims: List[int]
  collapsed_slice_dims: List[int]
  start_index_map: List[int]
  index_vector_dim: int


class ScatterDimensionNumbers:
  update_window_dims: List[int]
  inserted_window_dims: List[int]
  scatter_dims_to_operand_dims: List[int]
  index_vector_dim: int


class ReplicaGroup:
  replica_ids: List[int]


def make_replica_groups(
    replica_groups: Optional[Sequence[Sequence[int]]]) -> List[ReplicaGroup]:
  ...

def weakref_lru_cache(cache_context_fn: Callable, call: Callable, maxsize=...):
  ...
