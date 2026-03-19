# Copyright 2021 The JAX Authors
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

from collections.abc import Sequence
import enum
from typing import Any, overload

from jax.jaxlib import _jax

from . import _xla_builder

XlaComputation = _jax.XlaComputation
PrimitiveType = _jax.PrimitiveType
Shape = _jax.Shape
ShapeIndex = _jax.ShapeIndex

_ChannelHandle = Any
_ConvDimensionNumbers = Any
_DotDimensionNumbers = Any
_Layout = Any
_LiteralSlice = Any
_GatherDimensionNumbers = Any
_PaddingConfig = Any
_ReplicaGroup = Any
_ScatterDimensionNumbers = Any

class ShapeIndex:
  def __init__(self, indices: list[int]) -> None: ...
  def __eq__(self, other: Any) -> bool: ...
  def __ne__(self, other: Any) -> bool: ...
  def __hash__(self) -> int: ...
  def __repr__(self) -> str: ...

class FftType(enum.IntEnum):
  FFT = ...
  IFFT = ...
  RFFT = ...
  IRFFT = ...

class PrecisionConfig_Precision(enum.IntEnum):
  DEFAULT = ...
  HIGH = ...
  HIGHEST = ...

class TriangularSolveOptions_Transpose(enum.IntEnum):
  TRANSPOSE_INVALID = ...
  NO_TRANSPOSE = ...
  TRANSPOSE = ...
  ADJOINT = ...

class RandomAlgorithm(enum.IntEnum):
  RNG_DEFAULT = ...
  RNG_THREE_FRY = ...
  RNG_PHILOX = ...

class ResultAccuracy_Mode(enum.IntEnum):
  DEFAULT = ...
  HIGHEST = ...
  TOLERANCE = ...

class ResultAccuracy:
  mode: ResultAccuracy_Mode
  atol: float
  rtol: float
  ulps: int

class CustomCallSchedule(enum.IntEnum):
  SCHEDULE_NONE = ...
  SCHEDULE_LATEST = ...
  SCHEDULE_EARLIEST = ...

# TODO(b/189822916): Remove this enum when all clients are migrated to the
# status-returning API.
class CustomCallApiVersion(enum.IntEnum):
  API_VERSION_ORIGINAL = ...
  API_VERSION_STATUS_RETURNING = ...
  API_VERSION_STATUS_RETURNING_UNIFIED = ...
  API_VERSION_TYPED_FFI = ...

def AfterAll(
    builder: _xla_builder.XlaBuilder, tokens: Sequence[_xla_builder.XlaOp]
) -> _xla_builder.XlaOp: ...
def AllGather(
    operand: _xla_builder.XlaOp,
    all_gather_dimension: int,
    shard_count: int,
    replica_groups: Sequence[_ReplicaGroup] = ...,
    channel_id: _ChannelHandle | None = ...,
    shape_with_layout: _Layout | None = ...,
    use_global_device_ids: bool | None = ...,
) -> _xla_builder.XlaOp: ...
def AllReduce(
    operand: _xla_builder.XlaOp,
    computation: XlaComputation,
    replica_groups: Sequence[_ReplicaGroup] = ...,
    channel_id: _ChannelHandle | None = ...,
    shape_with_layout: _Layout | None = ...,
) -> _xla_builder.XlaOp: ...
def ApproxTopK(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    init_values: Sequence[_xla_builder.XlaOp],
    top_k: int,
    reduction_dim: int,
    comparator: XlaComputation,
    recall_target: float | None,
    aggregate_to_topk: bool | None,
    reduction_input_size_override: int | None,
) -> _xla_builder.XlaOp: ...
def ApproxTopKFallback(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    init_values: Sequence[_xla_builder.XlaOp],
    top_k: int,
    reduction_dim: int,
    comparator: XlaComputation,
    recall_target: float | None,
    aggregate_to_topk: bool | None,
    reduction_input_size_override: int | None,
) -> _xla_builder.XlaOp: ...
def ApproxTopKReductionOutputSize(
    input_size: int,
    rank: int,
    top_k: int,
    recall_target: float,
    aggregate_to_topk: bool | None = ...,
    input_size_override: int | None = ...,
) -> tuple[int, int]: ...
def ReduceScatter(
    operand: _xla_builder.XlaOp,
    computation: XlaComputation,
    scatter_dimension: int,
    shard_count: int,
    replica_groups: Sequence[_ReplicaGroup] = ...,
    channel_id: _ChannelHandle | None = ...,
    layout: _Layout | None = ...,
    use_global_device_ids: bool | None = ...,
) -> _xla_builder.XlaOp: ...
def AllToAll(
    operand: _xla_builder.XlaOp,
    split_dimension: int,
    concat_dimension: int,
    split_count: int,
    replica_groups: Sequence[_ReplicaGroup] = ...,
    layout: _Layout | None = ...,
    channel_id: _ChannelHandle | None = ...,
) -> _xla_builder.XlaOp: ...
def BitcastConvertType(
    operand: _xla_builder.XlaOp, new_element_type: PrimitiveType
) -> _xla_builder.XlaOp: ...
def Broadcast(
    operand: _xla_builder.XlaOp, sizes: Sequence[int]
) -> _xla_builder.XlaOp: ...
def BroadcastInDim(
    operand: _xla_builder.XlaOp,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> _xla_builder.XlaOp: ...
def Call(
    builder: _xla_builder.XlaBuilder,
    computation: XlaComputation,
    operands: Sequence[_xla_builder.XlaOp],
) -> _xla_builder.XlaOp: ...
def Cholesky(
    a: _xla_builder.XlaOp, lower: bool = ...
) -> _xla_builder.XlaOp: ...
def Clamp(
    min: _xla_builder.XlaOp,
    operand: _xla_builder.XlaOp,
    max: _xla_builder.XlaOp,
) -> _xla_builder.XlaOp: ...
def Collapse(
    operand: _xla_builder.XlaOp, dimensions: Sequence[int]
) -> _xla_builder.XlaOp: ...
def CollectivePermute(
    operand: _xla_builder.XlaOp,
    source_target_pairs: Sequence[tuple[int, int]],
    channel_id: _ChannelHandle | None = ...,
    inplace: bool = ...,
) -> _xla_builder.XlaOp: ...
def ConcatInDim(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    dimension: int,
) -> _xla_builder.XlaOp: ...
@overload
def Conditional(
    branch_index: _xla_builder.XlaOp,
    branch_computations: Sequence[XlaComputation],
    branch_operands: Sequence[_xla_builder.XlaOp],
) -> _xla_builder.XlaOp: ...
@overload
def Conditional(
    predicate: _xla_builder.XlaOp,
    true_operand: _xla_builder.XlaOp,
    true_computation: XlaComputation,
    false_operand: _xla_builder.XlaOp,
    false_computation: XlaComputation,
) -> _xla_builder.XlaOp: ...
def Constant(
    builder: _xla_builder.XlaBuilder, value: _LiteralSlice
) -> _xla_builder.XlaOp: ...
def ConstantLiteral(
    builder: _xla_builder.XlaBuilder, value: _LiteralSlice
) -> _xla_builder.XlaOp: ...
def ConvGeneralDilated(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
    dimension_numbers: _ConvDimensionNumbers,
    feature_group_count: int = ...,
    batch_group_count: int = ...,
    precision_config: PrecisionConfig_Precision | None = ...,
    preferred_element_type: PrimitiveType | None = ...,
    window_reversal: Sequence[bool] | None = ...,
) -> _xla_builder.XlaOp: ...
def ConvertElementType(
    operand: _xla_builder.XlaOp, new_element_type: PrimitiveType
) -> _xla_builder.XlaOp: ...
def CreateToken(builder: _xla_builder.XlaBuilder) -> _xla_builder.XlaOp: ...
def CrossReplicaSum(
    operand: _xla_builder.XlaOp, replica_groups: Sequence[_ReplicaGroup] = ...
) -> _xla_builder.XlaOp: ...
def CustomCall(
    builder: _xla_builder.XlaBuilder,
    call_target_name: bytes,
    operands: Sequence[_xla_builder.XlaOp],
    shape: Shape,
    opaque: bytes = ...,
    has_side_effect: bool = ...,
    schedule: CustomCallSchedule = ...,
    api_version: CustomCallApiVersion = ...,
) -> _xla_builder.XlaOp: ...
def CustomCallWithLayout(
    builder: _xla_builder.XlaBuilder,
    call_target_name: bytes,
    operands: Sequence[_xla_builder.XlaOp],
    shape_with_layout: Shape,
    operand_shapes_with_layout: Sequence[Shape],
    opaque: bytes = ...,
    has_side_effect: bool = ...,
    schedule: CustomCallSchedule = ...,
    api_version: CustomCallApiVersion = ...,
) -> _xla_builder.XlaOp: ...
def CustomCallWithAliasing(
    builder: _xla_builder.XlaBuilder,
    call_target_name: bytes,
    operands: Sequence[_xla_builder.XlaOp],
    shape_with_layout: Shape,
    operand_shapes_with_layout: Sequence[Shape],
    opaque: bytes = ...,
    has_side_effect: bool = ...,
    output_operand_aliasing: Sequence[
        tuple[ShapeIndex, tuple[int, ShapeIndex]]
    ] = ...,
    literal: _LiteralSlice = ...,
    schedule: CustomCallSchedule = ...,
    api_version: CustomCallApiVersion = ...,
) -> _xla_builder.XlaOp: ...
def Dot(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    precision_config: PrecisionConfig_Precision | None = ...,
    preferred_element_type: PrimitiveType | None = ...,
) -> _xla_builder.XlaOp: ...
def DotGeneral(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    dimensions_numbers: _DotDimensionNumbers,
    precision_config: PrecisionConfig_Precision | None = ...,
    preferred_element_type: PrimitiveType | None = ...,
) -> _xla_builder.XlaOp: ...
def DynamicReshape(
    operand: _xla_builder.XlaOp,
    dim_sizes: Sequence[_xla_builder.XlaOp],
    new_size_bounds: Sequence[int],
    dims_are_dynamic: Sequence[bool],
) -> _xla_builder.XlaOp: ...
def DynamicSlice(
    operand: _xla_builder.XlaOp,
    start_indices: Sequence[_xla_builder.XlaOp],
    slice_sizes: Sequence[int],
) -> _xla_builder.XlaOp: ...
def DynamicUpdateSlice(
    operand: _xla_builder.XlaOp,
    update: _xla_builder.XlaOp,
    start_indices: Sequence[_xla_builder.XlaOp],
) -> _xla_builder.XlaOp: ...
def Eigh(
    a: _xla_builder.XlaOp,
    lower: bool = ...,
    max_iter: int = ...,
    epsilon: float = ...,
    sort_eigenvalues: bool = ...,
) -> tuple[_xla_builder.XlaOp, _xla_builder.XlaOp]: ...
def Fft(
    operand: _xla_builder.XlaOp, fft_type: FftType, fft_length: Sequence[int]
) -> _xla_builder.XlaOp: ...
def Gather(
    a: _xla_builder.XlaOp,
    start_indices: _xla_builder.XlaOp,
    dimension_numbers: _GatherDimensionNumbers,
    slice_sizes: Sequence[int],
    indices_are_sorted: bool = ...,
) -> _xla_builder.XlaOp: ...
def GetDimensionSize(
    operand: _xla_builder.XlaOp, index: int
) -> _xla_builder.XlaOp: ...
def GetTupleElement(
    tuple_data: _xla_builder.XlaOp, index: int
) -> _xla_builder.XlaOp: ...
def InfeedWithToken(
    token: _xla_builder.XlaOp, shape: Shape, config: str | None = ...
) -> _xla_builder.XlaOp: ...
@overload
def Iota(
    builder: _xla_builder.XlaBuilder, shape: Shape, iota_dimension: int
) -> _xla_builder.XlaOp: ...
@overload
def Iota(
    builder: _xla_builder.XlaBuilder, type: PrimitiveType, size: int
) -> _xla_builder.XlaOp: ...
def LU(
    a: _xla_builder.XlaOp,
) -> tuple[_xla_builder.XlaOp, _xla_builder.XlaOp, _xla_builder.XlaOp]: ...
def Map(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    computation: XlaComputation,
    dimensions: Sequence[int],
    static_operands: Sequence[_xla_builder.XlaOp] = ...,
) -> _xla_builder.XlaOp: ...
def MultiCollectivePermute(
    operands: Sequence[_xla_builder.XlaOp],
    source_target_pairs: Sequence[tuple[int, int]],
    channel_id: _ChannelHandle | None = ...,
    inplace: bool = ...,
) -> _xla_builder.XlaOp: ...
def NextAfter(
    __from: _xla_builder.XlaOp, to: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def OutfeedWithToken(
    operand: _xla_builder.XlaOp,
    token: _xla_builder.XlaOp,
    shape_with_layout: Shape,
    outfeed_config: str | None = ...,
) -> _xla_builder.XlaOp: ...
def Pad(
    operand: _xla_builder.XlaOp,
    padding_value: _xla_builder.XlaOp,
    padding_config: _PaddingConfig,
) -> _xla_builder.XlaOp: ...
def Parameter(
    builder: _xla_builder.XlaBuilder,
    parameter_number: int,
    shape: Shape,
    name: str = ...,
    replicated_at_leaf_buffers: Sequence[bool] = ...,
) -> _xla_builder.XlaOp: ...
def ProductOfElementaryHouseholderReflectors(
    a: _xla_builder.XlaOp, taus: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def QR(
    a: _xla_builder.XlaOp, full_matrices: bool
) -> tuple[_xla_builder.XlaOp, _xla_builder.XlaOp]: ...
def QrDecomposition(
    a: _xla_builder.XlaOp,
) -> tuple[_xla_builder.XlaOp, _xla_builder.XlaOp]: ...
def Reduce(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    init_values: Sequence[_xla_builder.XlaOp],
    computation: XlaComputation,
    dimensions_to_reduce: Sequence[int],
) -> _xla_builder.XlaOp: ...
def ReducePrecision(
    operand: _xla_builder.XlaOp, exponent_bits: int, mantissa_bits: int
) -> _xla_builder.XlaOp: ...
@overload
def ReduceWindowWithGeneralPadding(
    operand: _xla_builder.XlaOp,
    init_value: _xla_builder.XlaOp,
    computation: XlaComputation,
    window_dimensions: Sequence[int],
    window_strides: Sequence[int],
    base_dilations: Sequence[int],
    window_dilations: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> _xla_builder.XlaOp: ...
@overload
def ReduceWindowWithGeneralPadding(
    operands: Sequence[_xla_builder.XlaOp],
    init_values: Sequence[_xla_builder.XlaOp],
    computation: XlaComputation,
    window_dimensions: Sequence[int],
    window_strides: Sequence[int],
    base_dilations: Sequence[int],
    window_dilations: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> _xla_builder.XlaOp: ...
def ReplicaId(builder: _xla_builder.XlaBuilder) -> _xla_builder.XlaOp: ...
def Reshape(
    operand: _xla_builder.XlaOp, new_sizes: Sequence[int]
) -> _xla_builder.XlaOp: ...
def Rev(
    operand: _xla_builder.XlaOp, dimensions: Sequence[int]
) -> _xla_builder.XlaOp: ...
def RngBitGenerator(
    algorithm: RandomAlgorithm, initial_state: _xla_builder.XlaOp, shape: Shape
) -> _xla_builder.XlaOp: ...
def RngNormal(
    mu: _xla_builder.XlaOp, sigma: _xla_builder.XlaOp, shape: Shape
) -> _xla_builder.XlaOp: ...
def RngUniform(
    a: _xla_builder.XlaOp, b: _xla_builder.XlaOp, shape: Shape
) -> _xla_builder.XlaOp: ...
@overload
def Scatter(
    input: _xla_builder.XlaOp,
    scatter_indices: _xla_builder.XlaOp,
    updates: _xla_builder.XlaOp,
    update_computation: XlaComputation,
    dimension_numbers: _ScatterDimensionNumbers,
    indices_are_sorted: bool = ...,
    unique_indices: bool = ...,
) -> _xla_builder.XlaOp: ...
@overload
def Scatter(
    inputs: Sequence[_xla_builder.XlaOp],
    scatter_indices: _xla_builder.XlaOp,
    updates: Sequence[_xla_builder.XlaOp],
    update_computation: XlaComputation,
    dimension_numbers: _ScatterDimensionNumbers,
    indices_are_sorted: bool = ...,
    unique_indices: bool = ...,
) -> _xla_builder.XlaOp: ...
def Select(
    pred: _xla_builder.XlaOp,
    on_true: _xla_builder.XlaOp,
    on_false: _xla_builder.XlaOp,
) -> _xla_builder.XlaOp: ...
def SelectAndScatterWithGeneralPadding(
    operand: _xla_builder.XlaOp,
    select: XlaComputation,
    window_dimensions: Sequence[int],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    source: _xla_builder.XlaOp,
    init_value: _xla_builder.XlaOp,
    scatter: XlaComputation,
) -> _xla_builder.XlaOp: ...
def Slice(
    operand: _xla_builder.XlaOp,
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int],
) -> _xla_builder.XlaOp: ...
def SliceInDim(
    operand: _xla_builder.XlaOp,
    start_index: int,
    limit_index: int,
    stride: int,
    dimno: int,
) -> _xla_builder.XlaOp: ...
def Sort(
    builder: _xla_builder.XlaBuilder,
    operands: Sequence[_xla_builder.XlaOp],
    comparator: XlaComputation | None = ...,
    dimension: int = ...,
    is_stable: bool = ...,
) -> _xla_builder.XlaOp: ...
def SVD(
    a: _xla_builder.XlaOp, max_iter: int = ..., epsilon: float = ...
) -> tuple[_xla_builder.XlaOp, _xla_builder.XlaOp, _xla_builder.XlaOp]: ...
def TopK(input: _xla_builder.XlaOp, k: int) -> _xla_builder.XlaOp: ...
def Transpose(
    operand: _xla_builder.XlaOp, permutation: Sequence[int]
) -> _xla_builder.XlaOp: ...
def TriangularSolve(
    a: _xla_builder.XlaOp,
    b: _xla_builder.XlaOp,
    left_side: bool,
    lower: bool,
    unit_diagonal: bool,
    transpose_a: TriangularSolveOptions_Transpose,
) -> _xla_builder.XlaOp: ...
def Tuple(
    builder: _xla_builder.XlaBuilder, elements: Sequence[_xla_builder.XlaOp]
) -> _xla_builder.XlaOp: ...
def While(
    condition: XlaComputation, body: XlaComputation, init: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def Igamma(
    a: _xla_builder.XlaOp, x: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def Igammac(
    a: _xla_builder.XlaOp, x: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def IgammaGradA(
    a: _xla_builder.XlaOp, x: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def RandomGammaGrad(
    a: _xla_builder.XlaOp, x: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def RegularizedIncompleteBeta(
    a: _xla_builder.XlaOp, b: _xla_builder.XlaOp, x: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def Zeta(
    a: _xla_builder.XlaOp, q: _xla_builder.XlaOp
) -> _xla_builder.XlaOp: ...
def Eq(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Ne(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Ge(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Gt(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Lt(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Le(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Add(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Sub(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Mul(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Div(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Rem(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Max(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Min(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def And(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Or(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Xor(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def ShiftLeft(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def ShiftRightArithmetic(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def ShiftRightLogical(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Atan2(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Pow(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Complex(
    lhs: _xla_builder.XlaOp,
    rhs: _xla_builder.XlaOp,
    broadcast_dimensions: Sequence[int] = ...,
) -> _xla_builder.XlaOp: ...
def Not(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def PopulationCount(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Clz(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Abs(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Exp(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Expm1(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Floor(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Ceil(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Round(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Log(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Log1p(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Sign(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Cos(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def OptimizationBarrier(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Sin(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Tan(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Tanh(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def IsFinite(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Neg(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Sqrt(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Rsqrt(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Cbrt(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def Square(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Reciprocal(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Erfc(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Erf(
    operand: _xla_builder.XlaOp, result_accuracy: ResultAccuracy = ...
) -> _xla_builder.XlaOp: ...
def ErfInv(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Lgamma(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Digamma(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def BesselI0e(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def BesselI1e(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Acos(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Asin(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Atan(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Acosh(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Asinh(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Atanh(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Cosh(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Sinh(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Real(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Imag(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
def Conj(__arg: _xla_builder.XlaOp) -> _xla_builder.XlaOp: ...
