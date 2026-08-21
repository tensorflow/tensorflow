# Copyright 2026 The OpenXLA Authors
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

"""Cost model functions for Pallas matmul benchmarks."""

from collections.abc import Sequence
import dataclasses
import enum
import functools
import math

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.core import platform_info


def _get_base_values(
    dim: int, min_block_size: int, min_value: int = 1
) -> np.ndarray:
  """Returns a list of base values for a given dimension.

    Here the base value of a window is the multiple of the smallest block size
    allowed for that dimension. E.g. since block_k must be a multiple of 256,
    a base value of 1 corresponds to a block size of 256.

    Using all possible base values for all three dimensions is too costly, so we
    use a heuristic that selects a subset of possible base values. We use
    smaller gaps at lower base values since they have a larger impact on the
    latency of the window, and progressively double the gap as the base value
    increases.

  Args:
    dim: The dimension size (e.g. m, k, or n).
    min_block_size: The base value of the window.
    min_value: The minimum base value to use.

  Returns:
    A list of base values to use for the given dimension.
  """
  max_value = dim // min_block_size
  start = min(max_value, min_value)
  end = 16
  step = 1
  values = np.arange(start, end + 1)
  while values[-1] < max_value:
    step = step * 2
    start = end + step
    end = end * 2
    next_values = np.arange(start, end + 1, step)
    values = np.concatenate([values, next_values])
  # The last value may be larger than dim, so we remove larger values and add
  # the base value corresponding to dim if necessary.
  if values[-1] > max_value:
    ind = np.argmax(values >= max_value)
    values = values[: ind + 1]
    if values[ind] != max_value:
      values[-1] = max_value
  return values


def _dma_latency(
    bytes_to_transfer: np.ndarray,
    base_latencies_ns: np.ndarray,
    bandwidth_gb_per_sec: int,
) -> np.ndarray:
  """Estimate the latency of a set of DMA operations executing in parallel.

  This works by assuming that the bandwidth constant and split evenly across all
  DMA operations. As each DMA operation completes, the freed up bandwidth is
  used for the remaining DMA operations. The final result is the sum of the
  total transfer time with the latency of the last DMA operation to complete.

  Args:
    bytes_to_transfer: The number of bytes to transfer for each DMA operation.
      The minor most dimension is the DMA operations, and the other dimensions
      are treated as batch dimensions.
    base_latencies_ns: The base latency of each DMA operation. This should have
      the same number of dimensions as `bytes_to_transfer`, with the minor most
      dimension being the DMA operations and the rest being 1. This is not a
      constant because HBM<->VMEM and VMEM<-HBM have different latencies.
    bandwidth_gb_per_sec: The bandwidth of the DMA operations.

  Returns:
      The estimated DMA latency in ns for all DMA operations to complete. The
      shape is the same as `bytes_to_transfer` with the minor most dimension
      removed.
  """
  total_transfer_time_ns = np.zeros(bytes_to_transfer.shape[:-1] + (1,))
  in_flight_latencies_ns = np.zeros(bytes_to_transfer.shape[:-1] + (1,))
  while (bytes_to_transfer > 0).any():
    bytes_to_transfer_nonzero = np.ma.masked_where(
        bytes_to_transfer == 0, bytes_to_transfer
    )
    bytes_transferred = np.ma.min(
        bytes_to_transfer_nonzero, axis=-1, keepdims=True
    ).filled(0)
    split_ways = (bytes_to_transfer > 0).sum(-1, keepdims=True)
    transfer_time_ns = bytes_transferred * split_ways / bandwidth_gb_per_sec
    total_transfer_time_ns += transfer_time_ns
    completed = (bytes_to_transfer > 0) & (
        bytes_transferred == bytes_to_transfer
    )
    in_flight_latencies_ns = np.maximum(
        (in_flight_latencies_ns - transfer_time_ns),
        (completed * base_latencies_ns).max(-1, keepdims=True),
    )
    bytes_to_transfer = np.maximum(bytes_to_transfer - bytes_transferred, 0)
  return (total_transfer_time_ns + in_flight_latencies_ns).squeeze(-1)


class Axis(enum.Enum):
  """The axes of the matmul, used to represent grid iteration order."""

  M = 0
  K = 1
  N = 2


@dataclasses.dataclass(frozen=True)
class Operand:
  """Represents a kernel operand for VMEM and DMA latency modeling.

  Attributes:
    size_bytes: Array of tile sizes in bytes over the candidate window grid.
    axes: The two matmul axes (e.g. (Axis.M, Axis.K)) that index this operand.
    is_output: Whether this is an output operand (transferred VMEM->HBM) rather
      than an input operand (transferred HBM->VMEM).
  """

  size_bytes: np.ndarray
  axes: tuple[Axis, Axis]
  is_output: bool = False


@dataclasses.dataclass(frozen=True)
class PipelineStageCost:
  """Throughput and latency of one stage of the compute pipeline.

  Each grid iteration is modeled as three pipelined stages of
  `input fusion -> MXU op -> output fusion`. Each subclass of `CostModel`
  specializes the cost of each stage. Throughput is tracked separately per
  hardware unit, because different stages may compete for the same unit (e.g.
  both input and output fusions issue into ALU slots). Each throughput field
  tracks total cycles the stage occupies that unit per grid iteration.

  Attributes:
    matmul_cycles: Total MXU vmatmul cycles per grid iteration (summed over all
      MXUs).
    latch_cycles: RHS latch overhead cycles per grid iteration (summed over all
      MXUs).
    alu_cycles: Vector ALU cycles per grid iteration (summed over all ALUs).
    xlu_cycles: Vector XLU cycles per grid iteration (summed over all XLUs).
    latency_cycles: The number of cycles for a single pipeline stage to fill and
      drain.
  """

  matmul_cycles: float | np.ndarray = 0.0
  latch_cycles: float | np.ndarray = 0.0
  alu_cycles: float | np.ndarray = 0.0
  xlu_cycles: float | np.ndarray = 0.0
  latency_cycles: float | np.ndarray = 0.0


class CostModel:
  """Base cost model for Pallas matmul kernels.

  Models a core matrix multiplication over a 3D grid of candidate window sizes
  (block_m, block_k, block_n) by estimating total kernel latency as the sum of:
  1. The latency of the initial input operand transfers from HBM to VMEM.
  2. The latency of each loop iteration as the maximum of the per-iteration
     compute latency and the operand transfer latencies for the next iteration.
  3. The latency of the final output operand transfers from VMEM to HBM.

  The per-iteration compute latency models the iteration over three stages of
  `input fusion -> MXU op -> output fusion`. Each stage declares its cycle
  count per hardware unit (e.g. MXU, ALU, XLU); the most heavily loaded unit
  dominates, and every stage contributes its fill/drain latency:

    compute_cycles = max_u(sum_s(s.u_cycles) / num_u) + sum_s(s.latency_cycles)

  for s in {input fusion, matmul, output fusion} and u in {matmul, latch, ALU,
  XLU}.

  Subclasses can specialize the cost model for specific matmul variants by
  overriding hooks for:
  - Core matmul hardware parameters (`_get_mxu_size`, `_get_tile_sizes`,
    `_get_matmul_cadence_cycles`, `_get_vmatmuls_per_latch`,
    `_get_latch_latency_cycles`).
  - Operand sizes and additional operands (`_get_lhs_size_bytes`,
    `_get_rhs_size_bytes`, `_get_out_size_bytes`, `_get_additional_operands`).
  - Input and output fusions (`_get_input_fusion_vmem_usage_bytes`,
    `_get_output_fusion_vmem_usage_bytes`, `_get_input_fusion_stage_cost`,
    `_get_output_fusion_stage_cost`).

  Attributes:
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    lhs_mem: The memory space of the first operand.
    rhs_mem: The memory space of the second operand.
    out_mem: The memory space of the output.
    lhs_dtype: The dtype of the first operand in memory.
    rhs_dtype: The dtype of the second operand in memory.
    out_dtype: The dtype of the output in memory.
    acc_dtype: The dtype of the accumulator in VMEM.
    matmul_lhs_dtype: The dtype of the LHS operand fed to the MXU matmul.
    matmul_rhs_dtype: The dtype of the RHS operand fed to the MXU matmul.
    subblock_m: Optional subblock size for the m dimension.
  """

  def __init__(
      self,
      m: int,
      k: int,
      n: int,
      lhs_mem: pltpu.MemorySpace,
      rhs_mem: pltpu.MemorySpace,
      out_mem: pltpu.MemorySpace,
      lhs_dtype: jax.typing.DTypeLike,
      rhs_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
      acc_dtype: jax.typing.DTypeLike,
      *,
      matmul_lhs_dtype: jax.typing.DTypeLike | None = None,
      matmul_rhs_dtype: jax.typing.DTypeLike | None = None,
      subblock_m: int | None = None,
      chip_version: pltpu.ChipVersion | None = None,
  ):
    self._platform_info: platform_info.PlatformInfo = (
        platform_info.get_platform_info(chip_version)
    )
    self.m = m
    self.k = k
    self.n = n
    self.lhs_mem = lhs_mem
    self.rhs_mem = rhs_mem
    self.out_mem = out_mem
    self.lhs_dtype = jnp.dtype(lhs_dtype)
    self.rhs_dtype = jnp.dtype(rhs_dtype)
    self.out_dtype = jnp.dtype(out_dtype)
    self.acc_dtype = jnp.dtype(acc_dtype)
    self.matmul_lhs_dtype = self._platform_info.get_emulated_dtype(
        matmul_lhs_dtype if matmul_lhs_dtype is not None else lhs_dtype,
        self._platform_info.matmul_cadence_cycles_by_dtype,
    )
    self.matmul_rhs_dtype = self._platform_info.get_emulated_dtype(
        matmul_rhs_dtype if matmul_rhs_dtype is not None else rhs_dtype,
        self._platform_info.latch_cadence_cycles_by_dtype,
    )
    self.subblock_m = subblock_m

    self._bits_lhs = jax.dtypes.itemsize_bits(self.lhs_dtype)
    self._bits_rhs = jax.dtypes.itemsize_bits(self.rhs_dtype)
    self._bits_out = jax.dtypes.itemsize_bits(self.out_dtype)
    self._bits_acc = jax.dtypes.itemsize_bits(self.acc_dtype)
    self._bits_matmul_lhs = jax.dtypes.itemsize_bits(self.matmul_lhs_dtype)
    self._bits_matmul_rhs = jax.dtypes.itemsize_bits(self.matmul_rhs_dtype)

    m_tile_size, k_tile_size, n_tile_size = self._get_tile_sizes()
    m_align = math.ceil(m / m_tile_size) * m_tile_size
    k_align = math.ceil(k / k_tile_size) * k_tile_size
    n_align = math.ceil(n / n_tile_size) * n_tile_size
    m_base, k_base, n_base = None, None, None
    if lhs_mem == pltpu.VMEM or out_mem == pltpu.VMEM:
      m_base = np.array([m_align // m_tile_size])[:, None, None]
    if lhs_mem == pltpu.VMEM or rhs_mem == pltpu.VMEM:
      k_base = np.array([k_align // k_tile_size])[None, :, None]
    if rhs_mem == pltpu.VMEM or out_mem == pltpu.VMEM:
      n_base = np.array([n_align // n_tile_size])[None, None, :]
    if m_base is None:
      m_base = _get_base_values(m_align, m_tile_size)[:, None, None]
    if k_base is None:
      k_base = _get_base_values(k_align, k_tile_size)[None, :, None]
    if n_base is None:
      # By setting min_value=2, we ensure that block_n is at least 512, which
      # tends to result in better MRB accumulation chain scheduling so that
      # each MXU gets a different RHS latch.
      n_base = _get_base_values(n_align, n_tile_size, min_value=2)[
          None, None, :
      ]
    self._grid_shape = (m_base.shape[0], k_base.shape[1], n_base.shape[2])
    self._block_m = m_base * m_tile_size
    self._block_k = k_base * k_tile_size
    self._block_n = n_base * n_tile_size
    self._block_specs = np.stack(
        [
            np.broadcast_to(self._block_m, self._grid_shape),
            np.broadcast_to(self._block_k, self._grid_shape),
            np.broadcast_to(self._block_n, self._grid_shape),
        ],
        axis=-1,
    )
    self._grid_m = np.ceil(m / self._block_m)
    self._grid_k = np.ceil(k / self._block_k)
    self._grid_n = np.ceil(n / self._block_n)

    self._lhs_size_bytes = self._get_lhs_size_bytes()
    self._rhs_size_bytes = self._get_rhs_size_bytes()
    self._out_size_bytes = self._get_out_size_bytes()
    self._operands: tuple[Operand, ...] = (
        Operand(self._lhs_size_bytes, (Axis.M, Axis.K), is_output=False),
        Operand(self._rhs_size_bytes, (Axis.K, Axis.N), is_output=False),
        Operand(self._out_size_bytes, (Axis.M, Axis.N), is_output=True),
        *self._get_additional_operands(),
    )

    # For now, iteration order is fixed, as the algorithm does not return better
    # windows for other orders (i.e. no other iteration order returns a window
    # with strictly lower estimated latency).
    iteration_order = (Axis.M, Axis.N, Axis.K)
    # Arrays internally store axes in the order M, K, N, regardless of iteration
    # order.
    self._idx_map = {Axis.M: 0, Axis.K: 1, Axis.N: 2}
    # Construct the inner most non-trivial axis for each window in the grid.
    # For most windows this will be K, but for windows where K is the full
    # dimension, it will be M or N instead.
    inner_most_non_trivial_axis = np.full(
        self._grid_shape, self._idx_map[iteration_order[2]]
    )
    indices = [slice(None)] * 3
    indices[self._idx_map[iteration_order[2]]] = slice(-1, None)
    inner_most_non_trivial_axis[*indices] = self._idx_map[iteration_order[1]]
    indices[self._idx_map[iteration_order[1]]] = slice(-1, None)
    inner_most_non_trivial_axis[*indices] = self._idx_map[iteration_order[0]]
    self._inner_most_non_trivial_axis = inner_most_non_trivial_axis

    # VMEM usage for each window in the grid.
    self._vmem_usage_bytes = self._compute_vmem_usage_bytes()

  def _get_mxu_size(self) -> tuple[int, int]:
    """Returns the effective (contracting_dim, non_contracting_dim) MXU size."""
    return self._platform_info.mxu_size_by_dtype(self.matmul_lhs_dtype)

  def _get_tile_sizes(self) -> tuple[int, int, int]:
    """Returns the min (m_tile_size, k_tile_size, n_tile_size) block sizes."""
    sublane_count, _ = self._platform_info.vreg_size
    mxu_contracting_dim, mxu_non_contracting_dim = self._get_mxu_size()
    # Window sizes must be multiples of sublane and lane counts, but we use
    # MXU dimensions instead of lane count to maximize FLOPS utilization.
    m_tile_size = sublane_count * (32 // self._bits_lhs)
    return m_tile_size, mxu_contracting_dim, mxu_non_contracting_dim

  def _get_lhs_size_bytes(self) -> np.ndarray:
    """Returns the tile size in bytes for the LHS operand across the grid."""
    return self._block_m * self._block_k * self._bits_lhs // 8

  def _get_rhs_size_bytes(self) -> np.ndarray:
    """Returns the tile size in bytes for the RHS operand across the grid."""
    return self._block_k * self._block_n * self._bits_rhs // 8

  def _get_out_size_bytes(self) -> np.ndarray:
    """Returns the tile size in bytes for the output operand across the grid."""
    return self._block_m * self._block_n * self._bits_out // 8

  def _get_additional_operands(self) -> Sequence[Operand]:
    """Returns additional operands beyond (lhs, rhs, out) for VMEM and DMA."""
    return ()

  def _needs_swap(self, operand: Operand) -> np.ndarray:
    """Returns a boolean array indicating whether the operand spans >1 tile."""
    dim_by_axis = {Axis.M: self.m, Axis.K: self.k, Axis.N: self.n}
    block_by_axis = {
        Axis.M: self._block_m,
        Axis.K: self._block_k,
        Axis.N: self._block_n,
    }
    ax0, ax1 = operand.axes
    return np.asarray(
        ~(
            (dim_by_axis[ax0] == block_by_axis[ax0])
            & (dim_by_axis[ax1] == block_by_axis[ax1])
        )
    )

  def _get_invariant_axis(self, operand: Operand) -> Axis:
    """Returns the matmul axis along which the operand does not vary."""
    for axis in Axis:
      if axis not in operand.axes:
        return axis
    raise ValueError(f"Invalid operand axes: {operand.axes}")

  def _get_lhs_conversion_vmem_usage_bytes(
      self, k_dim: int | np.ndarray | None = None
  ) -> int | np.ndarray:
    """Returns VMEM bytes spilled if the matmul LHS requires VPU conversion."""
    if k_dim is None:
      k_dim = self._block_k
    spill_bits = None
    if self.lhs_dtype != self.matmul_lhs_dtype:
      # When lhs_dtype != matmul_lhs_dtype, the compiler will spill the
      # converted operand to VMEM.
      spill_bits = self._bits_matmul_lhs
    elif (
        self.matmul_lhs_dtype == jnp.int4 and self._platform_info.generation < 8
    ):
      # Prior to generation 8, int4 LHS requires VPU for unpacking from
      # compressed format and repacking to interleaved format.
      spill_bits = 4
    if spill_bits is not None:
      block_m = self._block_m
      if self.subblock_m is not None:
        block_m = np.minimum(self._block_m, self.subblock_m)
      return block_m * k_dim * spill_bits // 8
    return 0

  def _get_input_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    """Returns additional VMEM bytes required by input fusions."""
    return self._get_lhs_conversion_vmem_usage_bytes()

  def _get_output_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    """Returns additional VMEM bytes required by output fusions."""
    return 0

  def _compute_vmem_usage_bytes(self) -> np.ndarray:
    """Calculates total VMEM usage in bytes across all candidate windows."""
    vmem_usage = np.zeros(self._grid_shape, dtype=np.int64)
    for op in self._operands:
      # Double-buffer if the operand doesn't fit in a single window.
      num_buffers = 1 + self._needs_swap(op)
      vmem_usage = vmem_usage + num_buffers * op.size_bytes
    # Only one accumulator tile is needed.
    acc_vmem_bytes = self._block_m * self._block_n * self._bits_acc // 8
    return (
        vmem_usage
        + acc_vmem_bytes
        + self._get_input_fusion_vmem_usage_bytes()
        + self._get_output_fusion_vmem_usage_bytes()
    )

  def _get_matmul_cadence_cycles(self) -> int | float:
    """Returns the matmul cadence cycles per MXU."""
    return self._platform_info.get_matmul_cadence_cycles(self.matmul_lhs_dtype)

  def _get_rows_per_vmatmul(self) -> int:
    """Returns the number of LHS rows processed by a single vmatmul."""
    sublane_count, _ = self._platform_info.vreg_size
    rows_per_vmatmul = sublane_count * (32 // self._bits_matmul_lhs)
    if self._bits_matmul_lhs < 8:
      # Sub-byte matmuls use double contracting dimension and half the rows.
      rows_per_vmatmul //= 2
    return rows_per_vmatmul

  def _get_vmatmuls_per_latch(self) -> np.ndarray:
    """Returns the number of vmatmul instructions per RHS latch."""
    return np.ceil(self._block_m / self._get_rows_per_vmatmul())

  def _get_num_latches_per_iteration(self) -> np.ndarray:
    """Returns the number of RHS latches per grid iteration."""
    mxu_contracting_dim, mxu_non_contracting_dim = self._get_mxu_size()
    return (self._block_k // mxu_contracting_dim) * (
        self._block_n // mxu_non_contracting_dim
    )

  def _get_num_mxu_ops_per_iteration(self) -> np.ndarray:
    """Returns the number of MXU ops (vmatmuls) per grid iteration."""
    return (
        self._get_vmatmuls_per_latch() * self._get_num_latches_per_iteration()
    )

  def _get_latch_latency_cycles(self) -> int | float:
    """Returns the latency in cycles for a single RHS latch."""
    sublane_count, _ = self._platform_info.vreg_size
    mxu_contracting_dim, _ = self._get_mxu_size()
    rows_per_latch = sublane_count * (32 // self._bits_matmul_rhs)
    latch_cadence_cycles = self._platform_info.get_latch_cadence_cycles(
        self.matmul_rhs_dtype
    )
    return latch_cadence_cycles * (mxu_contracting_dim // rows_per_latch)

  def _get_matmul_stage_cost(self) -> PipelineStageCost:
    """Returns the throughput and latency of the core matmul.

    This assumes that we are able to achieve maximum FLOPS utilization and
    ignores contributions from the final vpop + vst ops. Most of the latter can
    be pipelined with the final vmatmuls in the loop and should be relatively
    small compared to the matmul latency.

    Returns:
      The matmul stage cost over the grid.
    """
    mxu_contracting_dim, mxu_non_contracting_dim = self._get_mxu_size()
    matmul_cadence_cycles = self._get_matmul_cadence_cycles()
    mxu_blocks_contracting_dim = self._block_k // mxu_contracting_dim
    mxu_blocks_non_contracting_dim = self._block_n // mxu_non_contracting_dim
    num_latches = self._get_num_latches_per_iteration()
    vmatmuls_per_latch = self._get_vmatmuls_per_latch()
    total_matmul_cycles = (
        matmul_cadence_cycles * vmatmuls_per_latch * num_latches
    )
    # If there is only one vmatmul per latch, then currently we reuse the same
    # MRB entry, triggering a hazard for all vmatmuls in an accumulation chain
    # except the last one by 4 cycles.
    additional_matmul_cycles = np.where(vmatmuls_per_latch == 1, 4, 0)
    total_matmul_cycles += (
        additional_matmul_cycles
        * (mxu_blocks_contracting_dim - 1)
        * mxu_blocks_non_contracting_dim
    )
    latch_latency_cycles = self._get_latch_latency_cycles()
    total_latch_cycles = latch_latency_cycles * num_latches
    matmul_latency_cycles = self._platform_info.get_matmul_latency_cycles(
        self.matmul_lhs_dtype
    )
    return PipelineStageCost(
        matmul_cycles=total_matmul_cycles,
        latch_cycles=total_latch_cycles,
        latency_cycles=latch_latency_cycles + matmul_latency_cycles,
    )

  def _get_input_fusion_stage_cost(self) -> PipelineStageCost:
    """Returns the throughput and latency of input fusions."""
    return PipelineStageCost()

  def _get_output_fusion_stage_cost(self) -> PipelineStageCost:
    """Returns the throughput and latency of output fusions."""
    return PipelineStageCost()

  def _xlu_cycles(
      self,
      num_ops: int | float | np.ndarray,
      op: platform_info.XluOp,
      dtype: jax.typing.DTypeLike,
  ) -> np.ndarray:
    """Returns XLU cycles for `num_ops` of `op` on `dtype` inputs."""
    cadence = self._platform_info.get_xlu_cadence_cycles(op, dtype)
    return (num_ops * cadence) / self._platform_info.num_xlus

  def _compute_latency_ns_per_iteration(
      self, p_state: int | None
  ) -> np.ndarray:
    """Computes total compute latency in ns per iteration over the grid.

    The iteration is modeled as many pipelined instances of
    `input fusion -> MXU op -> output fusion`. Each stage's throughput cycles
    are summed per hardware unit across stages, since stages that share a unit
    compete for its issue slots. The most heavily loaded unit determines the
    steady-state time, and every stage adds its fill/drain latency once.

    Args:
      p_state: The P-state to use for clock speed.

    Returns:
      The compute latency in ns per iteration over the grid.
    """
    clock_speed_ghz = self._platform_info.clock_speed_ghz_by_p_state[p_state]
    stages = (
        self._get_input_fusion_stage_cost(),
        self._get_matmul_stage_cost(),
        self._get_output_fusion_stage_cost(),
    )
    num_mxus = self._platform_info.num_mxus
    num_alus = self._platform_info.num_alus
    num_xlus = self._platform_info.num_xlus
    throughput_cycles = functools.reduce(
        np.maximum,
        (
            sum(s.matmul_cycles for s in stages) / num_mxus,
            sum(s.latch_cycles for s in stages) / num_mxus,
            sum(s.alu_cycles for s in stages) / num_alus,
            sum(s.xlu_cycles for s in stages) / num_xlus,
        ),
    )
    latency_cycles = sum(s.latency_cycles for s in stages)
    return np.broadcast_to(
        (throughput_cycles + latency_cycles) / clock_speed_ghz,
        self._grid_shape,
    )

  def _get_initial_operand_transfer_latency(self) -> np.ndarray:
    """Compute the latency of the initial operand transfers from HBM to VMEM."""
    hbm_to_vmem_latency_ns = self._platform_info.hbm_to_vmem_latency_ns
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    input_operands = [op for op in self._operands if not op.is_output]
    dma_sizes_in = np.stack(
        [
            np.broadcast_to(op.size_bytes, self._grid_shape)
            for op in input_operands
        ],
        axis=-1,
    )
    dma_latencies_in = np.full(
        (1, 1, 1, len(input_operands)), hbm_to_vmem_latency_ns
    )
    return _dma_latency(
        dma_sizes_in, dma_latencies_in, hbm_to_vmem_bandwidth_gb_per_sec
    )

  def _get_final_operand_transfer_latency(self) -> np.ndarray:
    """Compute the latency of the final operand transfers from VMEM to HBM."""
    vmem_to_hbm_latency_ns = self._platform_info.vmem_to_hbm_latency_ns
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    output_operands = [op for op in self._operands if op.is_output]
    dma_sizes_out = np.stack(
        [
            np.broadcast_to(op.size_bytes, self._grid_shape)
            for op in output_operands
        ],
        axis=-1,
    )
    dma_latencies_out = np.full(
        (1, 1, 1, len(output_operands)), vmem_to_hbm_latency_ns
    )
    return _dma_latency(
        dma_sizes_out, dma_latencies_out, hbm_to_vmem_bandwidth_gb_per_sec
    )

  def _get_loop_latency_excluding_final_inner_most_iteration(
      self, compute_latency_ns_per_iteration: np.ndarray
  ) -> np.ndarray:
    """Compute the loop latency excluding the final inner most iterations.

    Every loop iteration requires swapping out operands that vary along the
    inner-most non-trivial axis, except for the final inner-most iteration,
    which swaps out all operands for the next outer loop iteration. This
    function computes the cost of such iterations as the maximum of the
    per-iteration compute latency and the operand transfer latencies, and then
    sums them over all such iterations (e.g. grid_m * grid_n * (grid_k - 1)
    when K is the inner most non-trivial axis).

    Operands that fit in the window are ignored since they do not need to be
    swapped out.

    Args:
      compute_latency_ns_per_iteration: The total compute latency in ns per
        iteration over the grid.

    Returns:
      The loop latency in ns summed over all iterations except the final inner
      most iteration for the full grid.
    """
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    hbm_to_vmem_latency_ns = self._platform_info.hbm_to_vmem_latency_ns
    vmem_to_hbm_latency_ns = self._platform_info.vmem_to_hbm_latency_ns
    grid_m, grid_k, grid_n = self._grid_m, self._grid_k, self._grid_n
    grid_by_axis = {Axis.M: grid_m, Axis.K: grid_k, Axis.N: grid_n}
    inner_most_non_trivial_axis = self._inner_most_non_trivial_axis
    idx_map = self._idx_map
    dma_sizes_per_iteration = []
    dma_latencies_per_iteration = []
    for op in self._operands:
      ax0, ax1 = op.axes
      swap = (grid_by_axis[ax0] != 1) | (grid_by_axis[ax1] != 1)
      invariant_axis = self._get_invariant_axis(op)
      dma_sizes_per_iteration.append(
          (inner_most_non_trivial_axis != idx_map[invariant_axis])
          * swap
          * op.size_bytes
      )
      dma_latencies_per_iteration.append(
          vmem_to_hbm_latency_ns if op.is_output else hbm_to_vmem_latency_ns
      )
    dma_sizes_per_iteration = np.stack(dma_sizes_per_iteration, axis=-1)
    dma_latencies_per_iteration = np.array(dma_latencies_per_iteration)[
        None, None, None, :
    ]
    dma_latency_per_iteration = _dma_latency(
        dma_sizes_per_iteration,
        dma_latencies_per_iteration,
        hbm_to_vmem_bandwidth_gb_per_sec,
    )
    num_iterations = (
        (grid_m - (inner_most_non_trivial_axis == idx_map[Axis.M]))
        * (grid_k - (inner_most_non_trivial_axis == idx_map[Axis.K]))
        * (grid_n - (inner_most_non_trivial_axis == idx_map[Axis.N]))
    )
    return (
        np.maximum(
            dma_latency_per_iteration,
            compute_latency_ns_per_iteration,
        )
        * num_iterations
    )

  def _get_loop_latency_final_inner_most_iteration(
      self, compute_latency_ns_per_iteration: np.ndarray
  ) -> np.ndarray:
    """Compute the loop latency of the final inner most iteration.

    This is the latency of the final inner most iteration over the full grid,
    which requires swapping out all operands (except for the final iteration of
    the entire loop). We estimate this by taking the maximum of the compute
    latency and the operand transfer latencies, multiplied by the number of such
    iterations (i.e. grid_m * grid_n - 1 when K is the inner-most non-trivial
    axis), plus the compute latency of the final iteration (which does not
    require swapping).

    Args:
      compute_latency_ns_per_iteration: The total compute latency in ns per
        iteration over the grid.

    Returns:
      The loop latency in ns summed over the final inner most iteration for the
      full grid.
    """
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    hbm_to_vmem_latency_ns = self._platform_info.hbm_to_vmem_latency_ns
    vmem_to_hbm_latency_ns = self._platform_info.vmem_to_hbm_latency_ns
    grid_m, grid_k, grid_n = self._grid_m, self._grid_k, self._grid_n
    inner_most_non_trivial_axis = self._inner_most_non_trivial_axis
    idx_map = self._idx_map
    shape = self._grid_shape
    dma_sizes_per_iteration = np.stack(
        [np.broadcast_to(op.size_bytes, shape) for op in self._operands],
        axis=-1,
    )
    dma_latencies_per_iteration = np.array([
        vmem_to_hbm_latency_ns if op.is_output else hbm_to_vmem_latency_ns
        for op in self._operands
    ])[None, None, None, :]
    dma_latency_per_iteration = _dma_latency(
        dma_sizes_per_iteration,
        dma_latencies_per_iteration,
        hbm_to_vmem_bandwidth_gb_per_sec,
    )
    num_final_iterations = (
        np.maximum(
            grid_m * (inner_most_non_trivial_axis != idx_map[Axis.M]),
            np.ones_like(grid_m),
        )
        * np.maximum(
            grid_k * (inner_most_non_trivial_axis != idx_map[Axis.K]),
            np.ones_like(grid_k),
        )
        * np.maximum(
            grid_n * (inner_most_non_trivial_axis != idx_map[Axis.N]),
            np.ones_like(grid_n),
        )
    )
    return (
        np.maximum(
            dma_latency_per_iteration,
            compute_latency_ns_per_iteration,
        )
        * (num_final_iterations - 1)
        + compute_latency_ns_per_iteration
    )

  def select_window(
      self, vmem_limit_bytes: int, p_state: int | None = None
  ) -> tuple[int, int, int]:
    """Select the best window size for the given parameters.

    Args:
      vmem_limit_bytes: The VMEM limit in bytes.
      p_state: The P-state to use.

    Returns:
      A tuple of (block_m, block_k, block_n) representing the window size.
    """
    # Account for internal scratch bytes required by the compiler, which also
    # comes from scoped VMEM.
    vmem_limit_bytes -= self._platform_info.default_internal_scratch_bytes
    if p_state is None:
      # Note that p_state may still be None if the platform doesn't support
      # P-states.
      p_state = self._platform_info.default_p_state
    compute_latency_ns_per_iteration = self._compute_latency_ns_per_iteration(
        p_state
    )
    dma_latency_in = self._get_initial_operand_transfer_latency()
    loop_latency = self._get_loop_latency_excluding_final_inner_most_iteration(
        compute_latency_ns_per_iteration,
    ) + self._get_loop_latency_final_inner_most_iteration(
        compute_latency_ns_per_iteration,
    )
    dma_latency_out = self._get_final_operand_transfer_latency()
    total_latency = dma_latency_in + loop_latency + dma_latency_out
    vmem_usage_arr = np.asarray(self._vmem_usage_bytes)
    within_mem_usage = np.asarray(vmem_usage_arr <= vmem_limit_bytes)
    if not within_mem_usage.any():
      ind = int(np.argmin(vmem_usage_arr.ravel()))
      return self._block_specs.reshape(-1, 3)[ind]
    ind = np.argmin(total_latency[within_mem_usage].ravel())
    min_latency = total_latency[within_mem_usage].ravel()[ind]
    min_block_spec = self._block_specs[within_mem_usage, :].reshape(-1, 3)[ind]

    # Heuristic: having the full K dimension tends to outperform windows with
    # similar latencies but smaller K dimensions, due to not having to
    # accumulate across windows. If there is a window with full K that is within
    # 5% of the minimum latency, use that window instead.
    full_k = self._block_k.ravel()[-1]
    full_k_inds = within_mem_usage & (
        self._block_specs[..., self._idx_map[Axis.K]] == full_k
    )
    if full_k_inds.any():
      ind_full_k = total_latency[full_k_inds].ravel().argmin()
      min_latency_full_k = total_latency[full_k_inds].ravel()[ind_full_k]
      min_block_spec_full_k = self._block_specs[full_k_inds, :].reshape(-1, 3)[
          ind_full_k
      ]
      if (min_latency_full_k - min_latency) / min_latency < 0.05:
        return min_block_spec_full_k
    return min_block_spec


class DenseCostModel(CostModel):
  """Cost model specialized for Pallas dense matmul benchmarks."""


class SubchannelCostModel(CostModel):
  """Cost model specialized for Pallas subchannel quantized matmul benchmarks.

  Models:
  - Core matmul using `lhs_quantized_dtype` and `rhs_quantized_dtype`.
  - Additional scale operands: `rhs_scales` (and `lhs_scales` when
    `pre_quantize_lhs=True`).
  - Input fusion: on-the-fly LHS quantization (`quantize`) when
    `pre_quantize_lhs=False` (combining VPU intra-lane operations and XLU
    cross-lane reductions), plus any VPU conversion spill for emulated or
    sub-byte `lhs_quantized_dtype`.
  - Output fusion: per-subblock dequantization (`result * x_safe_scales *
    y_safe_scales`) and outer accumulation (`acc_ref += ...`).
  """

  # All ALU ops modeled below have a latency of 2 cycles, except vsel which has
  # a latency of 1 cycle. We correct for this by using 3.5 quantize ops below.
  _ALU_LATENCY_CYCLES = 2

  # Number of ALU ops per VREG for dynamic LHS quantization (intra-lane
  # vmaxabs, scale multiply, clipping, and vpack cast).
  _NUM_QUANTIZE_ALU_OPS = 3.5
  # Number of ALU ops per VREG for converting bf16 inputs to f32 on platforms
  # where all ALU and XLU ops must be done in float32 (vunpack.c.l and
  # vunpack.c.u).
  _NUM_CONVERT_BF16_TO_F32_ALU_OPS = 1
  # Number of ALU ops per VREG for unpacking an emulated quantized dtype (e.g.
  # s4 -> f8e4m3fn via vunpack.c.l, vunpack.c.u, and vcvt).
  _NUM_UNPACK_ALU_OPS = 3
  # Number of ALU ops per VREG for subchannel output dequantization (vpack, 2x
  # vmul) and accumulation (vadd).
  _NUM_DEQUANT_ACCUM_ALU_OPS = 4
  _MAX_COMPUTE_TILE_N = 2048

  def __init__(
      self,
      m: int,
      k: int,
      n: int,
      lhs_mem: pltpu.MemorySpace,
      rhs_mem: pltpu.MemorySpace,
      out_mem: pltpu.MemorySpace,
      lhs_dtype: jax.typing.DTypeLike,
      rhs_dtype: jax.typing.DTypeLike,
      out_dtype: jax.typing.DTypeLike,
      outer_acc_dtype: jax.typing.DTypeLike,
      lhs_quantized_dtype: jax.typing.DTypeLike,
      rhs_quantized_dtype: jax.typing.DTypeLike,
      *,
      inner_acc_dtype: jax.typing.DTypeLike = jnp.dtype(jnp.float32),
      subchannel_size: int = 1024,
      pre_quantize_lhs: bool = False,
      chip_version: pltpu.ChipVersion | None = None,
  ):
    self.unquantized_lhs_dtype = jnp.dtype(lhs_dtype)
    self.rhs_scales_dtype = jnp.dtype(rhs_dtype)
    self.inner_acc_dtype = jnp.dtype(inner_acc_dtype)
    self.outer_acc_dtype = jnp.dtype(outer_acc_dtype)
    self.lhs_quantized_dtype = jnp.dtype(lhs_quantized_dtype)
    self.rhs_quantized_dtype = jnp.dtype(rhs_quantized_dtype)
    self.subchannel_size = subchannel_size
    self.pre_quantize_lhs = pre_quantize_lhs
    self._bits_lhs_scales = jax.dtypes.itemsize_bits(self.unquantized_lhs_dtype)
    self._bits_rhs_scales = jax.dtypes.itemsize_bits(self.rhs_scales_dtype)
    self._bits_inner_acc = jax.dtypes.itemsize_bits(self.inner_acc_dtype)

    operand_lhs_dtype = (
        self.lhs_quantized_dtype
        if pre_quantize_lhs
        else self.unquantized_lhs_dtype
    )
    pinfo = platform_info.get_platform_info(chip_version)
    if jnp.issubdtype(
        self.lhs_quantized_dtype, jnp.floating
    ) and jnp.issubdtype(self.rhs_quantized_dtype, jnp.integer):
      supported_latch_dtypes = platform_info.DtypeDict({
          dt: cycles
          for dt, cycles in pinfo.latch_cadence_cycles_by_dtype.items()
          if jnp.issubdtype(dt, jnp.floating)
      })
      matmul_rhs_dtype = pinfo.get_emulated_dtype(
          self.rhs_quantized_dtype, supported_latch_dtypes
      )
    else:
      matmul_rhs_dtype = self.rhs_quantized_dtype
    super().__init__(
        m=m,
        k=k,
        n=n,
        lhs_mem=lhs_mem,
        rhs_mem=rhs_mem,
        out_mem=out_mem,
        lhs_dtype=operand_lhs_dtype,
        rhs_dtype=self.rhs_quantized_dtype,
        out_dtype=out_dtype,
        acc_dtype=self.outer_acc_dtype,
        matmul_lhs_dtype=self.lhs_quantized_dtype,
        matmul_rhs_dtype=matmul_rhs_dtype,
        chip_version=chip_version,
    )

  def _get_tile_sizes(self) -> tuple[int, int, int]:
    m_tile_size, k_tile_size, n_tile_size = super()._get_tile_sizes()
    return m_tile_size, max(k_tile_size, self.subchannel_size), n_tile_size

  def _get_additional_operands(self) -> Sequence[Operand]:
    subblocks_per_tile = self._block_k // self.subchannel_size
    rhs_scales_bytes = (
        subblocks_per_tile * self._block_n * self._bits_rhs_scales // 8
    )
    operands = [
        Operand(rhs_scales_bytes, (Axis.K, Axis.N), is_output=False),
    ]
    if self.pre_quantize_lhs:
      lhs_scales_bytes = (
          self._block_m * subblocks_per_tile * self._bits_lhs_scales // 8
      )
      operands.append(
          Operand(lhs_scales_bytes, (Axis.M, Axis.K), is_output=False)
      )
    return operands

  def _get_input_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    vmem_bytes = self._get_lhs_conversion_vmem_usage_bytes(
        k_dim=self.subchannel_size
    )
    if not self.pre_quantize_lhs:
      # Dynamically quantized x_q_block of shape (block_m, subchannel_size) is
      # reused across steps_n in the inner loop. Avoid double-counting when
      # _get_lhs_conversion_vmem_usage_bytes already accounts for it.
      x_q_vmem_bytes = (
          self._block_m * self.subchannel_size * self._bits_matmul_lhs // 8
      )
      vmem_bytes = np.maximum(vmem_bytes, x_q_vmem_bytes)
    return vmem_bytes

  def _get_output_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    compute_tile_n = np.minimum(self._block_n, self._MAX_COMPUTE_TILE_N)
    subblock_acc_bits = max(self._bits_inner_acc, self._bits_acc)
    return self._block_m * compute_tile_n * subblock_acc_bits // 8

  def _get_input_fusion_dtype(self) -> jnp.dtype:
    """Returns the precision for input fusion VPU and XLU ops."""
    if self._platform_info.chip_version in (
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    ):
      return jnp.dtype(jnp.float32)
    if jnp.dtype(self.unquantized_lhs_dtype) == jnp.dtype(jnp.float32):
      return jnp.dtype(jnp.float32)
    return jnp.dtype(jnp.bfloat16)

  def _get_output_fusion_dtype(self) -> jnp.dtype:
    """Returns the precision for output fusion VPU ops."""
    if self._platform_info.chip_version in (
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    ):
      return jnp.dtype(jnp.float32)
    if jnp.dtype(self.outer_acc_dtype) == jnp.dtype(jnp.float32):
      return jnp.dtype(jnp.float32)
    return jnp.dtype(jnp.bfloat16)

  def _num_vregs(
      self,
      rows: int | float | np.ndarray,
      cols: int | float | np.ndarray,
      dtype: jax.typing.DTypeLike,
  ) -> np.ndarray:
    """Returns the number of VREGs for a (rows, cols) tile with precision `dtype`."""
    sublane_count, lane_count = self._platform_info.vreg_size
    bitwidth = jax.dtypes.itemsize_bits(dtype)
    rows_per_vreg = sublane_count * (32 // bitwidth)
    return np.ceil(rows / rows_per_vreg) * np.ceil(cols / lane_count)

  def _alu_cycles(
      self,
      rows: int | float | np.ndarray,
      cols: int | float | np.ndarray,
      num_ops: int | float,
      dtype: jax.typing.DTypeLike,
  ) -> np.ndarray:
    """Returns unnormalized ALU throughput cycles for `num_ops` on a (rows, cols) tile."""
    return (
        self._num_vregs(rows, cols, dtype) * num_ops * self._ALU_LATENCY_CYCLES
    )

  def _alu_latency_cycles(
      self,
      rows: int | float | np.ndarray,
      cols: int | float | np.ndarray,
      num_ops: int | float,
      dtype: jax.typing.DTypeLike,
  ) -> np.ndarray:
    """Returns latency cycles to execute `num_ops` on a (rows, cols) tile across ALUs."""
    return (
        self._alu_cycles(rows, cols, num_ops, dtype)
        / self._platform_info.num_alus
    )

  def _get_input_fusion_stage_cost(self) -> PipelineStageCost:
    mxu_contracting_dim, mxu_non_contracting_dim = self._get_mxu_size()
    # Rows of LHS fed to the first MXU op (block_m may be smaller than a full
    # vmatmul for narrow LHS dtypes).
    first_op_rows = np.minimum(self._block_m, self._get_rows_per_vmatmul())
    input_dtype = self._get_input_fusion_dtype()
    # VPU ops share the VALUs, so their throughput cycles add up (including
    # with the output fusion's VPU work). The XLU is a separate functional unit
    # that pipelines against the VPU; units are balanced against each other in
    # `_compute_latency_ns_per_iteration`.
    alu_cycles: float | np.ndarray = 0.0
    xlu_cycles: float | np.ndarray = 0.0
    latency_cycles: float | np.ndarray = 0.0
    if not self.pre_quantize_lhs:
      # The per-subchannel maxabs (`jnp.max(jnp.abs(x))` in the unquantized LHS
      # dtype) is an intra-lane maxabs on the VPU followed by one xlane reduce
      # max per vreg on the XLU.
      xlu_op = platform_info.XluOp.XLANE_REDUCE
      xlu_dtype = self._platform_info.get_xlu_dtype(xlu_op, input_dtype)
      sublane_count, _ = self._platform_info.vreg_size
      rows_per_xlu_op = sublane_count * (
          32 // jax.dtypes.itemsize_bits(xlu_dtype)
      )
      subblocks_per_tile = self._block_k // self.subchannel_size
      alu_cycles = alu_cycles + self._alu_cycles(
          self._block_m, self._block_k, self._NUM_QUANTIZE_ALU_OPS, input_dtype
      )
      num_xlu_ops = subblocks_per_tile * np.ceil(
          self._block_m / rows_per_xlu_op
      )
      xlu_cadence = self._platform_info.get_xlu_cadence_cycles(
          xlu_op, xlu_dtype
      )
      xlu_cycles = xlu_cycles + num_xlu_ops * xlu_cadence
      first_op_xlu_ops = np.ceil(first_op_rows / rows_per_xlu_op)
      # The scale depends on the amax over a full subchannel, so the first MXU
      # op has to wait for the quantization of a full subchannel of its rows,
      # including an xlane reduce max through the XLU pipeline.
      latency_cycles = (
          latency_cycles
          + self._alu_latency_cycles(
              first_op_rows,
              self.subchannel_size,
              self._NUM_QUANTIZE_ALU_OPS,
              input_dtype,
          )
          + (first_op_xlu_ops * xlu_cadence) / self._platform_info.num_xlus
          + self._platform_info.get_xlu_latency_cycles(xlu_op, xlu_dtype)
      )
      if self._platform_info.chip_version in (
          pltpu.ChipVersion.TPU_V5E,
          pltpu.ChipVersion.TPU_V5P,
      ) and self.unquantized_lhs_dtype == jnp.dtype(jnp.bfloat16):
        alu_cycles = alu_cycles + self._alu_cycles(
            self._block_m,
            self._block_k,
            self._NUM_CONVERT_BF16_TO_F32_ALU_OPS,
            input_dtype,
        )
        latency_cycles = latency_cycles + self._alu_latency_cycles(
            first_op_rows,
            self.subchannel_size,
            self._NUM_CONVERT_BF16_TO_F32_ALU_OPS,
            input_dtype,
        )
    if self.lhs_quantized_dtype != self.matmul_lhs_dtype:
      alu_cycles = alu_cycles + self._alu_cycles(
          self._block_m, self._block_k, self._NUM_UNPACK_ALU_OPS, input_dtype
      )
      latency_cycles = latency_cycles + self._alu_latency_cycles(
          first_op_rows,
          mxu_contracting_dim,
          self._NUM_UNPACK_ALU_OPS,
          input_dtype,
      )
    if self.rhs_quantized_dtype != self.matmul_rhs_dtype:
      alu_cycles = alu_cycles + self._alu_cycles(
          self._block_k, self._block_n, self._NUM_UNPACK_ALU_OPS, input_dtype
      )
      latency_cycles = latency_cycles + self._alu_latency_cycles(
          mxu_contracting_dim,
          mxu_non_contracting_dim,
          self._NUM_UNPACK_ALU_OPS,
          input_dtype,
      )
    return PipelineStageCost(
        alu_cycles=alu_cycles,
        xlu_cycles=xlu_cycles,
        latency_cycles=latency_cycles,
    )

  def _get_output_fusion_stage_cost(self) -> PipelineStageCost:
    _, mxu_non_contracting_dim = self._get_mxu_size()
    subblocks_per_tile = self._block_k // self.subchannel_size
    output_dtype = self._get_output_fusion_dtype()
    total_cycles = subblocks_per_tile * self._alu_cycles(
        self._block_m,
        self._block_n,
        self._NUM_DEQUANT_ACCUM_ALU_OPS,
        output_dtype,
    )
    # Draining the pipeline requires dequantizing and accumulating the result
    # tile of the last MXU op.
    last_op_rows = np.minimum(self._block_m, self._get_rows_per_vmatmul())
    return PipelineStageCost(
        alu_cycles=total_cycles,
        latency_cycles=self._alu_latency_cycles(
            last_op_rows,
            mxu_non_contracting_dim,
            self._NUM_DEQUANT_ACCUM_ALU_OPS,
            output_dtype,
        ),
    )
