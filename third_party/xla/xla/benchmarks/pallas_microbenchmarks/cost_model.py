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

import enum
import math

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.core import platform_info  # pylint: disable=g-direct-tensorflow-import


def _vmem_usage_bytes(
    m: int,
    k: int,
    n: int,
    block_m: int | np.ndarray,
    block_k: int | np.ndarray,
    block_n: int | np.ndarray,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    sparse_rhs: bool,
    sp_n: int,
    sp_m: int,
) -> int | np.ndarray:
  """Calculates the VMEM usage in bytes for a matmul with the given parameters.

  Args:
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    block_m: The block size for the m dimension, or an array of block sizes.
    block_k: The block size for the k dimension, or an array of block sizes.
    block_n: The block size for the n dimension, or an array of block sizes.
    lhs_dtype: The dtype of the first operand.
    rhs_dtype: The dtype of the second operand.
    out_dtype: The dtype of the output.
    acc_dtype: The dtype of the accumulator.
    sparse_rhs: Whether the RHS operand is sparse.
    sp_n: The N value of the N:M sparsity pattern.
    sp_m: The M value of the N:M sparsity pattern.

  Returns:
    The estimated VMEM usage in bytes, or an array of VMEM usages for all
    block size combinations.
  """

  def _vmem_for_operand(d1, d2, block_d1, block_d2, dtype):
    return (
        # Double-buffer if the operand doesn't fit in the window.
        (2 - ((d1 == block_d1) & (d2 == block_d2)))
        * block_d1
        * block_d2
        * jax.dtypes.itemsize_bits(dtype)
        // 8
    )

  rhs_vmem_usage = _vmem_for_operand(k, n, block_k, block_n, rhs_dtype)
  if sparse_rhs:
    rhs_sp_indices_vmem_usage = _vmem_for_operand(
        k, n, block_k, block_n, jnp.int2
    )
    rhs_vmem_usage = (
        (rhs_vmem_usage + rhs_sp_indices_vmem_usage) * sp_n // sp_m
    )
  return (
      _vmem_for_operand(m, k, block_m, block_k, lhs_dtype)
      + rhs_vmem_usage
      + _vmem_for_operand(m, n, block_m, block_n, out_dtype)
      # Only one accumulator tile is needed.
      + block_m * block_n * jax.dtypes.itemsize_bits(acc_dtype) // 8
  )


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


class CostModel:
  """A cost model for Pallas kernels.

  Initially handles simple kernels with a single matmul. The cost model works
  by estimating the total latency of the kernel as the sum of three parts:
  1. The latency of the initial operand transfers from HBM to VMEM.
  2. The latency of each loop iteration as the maximum of the matmul latency
  and the operand transfer latencies for the next iteration.
  3. The latency of the final operand transfer from VMEM to HBM.
  A best window is selected by doing a brute force computation over a grid of
  possible window sizes.

  Attributes:
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    lhs_mem: The memory space of the first operand.
    rhs_mem: The memory space of the second operand.
    out_mem: The memory space of the output.
    lhs_dtype: The dtype of the first operand.
    rhs_dtype: The dtype of the second operand.
    out_dtype: The dtype of the output.
    acc_dtype: The dtype of the accumulator.
    sparse_rhs: Whether the RHS operand is sparse.
    sp_n: The N value of the N:M sparsity pattern.
    sp_m: The M value of the N:M sparsity pattern.
    chip_version: The TPU chip version. If not provided, the chip version of the
      current device will be used.
  """

  def __init__(
      self,
      m: int,
      k: int,
      n: int,
      lhs_mem: pltpu.MemorySpace,
      rhs_mem: pltpu.MemorySpace,
      out_mem: pltpu.MemorySpace,
      lhs_dtype: jnp.dtype,
      rhs_dtype: jnp.dtype,
      out_dtype: jnp.dtype,
      acc_dtype: jnp.dtype,
      sparse_rhs: bool = False,
      sp_n: int = 1,
      sp_m: int = 1,
      chip_version: pltpu.ChipVersion | None = None,
  ):
    self._platform_info: platform_info.PlatformInfo = (
        platform_info.get_platform_info(chip_version)
    )
    sublane_count, _ = self._platform_info.vreg_size
    mxu_contracting_dim, mxu_non_contracting_dim = (
        self._platform_info.mxu_size_by_dtype(lhs_dtype)
    )

    # Window sizes must be multiples of sublane and lane counts, but we use
    # MXU dimensions instead of lane count to maximize FLOPS utilization.
    m_tile_size = sublane_count * (32 // jax.dtypes.itemsize_bits(lhs_dtype))
    k_tile_size = mxu_contracting_dim
    n_tile_size = mxu_non_contracting_dim
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
    self.lhs_dtype = lhs_dtype
    self.rhs_dtype = rhs_dtype
    self.out_dtype = out_dtype
    self.acc_dtype = acc_dtype
    self.sparse_rhs = sparse_rhs
    self.sp_n = sp_n
    self.sp_m = sp_m
    self._bits_lhs = jax.dtypes.itemsize_bits(lhs_dtype)
    self._bits_rhs = jax.dtypes.itemsize_bits(rhs_dtype)
    self._bits_out = jax.dtypes.itemsize_bits(out_dtype)
    self._lhs_size_bytes = self._block_m * self._block_k * self._bits_lhs // 8
    self._rhs_size_bytes = self._block_k * self._block_n * self._bits_rhs // 8
    if sparse_rhs:
      self._rhs_size_bytes = self._rhs_size_bytes * sp_n // sp_m
      sp_indices_bits = math.ceil(math.log2(sp_m))
      self._rhs_sp_indices_size_bytes = (
          self._block_k * self._block_n * sp_indices_bits // 8 * sp_n // sp_m
      )
    self._out_size_bytes = self._block_m * self._block_n * self._bits_out // 8

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
    self._vmem_usage_bytes = _vmem_usage_bytes(
        m,
        k,
        n,
        self._block_m,
        self._block_k,
        self._block_n,
        lhs_dtype,
        rhs_dtype,
        out_dtype,
        acc_dtype,
        sparse_rhs,
        sp_n,
        sp_m,
    )

  def _compute_matmul_latency_ns_per_iteration(
      self, p_state: int | None
  ) -> np.ndarray:
    """Compute the matmul latency in ns per iteration for the given P-state.

    This assumes that we are able to achieve maximum FLOPS utilization and
    ignores contributions from the initial latch and final vpop + vst ops.
    Most of the latter can be pipelined with the final vmatmuls in the loop
    and should be relatively small compared to the matmul latency.

    Args:
      p_state: The P-state to use for clock speed.

    Returns:
      The matmul latency in ns per iteration over the grid.
    """
    num_mxus = self._platform_info.num_mxus
    sublane_count, _ = self._platform_info.vreg_size
    mxu_contracting_dim, mxu_non_contracting_dim = (
        self._platform_info.mxu_size_by_dtype(self.lhs_dtype)
    )
    clock_speed_ghz = self._platform_info.clock_speed_ghz_by_p_state[p_state]
    matmul_cadence_cycles = self._platform_info.matmul_cadence_cycles_by_dtype[
        self.lhs_dtype
    ]
    mxu_blocks_contracting_dim = self._block_k // mxu_contracting_dim
    mxu_blocks_non_contracting_dim = self._block_n // mxu_non_contracting_dim
    num_latches = mxu_blocks_contracting_dim * mxu_blocks_non_contracting_dim
    rows_per_vmatmul = sublane_count * (32 / self._bits_lhs)
    vmatmuls_per_latch = np.ceil(self._block_m / rows_per_vmatmul)
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
    rows_per_latch = sublane_count * (32 // self._bits_rhs)
    latch_cadence_cycles = self._platform_info.latch_cadence_cycles_by_dtype[
        self.rhs_dtype
    ]
    latch_latency_cycles = latch_cadence_cycles * (
        mxu_contracting_dim // rows_per_latch
    )
    # Some of the latch cycles can be pipelined with the vmatmul.
    matmul_cycles_between_latches = (
        vmatmuls_per_latch * matmul_cadence_cycles + additional_matmul_cycles
    )
    latch_overhead_cycles = np.maximum(
        latch_latency_cycles - matmul_cycles_between_latches, 0
    )
    # The first latch can't be pipelined with the vmatmul, so it has full
    # overhead.
    latch_overhead_ns = (
        latch_latency_cycles
        + latch_overhead_cycles * (num_latches / num_mxus - 1)
    ) / clock_speed_ghz
    matmul_latency_cycles = self._platform_info.matmul_latency_cycles_by_dtype[
        self.lhs_dtype
    ]
    # The first and last matmul_latency_cycles of the iteration will not have
    # full MXU utilization. We approximate this by assuming linear rampup from
    # 0% to 100% utilization over the first and last matmul_latency_cycles. With
    # an average utilization of 50%, these two intervals add up to
    # matmul_latency_cycles of overhead.
    matmul_overhead_ns = matmul_latency_cycles / clock_speed_ghz
    # Assume matmul cycles can be evenly split across MXUs.
    return (
        total_matmul_cycles / num_mxus / clock_speed_ghz
        + latch_overhead_ns
        + matmul_overhead_ns
    )

  def _get_initial_operand_transfer_latency(self):
    """Compute the latency of the initial operand transfers from HBM to VMEM."""
    hbm_to_vmem_latency_ns = self._platform_info.hbm_to_vmem_latency_ns
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    dma_sizes_in = [
        np.broadcast_to(self._lhs_size_bytes, self._grid_shape),
        np.broadcast_to(self._rhs_size_bytes, self._grid_shape),
    ]
    dma_latencies_in = [hbm_to_vmem_latency_ns, hbm_to_vmem_latency_ns]
    if self.sparse_rhs:
      dma_sizes_in.append(
          np.broadcast_to(
              self._rhs_sp_indices_size_bytes, self._grid_shape
          )
      )
      dma_latencies_in.append(hbm_to_vmem_latency_ns)
    dma_sizes_in = np.stack(dma_sizes_in, axis=-1)
    dma_latencies_in = np.array(dma_latencies_in)[None, None, None, :]
    return _dma_latency(
        dma_sizes_in, dma_latencies_in, hbm_to_vmem_bandwidth_gb_per_sec
    )

  def _get_final_operand_transfer_latency(self):
    """Compute the latency of the final operand transfers from VMEM to HBM."""
    vmem_to_hbm_latency_ns = self._platform_info.vmem_to_hbm_latency_ns
    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    dma_sizes_out = self._out_size_bytes[..., None]
    dma_latencies_out = np.array([vmem_to_hbm_latency_ns])[None, None, None, :]
    return _dma_latency(
        dma_sizes_out, dma_latencies_out, hbm_to_vmem_bandwidth_gb_per_sec
    )

  def _get_loop_latency_excluding_final_inner_most_iteration(
      self, matmul_latency_ns_per_iteration: np.ndarray
  ) -> np.ndarray:
    """Compute the loop latency excluding the final inner most iterations.

    Every loop iteration requires swapping out two of the three operands,
    except for the final iteration, which swaps out all three for the next
    inner most loop. This function computes the cost of such iterations as the
    maximum of the matmul latency and the operand transfer latencies, and then
    sums them over all such iterations (e.g. grid_m * grid_n * (grid_k - 1)
    when K is the inner most non-trivial axis).

    Operands that fit in the window are ignored since they do not need to be
    swapped out.

    Args:
      matmul_latency_ns_per_iteration: The matmul latency in ns per iteration
        over the grid.

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
    swap_lhs = (grid_m != 1) | (grid_k != 1)
    swap_rhs = (grid_k != 1) | (grid_n != 1)
    swap_out = (grid_m != 1) | (grid_n != 1)
    inner_most_non_trivial_axis = self._inner_most_non_trivial_axis
    idx_map = self._idx_map
    dma_sizes_per_iteration = [
        (inner_most_non_trivial_axis != idx_map[Axis.N])
        * swap_lhs
        * self._lhs_size_bytes,
        (inner_most_non_trivial_axis != idx_map[Axis.M])
        * swap_rhs
        * self._rhs_size_bytes,
        (inner_most_non_trivial_axis != idx_map[Axis.K])
        * swap_out
        * self._out_size_bytes,
    ]
    dma_latencies_per_iteration = [
        hbm_to_vmem_latency_ns,
        hbm_to_vmem_latency_ns,
        vmem_to_hbm_latency_ns,
    ]
    if self.sparse_rhs:
      dma_sizes_per_iteration.append(
          (inner_most_non_trivial_axis != idx_map[Axis.M])
          * swap_rhs
          * self._rhs_sp_indices_size_bytes
      )
      dma_latencies_per_iteration.append(hbm_to_vmem_latency_ns)
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
            matmul_latency_ns_per_iteration,
        )
        * num_iterations
    )

  def _get_loop_latency_final_inner_most_iteration(
      self, matmul_latency_ns_per_iteration: np.ndarray
  ) -> np.ndarray:
    """Compute the loop latency of the final inner most iteration.

    This is the latency of the final inner most iteration over the full grid,
    which requires swapping out all three operands (except for the final
    iteration of the entire loop). We estimate this by taking the maximum of
    the matmul latency and the operand transfer latencies, multiplied by the
    number of such iterations (i.e. grid_m * grid_n - 1 when K is the inner-most
    non-trivial axis), plus the matmul latency of the final iteration (which
    does not require swapping).

    Args:
      matmul_latency_ns_per_iteration: The matmul latency in ns per iteration
        over the grid.

    Returns:
      The loop latency in ns summed over the final inner most iteration for the
      full grid.
    """

    hbm_to_vmem_bandwidth_gb_per_sec = (
        self._platform_info.hbm_to_vmem_bandwidth_gb_per_sec
    )
    hbm_to_vmem_latency_ns = self._platform_info.hbm_to_vmem_latency_ns
    vmem_to_hbm_latency_ns = self._platform_info.vmem_to_hbm_latency_ns
    lhs_size_bytes = self._lhs_size_bytes
    rhs_size_bytes = self._rhs_size_bytes
    out_size_bytes = self._out_size_bytes
    grid_m, grid_k, grid_n = self._grid_m, self._grid_k, self._grid_n
    inner_most_non_trivial_axis = self._inner_most_non_trivial_axis
    idx_map = self._idx_map
    shape = self._grid_shape
    dma_sizes_per_iteration = [
        np.broadcast_to(lhs_size_bytes, shape),
        np.broadcast_to(rhs_size_bytes, shape),
        np.broadcast_to(out_size_bytes, shape),
    ]
    dma_latencies_per_iteration = [
        hbm_to_vmem_latency_ns,
        hbm_to_vmem_latency_ns,
        vmem_to_hbm_latency_ns,
    ]
    if self.sparse_rhs:
      dma_sizes_per_iteration.append(
          np.broadcast_to(self._rhs_sp_indices_size_bytes, shape)
      )
      dma_latencies_per_iteration.append(hbm_to_vmem_latency_ns)
    dma_sizes_per_iteration = np.stack(dma_sizes_per_iteration, axis=-1)
    dma_latencies_per_iteration = np.array(dma_latencies_per_iteration)[
        None, None, None, :
    ]
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
            matmul_latency_ns_per_iteration,
        )
        * (num_final_iterations - 1)
        + matmul_latency_ns_per_iteration
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
    if p_state is None:
      # Note that p_state may still be None if the platform doesn't support
      # P-states.
      p_state = self._platform_info.default_p_state
    matmul_latency_ns_per_iteration = (
        self._compute_matmul_latency_ns_per_iteration(p_state)
    )
    dma_latency_in = self._get_initial_operand_transfer_latency()
    loop_latency = self._get_loop_latency_excluding_final_inner_most_iteration(
        matmul_latency_ns_per_iteration,
    ) + self._get_loop_latency_final_inner_most_iteration(
        matmul_latency_ns_per_iteration,
    )
    dma_latency_out = self._get_final_operand_transfer_latency()
    total_latency = dma_latency_in + loop_latency + dma_latency_out
    within_mem_usage = self._vmem_usage_bytes <= vmem_limit_bytes
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
