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

"""Script for benchmarking Pallas subchannel quantization matmuls on TPUs."""

import dataclasses
import math
from typing import Any, Callable, Sequence

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from xla.benchmarks.core import benchmark
from xla.benchmarks.core import flag_utils
from xla.benchmarks.core import platform_info
from xla.benchmarks.pallas_microbenchmarks import cost_model as pallas_cost_model
from xla.benchmarks.pallas_microbenchmarks import memory_utils

InputSpec = benchmark.InputSpec


_KERNEL_NAME_TEMPLATE = "subchannel_matmul_{m}_{k}_{n}_{lhs_dtype}-{lhs_quantized_dtype}_{rhs_dtype}-{rhs_quantized_dtype}_{out_dtype}"


def _get_dtype_max_val(dtype: jnp.dtype) -> float:
  """Returns the maximum value of a given dtype.

  Intentionally does not support float32, bfloat16, or other non-quantized
  dtypes.

  Args:
    dtype: The dtype to get the maximum value of.

  Returns:
    The maximum value of the given dtype.
  """
  if dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float4_e2m1fn):
    return float(jnp.finfo(dtype).max)
  elif dtype in (jnp.int8, jnp.int4):
    return float(jnp.iinfo(dtype).max)
  else:
    raise ValueError(f"Unsupported quantized dtype: {dtype}")


def quantize(arr: jax.Array, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
  """Quantizes an array to a given dtype.

  Args:
    arr: The array to quantize.
    dtype: The dtype to quantize to.

  Returns:
    A tuple of the quantized array and the scales for multiplying after the
    matmul block prior to accumulation.
  """
  max_val = _get_dtype_max_val(dtype)
  inv_range = 1.0 / max_val
  absmaxs = jnp.max(jnp.abs(arr), axis=1, keepdims=True)
  scales = absmaxs * inv_range
  safe_scales = jnp.where(scales == 0.0, 1.0, scales)
  inv_scales = 1.0 / safe_scales
  return (arr * inv_scales).clip(-max_val, max_val).astype(dtype), safe_scales


def select_window(
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
    lhs_quantized_dtype: jnp.dtype,
    rhs_quantized_dtype: jnp.dtype,
    pre_quantize_lhs: bool = False,
    chip_version: pltpu.ChipVersion | None = None,
) -> tuple[int, int, int]:
  """Derives the optimal window size for subchannel matmul using the cost model."""
  # Unused for now, may be used if `pre_quantize_rhs` is added in the future.
  del rhs_dtype
  default_vmem_kib = platform_info.get_default_vmem_limit_kib(chip_version)
  vmem_limit_kib = flag_utils.get_flag_value(
      "xla_tpu_scoped_vmem_limit_kib", default=default_vmem_kib, flag_type=int
  )
  vmem_limit_bytes = vmem_limit_kib * 1024
  p_state = flag_utils.get_flag_value(
      "xla_tpu_dvfs_p_state", default=None, flag_type=int
  )
  if p_state is not None and p_state < 0:
    p_state = None
  cost_lhs_dtype = lhs_quantized_dtype if pre_quantize_lhs else lhs_dtype
  cost_rhs_dtype = rhs_quantized_dtype
  block_m, block_k, block_n = pallas_cost_model.CostModel(
      m,
      k,
      n,
      lhs_mem,
      rhs_mem,
      out_mem,
      cost_lhs_dtype,
      cost_rhs_dtype,
      out_dtype,
      acc_dtype,
      chip_version=chip_version,
  ).select_window(
      vmem_limit_bytes,
      p_state,
  )
  return int(block_m), int(block_k), int(block_n)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubchannelMatmulConfig(benchmark.BenchmarkConfig):
  """Config for Pallas subchannel quantized matmul benchmark.

  Attributes:
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    block_m: The block size for the m dimension.
    block_k: The block size for the k dimension.
    block_n: The block size for the n dimension.
    subchannel_size: Subchannel tile size along the k dimension.
    lhs_mem: The memory space of the first operand.
    rhs_mem: The memory space of the second operand.
    out_mem: The memory space of the output.
    lhs_dtype: The unquantized dtype of the first operand.
    rhs_dtype: The unquantized dtype of the second operand.
    out_dtype: The dtype of the output.
    acc_dtype: The dtype of the accumulator.
    lhs_quantized_dtype: Quantized dtype of LHS.
    rhs_quantized_dtype: Quantized dtype of RHS.
    pre_quantize_lhs: Whether quantization of LHS happens outside kernel.
  """

  m: int
  k: int
  n: int
  block_m: int
  block_k: int
  block_n: int
  subchannel_size: int
  lhs_mem: pltpu.MemorySpace
  rhs_mem: pltpu.MemorySpace
  out_mem: pltpu.MemorySpace
  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  lhs_quantized_dtype: jnp.dtype
  rhs_quantized_dtype: jnp.dtype
  pre_quantize_lhs: bool = False

  def get_benchmark(self) -> benchmark.Benchmark:
    return SubchannelMatmulBenchmark(self)


def subchannel_matmul_kernel(
    cfg: SubchannelMatmulConfig,
    kernel_name: str,
) -> Callable[..., Any]:
  """Returns a Pallas kernel for subchannel quantized matmul."""
  m, k, n = cfg.m, cfg.k, cfg.n
  block_m, block_k, block_n = cfg.block_m, cfg.block_k, cfg.block_n
  subchannel_size = cfg.subchannel_size
  lhs_mem, rhs_mem, out_mem = cfg.lhs_mem, cfg.rhs_mem, cfg.out_mem
  acc_dtype = cfg.acc_dtype

  grid_m = math.ceil(m / block_m)
  grid_n = math.ceil(n / block_n)
  grid_k = math.ceil(k / block_k)

  subblocks_per_tile = block_k // subchannel_size
  compute_tile_n = min(block_n, 2048)
  steps_n = block_n // compute_tile_n

  @jax.jit
  def _subchannel_matmul_kernel(
      lhs_ref, rhs_ref, lhs_scales_ref, rhs_scales_ref, o_tile_ref, acc_ref
  ):

    @pl.when(pl.program_id(2) == 0)
    def init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    for sub_idx in range(subblocks_per_tile):
      k_start = sub_idx * subchannel_size
      k_end = (sub_idx + 1) * subchannel_size

      if cfg.pre_quantize_lhs:
        assert lhs_scales_ref is not None
        x_q_block = lhs_ref[:, k_start:k_end]
        x_safe_scales = lhs_scales_ref[:, sub_idx : sub_idx + 1]
      else:
        assert lhs_scales_ref is None
        x_block = lhs_ref[:, k_start:k_end]
        x_q_block, x_safe_scales = quantize(x_block, cfg.lhs_quantized_dtype)

      for n_idx in range(steps_n):
        n_start = n_idx * compute_tile_n
        n_end = (n_idx + 1) * compute_tile_n

        y_q_slice = rhs_ref[k_start:k_end, n_start:n_end]

        result = jnp.dot(
            x_q_block, y_q_slice, preferred_element_type=jnp.float32
        ).astype(acc_dtype)

        y_safe_scales = rhs_scales_ref[sub_idx : sub_idx + 1, n_start:n_end]
        acc_ref[:, n_start:n_end] += (
            result
            * x_safe_scales.astype(acc_dtype)
            * y_safe_scales.astype(acc_dtype)
        )

    @pl.when(pl.program_id(2) == grid_k - 1)
    def store():
      o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)

  lhs_bits = jax.dtypes.itemsize_bits(cfg.lhs_dtype)
  lhs_quant_bits = jax.dtypes.itemsize_bits(cfg.lhs_quantized_dtype)
  lhs_input_bytes_accessed = (
      m * k * lhs_quant_bits // 8 + m * subblocks_per_tile * lhs_bits // 8
      if cfg.pre_quantize_lhs
      else m * k * lhs_bits // 8
  )
  rhs_bits = jax.dtypes.itemsize_bits(cfg.rhs_dtype)
  rhs_quant_bits = jax.dtypes.itemsize_bits(cfg.rhs_quantized_dtype)
  rhs_input_bytes_accessed = (
      k * n * rhs_quant_bits // 8 + n * subblocks_per_tile * rhs_bits // 8
  )
  bytes_accessed = (
      lhs_input_bytes_accessed
      + rhs_input_bytes_accessed
      + m * n * jax.dtypes.itemsize_bits(cfg.out_dtype) // 8
  )

  if cfg.pre_quantize_lhs:
    lhs_scales_spec = pl.BlockSpec(
        (block_m, subblocks_per_tile),
        lambda i, _, k_idx: (i, k_idx),
    )
  else:
    lhs_scales_spec = None

  rhs_scales_spec = pl.BlockSpec(
      (subblocks_per_tile, block_n),
      lambda _, j, k_idx: (k_idx, j),
  )
  in_specs = [
      pl.BlockSpec(
          (block_m, block_k),
          lambda i, _, k_idx: (i, k_idx),
      ),
      pl.BlockSpec(
          (block_k, block_n),
          lambda _, j, k_idx: (k_idx, j),
      ),
      lhs_scales_spec,
      rhs_scales_spec,
  ]

  out_specs = pl.BlockSpec(
      (block_m, block_n),
      lambda i, j, _: (i, j),
  )

  pallas_func = pl.pallas_call(
      _subchannel_matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), cfg.out_dtype)
      if out_mem == pltpu.HBM
      else out_mem((m, n), cfg.out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=in_specs,
          out_specs=out_specs,
          grid=(grid_m, grid_n, grid_k),
          scratch_shapes=[pltpu.VMEM((block_m, block_n), acc_dtype)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          internal_scratch_in_bytes=platform_info.get_default_internal_scratch_bytes(),
      ),
      cost_estimate=pl.CostEstimate(
          flops=2 * m * k * n,
          transcendentals=0,
          bytes_accessed=int(bytes_accessed),
      ),
      name=kernel_name,
  )

  @jax.jit
  def _target_fn(lhs, rhs, lhs_scales, rhs_scales):
    if lhs_mem == pltpu.VMEM:
      lhs = memory_utils.copy_to_vmem(lhs)
      if lhs_scales is not None:
        lhs_scales = memory_utils.copy_to_vmem(lhs_scales)
    lhs = memory_utils.with_large_2nd_minor_layout(lhs)
    if rhs_mem == pltpu.VMEM:
      rhs = memory_utils.copy_to_vmem(rhs)
      if rhs_scales is not None:
        rhs_scales = memory_utils.copy_to_vmem(rhs_scales)
    rhs = memory_utils.with_large_2nd_minor_layout(rhs)
    return pallas_func(lhs, rhs, lhs_scales, rhs_scales)

  return _target_fn


def subchannel_matmul_jax(
    cfg: SubchannelMatmulConfig,
) -> Callable[..., Any]:
  """Returns a reference implementation for subchannel quantized matmul.

  Args:
    cfg: The config for the subchannel matmul benchmark.

  Returns:
    A reference implementation for subchannel quantized matmul.
  """
  m, n = cfg.m, cfg.n
  subchannel_size = cfg.subchannel_size
  out_dtype = cfg.out_dtype

  def _jax_impl(lhs, rhs, lhs_scales, rhs_scales):
    if cfg.pre_quantize_lhs:
      assert lhs_scales is not None
      lhs_q, lhs_s = lhs, lhs_scales
    else:
      assert lhs_scales is None
      lhs_reshaped = lhs.reshape(m, -1, subchannel_size)
      lhs_q, lhs_s = quantize(lhs_reshaped, cfg.lhs_quantized_dtype)

    rhs_reshaped = rhs.reshape(-1, subchannel_size, n)

    dot_blocks = jnp.einsum(
        "mks, ksn -> mkn",
        lhs_q.astype(jnp.float32),
        rhs_reshaped.astype(jnp.float32),
    )
    scaled_blocks = dot_blocks * lhs_s * rhs_scales

    out = jnp.sum(scaled_blocks, axis=1)

    return out.astype(out_dtype)

  return _jax_impl


class SubchannelMatmulBenchmark(benchmark.Benchmark):
  """Pallas subchannel quantization matmul benchmark."""

  def __init__(self, cfg: SubchannelMatmulConfig):
    """Initializes the subchannel matmul benchmark.

    Args:
      cfg: The config for the subchannel matmul benchmark.
    """
    super().__init__()
    self._cfg = cfg
    self._kernel_name = _KERNEL_NAME_TEMPLATE.format(
        m=cfg.m,
        k=cfg.k,
        n=cfg.n,
        lhs_dtype=benchmark.dtype_to_str(cfg.lhs_dtype),
        lhs_quantized_dtype=benchmark.dtype_to_str(cfg.lhs_quantized_dtype),
        rhs_dtype=benchmark.dtype_to_str(cfg.rhs_dtype),
        rhs_quantized_dtype=benchmark.dtype_to_str(cfg.rhs_quantized_dtype),
        out_dtype=benchmark.dtype_to_str(cfg.out_dtype),
    )

  def get_input_shapes_and_dtypes(self) -> Sequence[InputSpec | None]:
    cfg = self._cfg
    num_sub_blocks = cfg.k // cfg.subchannel_size
    if cfg.pre_quantize_lhs:
      lhs_dtype = cfg.lhs_quantized_dtype
      lhs_scales = InputSpec(shape=(cfg.m, num_sub_blocks), dtype=cfg.lhs_dtype)
    else:
      lhs_dtype = cfg.lhs_dtype
      lhs_scales = None
    rhs_dtype = cfg.rhs_quantized_dtype
    rhs_scales = InputSpec(shape=(num_sub_blocks, cfg.n), dtype=cfg.rhs_dtype)
    return [
        InputSpec(shape=(cfg.m, cfg.k), dtype=lhs_dtype),
        InputSpec(shape=(cfg.k, cfg.n), dtype=rhs_dtype),
        lhs_scales,
        rhs_scales,
    ]

  def kernel_name(self) -> str:
    return self._kernel_name

  def target_fn(self) -> Callable[..., Any]:
    cfg = self._cfg
    return subchannel_matmul_kernel(
        cfg=cfg,
        kernel_name=self._kernel_name,
    )

  def reference_fn(self) -> Callable[..., Any] | None:
    cfg = self._cfg
    return subchannel_matmul_jax(cfg)
