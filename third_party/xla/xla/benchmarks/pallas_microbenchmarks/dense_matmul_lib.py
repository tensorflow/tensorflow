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

"""Script for benchmarking Pallas matmuls on TPUs."""

import dataclasses
import math
from typing import Any, Callable, Sequence

from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from xla.benchmarks.core import benchmark  # pylint: disable=g-direct-tensorflow-import
from xla.benchmarks.pallas_microbenchmarks import memory_utils  # pylint: disable=g-direct-tensorflow-import

Benchmark = benchmark.Benchmark
InputSpec = benchmark.InputSpec


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenseMatmulConfig:
  """Config for Pallas dense matmul benchmark.

  Attributes:
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    block_m: The block size for the m dimension.
    block_k: The block size for the k dimension.
    block_n: The block size for the n dimension.
    lhs_mem: The memory space of the first operand.
    rhs_mem: The memory space of the second operand.
    out_mem: The memory space of the output.
    lhs_dtype: The dtype of the first operand.
    rhs_dtype: The dtype of the second operand.
    out_dtype: The dtype of the output.
    acc_dtype: The dtype of the accumulator.
  """
  m: int
  k: int
  n: int
  block_m: int
  block_k: int
  block_n: int
  lhs_mem: pltpu.MemorySpace
  rhs_mem: pltpu.MemorySpace
  out_mem: pltpu.MemorySpace
  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype


def dense_matmul_kernel(
    cfg: DenseMatmulConfig, internal_scratch_in_bytes: int, kernel_name: str
) -> Callable[..., Any]:
  """Returns a Pallas kernel for dense matmul benchmark.

  Args:
    cfg: The config for the dense matmul benchmark.
    internal_scratch_in_bytes: The size of the internal scratch in bytes.
    kernel_name: The name of the kernel.

  Returns:
    A Pallas kernel for dense matmul benchmark accepting two arguments
    (lhs, rhs) and returning the result of the matmul.
  """
  m, k, n = cfg.m, cfg.k, cfg.n
  block_m, block_k, block_n = cfg.block_m, cfg.block_k, cfg.block_n
  lhs_mem, rhs_mem, out_mem = cfg.lhs_mem, cfg.rhs_mem, cfg.out_mem
  lhs_dtype, rhs_dtype, out_dtype = (
      cfg.lhs_dtype,
      cfg.rhs_dtype,
      cfg.out_dtype,
  )
  acc_dtype = cfg.acc_dtype

  grid_m = math.ceil(m / block_m)
  grid_n = math.ceil(n / block_n)
  grid_k = math.ceil(k / block_k)

  logging.info(
      "block_m: %s, block_k: %s, block_n: %s", block_m, block_k, block_n
  )
  logging.info("grid_m: %s, grid_k: %s, grid_n: %s", grid_m, grid_k, grid_n)

  @jax.jit
  def _matmul_kernel_inner(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref):

    @pl.when(pl.program_id(2) == 0)
    def init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    matmul = jnp.dot(
        x_tile_ref[...],
        y_tile_ref[...],
        preferred_element_type=acc_ref.dtype,
    )
    acc_ref[...] = acc_ref[...] + matmul

    @pl.when(pl.program_id(2) == grid_k - 1)
    def store():
      o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)

  pallas_func = pl.pallas_call(
      _matmul_kernel_inner,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype)
      if out_mem == pltpu.HBM
      else out_mem((m, n), out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, block_k), lambda i, _, k: (i, k)),
              pl.BlockSpec((block_k, block_n), lambda _, j, k: (k, j)),
          ],
          out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
          grid=(grid_m, grid_n, grid_k),
          scratch_shapes=[pltpu.VMEM((block_m, block_n), acc_dtype)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          internal_scratch_in_bytes=internal_scratch_in_bytes,
      ),
      cost_estimate=pl.CostEstimate(
          flops=2 * m * k * n,
          transcendentals=0,
          bytes_accessed=(
              m * k * jax.dtypes.itemsize_bits(lhs_dtype) // 8
              + k * n * jax.dtypes.itemsize_bits(rhs_dtype) // 8
              + m * n * jax.dtypes.itemsize_bits(out_dtype) // 8
          ),
      ),
      name=kernel_name,
  )

  @jax.jit
  def _target_fn(lhs, rhs):
    if lhs_mem == pltpu.VMEM:
      lhs = memory_utils.copy_to_vmem(lhs)
    if rhs_mem == pltpu.VMEM:
      rhs = memory_utils.copy_to_vmem(rhs)
    return pallas_func(lhs, rhs)

  return _target_fn.lower(
      jax.ShapeDtypeStruct((m, k), lhs_dtype),
      jax.ShapeDtypeStruct((k, n), rhs_dtype),
  ).compile({"xla_detailed_logging": True})


def dense_matmul_reference(cfg: DenseMatmulConfig) -> Callable[..., Any]:
  """Returns a reference implementation for dense matmul benchmark.

  Args:
    cfg: The config for the dense matmul benchmark.

  Returns:
    A function accepting two arguments (lhs, rhs) that implements the reference
    for dense matmul benchmark.
  """

  def _reference_fn(lhs, rhs):
    return jnp.dot(lhs, rhs, preferred_element_type=cfg.acc_dtype).astype(
        cfg.out_dtype
    )

  return _reference_fn


class DenseMatmulBenchmark(Benchmark):
  """Pallas dense matmul benchmark subclassing core Benchmark."""

  def __init__(
      self,
      cfg: DenseMatmulConfig,
      internal_scratch_in_bytes: int,
      kernel_name: str,
  ):
    """Initializes the dense matmul benchmark.

    Args:
      cfg: The config for the dense matmul benchmark.
      internal_scratch_in_bytes: The size of the internal scratch in bytes.
      kernel_name: The name of the kernel.
    """
    super().__init__()
    self._cfg = cfg
    self._internal_scratch_in_bytes = internal_scratch_in_bytes
    self._kernel_name = kernel_name

  def get_input_shapes_and_dtypes(self) -> Sequence[InputSpec | None]:
    cfg = self._cfg
    return [
        InputSpec(shape=(cfg.m, cfg.k), dtype=cfg.lhs_dtype),
        InputSpec(shape=(cfg.k, cfg.n), dtype=cfg.rhs_dtype),
    ]

  def target_fn(self) -> Callable[..., Any]:
    return dense_matmul_kernel(
        self._cfg, self._internal_scratch_in_bytes, self._kernel_name
    )

  def kernel_name(self) -> str:
    return self._kernel_name

  def reference_fn(self) -> Callable[..., Any] | None:
    return dense_matmul_reference(self._cfg)
