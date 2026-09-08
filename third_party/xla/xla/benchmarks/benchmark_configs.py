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

"""Preconfigured benchmark configs and shared cost model derivation utilities."""

from collections.abc import Callable
from typing import Any, Mapping

import immutabledict
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from xla.benchmarks.jax_microbenchmarks import matmul_lib
from xla.benchmarks.pallas_microbenchmarks import dense_matmul_lib
from xla.benchmarks.pallas_microbenchmarks import subchannel_matmul_lib


_DIM_VALUES = (1024, 2048, 4096, 8192, 16384, 32768)

_LHS_RHS_DTYPE_PAIRS = (
    (jnp.bfloat16, jnp.bfloat16),
    (jnp.bfloat16, jnp.float8_e4m3fn),
    (jnp.bfloat16, jnp.int4),
    (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
    (jnp.float8_e4m3fn, jnp.int4),
)

_OUT_DTYPE_PAIRS = (
    jnp.float32,
    jnp.bfloat16,
)


def get_dense_matmul_configs(
    chip_version: pltpu.ChipVersion | None = None,
) -> list[dense_matmul_lib.DenseMatmulConfig]:
  """Generates preconfigured dense matmul benchmark configs."""
  configs = []
  acc_dtype = jnp.float32
  subblock_m = dense_matmul_lib.get_default_subblock_m(chip_version)

  for m in _DIM_VALUES:
    n = k = m
    mem_options = [pltpu.HBM, pltpu.VMEM] if m in (1024, 2048) else [pltpu.HBM]
    for lhs_dtype, rhs_dtype in _LHS_RHS_DTYPE_PAIRS:
      for out_dtype in _OUT_DTYPE_PAIRS:
        for mem in mem_options:
          block_m, block_k, block_n = dense_matmul_lib.select_window(
              m=m,
              k=k,
              n=n,
              lhs_mem=mem,
              rhs_mem=mem,
              out_mem=mem,
              lhs_dtype=lhs_dtype,
              rhs_dtype=rhs_dtype,
              out_dtype=out_dtype,
              acc_dtype=acc_dtype,
              subblock_m=subblock_m,
              chip_version=chip_version,
          )
          configs.append(
              dense_matmul_lib.DenseMatmulConfig(
                  m=m,
                  k=k,
                  n=n,
                  block_m=int(block_m),
                  block_k=int(block_k),
                  block_n=int(block_n),
                  lhs_mem=mem,
                  rhs_mem=mem,
                  out_mem=mem,
                  lhs_dtype=lhs_dtype,
                  rhs_dtype=rhs_dtype,
                  out_dtype=out_dtype,
                  acc_dtype=acc_dtype,
                  subblock_m=subblock_m,
              )
          )
  return configs


def get_subchannel_matmul_configs(
    chip_version: pltpu.ChipVersion | None = None,
) -> list[subchannel_matmul_lib.SubchannelMatmulConfig]:
  """Generates preconfigured subchannel matmul benchmark configs."""
  configs = []
  m, k, n = 128, 8192, 4096
  lhs_dtype = rhs_dtype = out_dtype = jnp.bfloat16
  acc_dtype = jnp.bfloat16
  subchannel_size = 1024
  lhs_quantized_dtype = jnp.float8_e4m3fn
  rhs_quantized_dtype = jnp.int4
  pre_quantize_lhs = False

  for mem in [pltpu.HBM, pltpu.VMEM]:
    block_m, block_k, block_n = subchannel_matmul_lib.select_window(
        m=m,
        k=k,
        n=n,
        lhs_mem=mem,
        rhs_mem=mem,
        out_mem=mem,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        lhs_quantized_dtype=lhs_quantized_dtype,
        rhs_quantized_dtype=rhs_quantized_dtype,
        pre_quantize_lhs=pre_quantize_lhs,
        chip_version=chip_version,
    )
    configs.append(
        subchannel_matmul_lib.SubchannelMatmulConfig(
            m=m,
            k=k,
            n=n,
            block_m=int(block_m),
            block_k=int(block_k),
            block_n=int(block_n),
            subchannel_size=subchannel_size,
            lhs_mem=mem,
            rhs_mem=mem,
            out_mem=mem,
            lhs_dtype=lhs_dtype,
            rhs_dtype=rhs_dtype,
            out_dtype=out_dtype,
            acc_dtype=acc_dtype,
            lhs_quantized_dtype=lhs_quantized_dtype,
            rhs_quantized_dtype=rhs_quantized_dtype,
            pre_quantize_lhs=pre_quantize_lhs,
        )
    )
  return configs


def get_jax_matmul_configs(
    chip_version: pltpu.ChipVersion | None = None,
) -> list[matmul_lib.JaxMatmulConfig]:
  """Generates preconfigured JAX matmul benchmark configs."""
  del chip_version  # Unused.
  configs = []
  for m in _DIM_VALUES:
    n = k = m
    for lhs_dtype, rhs_dtype in _LHS_RHS_DTYPE_PAIRS:
      for out_dtype in _OUT_DTYPE_PAIRS:
        configs.append(
            matmul_lib.JaxMatmulConfig(
                b=1,
                m=m,
                k=k,
                n=n,
                lhs_dtype=lhs_dtype,
                rhs_dtype=rhs_dtype,
                out_dtype=out_dtype,
            )
        )
  return configs


BENCHMARK_FACTORIES: Mapping[
    str, Callable[[pltpu.ChipVersion | None], list[Any]]
] = immutabledict.immutabledict({
    "dense_matmul": get_dense_matmul_configs,
    "subchannel_matmul": get_subchannel_matmul_configs,
    "jax_matmul": get_jax_matmul_configs,
})
