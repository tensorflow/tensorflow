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

from xla.benchmarks.core import platform_info
from xla.benchmarks.dma_microbenchmarks import chip_to_chip_dma_benchmark
from xla.benchmarks.dma_microbenchmarks import chiplet_to_chiplet_dma_benchmark
from xla.benchmarks.dma_microbenchmarks import host_dma_benchmark
from xla.benchmarks.dma_microbenchmarks import local_dma_benchmark
from xla.benchmarks.dma_microbenchmarks import memory_base
from xla.benchmarks.jax_microbenchmarks import matmul_lib
from xla.benchmarks.pallas_microbenchmarks import dense_matmul_lib
from xla.benchmarks.pallas_microbenchmarks import subchannel_matmul_lib


_DIM_VALUES = (1024, 2048, 4096, 8192, 16384, 32768)

_SUBCHANNEL_DIM_VALUES = (4096, 8192, 16384)

_OUT_DTYPE_PAIRS = (
    jnp.float32,
    jnp.bfloat16,
)


def _lhs_rhs_dtype_pairs(
    low_precision_dtype: jnp.dtype,
) -> list[tuple[jnp.dtype, jnp.dtype]]:
  return [  # pyrefly: ignore[bad-return]
      (jnp.bfloat16, jnp.bfloat16),
      (jnp.bfloat16, low_precision_dtype),
      (low_precision_dtype, low_precision_dtype),
  ]


def get_dense_matmul_configs(
    chip_version: pltpu.ChipVersion | None = None,
) -> list[dense_matmul_lib.DenseMatmulConfig]:
  """Generates preconfigured dense matmul benchmark configs."""
  configs = []
  pinfo = platform_info.get_platform_info(chip_version)
  if jnp.int4 in pinfo.matmul_cadence_cycles_by_dtype:
    low_precision_dtype = jnp.int4
  else:
    low_precision_dtype = jnp.float8_e4m3fn
  subblock_m = dense_matmul_lib.get_default_subblock_m(chip_version)

  for m in _DIM_VALUES:
    n = k = m
    mem_options = [pltpu.HBM, pltpu.VMEM] if m in (1024, 2048) else [pltpu.HBM]
    for lhs_dtype, rhs_dtype in _lhs_rhs_dtype_pairs(
        low_precision_dtype,  # pyrefly: ignore[bad-argument-type]
    ):
      if jnp.issubdtype(lhs_dtype, jnp.integer) and pinfo.generation < 8:
        acc_dtype = jnp.int32
      else:
        acc_dtype = jnp.float32
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
  pinfo = platform_info.get_platform_info(chip_version)
  m = 128
  lhs_dtype = rhs_dtype = out_dtype = jnp.bfloat16
  outer_acc_dtype = jnp.bfloat16
  subchannel_size = 1024
  if jnp.int4 in pinfo.matmul_cadence_cycles_by_dtype:
    lhs_quantized_dtype = jnp.int4
    inner_acc_dtype = jnp.int32
  else:
    lhs_quantized_dtype = jnp.float8_e4m3fn
    inner_acc_dtype = jnp.float32
  rhs_quantized_dtype = jnp.int4
  pre_quantize_lhs = False
  for k in _SUBCHANNEL_DIM_VALUES:
    n = k
    mem_options = [pltpu.HBM, pltpu.VMEM] if k == 4096 else [pltpu.HBM]
    for mem in mem_options:
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
          outer_acc_dtype=outer_acc_dtype,
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
              inner_acc_dtype=inner_acc_dtype,
              outer_acc_dtype=outer_acc_dtype,
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
  configs = []
  pinfo = platform_info.get_platform_info(chip_version)
  if jnp.int4 in pinfo.matmul_cadence_cycles_by_dtype:
    low_precision_dtype = jnp.int4
  else:
    low_precision_dtype = jnp.float8_e4m3fn
  for m in _DIM_VALUES:
    n = k = m
    for lhs_dtype, rhs_dtype in _lhs_rhs_dtype_pairs(
        low_precision_dtype,  # pyrefly: ignore[bad-argument-type]
    ):
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


_DMA_SUITES: tuple[tuple[str, type[memory_base.MemoryBenchmarks]], ...] = (
    ("local_dma", local_dma_benchmark.LocalDmaBenchmarks),
    ("host_dma", host_dma_benchmark.HostDmaBenchmarks),
    ("chip_to_chip_dma", chip_to_chip_dma_benchmark.ChipToChipBenchmarks),
    (
        "chiplet_to_chiplet_dma",
        chiplet_to_chiplet_dma_benchmark.ChipletToChipletBenchmarks,
    ),
)


def get_dma_configs(
    chip_version: pltpu.ChipVersion | None = None,
) -> list[memory_base.DmaBenchmarkConfig]:
  """Generates configs for all DMA microbenchmarks."""
  del chip_version  # Unused.
  configs = []
  for suite, test_class in _DMA_SUITES:
    for name in sorted(dir(test_class)):
      if name.startswith("test_"):
        configs.append(
            memory_base.DmaBenchmarkConfig(
                suite=suite, test_class=test_class, test_name=name
            )
        )
  return configs


BENCHMARK_FACTORIES: Mapping[
    str, Callable[[pltpu.ChipVersion | None], list[Any]]
] = immutabledict.immutabledict({
    "dense_matmul": get_dense_matmul_configs,
    "subchannel_matmul": get_subchannel_matmul_configs,
    "jax_matmul": get_jax_matmul_configs,
    "dma": get_dma_configs,
})
