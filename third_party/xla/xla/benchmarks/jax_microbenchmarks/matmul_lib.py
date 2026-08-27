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

"""Library for JAX matmul microbenchmarks."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from xla.benchmarks.core import benchmark  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass(frozen=True, kw_only=True)
class JaxMatmulConfig(benchmark.BenchmarkConfig):
  """Config for JAX matmul benchmark.

  Attributes:
    b: Batch size.
    m: The number of rows in the first operand.
    k: The contraction dimension.
    n: The number of columns in the second operand.
    lhs_dtype: The dtype of the first operand.
    rhs_dtype: The dtype of the second operand.
    out_dtype: The dtype of the output.
  """
  b: int = 1
  m: int = 1024
  k: int = 1024
  n: int = 1024
  lhs_dtype: jnp.dtype = jnp.bfloat16
  rhs_dtype: jnp.dtype = jnp.bfloat16
  out_dtype: jnp.dtype = jnp.bfloat16

  def get_benchmark(self) -> benchmark.Benchmark:
    return JaxMatmulBenchmark(self)


class JaxMatmulBenchmark(benchmark.Benchmark):
  """JAX matmul benchmark subclassing core Benchmark."""

  def __init__(
      self,
      cfg: JaxMatmulConfig,
  ):
    super().__init__()
    self._cfg = cfg
    lhs_str = benchmark.dtype_to_str(cfg.lhs_dtype)
    rhs_str = benchmark.dtype_to_str(cfg.rhs_dtype)
    out_str = benchmark.dtype_to_str(cfg.out_dtype)
    self._kernel_name = (
        f"matmul_{cfg.m}_{cfg.k}_{cfg.n}_{lhs_str}_{rhs_str}_{out_str}"
    )

  def get_input_shapes_and_dtypes(self) -> Sequence[benchmark.InputSpec | None]:
    cfg = self._cfg
    return [
        benchmark.InputSpec(shape=(cfg.b, cfg.m, cfg.k), dtype=cfg.lhs_dtype),
        benchmark.InputSpec(shape=(cfg.b, cfg.k, cfg.n), dtype=cfg.rhs_dtype),
    ]

  def target_fn(self) -> Callable[..., Any]:
    cfg = self._cfg
    out_dtype = cfg.out_dtype

    def kernel_fn(lhs_in, rhs_in):
      return jnp.matmul(lhs_in, rhs_in, preferred_element_type=out_dtype)

    named_kernel_fn = jax.named_call(kernel_fn, name=self._kernel_name)
    return jax.jit(named_kernel_fn)

  def kernel_name(self) -> str:
    return self._kernel_name
