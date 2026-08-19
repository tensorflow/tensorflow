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

from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.jax_microbenchmarks import jax_profiler_utils  # pylint: disable=g-direct-tensorflow-import

DTYPE_MAPPING = {
    "bf16": jnp.bfloat16,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "s8": jnp.int8,
    "s16": jnp.int16,
    "s32": jnp.int32,
    "u8": jnp.uint8,
    "u16": jnp.uint16,
    "u32": jnp.uint32,
}


def dtype_to_str(dtype: Any) -> str:
  """Converts a JAX/NumPy dtype to its string representation."""
  for k, v in DTYPE_MAPPING.items():
    if v == dtype:
      return k
  return getattr(dtype, "name", str(dtype))


def make_inputs(
    b: int,
    m: int,
    k: int,
    n: int,
    lhs_dtype: Any,
    rhs_dtype: Any,
    use_random_data: bool,
) -> tuple[jax.Array, jax.Array]:
  """Generates random or zero inputs for the matmul."""
  if not use_random_data:
    lhs = jnp.zeros((b, m, k), dtype=lhs_dtype)
    rhs = jnp.zeros((b, k, n), dtype=rhs_dtype)
    return lhs, rhs

  rng = np.random.default_rng(1234)

  def _generate(dtype, shape):
    if jnp.issubdtype(dtype, jnp.floating):
      arr = rng.normal(size=shape).astype(dtype)
    else:
      arr = rng.integers(127, size=shape).astype(dtype)
    return jnp.asarray(arr)

  lhs = _generate(lhs_dtype, (b, m, k))
  rhs = _generate(rhs_dtype, (b, k, n))
  return lhs, rhs


def run_matmul_jax(
    b: int,
    m: int,
    k: int,
    n: int,
    lhs_dtype: Any,
    rhs_dtype: Any,
    out_dtype: Any,
    repeat: int,
    runs: int,
    use_random_data: bool,
) -> None:
  """Runs the JAX matmul benchmark."""
  logging.info(
      "Running Matmul configuration: B=%d, M=%d, K=%d, N=%d", b, m, k, n
  )
  lhs, rhs = make_inputs(b, m, k, n, lhs_dtype, rhs_dtype, use_random_data)

  lhs_str = dtype_to_str(lhs_dtype)
  rhs_str = dtype_to_str(rhs_dtype)
  out_str = dtype_to_str(out_dtype)
  kernel_name = f"matmul_{m}_{k}_{n}_{lhs_str}_{rhs_str}_{out_str}"

  def kernel_fn(lhs_in, rhs_in):
    return jnp.matmul(lhs_in, rhs_in, preferred_element_type=out_dtype)

  named_kernel_fn = jax.named_call(kernel_fn, name=kernel_name)
  f_compiled = jax.jit(named_kernel_fn).lower(lhs, rhs).compile()

  if runs <= 0:
    logging.warning(
        "--runs was 0 or less, so we will only compile the matmul but not"
        " run it."
    )
    return

  for _ in range(runs):
    with jax_profiler_utils.JaxProfiler(kernel_name) as profiler:
      res = f_compiled(lhs, rhs)
      res.block_until_ready()
      for _ in range(repeat - 1):
        res = f_compiled(lhs, rhs)
        res.block_until_ready()

    if profiler.result is not None:
      logging.info(
          "Profiler results:\n%s", profiler.result.as_dataframe().to_string()
      )
    else:
      logging.warning("No profiler results found.")
