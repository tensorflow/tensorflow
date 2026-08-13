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

"""JAX implementation of the parameterized matmul microbenchmark."""

from absl import app
from absl import flags
import jax.numpy as jnp

from xla.benchmarks.jax_microbenchmarks import matmul_lib

_DIM = flags.DEFINE_list(
    "dim",
    ["1", "1024", "1024", "1024"],
    "Dimension sizes for op parameters (ex: B,M,K,N)",
)
_FMT = flags.DEFINE_list(
    "fmt",
    ["bf16", "bf16", "bf16"],
    "Formats for op parameters and output (ex: bf16,s8,bf16). Always (operand0,"
    " operand1, output).",
)
_USE_RANDOM_DATA = flags.DEFINE_bool(
    "use_random_data",
    True,
    "Use random data for input tensors, or zero if false.",
)

_REPEAT = flags.DEFINE_integer(
    "repeat",
    10,
    "Repeats the matmul execution many times in Python, using the same operand"
    " literals, and timing the aggregate duration.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Runs the entire matmul program end-to-end, including repeating copying"
    " literals to the device and copying off the result.",
)


def parse_dtype(dtype_str: str) -> jnp.dtype:
  if dtype_str not in matmul_lib.DTYPE_MAPPING:
    raise ValueError(f"Unsupported dtype: {dtype_str}")
  return matmul_lib.DTYPE_MAPPING[dtype_str]


def main(_):
  if len(_FMT.value) != 3:
    raise ValueError(f"Expected 3 formats, got {len(_FMT.value)}")
  lhs_dtype = parse_dtype(_FMT.value[0])
  rhs_dtype = parse_dtype(_FMT.value[1])
  out_dtype = parse_dtype(_FMT.value[2])

  if len(_DIM.value) != 4:
    raise ValueError(f"Expected 4 dims, got {len(_DIM.value)}")

  b_val, m_val, k_val, n_val = map(int, _DIM.value)
  matmul_lib.run_matmul_jax(
      b=b_val,
      m=m_val,
      k=k_val,
      n=n_val,
      lhs_dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      out_dtype=out_dtype,
      repeat=_REPEAT.value,
      runs=_RUNS.value,
      use_random_data=_USE_RANDOM_DATA.value,
  )


if __name__ == "__main__":
  app.run(main)
