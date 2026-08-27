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

from absl import app
from absl import flags
from absl import logging
import immutabledict
from jax.experimental.pallas import tpu as pltpu

from xla.benchmarks.core import benchmark
from xla.benchmarks.pallas_microbenchmarks import subchannel_matmul_lib

immutabledict = immutabledict.immutabledict

_DIM = flags.DEFINE_list(
    "dim",
    ["1", "64", "8192", "8192"],
    "Dimension sizes for op parameters (ex: B,M,K,N)",
)
_FMT = flags.DEFINE_list(
    "fmt",
    ["bf16", "bf16", "bf16"],
    "Formats for op parameters and output (ex: bf16,s4,bf16). Always (operand0,"
    " operand1, output).",
)
_ACC_DTYPE = flags.DEFINE_string(
    "acc_dtype",
    "bf16",
    "Format for accumulator.",
)
_MEM = flags.DEFINE_list(
    "mem",
    ["0", "0", "0"],
    "comma-separated list of memory space IDs (ex: 0,0,3). "
    "0 is HBM, 3 is CMEM, 1 is VMEM, 2 is Sync flag.",
)
_WINDOW = flags.DEFINE_list(
    "window",
    None,
    "Comma-separated list of window tile sizes (ex: 1,64,8192,8192).\n\nThe"
    " order corresponds to the same B,M,K,N used by --dim.",
)
_USE_RANDOM_DATA = flags.DEFINE_bool(
    "use_random_data",
    True,
    "Use random data for input tensors, or zero if false.",
)
_REPEAT = flags.DEFINE_integer(
    "repeat",
    1,
    "Repeats the matmul Pallas operation many times, one after each "
    "other, using the same operand literals and using the last "
    "repetition as the final result.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Number of times to run the entire matmul program end-to-end with different"
    " input tensors.",
)
_CHECK_NUMERICS = flags.DEFINE_bool(
    "check_numerics",
    False,
    "Whether to check numerics of the result against the reference"
    " implementation.",
)

_LHS_QUANTIZED_DTYPE = flags.DEFINE_string(
    "lhs_quantized_dtype",
    "f8e4m3fn",
    "Quantized dtype for LHS.",
)
_RHS_QUANTIZED_DTYPE = flags.DEFINE_string(
    "rhs_quantized_dtype",
    "s4",
    "Quantized dtype for RHS.",
)
_PRE_QUANTIZE_LHS = flags.DEFINE_bool(
    "pre_quantize_lhs",
    False,
    "Whether LHS quantization happens outside the kernel (pre-quantized) or"
    " inside.",
)
_SUBCHANNEL_SIZE = flags.DEFINE_integer(
    "subchannel_size",
    1024,
    "Subchannel block size for quantization.",
)

FLAGS = flags.FLAGS

FLAGS.mark_as_parsed()


def main(_):
  if len(_FMT.value) != 3:
    raise ValueError(f"Expected 3 formats, got {len(_FMT.value)}")
  lhs_dtype = benchmark.str_to_dtype(_FMT.value[0])
  rhs_dtype = benchmark.str_to_dtype(_FMT.value[1])
  out_dtype = benchmark.str_to_dtype(_FMT.value[2])
  acc_dtype = benchmark.str_to_dtype(_ACC_DTYPE.value)

  lhs_quantized_dtype = benchmark.str_to_dtype(_LHS_QUANTIZED_DTYPE.value)
  rhs_quantized_dtype = benchmark.str_to_dtype(_RHS_QUANTIZED_DTYPE.value)
  pre_quantize_lhs = _PRE_QUANTIZE_LHS.value
  subchannel_size = _SUBCHANNEL_SIZE.value

  if len(_DIM.value) != 4:
    raise ValueError(f"Expected 4 dims, got {len(_DIM.value)}")
  b, m, k, n = map(int, _DIM.value)
  if b != 1:
    raise ValueError("Currently only batch size 1 is supported.")
  if len(_MEM.value) != 3:
    raise ValueError(f"Expected 3 memory spaces, got {len(_MEM.value)}")
  lhs_mem, rhs_mem, out_mem = map(int, _MEM.value)
  if not {lhs_mem, rhs_mem, out_mem}.issubset({0, 1}):
    raise ValueError(
        "Currently only HBM (0) and VMEM (1) are supported, but got"
        f" {_MEM.value}."
    )
  mem_spaces = {
      0: pltpu.HBM,
      1: pltpu.VMEM,
  }
  lhs_mem = mem_spaces[lhs_mem]
  rhs_mem = mem_spaces[rhs_mem]
  out_mem = mem_spaces[out_mem]
  if _WINDOW.value is None:
    block_m, block_k, block_n = subchannel_matmul_lib.select_window(
        m=m,
        k=k,
        n=n,
        lhs_mem=lhs_mem,
        rhs_mem=rhs_mem,
        out_mem=out_mem,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        lhs_quantized_dtype=lhs_quantized_dtype,
        rhs_quantized_dtype=rhs_quantized_dtype,
        pre_quantize_lhs=pre_quantize_lhs,
    )
    logging.info(
        "Selected window size from Pallas cost model: %s, %s, %s",
        block_m,
        block_k,
        block_n,
    )
  else:
    if len(_WINDOW.value) != 4:
      raise ValueError(f"Expected 4 window sizes, got {len(_WINDOW.value)}")
    block_b, block_m, block_k, block_n = map(int, _WINDOW.value)
    if block_b != 1:
      raise ValueError("Currently only batch size 1 is supported.")
  if _REPEAT.value < 1:
    raise ValueError("Repeat must be at least 1.")
  if _RUNS.value < 1:
    raise ValueError("Runs must be at least 1.")

  cfg = subchannel_matmul_lib.SubchannelMatmulConfig(
      m=m,
      k=k,
      n=n,
      block_m=block_m,
      block_k=block_k,
      block_n=block_n,
      subchannel_size=subchannel_size,
      lhs_mem=lhs_mem,
      rhs_mem=rhs_mem,
      out_mem=out_mem,
      lhs_dtype=lhs_dtype,
      rhs_dtype=rhs_dtype,
      out_dtype=out_dtype,
      acc_dtype=acc_dtype,
      lhs_quantized_dtype=lhs_quantized_dtype,
      rhs_quantized_dtype=rhs_quantized_dtype,
      pre_quantize_lhs=pre_quantize_lhs,
  )
  bm = cfg.get_benchmark()
  bm.run(
      repeat=_REPEAT.value,
      runs=_RUNS.value,
      use_random_data=_USE_RANDOM_DATA.value,
      check_numerics=_CHECK_NUMERICS.value,
  )


if __name__ == "__main__":
  app.run(main)
