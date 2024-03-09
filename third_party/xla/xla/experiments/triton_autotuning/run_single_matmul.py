#!/usr/bin/python3
# Copyright 2023 The OpenXLA Authors.
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
# ==============================================================================

"""Runs a single matmul with a supplied configuration."""
import sys

from absl import app
from absl import flags
from matmul_lib import benchmark_cublas
from matmul_lib import benchmark_matmul
from matmul_lib import MatmulSize
from matmul_lib import MatmulTiling
from matmul_lib import MatrixLayout
from matmul_lib import print_roofline_performance
from matmul_lib import QuantizedInputType
import torch
import tqdm


_M = flags.DEFINE_integer('m', 64, 'Size of first matrix')
_K = flags.DEFINE_integer('k', 64, 'Size of contracting dimension')
_N = flags.DEFINE_integer('n', 64, 'Size of second matrix')
_QUANTIZED_LHS = flags.DEFINE_enum_class(
    'quantized_lhs',
    QuantizedInputType.BFLOAT16,
    QuantizedInputType,
    'Type to use for LHS quantization',
)
_QUANTIZED_RHS = flags.DEFINE_enum_class(
    'quantized_rhs',
    QuantizedInputType.BFLOAT16,
    QuantizedInputType,
    'Type to use for RHS quantization',
)

_BLOCK_M = flags.DEFINE_integer('block_m', 16, 'Tiling in M-dimension')
_BLOCK_N = flags.DEFINE_integer('block_n', 16, 'Tiling in N-dimension')
_BLOCK_K = flags.DEFINE_integer('block_k', 16, 'Tiling in K-dimension')
_SPLIT_K = flags.DEFINE_integer(
    'split_k', 1, 'Number of splits for contracting dimension'
)
_LHS_LAYOUT = flags.DEFINE_enum_class(
    'lhs_layout',
    MatrixLayout.ROW_MAJOR,
    MatrixLayout,
    'Layout to use for LHS',
)
_RHS_LAYOUT = flags.DEFINE_enum_class(
    'rhs_layout',
    MatrixLayout.ROW_MAJOR,
    MatrixLayout,
    'Layout to use for RHS',
)
_RESULT_LAYOUT = flags.DEFINE_enum_class(
    'result_layout',
    MatrixLayout.ROW_MAJOR,
    MatrixLayout,
    'Layout to use for the result',
)
_NUM_STAGES = flags.DEFINE_integer(
    'num_stages', 1, 'Number of pipelining stages'
)
_NUM_WARPS = flags.DEFINE_integer(
    'num_warps', 4, 'Number of warps to allocate in a given block'
)

_DEBUG = flags.DEFINE_bool('debug', False, 'Print debug information')


def main():
  s = torch.cuda.Stream()
  pbar = tqdm.tqdm(ncols=0)
  dims = MatmulSize(
      M=_M.value,
      N=_N.value,
      K=_K.value,
      quantized_lhs=_QUANTIZED_LHS.value,
      quantized_rhs=_QUANTIZED_RHS.value,
  )
  timing = benchmark_matmul(
      dims=dims,
      pbar=pbar,
      shared_stream=s,
      tilings=[
          MatmulTiling(
              _BLOCK_M.value,
              _BLOCK_N.value,
              _BLOCK_K.value,
              _SPLIT_K.value,
              _LHS_LAYOUT.value,
              _RHS_LAYOUT.value,
              _RESULT_LAYOUT.value,
              _NUM_STAGES.value,
              _NUM_WARPS.value,
          )
      ],
      repetitions_ms=300,
      debug=_DEBUG.value,
  )
  if len(timing) != 1:
    print('Failed to find working configuration')
    sys.exit(1)
  t = timing[0]
  print(f'Timing: {t}')
  print_roofline_performance(dims, t.min_time_ms)
  cublas_time = benchmark_cublas(dims)
  print(f'Reference cuBLAS time (bf16xbf16->bf16): {cublas_time:0.4f}ms')


if __name__ == '__main__':
  app.parse_flags_with_usage(sys.argv)
  main()
