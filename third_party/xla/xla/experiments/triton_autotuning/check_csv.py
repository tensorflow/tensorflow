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

"""Measures timings of tilings provided in a CSV file."""
import sys

from absl import app
from absl import flags
from matmul_lib import benchmark_matmul
from matmul_lib import MatmulSize
from matmul_lib import MatmulTiling
from matmul_lib import MatrixLayout
from matmul_lib import QuantizedInputType
import pandas as pd
import torch
import tqdm

_DATA = flags.DEFINE_string('data', '', 'Data to check')
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', '/tmp/checked.csv', 'File to write output data to'
)
_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples', 100, 'Number of samples to check'
)
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


def get_actual_time(r, s, pbar):
  dims = MatmulSize(
      M=_M.value,
      N=_N.value,
      K=_K.value,
      quantized_lhs=_QUANTIZED_LHS.value,
      quantized_rhs=_QUANTIZED_RHS.value,
  )
  return benchmark_matmul(
      dims=dims,
      pbar=pbar,
      shared_stream=s,
      tilings=[
          MatmulTiling(
              r.block_m,
              r.block_n,
              r.block_k,
              r.split_k,
              MatrixLayout(r.lhs_layout),
              MatrixLayout(r.rhs_layout),
              MatrixLayout(r.result_layout),
              r.num_stages,
              r.num_warps,
          )
      ],
      repetitions_ms=300,
  )[0].min_time_ms


def main():
  df = pd.read_csv(_DATA.value).sample(_NUM_SAMPLES.value)
  shared_stream = torch.cuda.Stream()
  measured_times = []
  pbar = tqdm.tqdm(total=_NUM_SAMPLES.value, ncols=0)
  with torch.cuda.stream(shared_stream):
    for _, r in df.iterrows():
      measured_times.append(get_actual_time(r, shared_stream, pbar))
  df = df.assign(measured_min_time_ms=measured_times)
  pbar.close()

  def absolute_error(r):
    return abs(r.measured_min_time_ms - r.min_time_ms)

  def relative_error(r):
    return absolute_error(r) / r.min_time_ms

  errors = df.assign(absolute_error=absolute_error).assign(
      relative_error=relative_error
  )[['absolute_error', 'relative_error']]
  print(errors)
  print(errors.describe())
  df.to_csv(_OUTPUT_FILE.value)


if __name__ == '__main__':
  app.parse_flags_with_usage(sys.argv)
  main()
