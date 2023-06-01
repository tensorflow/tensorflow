#!/usr/bin/python3
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import sys

from absl import app
from absl import flags
import torch
import pandas as pd
import tqdm

import triton
import triton.language as tl

FLAGS = flags.FLAGS

flags.DEFINE_string('data', '', 'Data to check')
flags.DEFINE_string('output_file', '/tmp/checked.csv',
                    'File to write output data to')
flags.DEFINE_integer('num_samples', 100, 'Number of samples to check')
flags.DEFINE_float('time_cutoff', 0.02,
                   'Only consider samples with min_time_ms larger than cutoff')

from matmul_lib import benchmark_matmul

def get_actual_time(r, s, pbar):
  dims = MatmulSize(FLAGS.m, FLAGS.n, FLAGS.k, FLAGS.quantized_lhs)
  return benchmark_matmul(dims=dims,
                        pbar=pbar,
                        shared_stream=s,
                       tilings=[MatmulTiling(r.block_m, r.block_n, r.block_k,
                       r.split_k, r.num_stages, r.num_warps)],
                          repetitions=20)[0].min_time_ms

def main():
  df = pd.read_csv(FLAGS.data).sample(FLAGS.num_samples)
  shared_stream = torch.cuda.Stream()
  measured_times = []
  pbar = tqdm.tqdm(total=FLAGS.num_samples)
  with torch.cuda.stream(shared_stream):
    for index, r in df.iterrows():
      measured_times.append(get_actual_time(r, shared_stream, s))
  df = df.assign(measured_min_time_ms=measured_times)
  pbar.close()

  errors = df.assign(
      absolute_error=lambda r: abs(r.measured_min_time_ms - r.min_time_ms)
  ).assign(
      relative_error=lambda r: abs(r.measured_min_time_ms - r.min_time_ms) / r.min_time_ms
  )[["absolute_error", "relative_error"]]
  print(errors)
  print(errors.describe())
  df.to_csv(FLAGS.output_file)


if __name__ == '__main__':
  app.parse_flags_with_usage(sys.argv)
  triton.compiler.init_cuda_utils()
  main()
