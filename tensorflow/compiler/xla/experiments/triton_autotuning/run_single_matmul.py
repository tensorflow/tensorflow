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

"""Runs a single matmul with a supplied configuration."""
import sys

from absl import app
from absl import flags
import triton
import torch
import tqdm

from matmul_lib import benchmark_matmul, print_roofline_performance, benchmark_cublas, MatmulSize, MatmulTiling

FLAGS = flags.FLAGS

flags.DEFINE_integer('m', 64, 'Size of first matrix')
flags.DEFINE_integer('k', 64, 'Size of contracting dimension')
flags.DEFINE_integer('n', 64, 'Size of second matrix')
flags.DEFINE_integer('quantized_lhs', 0, 'Whether LHS is in int8')

flags.DEFINE_integer('block_m', 16, 'Tiling in M-dimension')
flags.DEFINE_integer('block_n', 16, 'Tiling in N-dimension')
flags.DEFINE_integer('block_k', 16, 'Tiling in K-dimension')

flags.DEFINE_integer('split_k', 1, 'Number of splits for contracting dimension')
flags.DEFINE_integer('num_stages', 1, 'Number of pipelining stages')
flags.DEFINE_integer('num_warps', 4, 'Number of warps to allocate in a given block')
flags.DEFINE_bool('debug', False, 'Print debug information')


def main():
  s = torch.cuda.Stream()
  pbar = tqdm.tqdm()
  dims = MatmulSize(FLAGS.m, FLAGS.n, FLAGS.k, FLAGS.quantized_lhs)
  timing = benchmark_matmul(dims=dims,
                        pbar=pbar,
                        shared_stream=s,
                       tilings=[MatmulTiling(FLAGS.block_m, FLAGS.block_n, FLAGS.block_k,
                       FLAGS.split_k, FLAGS.num_stages, FLAGS.num_warps)],
                       repetitions=20,
                       debug=FLAGS.debug)
  if len(timing) != 1:
      print("Failed to find working configuration")
      sys.exit(1)
  t = timing[0]
  print(f"Timing: {t}")
  print_roofline_performance(dims, t.min_time_ms)
  cublas_time = benchmark_cublas(dims)
  print(f"Reference cuBLAS time (bf16xbf16->bf16): {cublas_time:0.4f}ms")


if __name__ == "__main__":
  triton.compiler.init_cuda_utils()
  app.parse_flags_with_usage(sys.argv)
  main()
