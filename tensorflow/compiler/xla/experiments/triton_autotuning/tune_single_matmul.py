#!/usr/bin/env python
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

"""Finds best tuning for a single matmul"""
import sys
import typing
import itertools

import torch
from absl import app
from absl import flags
import tqdm
import triton

from matmul_lib import MatmulSize, MatmulTiming, benchmark_matmul, generate_tiling_configs, benchmark_cublas, print_roofline_performance, parse_int_list

FLAGS = flags.FLAGS

flags.DEFINE_integer('m', 64, 'Size of first matrix')
flags.DEFINE_integer('k', 64, 'Size of contracting dimension')
flags.DEFINE_integer('n', 64, 'Size of second matrix')
flags.DEFINE_integer('quantized_lhs', 0, 'Whether LHS is in int8')

flags.DEFINE_string('tilings_m', '32, 64, 128, 256', 'Tilings to try for M')
flags.DEFINE_string('tilings_n', '32, 64, 128, 256', 'Tilings to try for N')
flags.DEFINE_string('tilings_k', '32, 64, 128, 256, 512', 'Tilings to try for K')
flags.DEFINE_string(
    'num_stages', '1,2,3', 'Number of stages to try'
)
flags.DEFINE_string('num_warps', '4,8', 'Number of warps to try')
flags.DEFINE_string(
    'split_ks', '1,2,3,4,5', 'Number of split_k values to try'
)
flags.DEFINE_bool('debug', False, 'Print debug information')


def main() -> None:
  dims = MatmulSize(M=FLAGS.m, N=FLAGS.n, K=FLAGS.k, quantized_lhs=FLAGS.quantized_lhs)
  s = torch.cuda.Stream()
  tilings = generate_tiling_configs(
      parse_int_list(FLAGS.tilings_m),
      parse_int_list(FLAGS.tilings_n),
      parse_int_list(FLAGS.tilings_k),
      parse_int_list(FLAGS.split_ks),
      parse_int_list(FLAGS.num_stages),
      parse_int_list(FLAGS.num_warps))
  pbar = tqdm.tqdm(total=len(tilings))
  timings = sorted(benchmark_matmul(dims, pbar, s, tilings, repetitions=10,
                                    debug=FLAGS.debug),
                   key=lambda t: t.min_time_ms)
  fastest : MatmulTiming = timings[0]
  print(f"Fastest configuration: {fastest}")

  features = {'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'SPLIT_K', 'num_stages', 'num_warps'}
  for f in features:
    other_features = features - {f}

    other_features_equal_to_best = lambda t: all(
        getattr(fastest.tiling, of) == getattr(
            t.tiling, of) for of in other_features)

    # Keep everyting but the currently evaluated feature fixed to the best value.
    others_fixed = [t for t in timings if other_features_equal_to_best(t)]

    # TODO(cheshire): Visualize.
    print(f"Varying feature {f}:",
          ", ".join(
        f"{t.min_time_ms:0.4f} @ {f}={getattr(t.tiling, f)}" for t in others_fixed))

  print_roofline_performance(dims, fastest.min_time_ms)
  cublas_time = benchmark_cublas(dims)
  print(f"Reference cuBLAS time (bf16xbf16->bf16): {cublas_time:0.4f}ms")


if __name__ == '__main__':
  app.parse_flags_with_usage(sys.argv)
  triton.compiler.init_cuda_utils()
  main()
