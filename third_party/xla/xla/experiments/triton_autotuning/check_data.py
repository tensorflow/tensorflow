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

"""Plot actual min time vs estimated min time from Triton performance model."""

from collections.abc import Sequence
from absl import app
import pandas as pd
import plotext as plt
import torch
import triton


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError('Incorrect number of command-line arguments.')
  f = argv[1]

  df = pd.read_csv(
      f,
      dtype={
          'M': int,
          'N': int,
          'K': int,
          'BLOCK_M': int,
          'BLOCK_N': int,
          'BLOCK_K': int,
          'SPLIT_K': int,
          'num_stages': int,
          'num_warps': int,
          'min_time_ms': float,
      },
  )
  grouped_df = df.groupby(['M', 'N', 'K']).min().sort_values('min_time_ms')

  estimated_times = []
  actual_times = []

  matrix = torch.randn(1, 1, device='cuda', dtype=torch.float16)
  for dims, r in grouped_df.iterrows():
    m, n, k = dims
    estimated_time = triton.ops.matmul_perf_model.estimate_matmul_time(
        num_warps=r.num_warps,
        num_stages=r.num_stages,
        A=matrix,
        B=matrix,
        C=matrix,
        M=m,
        N=n,
        K=k,
        BLOCK_M=r.BLOCK_M,
        BLOCK_N=r.BLOCK_N,
        BLOCK_K=r.BLOCK_K,
        SPLIT_K=r.SPLIT_K,
    )
    actual_times.append(r.min_time_ms)
    estimated_times.append(estimated_time)

  plt.theme('dark')
  plt.plot(actual_times, estimated_times)
  plt.xlabel('Actual Time (ms)')
  plt.ylabel('Estimated Time (ms)')
  plt.title('Estimated time as a function of actual time')
  plt.show()


if __name__ == '__main__':
  app.run(main)
