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

"""Launch Triton search for good tiling sizes, save to CSV."""

import concurrent.futures
import csv
import itertools
import logging
import os
import random
import sys
import time
import typing

from absl import app
from absl import flags
from matmul_lib import benchmark_matmul
from matmul_lib import generate_tiling_configs
from matmul_lib import MatmulSize
from matmul_lib import MatmulTiming
from matmul_lib import parse_int_list
from matmul_lib import parse_layout_list
from matmul_lib import QuantizedInputType
import numpy as np
import torch
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)

_OUTPUT_FILE = flags.DEFINE_string(
    'output_file',
    'out.csv',
    """File to generate output into.

1) Output is streamed: for each point processed, incremental output is written
out.
2) Restarts with checkpointing are supported: the script will not regenerate data
for files already present.
""",
)
_MAX_WORKERS = flags.DEFINE_integer(
    'max_workers', 64, 'Number of threads to use'
)
_REPETITIONS_MS = flags.DEFINE_integer(
    'repetitions_ms', 300, 'Number of requests'
)
_NUM_SAMPLES = flags.DEFINE_integer('num_samples', 1000, 'Number of samples ')
_TILINGS_M = flags.DEFINE_string(
    'tilings_m', '32, 64, 128, 256', 'Tilings to try for M'
)
_TILINGS_N = flags.DEFINE_string(
    'tilings_n', '32, 64, 128, 256', 'Tilings to try for N'
)
_TILINGS_K = flags.DEFINE_string(
    'tilings_k', '32, 64, 128, 256, 512', 'Tilings to try for K'
)
_LHS_LAYOUTS = flags.DEFINE_string(
    'lhs_layouts', 'row_major', 'Layouts to try for LHS'
)
_RHS_LAYOUTS = flags.DEFINE_string(
    'rhs_layouts', 'row_major', 'Layouts to try for RHS'
)
_RESULT_LAYOUTS = flags.DEFINE_string(
    'result_layouts', 'row_major', 'Layouts to try for the result'
)
_NUM_STAGES = flags.DEFINE_string(
    'num_stages', '1,2,3', 'Number of stages to try'
)
_NUM_WARPS = flags.DEFINE_string('num_warps', '4,8', 'Number of warps to try')
_SPLIT_KS = flags.DEFINE_string(
    'split_ks', '1,2,3,4,5', 'Number of split_k values to try'
)

logging.basicConfig(
    format=(
        '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d]'
        ' %(threadName)15s: %(message)s'
    ),
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)

# pylint: disable=g-long-lambda
# pylint: disable=g-complex-comprehension
# pylint: disable=cell-var-from-loop


def read_timings() -> typing.Set[MatmulSize]:
  """Find timings already existing in the file."""
  out: typing.Set[MatmulSize] = set()
  with open(_OUTPUT_FILE.value) as f:
    reader = csv.reader(f)
    for row in reader:
      if row[0].isdigit():
        # M, N, K + quantized_lhs
        out.add(MatmulSize(*map(int, row[:4])))
  return out


def write_csv_header() -> None:
  """Write CSV file header."""
  with open(_OUTPUT_FILE.value, 'w') as f:
    fieldnames = [
        'M',
        'N',
        'K',
        'quantized_lhs',
        'BLOCK_M',
        'BLOCK_N',
        'BLOCK_K',
        'SPLIT_K',
        'lhs_layout',
        'rhs_layout',
        'result_layout',
        'num_stages',
        'num_warps',
        'min_time_ms',
    ]
    writer = csv.writer(f)
    writer.writerow(fieldnames)


def write_timings(timings: typing.Sequence[MatmulTiming]) -> None:
  """Write matmul timing data to CSV output."""
  with open(_OUTPUT_FILE.value, 'a') as f:
    writer = csv.writer(f)
    for d in timings:
      writer.writerow([
          d.dims.M,
          d.dims.N,
          d.dims.K,
          d.dims.quantized_lhs,
          d.tiling.BLOCK_M,
          d.tiling.BLOCK_N,
          d.tiling.BLOCK_K,
          d.tiling.SPLIT_K,
          d.tiling.num_stages,
          d.tiling.num_warps,
          d.min_time_ms,
      ])


def generate_samples() -> typing.List[MatmulSize]:
  """Generate a list of matmuls we will be benchmarking."""
  m_axis = np.unique(np.logspace(4, 13, num=200, dtype=np.int64, base=2))
  n_axis = np.unique(np.logspace(4, 13, num=200, dtype=np.int64, base=2))
  k_axis = np.unique(np.logspace(4, 13, num=200, dtype=np.int64, base=2))
  quants = [QuantizedInputType.BFLOAT16, QuantizedInputType.INT8]
  prod = itertools.product(m_axis, n_axis, k_axis, quants[1], quants[0])
  out = random.choices((MatmulSize(*p) for p in prod), k=_NUM_SAMPLES.value)
  return out


def run_search(
    existing_samples: typing.Set[MatmulSize],
) -> typing.Sequence[MatmulTiming]:
  """Run search on a list of matmul configurations."""
  samples: typing.Sequence[MatmulSize] = [
      s for s in generate_samples() if s not in existing_samples
  ]
  t0 = time.time()
  shared_stream = torch.cuda.Stream()
  tilings = generate_tiling_configs(
      parse_int_list(_TILINGS_M.value),
      parse_int_list(_TILINGS_N.value),
      parse_int_list(_TILINGS_K.value),
      parse_layout_list(_LHS_LAYOUTS.value),
      parse_layout_list(_RHS_LAYOUTS.value),
      parse_layout_list(_RESULT_LAYOUTS.value),
      parse_int_list(_SPLIT_KS.value),
      parse_int_list(_NUM_STAGES.value),
      parse_int_list(_NUM_WARPS.value),
  )

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_MAX_WORKERS.value
  ) as executor:
    pbar = tqdm.tqdm(total=len(samples) * len(tilings), ncols=0)
    results = []
    with logging_redirect_tqdm():
      if _MAX_WORKERS.value == 1:
        for c in samples:
          res = benchmark_matmul(
              c, pbar, shared_stream, tilings, _REPETITIONS_MS.value
          )
          results.extend(res)
          write_timings(res)
      else:
        future_to_dims = {
            executor.submit(
                benchmark_matmul,
                c,
                pbar,
                shared_stream,
                tilings,
                _REPETITIONS_MS.value,
            ): c
            for c in samples
        }
        for future in concurrent.futures.as_completed(future_to_dims):
          res = future.result()
          results.extend(res)
          write_timings(res)

    pbar.close()

  LOG.info('%d datapoints generated in %.2fs', len(results), (time.time() - t0))
  return results


def main() -> None:
  existing_samples: typing.Set[MatmulSize] = set()
  if os.path.isfile(_OUTPUT_FILE.value):
    existing_samples = read_timings()
  else:
    write_csv_header()

  run_search(existing_samples)


if __name__ == '__main__':
  random.seed(42)
  app.parse_flags_with_usage(sys.argv)
  main()
