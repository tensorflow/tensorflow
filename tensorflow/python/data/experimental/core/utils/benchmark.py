# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Benchmark utils."""

import time
from typing import Any, Dict, Optional, Union

from absl import logging
import dataclasses
import tensorflow as tf
from tensorflow.data.experimental.core.utils import tqdm_utils

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
  pd = Any

# pylint: disable=logging-format-interpolation

StatDict = Dict[str, Union[int, float]]


@dataclasses.dataclass
class BenchmarkResult:
  stats: 'pd.DataFrame'
  raw_stats: 'pd.DataFrame'


def benchmark(
    ds: tf.data.Dataset,
    *,
    num_iter: Optional[int] = None,
    batch_size: int = 1,
) -> BenchmarkResult:
  """Benchmarks a `tf.data.Dataset`.

  Usage:

  ```py
  ds = tfds.load('mnist', split='train').batch(32).prefetch()
  tfds.core.benchmark(ds, batch_size=32)
  ```

  Reports:

  - Total execution time
  - Setup time (first warmup batch)
  - Number of examples/sec

  Args:
    ds: Dataset to benchmark
    num_iter: Number of iteration to perform (iteration might be batched)
    batch_size: Batch size of the dataset, used to normalize iterations

  Returns:
    statistics: The recorded statistics, for eventual post-processing
  """
  # Benchmark the first batch separatelly (setup overhead)
  start_time = time.perf_counter()  # pytype: disable=module-attr
  ds_iter = iter(ds)  # pytype: disable=wrong-arg-types
  next(ds_iter)  # First warmup batch
  first_batch_time = time.perf_counter()  # pytype: disable=module-attr

  # Benchmark the following batches
  i = None
  for i, _ in tqdm_utils.tqdm(enumerate(ds_iter)):
    if num_iter and i > num_iter:
      break
  end_time = time.perf_counter()  # pytype: disable=module-attr

  if num_iter and i < num_iter:
    logging.warning(
        'Number of iteration shorter than expected ({} vs {})'.format(
            i, num_iter
        )
    )

  print('\n************ Summary ************\n')
  num_examples = (i + 1) * batch_size
  stats = {
      'first+last': _log_stats(
          'First included', start_time, end_time, num_examples + batch_size
      ),
      'first': _log_stats(
          'First only', start_time, first_batch_time, batch_size
      ),
      'last': _log_stats(
          'First excluded', first_batch_time, end_time, num_examples
      )
  }
  raw_stats = {
      'start_time': start_time,
      'first_batch_time': first_batch_time,
      'end_time': end_time,
      'num_iter': i + 2,  # First batch and zero-shifted
  }
  stats = pd.DataFrame.from_dict(stats, orient='index')
  raw_stats = pd.DataFrame.from_dict(raw_stats, orient='index',
                                     columns=['duration'])
  return BenchmarkResult(stats=stats, raw_stats=raw_stats)


def _log_stats(
    msg: str, start: float, end: float, num_examples: int
) -> StatDict:
  """Log and returns stats."""
  total_time = end - start
  stats = {
      'duration': total_time,
      'num_examples': num_examples,
      'avg': num_examples / total_time,
  }
  print(
      'Examples/sec ({}) {avg:.2f} ex/sec (total: {num_examples} ex, '
      '{duration:.2f} sec)'.format(msg, **stats)
  )
  return stats
