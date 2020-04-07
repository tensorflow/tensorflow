#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.experimental.CsvDataset`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import tempfile
import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class CsvDatasetBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.experimental.CsvDataset`."""

  FLOAT_VAL = '1.23456E12'
  STR_VAL = string.ascii_letters * 10

  def _set_up(self, str_val):
    # Since this isn't test.TestCase, have to manually create a test dir
    gfile.MakeDirs(googletest.GetTempDir())
    self._temp_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())

    self._num_cols = [4, 64, 256]
    self._num_per_iter = 5000
    self._filenames = []
    for n in self._num_cols:
      fn = os.path.join(self._temp_dir, 'file%d.csv' % n)
      with open(fn, 'w') as f:
        # Just write 100 rows and use `repeat`... Assumes the cost
        # of creating an iterator is not significant
        row = ','.join(str_val for _ in range(n))
        f.write('\n'.join(row for _ in range(100)))
      self._filenames.append(fn)

  def _tear_down(self):
    gfile.DeleteRecursively(self._temp_dir)

  def _run_benchmark(self, dataset, num_cols, prefix):
    dataset = dataset.skip(self._num_per_iter - 1)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    deltas = []
    for _ in range(10):
      next_element = dataset_ops.make_one_shot_iterator(dataset).get_next()
      with session.Session() as sess:
        start = time.time()
        # NOTE: This depends on the underlying implementation of skip, to have
        # the net effect of calling `GetNext` num_per_iter times on the
        # input dataset. We do it this way (instead of a python for loop, or
        # batching N inputs in one iter) so that the overhead from session.run
        # or batch doesn't dominate. If we eventually optimize skip, this has
        # to change.
        sess.run(next_element)
        end = time.time()
      deltas.append(end - start)
    # Median wall time per CSV record read and decoded
    median_wall_time = np.median(deltas) / self._num_per_iter
    self.report_benchmark(
        iters=self._num_per_iter,
        wall_time=median_wall_time,
        name='%s_with_cols_%d' % (prefix, num_cols))

  def benchmark_map_with_floats(self):
    self._set_up(self.FLOAT_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [[0.0]] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))  # pylint: disable=cell-var-from-loop
      self._run_benchmark(dataset, num_cols, 'csv_float_map_decode_csv')
    self._tear_down()

  def benchmark_map_with_strings(self):
    self._set_up(self.STR_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [['']] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = dataset.map(lambda l: parsing_ops.decode_csv(l, **kwargs))  # pylint: disable=cell-var-from-loop
      self._run_benchmark(dataset, num_cols, 'csv_strings_map_decode_csv')
    self._tear_down()

  def benchmark_csv_dataset_with_floats(self):
    self._set_up(self.FLOAT_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [[0.0]] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()  # pylint: disable=cell-var-from-loop
      self._run_benchmark(dataset, num_cols, 'csv_float_fused_dataset')
    self._tear_down()

  def benchmark_csv_dataset_with_strings(self):
    self._set_up(self.STR_VAL)
    for i in range(len(self._filenames)):
      num_cols = self._num_cols[i]
      kwargs = {'record_defaults': [['']] * num_cols}
      dataset = core_readers.TextLineDataset(self._filenames[i]).repeat()
      dataset = readers.CsvDataset(self._filenames[i], **kwargs).repeat()  # pylint: disable=cell-var-from-loop
      self._run_benchmark(dataset, num_cols, 'csv_strings_fused_dataset')
    self._tear_down()

if __name__ == '__main__':
  test.main()
