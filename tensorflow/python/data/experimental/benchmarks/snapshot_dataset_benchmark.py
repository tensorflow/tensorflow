# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.experimental.snapshot()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensorflow.python.client import session
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class SnapshotDatasetBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.experimental.snapshot()`."""

  def _makeSnapshotDirectory(self):
    tmp_dir = test.get_temp_dir()
    tmp_dir = os.path.join(tmp_dir, "snapshot")
    if os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    return tmp_dir

  def _createSimpleDataset(self, num_elems, tmp_dir=None,
                           compression=snapshot.COMPRESSION_NONE):
    if not tmp_dir:
      tmp_dir = self._makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    dataset = dataset.map(
        lambda x: gen_array_ops.broadcast_to(x, [50, 50, 3]))
    dataset = dataset.repeat(num_elems)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmp_dir, compression=compression))

    return dataset

  def _consumeDataset(self, dataset, num_elems):
    dataset = dataset.skip(num_elems)
    next_element = dataset_ops.make_one_shot_iterator(dataset).get_next()
    with session.Session() as sess:
      try:
        sess.run(next_element)
      except errors.OutOfRangeError:
        pass

  def benchmarkWriteSnapshotGzipCompression(self):
    num_elems = 500000
    dataset = self._createSimpleDataset(
        num_elems, compression=snapshot.COMPRESSION_GZIP)

    self.run_and_report_benchmark(dataset, num_elems, "write_gzip",
                                  warmup=False, iters=1)

  def benchmarkWriteSnapshotSnappyCompression(self):
    num_elems = 500000
    dataset = self._createSimpleDataset(
        num_elems, compression=snapshot.COMPRESSION_SNAPPY)

    self.run_and_report_benchmark(
        dataset, num_elems, "write_snappy", warmup=False, iters=1)

  def benchmarkWriteSnapshotSimple(self):
    num_elems = 500000
    dataset = self._createSimpleDataset(num_elems)

    # We only run one iteration here because running multiple iterations will
    # cause the later iterations to simply read from the already written
    # snapshot rather than write a new one.
    self.run_and_report_benchmark(dataset, num_elems, "write_simple",
                                  warmup=False, iters=1)

  def benchmarkPassthroughSnapshotSimple(self):
    num_elems = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(num_elems, tmp_dir)

    # Consume only 1 element, thus making sure we don't finalize.
    self._consumeDataset(dataset, 1)

    self.run_and_report_benchmark(dataset, num_elems, "passthrough_simple")

  def benchmarkReadSnapshotSimple(self):
    num_elems = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(num_elems, tmp_dir)

    # consume all the elements to let snapshot write things to disk
    self._consumeDataset(dataset, num_elems)

    self.run_and_report_benchmark(dataset, num_elems, "read_simple")

  def benchmarkReadSnapshotGzipCompression(self):
    num_elems = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elems, tmp_dir, compression=snapshot.COMPRESSION_GZIP)

    self._consumeDataset(dataset, num_elems)
    self.run_and_report_benchmark(dataset, num_elems, "read_gzip")

  def benchmarkReadSnapshotSnappyCompression(self):
    num_elems = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elems, tmp_dir, compression=snapshot.COMPRESSION_SNAPPY)

    self._consumeDataset(dataset, num_elems)
    self.run_and_report_benchmark(dataset, num_elems, "read_snappy")


if __name__ == "__main__":
  test.main()
