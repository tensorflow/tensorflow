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

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class SnapshotDatasetBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.experimental.snapshot()`."""

  def _makeSnapshotDirectory(self):
    tmp_dir = test.get_temp_dir()
    tmp_dir = os.path.join(tmp_dir, "snapshot")
    if os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    return tmp_dir

  def _createSimpleDataset(self,
                           num_elements,
                           tmp_dir=None,
                           compression=snapshot.COMPRESSION_NONE):
    if not tmp_dir:
      tmp_dir = self._makeSnapshotDirectory()

    dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    dataset = dataset.map(
        lambda x: gen_array_ops.broadcast_to(x, [50, 50, 3]))
    dataset = dataset.repeat(num_elements)
    dataset = dataset.apply(
        snapshot.legacy_snapshot(tmp_dir, compression=compression))

    return dataset

  def benchmarkWriteSnapshotGzipCompression(self):
    num_elements = 500000
    dataset = self._createSimpleDataset(
        num_elements=num_elements, compression=snapshot.COMPRESSION_GZIP)

    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        name="write_gzip",
        warmup=False,
        iters=1)

  def benchmarkWriteSnapshotSnappyCompression(self):
    num_elements = 500000
    dataset = self._createSimpleDataset(
        num_elements=num_elements, compression=snapshot.COMPRESSION_SNAPPY)

    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        name="write_snappy",
        warmup=False,
        iters=1)

  def benchmarkWriteSnapshotSimple(self):
    num_elements = 500000
    dataset = self._createSimpleDataset(num_elements=num_elements)

    # We only run one iteration here because running multiple iterations will
    # cause the later iterations to simply read from the already written
    # snapshot rather than write a new one.
    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        name="write_simple",
        warmup=False,
        iters=1)

  def benchmarkPassthroughSnapshotSimple(self):
    num_elements = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elements=num_elements, tmp_dir=tmp_dir)

    # Consume only 1 element, thus making sure we don't finalize.
    self.run_benchmark(
        dataset=dataset,
        num_elements=1,
        iters=1,
        warmup=False,
        apply_default_optimizations=True)
    # Now run the actual benchmarks and report them
    self.run_and_report_benchmark(
        dataset=dataset, num_elements=num_elements, name="passthrough_simple")

  def benchmarkReadSnapshotSimple(self):
    num_elements = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elements=num_elements, tmp_dir=tmp_dir)

    # consume all the elements to let snapshot write things to disk
    self.run_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=1,
        warmup=False,
        apply_default_optimizations=True)
    # Now run the actual benchmarks and report them
    self.run_and_report_benchmark(
        dataset=dataset, num_elements=num_elements, name="read_simple")

  def benchmarkReadSnapshotGzipCompression(self):
    num_elements = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elements=num_elements,
        tmp_dir=tmp_dir,
        compression=snapshot.COMPRESSION_GZIP)

    # consume all the elements to let snapshot write things to disk
    self.run_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=1,
        warmup=False,
        apply_default_optimizations=True)
    # Now run the actual benchmarks and report them
    self.run_and_report_benchmark(
        dataset=dataset, num_elements=num_elements, name="read_gzip")

  def benchmarkReadSnapshotSnappyCompression(self):
    num_elements = 100000
    tmp_dir = self._makeSnapshotDirectory()
    dataset = self._createSimpleDataset(
        num_elements=num_elements,
        tmp_dir=tmp_dir,
        compression=snapshot.COMPRESSION_SNAPPY)

    # consume all the elements to let snapshot write things to disk
    self.run_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=1,
        warmup=False,
        apply_default_optimizations=True)
    # Now run the actual benchmarks and report them
    self.run_and_report_benchmark(
        dataset=dataset, num_elements=num_elements, name="read_snappy")


if __name__ == "__main__":
  benchmark_base.test.main()
