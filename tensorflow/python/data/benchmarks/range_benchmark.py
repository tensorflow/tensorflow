# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for `tf.data.Dataset.range()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops


class RangeBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.range()`."""

  def benchmark_range(self):
    for modeling_enabled in [False, True]:
      num_elements = 10000000 if modeling_enabled else 50000000
      options = dataset_ops.Options()
      options.experimental_autotune = modeling_enabled
      dataset = dataset_ops.Dataset.range(num_elements)
      dataset = dataset.with_options(options)

      self.run_and_report_benchmark(
          dataset,
          num_elements=num_elements,
          name="modeling_%s" % ("on" if modeling_enabled else "off"))


if __name__ == "__main__":
  benchmark_base.test.main()
