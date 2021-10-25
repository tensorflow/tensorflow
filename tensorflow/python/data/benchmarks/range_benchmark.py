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
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib


class RangeBenchmark(benchmark_base.DatasetBenchmarkBase):
  """Benchmarks for `tf.data.Dataset.range()`."""

  def _benchmark_range(self, num_elements, autotune, benchmark_id):
    options = options_lib.Options()
    options.autotune.enabled = autotune
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.with_options(options)

    self.run_and_report_benchmark(
        dataset,
        num_elements=num_elements,
        extras={
            "model_name": "range.benchmark.%d" % benchmark_id,
            "parameters": "%d.%s" % (num_elements, autotune),
        },
        name="modeling_%s" % ("on" if autotune else "off"))

  def benchmark_range_with_modeling(self):
    self._benchmark_range(num_elements=10000000, autotune=True, benchmark_id=1)

  def benchmark_range_without_modeling(self):
    self._benchmark_range(num_elements=50000000, autotune=False, benchmark_id=2)


if __name__ == "__main__":
  benchmark_base.test.main()
