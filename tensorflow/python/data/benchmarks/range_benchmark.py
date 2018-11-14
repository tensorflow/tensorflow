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

import time

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test

_NUMPY_RANDOM_SEED = 42


class RangeBenchmark(test.Benchmark):
  """Benchmarks for `tf.data.Dataset.range()`."""

  def _benchmarkRangeHelper(self, modeling_enabled):
    num_elements = 10000000 if modeling_enabled else 50000000
    options = dataset_ops.Options()
    options.experimental_autotune = modeling_enabled

    # Use `Dataset.skip()` and `Dataset.take()` to perform the iteration in
    # C++, and focus on the minimal overheads (excluding Python invocation
    # costs).
    dataset = dataset_ops.Dataset.range(num_elements).skip(
        num_elements - 1).take(1).with_options(options)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      # Run once to warm up the session caches.
      sess.run(iterator.initializer)
      sess.run(next_element)

      # Run once for timing.
      sess.run(iterator.initializer)
      start = time.time()
      sess.run(next_element)
      end = time.time()

      time_per_element = (end - start) / num_elements
      print("Average time per element (%s modeling): %f nanoseconds" % (
          "with" if modeling_enabled else "without", time_per_element * 1e9))
      self.report_benchmark(iters=num_elements, wall_time=time_per_element,
                            name="benchmark_tf_data_dataset_range%s"
                            % ("_with_modeling" if modeling_enabled else ""))

  def benchmarkRange(self):
    for modeling_enabled in [False, True]:
      self._benchmarkRangeHelper(modeling_enabled)


if __name__ == "__main__":
  test.main()
