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
"""Local CPU benchmarks for collective ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test


class CollectiveOpBenchmark(test.Benchmark):
  """Benchmarks for local CPU collective op execution."""

  def benchmark_collective(self):
    """Measures the performance of local CPU collective execution."""
    shapes = [(10,), (1000,), (1000000,)]
    devices = [2, 4, 8]
    collective_key_counter = 0

    for group_size in devices:
      group_key = collective_key_counter
      instance_key = collective_key_counter
      collective_key_counter += 1

      for shape in shapes:
        config = config_pb2.ConfigProto(device_count={"CPU": group_size})
        with session.Session(config=config) as sess:
          # Use a C++ callable to minimize the Python overhead in the benchmark.
          callable_opts = config_pb2.CallableOptions()
          reduce_ops = []
          for device in range(group_size):
            with ops.device("CPU:{}".format(device)):
              t = constant_op.constant(np.multiply(range(shape[0]), 1.0))
              r = collective_ops.all_reduce(t, group_size, group_key,
                                            instance_key, "Add", "Div")
              reduce_ops.append(r)
              callable_opts.target.append(r.name)
          op_callable = sess._make_callable_from_options(callable_opts)  # pylint: disable=protected-access

          # Run five steps to warm up the session caches and do collective param
          # resolution before taking the first measurement.
          for _ in range(5):
            op_callable()
          deltas = []
          overall_start = time.time()
          # Run at least five repetitions and for at least five seconds.
          while len(deltas) < 5 or time.time() - overall_start < 5.0:
            start = time.time()
            for _ in range(100):
              op_callable()
            end = time.time()
            deltas.append(end - start)
          del op_callable

        median_wall_time = np.median(deltas) / 100.0
        iters = len(deltas) * 100

        self.report_benchmark(
            iters=iters, wall_time=median_wall_time,
            name="num_elements_{}_num_devices_{}".format(np.prod(shape),
                                                         group_size))


if __name__ == "__main__":
  test.main()
