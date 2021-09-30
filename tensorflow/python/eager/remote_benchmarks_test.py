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
r"""Benchmarks for remote worker eager execution.

To run CPU benchmarks:
  bazel run -c opt remote_benchmarks_test -- --benchmarks=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" remote_benchmarks_test -- \
    --benchmarks=.
"""

import gc
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import server_lib


def run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to maybe warm up the GPU
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class Foo(object):

  def __init__(self, num_vars):
    self._num_vars = num_vars
    self._v = []

  def __call__(self, inputs):
    if not self._v:
      for _ in range(self._num_vars):
        self._v.append(variables.Variable(
            random_ops.random_uniform([]), shape=[]))
    for v in self._v:
      inputs = inputs * v
    return inputs


class RemoteWorkerMicroBenchmarks(test.Benchmark):

  def __init__(self):
    # used for remote benchmarks
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server_target1 = self._cached_server1.target[len("grpc://"):]
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server_target2 = self._cached_server2.target[len("grpc://"):]

  def _run(self, func, num_iters=1000, execution_mode=context.ASYNC):
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / total_time})

  def benchmark_send(self):
    remote.connect_to_remote_host(self._cached_server_target1)

    x = random_ops.random_uniform((2, 2)).cpu()

    @def_function.function
    def remote_func(m):
      return math_ops.matmul(m, m)

    def func(m):
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        return remote_func(m)

    self._run(lambda: func(x))
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()

  def benchmark_worker_recv(self):
    remote.connect_to_remote_host(
        [self._cached_server_target1, self._cached_server_target2])

    with ops.device("job:worker/replica:0/task:1/device:CPU:0"):
      v = variables.Variable(1.0)

    @def_function.function
    def remote_func():
      return 1.0 + v

    def func():
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        return remote_func()

    self._run(func)
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()

  def benchmark_create_vars_inside_function(self):
    remote.connect_to_remote_host(self._cached_server_target1)

    def func():
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        layer = Foo(50)

        @def_function.function
        def remote_func():
          with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
            return layer(random_ops.random_uniform([]))

        return remote_func()

    self._run(func, execution_mode=context.ASYNC, num_iters=100)
    # NOTE(b/136184459): Force garbage collecting hanging resources before
    # subsequent calls to set_server_def, to ensure the destroy resource ops are
    # executed when their corresponding device and manager are still available.
    gc.collect()


if __name__ == "__main__":
  test.main()
