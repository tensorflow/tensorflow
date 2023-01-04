# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and benchmarks for interacting with the `tf.compat.v1.Session`."""

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SessionBenchmark(test.Benchmark):
  """Tests and benchmarks for interacting with the `tf.compat.v1.Session`."""

  def _benchmarkFeed(self, name, target, size, iters):
    """Runs a microbenchmark to measure the cost of feeding a tensor.

    Reports the median cost of feeding a tensor of `size` * `sizeof(float)`
    bytes.

    Args:
      name: A human-readable name for logging the output.
      target: The session target to use for the benchmark.
      size: The number of floating-point numbers to be feed.
      iters: The number of iterations to perform.
    """
    feed_val = np.random.rand(size).astype(np.float32)
    times = []
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=[size])
      # Fetch the operation rather than the tensor, to avoid measuring the time
      # to fetch back the value.
      no_op = array_ops.identity(p).op
      with session.Session(target) as sess:
        sess.run(no_op, feed_dict={p: feed_val})  # Warm-up run.
        for _ in range(iters):
          start_time = time.time()
          sess.run(no_op, feed_dict={p: feed_val})
          end_time = time.time()
          times.append(end_time - start_time)
    print("%s %d %f" % (name, size, np.median(times)))
    self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

  def _benchmarkFetch(self, name, target, size, iters):
    """Runs a microbenchmark to measure the cost of fetching a tensor.

    Reports the median cost of fetching a tensor of `size` * `sizeof(float)`
    bytes.

    Args:
      name: A human-readable name for logging the output.
      target: The session target to use for the benchmark.
      size: The number of floating-point numbers to be fetched.
      iters: The number of iterations to perform.
    """
    times = []
    with ops.Graph().as_default():
      # Define the tensor to be fetched as a variable, to avoid
      # constant-folding.
      v = variables.Variable(random_ops.random_normal([size]))
      with session.Session(target) as sess:
        sess.run(v.initializer)
        sess.run(v)  # Warm-up run.
        for _ in range(iters):
          start_time = time.time()
          sess.run(v)
          end_time = time.time()
          times.append(end_time - start_time)
    print("%s %d %f" % (name, size, np.median(times)))
    self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

  def _benchmarkFetchPrebuilt(self, name, target, size, iters):
    """Runs a microbenchmark to measure the cost of fetching a tensor.

    Reports the median cost of fetching a tensor of `size` * `sizeof(float)`
    bytes.

    Args:
      name: A human-readable name for logging the output.
      target: The session target to use for the benchmark.
      size: The number of floating-point numbers to be fetched.
      iters: The number of iterations to perform.
    """
    times = []
    with ops.Graph().as_default():
      # Define the tensor to be fetched as a variable, to avoid
      # constant-folding.
      v = variables.Variable(random_ops.random_normal([size]))
      with session.Session(target) as sess:
        sess.run(v.initializer)
        runner = sess.make_callable(v)
        runner()  # Warm-up run.
        for _ in range(iters):
          start_time = time.time()
          runner()
          end_time = time.time()
          times.append(end_time - start_time)
    print("%s %d %f" % (name, size, np.median(times)))
    self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

  def _benchmarkRunOp(self, name, target, iters):
    """Runs a microbenchmark to measure the cost of running an op.

    Reports the median cost of running a trivial (Variable) op.

    Args:
      name: A human-readable name for logging the output.
      target: The session target to use for the benchmark.
      iters: The number of iterations to perform.
    """
    times = []
    with ops.Graph().as_default():
      # Define the op to be run as a variable, to avoid
      # constant-folding.
      v = variables.Variable(random_ops.random_normal([]))
      with session.Session(target) as sess:
        sess.run(v.initializer)
        sess.run(v.op)  # Warm-up run.
        for _ in range(iters):
          start_time = time.time()
          sess.run(v.op)
          end_time = time.time()
          times.append(end_time - start_time)
    print("%s %f" % (name, np.median(times)))
    self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

  def _benchmarkRunOpPrebuilt(self, name, target, iters):
    """Runs a microbenchmark to measure the cost of running an op.

    Reports the median cost of running a trivial (Variable) op.

    Args:
      name: A human-readable name for logging the output.
      target: The session target to use for the benchmark.
      iters: The number of iterations to perform.
    """
    times = []
    with ops.Graph().as_default():
      # Define the op to be run as a variable, to avoid
      # constant-folding.
      v = variables.Variable(random_ops.random_normal([]))
      with session.Session(target) as sess:
        sess.run(v.initializer)
        runner = sess.make_callable(v.op)
        runner()  # Warm-up run.
        for _ in range(iters):
          start_time = time.time()
          runner()
          end_time = time.time()
          times.append(end_time - start_time)
    print("%s %f" % (name, np.median(times)))
    self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

  def benchmarkGrpcSession(self):
    server = server_lib.Server.create_local_server()
    self._benchmarkFeed("benchmark_session_feed_grpc_4B", server.target, 1,
                        30000)
    session.Session.reset(server.target)
    self._benchmarkFeed("benchmark_session_feed_grpc_4MB", server.target,
                        1 << 20, 25000)
    session.Session.reset(server.target)
    self._benchmarkFetch("benchmark_session_fetch_grpc_4B", server.target, 1,
                         40000)
    session.Session.reset(server.target)
    self._benchmarkFetch("benchmark_session_fetch_grpc_4MB", server.target,
                         1 << 20, 20000)
    session.Session.reset(server.target)
    self._benchmarkFetchPrebuilt("benchmark_session_fetchprebuilt_grpc_4B",
                                 server.target, 1, 50000)
    session.Session.reset(server.target)
    self._benchmarkFetchPrebuilt("benchmark_session_fetchprebuilt_grpc_4MB",
                                 server.target, 1 << 20, 50000)
    session.Session.reset(server.target)
    self._benchmarkRunOp("benchmark_session_runop_grpc", server.target, 50000)
    session.Session.reset(server.target)
    self._benchmarkRunOpPrebuilt("benchmark_session_runopprebuilt_grpc",
                                 server.target, 100000)
    session.Session.reset(server.target)

  def benchmarkDirectSession(self):
    self._benchmarkFeed("benchmark_session_feed_direct_4B", "", 1, 80000)
    self._benchmarkFeed("benchmark_session_feed_direct_4MB", "", 1 << 20, 20000)
    self._benchmarkFetch("benchmark_session_fetch_direct_4B", "", 1, 100000)
    self._benchmarkFetch("benchmark_session_fetch_direct_4MB", "", 1 << 20,
                         20000)
    self._benchmarkFetchPrebuilt("benchmark_session_fetchprebuilt_direct_4B",
                                 "", 1, 200000)
    self._benchmarkFetchPrebuilt("benchmark_session_fetchprebuilt_direct_4MB",
                                 "", 1 << 20, 200000)
    self._benchmarkRunOp("benchmark_session_runop_direct", "", 200000)
    self._benchmarkRunOpPrebuilt("benchmark_session_runopprebuilt_direct", "",
                                 200000)


if __name__ == "__main__":
  test.main()
