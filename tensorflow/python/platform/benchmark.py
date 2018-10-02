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

"""Utilities to run benchmarks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import os
import re
import sys
import time

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


# When a subclass of the Benchmark class is created, it is added to
# the registry automatically
GLOBAL_BENCHMARK_REGISTRY = set()

# Environment variable that determines whether benchmarks are written.
# See also tensorflow/core/util/reporter.h TestReporter::kTestReporterEnv.
TEST_REPORTER_TEST_ENV = "TEST_REPORT_FILE_PREFIX"


def _global_report_benchmark(
    name, iters=None, cpu_time=None, wall_time=None,
    throughput=None, extras=None):
  """Method for recording a benchmark directly.

  Args:
    name: The BenchmarkEntry name.
    iters: (optional) How many iterations were run
    cpu_time: (optional) Total cpu time in seconds
    wall_time: (optional) Total wall time in seconds
    throughput: (optional) Throughput (in MB/s)
    extras: (optional) Dict mapping string keys to additional benchmark info.

  Raises:
    TypeError: if extras is not a dict.
    IOError: if the benchmark output file already exists.
  """
  if extras is not None:
    if not isinstance(extras, dict):
      raise TypeError("extras must be a dict")

  logging.info("Benchmark [%s] iters: %d, wall_time: %g, cpu_time: %g,"
               "throughput: %g %s", name, iters if iters is not None else -1,
               wall_time if wall_time is not None else -1, cpu_time if
               cpu_time is not None else -1, throughput if
               throughput is not None else -1, str(extras) if extras else "")

  entries = test_log_pb2.BenchmarkEntries()
  entry = entries.entry.add()
  entry.name = name
  if iters is not None:
    entry.iters = iters
  if cpu_time is not None:
    entry.cpu_time = cpu_time
  if wall_time is not None:
    entry.wall_time = wall_time
  if throughput is not None:
    entry.throughput = throughput
  if extras is not None:
    for (k, v) in extras.items():
      if isinstance(v, numbers.Number):
        entry.extras[k].double_value = v
      else:
        entry.extras[k].string_value = str(v)

  test_env = os.environ.get(TEST_REPORTER_TEST_ENV, None)
  if test_env is None:
    # Reporting was not requested, just print the proto
    print(str(entries))
    return

  serialized_entry = entries.SerializeToString()

  mangled_name = name.replace("/", "__")
  output_path = "%s%s" % (test_env, mangled_name)
  if gfile.Exists(output_path):
    raise IOError("File already exists: %s" % output_path)
  with gfile.GFile(output_path, "wb") as out:
    out.write(serialized_entry)


class _BenchmarkRegistrar(type):
  """The Benchmark class registrar.  Used by abstract Benchmark class."""

  def __new__(mcs, clsname, base, attrs):
    newclass = super(mcs, _BenchmarkRegistrar).__new__(
        mcs, clsname, base, attrs)
    if not newclass.is_abstract():
      GLOBAL_BENCHMARK_REGISTRY.add(newclass)
    return newclass


class Benchmark(six.with_metaclass(_BenchmarkRegistrar, object)):
  """Abstract class that provides helper functions for running benchmarks.

  Any class subclassing this one is immediately registered in the global
  benchmark registry.

  Only methods whose names start with the word "benchmark" will be run during
  benchmarking.
  """

  @classmethod
  def is_abstract(cls):
    # mro: (_BenchmarkRegistrar, Benchmark) means this is Benchmark
    return len(cls.mro()) <= 2

  def _get_name(self, overwrite_name=None):
    """Returns full name of class and method calling report_benchmark."""

    # Find the caller method (outermost Benchmark class)
    stack = tf_inspect.stack()
    calling_class = None
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, Benchmark):
        calling_class = f_self  # Get the outermost stack Benchmark call
        name = frame[3]  # Get the method name
        break
    if calling_class is None:
      raise ValueError("Unable to determine calling Benchmark class.")

    # Use the method name, or overwrite_name is provided.
    name = overwrite_name or name
    # Prefix the name with the class name.
    class_name = type(calling_class).__name__
    name = "%s.%s" % (class_name, name)
    return name

  def report_benchmark(
      self,
      iters=None,
      cpu_time=None,
      wall_time=None,
      throughput=None,
      extras=None,
      name=None):
    """Report a benchmark.

    Args:
      iters: (optional) How many iterations were run
      cpu_time: (optional) median or mean cpu time in seconds.
      wall_time: (optional) median or mean wall time in seconds.
      throughput: (optional) Throughput (in MB/s)
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
    """
    name = self._get_name(overwrite_name=name)
    _global_report_benchmark(
        name=name, iters=iters, cpu_time=cpu_time, wall_time=wall_time,
        throughput=throughput, extras=extras)


@tf_export("test.Benchmark")
class TensorFlowBenchmark(Benchmark):
  """Abstract class that provides helpers for TensorFlow benchmarks."""

  @classmethod
  def is_abstract(cls):
    # mro: (_BenchmarkRegistrar, Benchmark, TensorFlowBenchmark) means
    # this is TensorFlowBenchmark.
    return len(cls.mro()) <= 3

  def run_op_benchmark(self,
                       sess,
                       op_or_tensor,
                       feed_dict=None,
                       burn_iters=2,
                       min_iters=10,
                       store_trace=False,
                       store_memory_usage=True,
                       name=None,
                       extras=None,
                       mbs=0):
    """Run an op or tensor in the given session.  Report the results.

    Args:
      sess: `Session` object to use for timing.
      op_or_tensor: `Operation` or `Tensor` to benchmark.
      feed_dict: A `dict` of values to feed for each op iteration (see the
        `feed_dict` parameter of `Session.run`).
      burn_iters: Number of burn-in iterations to run.
      min_iters: Minimum number of iterations to use for timing.
      store_trace: Boolean, whether to run an extra untimed iteration and
        store the trace of iteration in returned extras.
        The trace will be stored as a string in Google Chrome trace format
        in the extras field "full_trace_chrome_format". Note that trace
        will not be stored in test_log_pb2.TestResults proto.
      store_memory_usage: Boolean, whether to run an extra untimed iteration,
        calculate memory usage, and store that in extras fields.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      mbs: (optional) The number of megabytes moved by this op, used to
        calculate the ops throughput.

    Returns:
      A `dict` containing the key-value pairs that were passed to
      `report_benchmark`. If `store_trace` option is used, then
      `full_chrome_trace_format` will be included in return dictionary even
      though it is not passed to `report_benchmark` with `extras`.
    """
    for _ in range(burn_iters):
      sess.run(op_or_tensor, feed_dict=feed_dict)

    deltas = [None] * min_iters

    for i in range(min_iters):
      start_time = time.time()
      sess.run(op_or_tensor, feed_dict=feed_dict)
      end_time = time.time()
      delta = end_time - start_time
      deltas[i] = delta

    extras = extras if extras is not None else {}
    unreported_extras = {}
    if store_trace or store_memory_usage:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(op_or_tensor, feed_dict=feed_dict,
               options=run_options, run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)

      if store_trace:
        unreported_extras["full_trace_chrome_format"] = (
            tl.generate_chrome_trace_format())

      if store_memory_usage:
        step_stats_analysis = tl.analyze_step_stats(show_memory=True)
        allocator_maximums = step_stats_analysis.allocator_maximums
        for k, v in allocator_maximums.items():
          extras["allocator_maximum_num_bytes_%s" % k] = v.num_bytes

    def _median(x):
      if not x:
        return -1
      s = sorted(x)
      l = len(x)
      lm1 = l - 1
      return (s[l//2] + s[lm1//2]) / 2.0

    median_delta = _median(deltas)

    benchmark_values = {
        "iters": min_iters,
        "wall_time": median_delta,
        "extras": extras,
        "name": name,
        "throughput": mbs / median_delta
    }
    self.report_benchmark(**benchmark_values)
    benchmark_values["extras"].update(unreported_extras)
    return benchmark_values


def _run_benchmarks(regex):
  """Run benchmarks that match regex `regex`.

  This function goes through the global benchmark registry, and matches
  benchmark class and method names of the form
  `module.name.BenchmarkClass.benchmarkMethod` to the given regex.
  If a method matches, it is run.

  Args:
    regex: The string regular expression to match Benchmark classes against.
  """
  registry = list(GLOBAL_BENCHMARK_REGISTRY)

  # Match benchmarks in registry against regex
  for benchmark in registry:
    benchmark_name = "%s.%s" % (benchmark.__module__, benchmark.__name__)
    attrs = dir(benchmark)
    # Don't instantiate the benchmark class unless necessary
    benchmark_instance = None

    for attr in attrs:
      if not attr.startswith("benchmark"):
        continue
      candidate_benchmark_fn = getattr(benchmark, attr)
      if not callable(candidate_benchmark_fn):
        continue
      full_benchmark_name = "%s.%s" % (benchmark_name, attr)
      if regex == "all" or re.search(regex, full_benchmark_name):
        # Instantiate the class if it hasn't been instantiated
        benchmark_instance = benchmark_instance or benchmark()
        # Get the method tied to the class
        instance_benchmark_fn = getattr(benchmark_instance, attr)
        # Call the instance method
        instance_benchmark_fn()


def benchmarks_main(true_main, argv=None):
  """Run benchmarks as declared in argv.

  Args:
    true_main: True main function to run if benchmarks are not requested.
    argv: the command line arguments (if None, uses sys.argv).
  """
  if argv is None:
    argv = sys.argv
  found_arg = [arg for arg in argv
               if arg.startswith("--benchmarks=")
               or arg.startswith("-benchmarks=")]
  if found_arg:
    # Remove --benchmarks arg from sys.argv
    argv.remove(found_arg[0])

    regex = found_arg[0].split("=")[1]
    app.run(lambda _: _run_benchmarks(regex), argv=argv)
  else:
    true_main()
