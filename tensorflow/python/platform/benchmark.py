# Copyright 2016 Google Inc. All Rights Reserved.
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

import inspect
import numbers
import os
import re


from google.protobuf import text_format
from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import gfile

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

  test_env = os.environ.get(TEST_REPORTER_TEST_ENV, None)
  if test_env is None:
    # Reporting was not requested
    return

  entry = test_log_pb2.BenchmarkEntry()
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

  serialized_entry = text_format.MessageToString(entry)

  mangled_name = name.replace("/", "__")
  output_path = "%s%s" % (test_env, mangled_name)
  if gfile.Exists(output_path):
    raise IOError("File already exists: %s" % output_path)
  with gfile.GFile(output_path, "w") as out:
    out.write(serialized_entry)


class _BenchmarkRegistrar(type):
  """The Benchmark class registrar.  Used by abstract Benchmark class."""

  def __new__(mcs, clsname, base, attrs):
    newclass = super(mcs, _BenchmarkRegistrar).__new__(
        mcs, clsname, base, attrs)
    if len(newclass.mro()) > 2:
      # Only the base Benchmark abstract class has mro length 2.
      # The rest subclass from it and are therefore registered.
      GLOBAL_BENCHMARK_REGISTRY.add(newclass)
    return newclass


class Benchmark(object):
  """Abstract class that provides helper functions for running benchmarks.

  Any class subclassing this one is immediately registered in the global
  benchmark registry.

  Only methods whose names start with the word "benchmark" will be run during
  benchmarking.
  """
  __metaclass__ = _BenchmarkRegistrar

  def _get_name(self, overwrite_name):
    """Returns full name of class and method calling report_benchmark."""

    # Expect that the caller called report_benchmark, which called _get_name.
    caller = inspect.stack()[2]
    calling_class = caller[0].f_locals.get("self", None)
    # Use the method name, or overwrite_name is provided.
    name = overwrite_name if overwrite_name is not None else caller[3]
    if calling_class is not None:
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
      cpu_time: (optional) Total cpu time in seconds
      wall_time: (optional) Total wall time in seconds
      throughput: (optional) Throughput (in MB/s)
      extras: (optional) Dict mapping string keys to additional benchmark info.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the calling class and top-level
        method name.
    """
    name = self._get_name(overwrite_name=name)
    _global_report_benchmark(
        name=name, iters=iters, cpu_time=cpu_time, wall_time=wall_time,
        throughput=throughput, extras=extras)


def _run_specific_benchmark(benchmark_class):
  benchmark = benchmark_class()
  attrs = dir(benchmark)
  # Only run methods of this class whose names start with "benchmark"
  for attr in attrs:
    if not attr.startswith("benchmark"):
      continue
    benchmark_fn = getattr(benchmark, attr)
    if not callable(benchmark_fn):
      continue
    # Call this benchmark method
    benchmark_fn()


def run_benchmarks(args, kwargs):
  """Run benchmarks as declared in args.

  Args:
    args: List of args to main()
    kwargs: List of kwargs to main()

  Returns:
    Tuple (early_exit, new_args, kwargs), where
    early_exit: Bool, whether main() should now exit
    new_args: Updated args for the remainder (having removed benchmark flags)
    kwargs: Same as input kwargs.
  """
  exit_early = False

  registry = list(GLOBAL_BENCHMARK_REGISTRY)

  new_args = []
  for arg in args:
    if arg.startswith("--benchmarks="):
      exit_early = True
      regex = arg.split("=")[1]

      # Match benchmarks in registry against regex
      for benchmark in registry:
        benchmark_name = "%s.%s" % (benchmark.__module__, benchmark.__name__)
        if re.search(regex, benchmark_name):
          # Found a match
          _run_specific_benchmark(benchmark)
    else:
      new_args.append(arg)

  return (exit_early, new_args, kwargs)
