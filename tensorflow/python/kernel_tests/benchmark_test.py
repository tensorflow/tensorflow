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
"""Tests for tensorflow.python.framework.importer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

# Used by SomeRandomBenchmark class below.
_ran_somebenchmark_1 = [False]
_ran_somebenchmark_2 = [False]
_ran_somebenchmark_but_shouldnt = [False]


class SomeRandomBenchmark(test.Benchmark):
  """This Benchmark should automatically be registered in the registry."""

  def _dontRunThisBenchmark(self):
    _ran_somebenchmark_but_shouldnt[0] = True

  def notBenchmarkMethod(self):
    _ran_somebenchmark_but_shouldnt[0] = True

  def benchmark1(self):
    _ran_somebenchmark_1[0] = True

  def benchmark2(self):
    _ran_somebenchmark_2[0] = True


class TestReportingBenchmark(test.Benchmark):
  """This benchmark (maybe) reports some stuff."""

  def benchmarkReport1(self):
    self.report_benchmark(iters=1)

  def benchmarkReport2(self):
    self.report_benchmark(
        iters=2,
        name="custom_benchmark_name",
        extras={"number_key": 3,
                "other_key": "string"})

  def benchmark_times_an_op(self):
    with session.Session() as sess:
      a = constant_op.constant(0.0)
      a_plus_a = a + a
      self.run_op_benchmark(
          sess, a_plus_a, min_iters=1000, store_trace=True, name="op_benchmark")


class BenchmarkTest(test.TestCase):

  def testGlobalBenchmarkRegistry(self):
    registry = list(benchmark.GLOBAL_BENCHMARK_REGISTRY)
    self.assertEqual(len(registry), 2)
    self.assertTrue(SomeRandomBenchmark in registry)
    self.assertTrue(TestReportingBenchmark in registry)

  def testRunSomeRandomBenchmark(self):
    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run other benchmarks, but this wont run the one we care about
    benchmark._run_benchmarks("unrelated")

    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run all the benchmarks, avoid generating any reports
    if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
      del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
    benchmark._run_benchmarks("SomeRandom")

    # Validate that SomeRandomBenchmark ran correctly
    self.assertTrue(_ran_somebenchmark_1[0])
    self.assertTrue(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    _ran_somebenchmark_1[0] = False
    _ran_somebenchmark_2[0] = False
    _ran_somebenchmark_but_shouldnt[0] = False

    # Test running a specific method of SomeRandomBenchmark
    if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
      del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
    benchmark._run_benchmarks("SomeRandom.*1$")

    self.assertTrue(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

  def testReportingBenchmark(self):
    tempdir = test.get_temp_dir()
    try:
      gfile.MakeDirs(tempdir)
    except OSError as e:
      # It's OK if the directory already exists.
      if " exists:" not in str(e):
        raise e

    prefix = os.path.join(tempdir,
                          "reporting_bench_%016x_" % random.getrandbits(64))
    expected_output_file = "%s%s" % (prefix,
                                     "TestReportingBenchmark.benchmarkReport1")
    expected_output_file_2 = "%s%s" % (
        prefix, "TestReportingBenchmark.custom_benchmark_name")
    expected_output_file_3 = "%s%s" % (prefix,
                                       "TestReportingBenchmark.op_benchmark")
    try:
      self.assertFalse(gfile.Exists(expected_output_file))
      # Run benchmark but without env, shouldn't write anything
      if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
        del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should run without writing anything
      self.assertFalse(gfile.Exists(expected_output_file))

      # Runbenchmark with env, should write
      os.environ[benchmark.TEST_REPORTER_TEST_ENV] = prefix

      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should write
      reporting.benchmarkReport2()  # This should write
      reporting.benchmark_times_an_op()  # This should write

      # Check the files were written
      self.assertTrue(gfile.Exists(expected_output_file))
      self.assertTrue(gfile.Exists(expected_output_file_2))
      self.assertTrue(gfile.Exists(expected_output_file_3))

      # Check the contents are correct
      expected_1 = test_log_pb2.BenchmarkEntry()
      expected_1.name = "TestReportingBenchmark.benchmarkReport1"
      expected_1.iters = 1

      expected_2 = test_log_pb2.BenchmarkEntry()
      expected_2.name = "TestReportingBenchmark.custom_benchmark_name"
      expected_2.iters = 2
      expected_2.extras["number_key"].double_value = 3
      expected_2.extras["other_key"].string_value = "string"

      expected_3 = test_log_pb2.BenchmarkEntry()
      expected_3.name = "TestReportingBenchmark.op_benchmark"
      expected_3.iters = 1000

      def read_benchmark_entry(f):
        s = gfile.GFile(f, "rb").read()
        entries = test_log_pb2.BenchmarkEntries.FromString(s)
        self.assertEquals(1, len(entries.entry))
        return entries.entry[0]

      read_benchmark_1 = read_benchmark_entry(expected_output_file)
      self.assertProtoEquals(expected_1, read_benchmark_1)

      read_benchmark_2 = read_benchmark_entry(expected_output_file_2)
      self.assertProtoEquals(expected_2, read_benchmark_2)

      read_benchmark_3 = read_benchmark_entry(expected_output_file_3)
      self.assertEquals(expected_3.name, read_benchmark_3.name)
      self.assertEquals(expected_3.iters, read_benchmark_3.iters)
      self.assertGreater(read_benchmark_3.wall_time, 0)
      full_trace = read_benchmark_3.extras["full_trace_chrome_format"]
      json_trace = json.loads(full_trace.string_value)
      self.assertTrue(isinstance(json_trace, dict))
      self.assertTrue("traceEvents" in json_trace.keys())
      allocator_keys = [k for k in read_benchmark_3.extras.keys()
                        if k.startswith("allocator_maximum_num_bytes_")]
      self.assertGreater(len(allocator_keys), 0)
      for k in allocator_keys:
        self.assertGreater(read_benchmark_3.extras[k].double_value, 0)

    finally:
      gfile.DeleteRecursively(tempdir)


if __name__ == "__main__":
  test.main()
