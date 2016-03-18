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

"""Tests for tensorflow.python.framework.importer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import benchmark


# Used by SomeRandomBenchmark class below.
_ran_somebenchmark_1 = [False]
_ran_somebenchmark_2 = [False]
_ran_somebenchmark_but_shouldnt = [False]


class SomeRandomBenchmark(tf.test.Benchmark):
  """This Benchmark should automatically be registered in the registry."""

  def _dontRunThisBenchmark(self):
    _ran_somebenchmark_but_shouldnt[0] = True

  def notBenchmarkMethod(self):
    _ran_somebenchmark_but_shouldnt[0] = True

  def benchmark1(self):
    _ran_somebenchmark_1[0] = True

  def benchmark2(self):
    _ran_somebenchmark_2[0] = True


class TestReportingBenchmark(tf.test.Benchmark):
  """This benchmark (maybe) reports some stuff."""

  def benchmarkReport1(self):
    self.report_benchmark(iters=1)

  def benchmarkReport2(self):
    self.report_benchmark(
        iters=2, name="custom_benchmark_name",
        extras={"number_key": 3, "other_key": "string"})


class BenchmarkTest(tf.test.TestCase):

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

    # Don't run any benchmarks.
    (exit_early, args, kwargs) = benchmark.run_benchmarks(
        ["--flag1", "--flag3"], {"a": 3})

    self.assertEqual(exit_early, False)
    self.assertEqual(args, ["--flag1", "--flag3"])
    self.assertEqual(kwargs, {"a": 3})

    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run other benchmarks, but this wont run the one we care about
    (exit_early, args, kwargs) = benchmark.run_benchmarks(
        ["--flag1", "--benchmarks=__unrelated__", "--flag3"], {"a": 3})

    self.assertEqual(exit_early, True)
    self.assertEqual(args, ["--flag1", "--flag3"])
    self.assertEqual(kwargs, {"a": 3})

    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run all the benchmarks, avoid generating any reports
    if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
      del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
    (exit_early, args, kwargs) = benchmark.run_benchmarks(
        ["--flag1", "--benchmarks=.", "--flag3"], {"a": 3})

    # Validate the output of run_benchmarks
    self.assertEqual(exit_early, True)
    self.assertEqual(args, ["--flag1", "--flag3"])
    self.assertEqual(kwargs, {"a": 3})

    # Validate that SomeRandomBenchmark ran correctly
    self.assertTrue(_ran_somebenchmark_1[0])
    self.assertTrue(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

  def testReportingBenchmark(self):
    tempdir = tf.test.get_temp_dir()
    try:
      tf.gfile.MakeDirs(tempdir)
    except OSError as e:
      # It's OK if the directory already exists.
      if " exists:" not in str(e):
        raise e

    prefix = os.path.join(
        tempdir, "reporting_bench_%016x_" % random.getrandbits(64))
    expected_output_file = "%s%s" % (
        prefix, "TestReportingBenchmark.benchmarkReport1")
    expected_output_file_2 = "%s%s" % (
        prefix, "TestReportingBenchmark.custom_benchmark_name")
    try:
      self.assertFalse(tf.gfile.Exists(expected_output_file))
      # Run benchmark but without env, shouldn't write anything
      if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
        del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should run without writing anything
      self.assertFalse(tf.gfile.Exists(expected_output_file))

      # Runbenchmark with env, should write
      os.environ[benchmark.TEST_REPORTER_TEST_ENV] = prefix

      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should write
      reporting.benchmarkReport2()  # This should write

      # Check the files were written
      self.assertTrue(tf.gfile.Exists(expected_output_file))
      self.assertTrue(tf.gfile.Exists(expected_output_file_2))

      # Check the contents are correct
      expected_1 = test_log_pb2.BenchmarkEntry()
      expected_1.name = "TestReportingBenchmark.benchmarkReport1"
      expected_1.iters = 1

      expected_2 = test_log_pb2.BenchmarkEntry()
      expected_2.name = "TestReportingBenchmark.custom_benchmark_name"
      expected_2.iters = 2
      expected_2.extras["number_key"].double_value = 3
      expected_2.extras["other_key"].string_value = "string"

      read_benchmark_1 = tf.gfile.GFile(expected_output_file, "r").read()
      read_benchmark_1 = text_format.Merge(
          read_benchmark_1, test_log_pb2.BenchmarkEntry())
      self.assertProtoEquals(expected_1, read_benchmark_1)

      read_benchmark_2 = tf.gfile.GFile(expected_output_file_2, "r").read()
      read_benchmark_2 = text_format.Merge(
          read_benchmark_2, test_log_pb2.BenchmarkEntry())
      self.assertProtoEquals(expected_2, read_benchmark_2)

    finally:
      tf.gfile.DeleteRecursively(tempdir)


if __name__ == "__main__":
  tf.test.main()
