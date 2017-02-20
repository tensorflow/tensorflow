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
"""Library for getting system information during TensorFlow tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shlex
import subprocess
import tempfile
import time

from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import gfile
from tensorflow.tools.test import system_info_lib


class MissingLogsError(Exception):
  pass


def get_git_commit_sha():
  """Get git commit SHA for this build.

  Attempt to get the SHA from environment variable GIT_COMMIT, which should
  be available on Jenkins build agents.

  Returns:
    SHA hash of the git commit used for the build, if available
  """

  return os.getenv("GIT_COMMIT")


def process_test_logs(name, test_name, test_args, start_time, run_time,
                      log_files):
  """Gather test information and put it in a TestResults proto.

  Args:
    name: Benchmark target identifier.
    test_name:  A unique bazel target, e.g. "//path/to:test"
    test_args:  A string containing all arguments to run the target with.

    start_time: Test starting time (epoch)
    run_time:   Wall time that the test ran for
    log_files:  Paths to the log files

  Returns:
    A TestResults proto
  """

  results = test_log_pb2.TestResults()
  results.name = name
  results.target = test_name
  results.start_time = start_time
  results.run_time = run_time

  # Gather source code information
  git_sha = get_git_commit_sha()
  if git_sha:
    results.commit_id.hash = git_sha

  results.entries.CopyFrom(process_benchmarks(log_files))
  results.run_configuration.argument.extend(test_args)
  results.machine_configuration.CopyFrom(
      system_info_lib.gather_machine_configuration())
  return results


def process_benchmarks(log_files):
  benchmarks = test_log_pb2.BenchmarkEntries()
  for f in log_files:
    content = gfile.GFile(f, "rb").read()
    if benchmarks.MergeFromString(content) != len(content):
      raise Exception("Failed parsing benchmark entry from %s" % f)
  return benchmarks


def run_and_gather_logs(name, test_name, test_args):
  """Run the bazel test given by test_name.  Gather and return the logs.

  Args:
    name: Benchmark target identifier.
    test_name: A unique bazel target, e.g. "//path/to:test"
    test_args: A string containing all arguments to run the target with.

  Returns:
    A tuple (test_results, mangled_test_name), where
    test_results: A test_log_pb2.TestResults proto
    mangled_test_name: A string, the mangled test name.

  Raises:
    ValueError: If the test_name is not a valid target.
    subprocess.CalledProcessError: If the target itself fails.
    IOError: If there are problems gathering test log output from the test.
    MissingLogsError: If we couldn't find benchmark logs.
  """
  if not (test_name and test_name.startswith("//") and ".." not in test_name and
          not test_name.endswith(":") and not test_name.endswith(":all") and
          not test_name.endswith("...") and len(test_name.split(":")) == 2):
    raise ValueError("Expected test_name parameter with a unique test, e.g.: "
                     "--test_name=//path/to:test")
  test_executable = test_name.rstrip().strip("/").replace(":", "/")

  if gfile.Exists(os.path.join("bazel-bin", test_executable)):
    # Running in standalone mode from core of the repository
    test_executable = os.path.join("bazel-bin", test_executable)
  else:
    # Hopefully running in sandboxed mode
    test_executable = os.path.join(".", test_executable)

  temp_directory = tempfile.mkdtemp(prefix="run_and_gather_logs")
  mangled_test_name = test_name.strip("/").replace("/", "_").replace(":", "_")
  test_file_prefix = os.path.join(temp_directory, mangled_test_name)
  test_file_prefix = "%s." % test_file_prefix

  try:
    if not gfile.Exists(test_executable):
      raise ValueError("Executable does not exist: %s" % test_executable)
    test_args = shlex.split(test_args)

    # This key is defined in tf/core/util/reporter.h as
    # TestReporter::kTestReporterEnv.
    os.environ["TEST_REPORT_FILE_PREFIX"] = test_file_prefix
    start_time = time.time()
    subprocess.check_call([test_executable] + test_args)
    run_time = time.time() - start_time
    log_files = gfile.Glob("{}*".format(test_file_prefix))
    if not log_files:
      raise MissingLogsError("No log files found at %s." % test_file_prefix)

    return (process_test_logs(
        name,
        test_name,
        test_args,
        start_time=int(start_time),
        run_time=run_time,
        log_files=log_files), mangled_test_name)

  finally:
    try:
      gfile.DeleteRecursively(temp_directory)
    except OSError:
      pass
