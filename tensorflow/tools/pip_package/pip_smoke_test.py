# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""This pip smoke test verifies dependency files exist in the pip package.

This script runs bazel queries to see what python files are required by the
tests and ensures they are in the pip package superset.
"""

import os
import subprocess

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

PIP_PACKAGE_QUERY_EXPRESSION = (
    "deps(//tensorflow/tools/pip_package:build_pip_package)")

# List of file paths containing BUILD files that should not be included for the
# pip smoke test.
BUILD_DENYLIST = [
    "tensorflow/lite",
    "tensorflow/compiler/mlir/lite",
    "tensorflow/compiler/mlir/tfrt",
    "tensorflow/core/runtime_fallback",
    "tensorflow/core/tfrt",
    "tensorflow/python/kernel_tests/signal",
    "tensorflow/examples",
    "tensorflow/tools/android",
    "tensorflow/tools/toolchains",
    "tensorflow/python/autograph/tests",
    "tensorflow/python/eager/benchmarks",
]


def GetBuild(dir_base):
  """Get the list of BUILD file all targets recursively startind at dir_base."""
  items = []
  for root, _, files in os.walk(dir_base):
    for name in files:
      if (name == "BUILD" and not any(x in root for x in BUILD_DENYLIST)):
        items.append("//" + root + ":all")
  return items


def BuildPyTestDependencies():
  python_targets = GetBuild("tensorflow/python")
  tensorflow_targets = GetBuild("tensorflow")
  # Build list of test targets,
  # python - attr(manual|pno_pip)
  targets = " + ".join(python_targets)
  targets += ' - attr(tags, "manual|no_pip", %s)' % " + ".join(
      tensorflow_targets)
  query_kind = "kind(py_test, %s)" % targets
  # Skip benchmarks etc.
  query_filter = 'filter("^((?!benchmark).)*$", %s)' % query_kind
  # Get the dependencies
  query_deps = "deps(%s, 1)" % query_filter

  return python_targets, query_deps


PYTHON_TARGETS, PY_TEST_QUERY_EXPRESSION = BuildPyTestDependencies()

# TODO(amitpatankar): Clean up denylist.
# List of dependencies that should not included in the pip package.
DEPENDENCY_DENYLIST = [
    "//tensorflow/cc/saved_model:saved_model_test_files",
    "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    "//tensorflow:no_tensorflow_py_deps",
    "//tensorflow/tools/pip_package:win_pip_package_marker",
    "//tensorflow/core:image_testdata",
    "//tensorflow/core/lib/lmdb:lmdb_testdata",
    "//tensorflow/core/lib/lmdb/testdata:lmdb_testdata",
    "//tensorflow/core/kernels/cloud:bigquery_reader_ops",
    "//tensorflow/python:extra_py_tests_deps",
    "//tensorflow/python:mixed_precision",
    "//tensorflow/python:tf_optimizer",
    "//tensorflow/python:compare_test_proto_py",
    "//tensorflow/python/framework:test_ops_2",
    "//tensorflow/python/framework:test_file_system.so",
    "//tensorflow/python/debug:grpc_tensorflow_server.par",
    "//tensorflow/python/feature_column:vocabulary_testdata",
    "//tensorflow/python/util:nest_test_main_lib",
    # lite
    "//tensorflow/lite/experimental/examples/lstm:rnn_cell",
    "//tensorflow/lite/experimental/examples/lstm:rnn_cell.py",
    "//tensorflow/lite/experimental/examples/lstm:unidirectional_sequence_lstm_test",  # pylint:disable=line-too-long
    "//tensorflow/lite/experimental/examples/lstm:unidirectional_sequence_lstm_test.py",  # pylint:disable=line-too-long
    "//tensorflow/lite/python:interpreter",
    "//tensorflow/lite/python:interpreter_test",
    "//tensorflow/lite/python:interpreter.py",
    "//tensorflow/lite/python:interpreter_test.py",
]


def main():
  """This script runs the pip smoke test.

  Raises:
    RuntimeError: If any dependencies for py_tests exist in subSet

  Prerequisites:
      1. Bazel is installed.
      2. Running in github repo of tensorflow.
      3. Configure has been run.

  """

  # pip_package_dependencies_list is the list of included files in pip packages
  pip_package_dependencies = subprocess.check_output([
      "bazel", "cquery", "--experimental_cc_shared_library",
      PIP_PACKAGE_QUERY_EXPRESSION
  ])
  if isinstance(pip_package_dependencies, bytes):
    pip_package_dependencies = pip_package_dependencies.decode("utf-8")
  pip_package_dependencies_list = pip_package_dependencies.strip().split("\n")
  pip_package_dependencies_list = [
      x.split()[0] for x in pip_package_dependencies_list
  ]
  print("Pip package superset size: %d" % len(pip_package_dependencies_list))

  # tf_py_test_dependencies is the list of dependencies for all python
  # tests in tensorflow
  tf_py_test_dependencies = subprocess.check_output([
      "bazel", "cquery", "--experimental_cc_shared_library",
      PY_TEST_QUERY_EXPRESSION
  ])
  if isinstance(tf_py_test_dependencies, bytes):
    tf_py_test_dependencies = tf_py_test_dependencies.decode("utf-8")
  tf_py_test_dependencies_list = tf_py_test_dependencies.strip().split("\n")
  tf_py_test_dependencies_list = [
      x.split()[0] for x in tf_py_test_dependencies.strip().split("\n")
  ]
  print("Pytest dependency subset size: %d" % len(tf_py_test_dependencies_list))

  missing_dependencies = []
  # File extensions and endings to ignore
  ignore_extensions = [
      "_test", "_test.py", "_test_cpu", "_test_cpu.py", "_test_gpu",
      "_test_gpu.py", "_test_lib"
  ]

  ignored_files_count = 0
  denylisted_dependencies_count = len(DEPENDENCY_DENYLIST)
  # Compare dependencies
  for dependency in tf_py_test_dependencies_list:
    if dependency and dependency.startswith("//tensorflow"):
      ignore = False
      # Ignore extensions
      if any(dependency.endswith(ext) for ext in ignore_extensions):
        ignore = True
        ignored_files_count += 1

      # Check if the dependency is in the pip package, the dependency denylist,
      # or should be ignored because of its file extension.
      if not (ignore or dependency in pip_package_dependencies_list or
              dependency in DEPENDENCY_DENYLIST):
        missing_dependencies.append(dependency)

  print("Ignored files count: %d" % ignored_files_count)
  print("Denylisted dependencies count: %d" % denylisted_dependencies_count)
  if missing_dependencies:
    print("Missing the following dependencies from pip_packages:")
    for missing_dependency in missing_dependencies:
      print("\nMissing dependency: %s " % missing_dependency)
      print("Affected Tests:")
      rdep_query = ("rdeps(kind(py_test, %s), %s)" %
                    (" + ".join(PYTHON_TARGETS), missing_dependency))
      affected_tests = subprocess.check_output(
          ["bazel", "cquery", "--experimental_cc_shared_library", rdep_query])
      affected_tests_list = affected_tests.split("\n")[:-2]
      print("\n".join(affected_tests_list))

    raise RuntimeError("""
    One or more added test dependencies are not in the pip package.
If these test dependencies need to be in TensorFlow pip package, please add them to //tensorflow/tools/pip_package/BUILD.
Else add no_pip tag to the test.""")

  else:
    print("TEST PASSED")


if __name__ == "__main__":
  main()
