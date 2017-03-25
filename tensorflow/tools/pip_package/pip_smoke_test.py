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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

PIP_PACKAGE_QUERY = """bazel query \
  'deps(//tensorflow/tools/pip_package:build_pip_package)'"""

PY_TEST_QUERY = """bazel query 'filter("^((?!(benchmark|manual|no_pip)).)*$", \
  deps(kind(py_test,\
  //tensorflow/python/... + \
  //tensorflow/tensorboard/...), 1))'"""

# Hard-coded blacklist of files if not included in pip package
# TODO(amitpatankar): Clean up blacklist.
BLACKLIST = [
    "//tensorflow/python:extra_py_tests_deps",
    "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    "//tensorflow:no_tensorflow_py_deps",
    "//tensorflow/python:test_ops_2",
    "//tensorflow/python:compare_test_proto_py",
    "//tensorflow/core:image_testdata",
    "//tensorflow/core/kernels/cloud:bigquery_reader_ops",
    "//tensorflow/python:framework/test_file_system.so"
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
  pip_package_dependencies = subprocess.check_output(
      PIP_PACKAGE_QUERY, shell=True)
  pip_package_dependencies_list = pip_package_dependencies.strip().split("\n")
  print("Pip package superset size: %d" % len(pip_package_dependencies_list))

  # tf_py_test_dependencies is the list of dependencies for all python
  # tests in tensorflow
  tf_py_test_dependencies = subprocess.check_output(
      PY_TEST_QUERY, shell=True)
  tf_py_test_dependencies_list = tf_py_test_dependencies.strip().split("\n")
  print("Pytest dependency subset size: %d" % len(tf_py_test_dependencies_list))

  missing_dependencies = []
  # File extensions and endings to ignore
  ignore_extensions = ["_test", "_test.py"]

  ignored_files = 0
  blacklisted_files = len(BLACKLIST)
  # Compare dependencies
  for dependency in tf_py_test_dependencies_list:
    if dependency and dependency.startswith("//tensorflow"):
      ignore = False
      # Ignore extensions
      if any(dependency.endswith(ext) for ext in ignore_extensions):
        ignore = True
        ignored_files += 1

      # Check if the dependency is in the pip package, the blacklist, or
      # should be ignored because of its file extension
      if (ignore or
          dependency in pip_package_dependencies_list or
          dependency in BLACKLIST):
        continue
      else:
        missing_dependencies.append(dependency)

  print("Ignored files: %d" % ignored_files)
  print("Blacklisted files: %d" % blacklisted_files)
  if missing_dependencies:
    print("Missing the following dependencies from pip_packages:")
    for missing_dependency in missing_dependencies:
      print("\nMissing dependency: %s " % missing_dependency)
      print("Affected Tests:")
      rdep_query = """bazel query 'rdeps(kind(py_test, \
      //tensorflow/python/...), %s)'""" % missing_dependency
      affected_tests = subprocess.check_output(rdep_query, shell=True)
      affected_tests_list = affected_tests.split("\n")[:-2]
      print("\n".join(affected_tests_list))

    raise RuntimeError("One or more dependencies are not in the pip package.")

  else:
    print("TEST PASSED")

if __name__ == "__main__":
  main()
