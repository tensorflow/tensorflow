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

import os
import subprocess

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

PIP_PACKAGE_QUERY_EXPRESSION = (
    "deps(//tensorflow/tools/pip_package:build_pip_package)")

# pylint: disable=g-backslash-continuation
PY_TEST_QUERY_EXPRESSION = 'deps(\
  filter("^((?!benchmark).)*$",\
  kind(py_test,\
  //tensorflow/python/... \
  + //tensorflow/contrib/... \
  - //tensorflow/contrib/tensorboard/... \
  - attr(tags, "manual|no_pip", //tensorflow/...))), 1)'
# pylint: enable=g-backslash-continuation

# Hard-coded blacklist of files if not included in pip package
# TODO(amitpatankar): Clean up blacklist.
BLACKLIST = [
    "//tensorflow/python:extra_py_tests_deps",
    "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    "//tensorflow:no_tensorflow_py_deps",
    "//tensorflow/tools/pip_package:win_pip_package_marker",
    "//tensorflow/python:test_ops_2",
    "//tensorflow/python:tf_optimizer",
    "//tensorflow/python:compare_test_proto_py",
    "//tensorflow/core:image_testdata",
    "//tensorflow/core:lmdb_testdata",
    "//tensorflow/core/kernels/cloud:bigquery_reader_ops",
    "//tensorflow/python/feature_column:vocabulary_testdata",
    "//tensorflow/python:framework/test_file_system.so",
    # contrib
    "//tensorflow/contrib/session_bundle:session_bundle_half_plus_two",
    "//tensorflow/contrib/keras:testing_utils",
    "//tensorflow/contrib/lite/python:interpreter",
    "//tensorflow/contrib/lite/python:interpreter_test",
    "//tensorflow/contrib/lite/python:interpreter.py",
    "//tensorflow/contrib/lite/python:interpreter_test.py",
    "//tensorflow/contrib/ffmpeg:test_data",
    "//tensorflow/contrib/factorization/examples:mnist",
    "//tensorflow/contrib/factorization/examples:mnist.py",
    "//tensorflow/contrib/factorization:factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",  # pylint:disable=line-too-long
    "//tensorflow/contrib/framework:checkpoint_ops_testdata",
    "//tensorflow/contrib/bayesflow:reinforce_simple_example",
    "//tensorflow/contrib/bayesflow:examples/reinforce_simple/reinforce_simple_example.py",  # pylint:disable=line-too-long
    "//tensorflow/contrib/timeseries/examples:predict",
    "//tensorflow/contrib/timeseries/examples:multivariate",
    "//tensorflow/contrib/timeseries/examples:known_anomaly",
    "//tensorflow/contrib/timeseries/examples:data/period_trend.csv",  # pylint:disable=line-too-long
    "//tensorflow/contrib/timeseries/python/timeseries:test_utils",
    "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:test_utils",  # pylint:disable=line-too-long
    "//tensorflow/contrib/image:sparse_image_warp_test_data",
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
      ["bazel", "query", PIP_PACKAGE_QUERY_EXPRESSION])
  pip_package_dependencies_list = pip_package_dependencies.strip().split("\n")
  print("Pip package superset size: %d" % len(pip_package_dependencies_list))

  # tf_py_test_dependencies is the list of dependencies for all python
  # tests in tensorflow
  tf_py_test_dependencies = subprocess.check_output(
      ["bazel", "query", PY_TEST_QUERY_EXPRESSION])
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
      if not (ignore or dependency in pip_package_dependencies_list or
              dependency in BLACKLIST):
        missing_dependencies.append(dependency)

  print("Ignored files: %d" % ignored_files)
  print("Blacklisted files: %d" % blacklisted_files)
  if missing_dependencies:
    print("Missing the following dependencies from pip_packages:")
    for missing_dependency in missing_dependencies:
      print("\nMissing dependency: %s " % missing_dependency)
      print("Affected Tests:")
      rdep_query = ("rdeps(kind(py_test, //tensorflow/python/...), %s)" %
                    missing_dependency)
      affected_tests = subprocess.check_output(["bazel", "query", rdep_query])
      affected_tests_list = affected_tests.split("\n")[:-2]
      print("\n".join(affected_tests_list))

    raise RuntimeError("""One or more dependencies are not in the pip package.
Please either blacklist the dependencies in
//tensorflow/tools/pip_package/pip_smoke_test.py
or add them to //tensorflow/tools/pip_package/BUILD.""")

  else:
    print("TEST PASSED")


if __name__ == "__main__":
  main()
