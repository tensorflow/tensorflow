# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

# Description:
#   Tensorflow builder compatibility checker.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/tools/tensorflow_builder:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "compat_checker",
    srcs = ["compat_checker.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_test(
    name = "compat_checker_test",
    srcs = ["compat_checker_test.py"],
    data = [
        ":test_config",
    ],
    strict_deps = True,
    tags = ["no_pip"],
    deps = [
        ":compat_checker",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/platform:test",
    ],
)

filegroup(
    name = "test_config",
    srcs = ["test_config.ini"],
)
