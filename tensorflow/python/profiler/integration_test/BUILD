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
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "mnist_testing_utils",
    srcs = ["mnist_testing_utils.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:extra_py_tests_deps",
    ],
)

cuda_py_strict_test(
    name = "profiler_api_test",
    srcs = ["profiler_api_test.py"],
    tags = [
        "no_oss",  # TODO(b/283175845)
        "no_pip",
        "no_windows",  # TODO(b/192257727)
    ],
    deps = [
        ":mnist_testing_utils",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "@pypi//portpicker",
    ],
)
