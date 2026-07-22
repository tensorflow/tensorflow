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

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_binary(
    name = "micro_benchmarks",
    srcs = ["micro_benchmarks.py"],
    strict_deps = True,
    deps = [
        ":numpy_mlp",
        ":tf_numpy_mlp",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python:extra_py_tests_deps",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "numpy_mlp",
    srcs = ["numpy_mlp.py"],
    strict_deps = True,
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tf_numpy_mlp",
    srcs = ["tf_numpy_mlp.py"],
    strict_deps = True,
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)
