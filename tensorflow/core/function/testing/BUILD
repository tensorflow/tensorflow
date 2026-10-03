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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")

visibility = ["//tensorflow/core/function:__subpackages__"]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
)

licenses(["notice"])

cc_library(
    name = "test_pass_cc",
    testonly = 1,
    hdrs = [
        "test_pass.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:logging",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_python_pybind_extension(
    name = "test_pass_pybind",
    testonly = 1,
    srcs = ["test_pass_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "test_pass_pybind.pyi",
    ],
    deps = [
        ":test_pass_cc",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "test_pass_py",
    testonly = 1,
    srcs = [
        "test_pass.py",
    ],
    visibility = visibility,
    deps = [
        ":test_pass_pybind",
        "//tensorflow/python:pywrap_tensorflow",  # buildcleaner: keep (required for TF pybind)
    ],
)
