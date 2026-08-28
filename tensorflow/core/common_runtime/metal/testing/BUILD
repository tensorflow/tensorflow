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
#
# The op sweep: every op the Metal backend registers, called through
# TensorFlow's own dispatch and compared against the CPU kernel. It needs a
# machine with an Apple silicon GPU, so it is a binary rather than a test:
# nothing in the ordinary CI matrix can run it.

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "op_recipes",
    srcs = ["op_recipes.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python:tensorflow_py"],
)

py_binary(
    name = "op_sweep",
    srcs = ["op_sweep.py"],
    data = ["registered_ops.txt"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_recipes",
        "//tensorflow/python:tensorflow_py",
    ],
)
