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

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Point both runtimes to the same python binary to ensure we always
# use the python binary specified by ./configure.py script.
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")

py_runtime(
    name = "py3_runtime",
    interpreter = "%{PYTHON_INTERPRETER}",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
    target_compatible_with = [%{PLATFORM_CONSTRAINT}],
    exec_compatible_with = [%{PLATFORM_CONSTRAINT}],
)

alias(name = "python_headers",
      actual = "@rules_python//python/cc:current_py_cc_headers")

# This alias is exists for the use of targets in the @llvm-project dependency,
# which expect a python_headers target called @python_runtime//:headers. We use
# a repo_mapping to alias python_runtime to this package, and an alias to create
# the correct target.
alias(
    name = "headers",
    actual = ":python_headers",
)


config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)