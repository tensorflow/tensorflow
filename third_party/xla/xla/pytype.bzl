# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

"""Default (OSS) build versions of Python pytype rules."""

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

# Placeholder to use until bazel supports pytype_library.
def pytype_library(name, pytype_deps = [], pytype_srcs = [], **kwargs):
    _ = (pytype_deps, pytype_srcs)  # @unused
    py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, pytype_deps = [], pytype_srcs = [], **kwargs):
    _ = (pytype_deps, pytype_srcs)  # @unused
    py_library(name = name, **kwargs)
