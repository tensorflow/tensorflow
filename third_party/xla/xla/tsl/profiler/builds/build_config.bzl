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

"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PROFILER_BUILDS_BUILD_CONFIG_USERS",
)
load("//xla/tsl:tsl.bzl", "clean_dep")
load(
    "//xla/tsl/profiler/builds/oss:build_config.bzl",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PROFILER_BUILDS_BUILD_CONFIG_USERS)

tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_alias = _tf_profiler_alias

def tf_profiler_copts():
    return []

def if_profiler_oss(if_true, if_false = []):
    return select({
        clean_dep("//xla/tsl/profiler/builds:profiler_build_oss"): if_true,
        "//conditions:default": if_false,
    })
