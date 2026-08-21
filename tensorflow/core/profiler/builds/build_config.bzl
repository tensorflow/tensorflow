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

"""Provides a redirection point for platform specific implementations of Starlark utilities."""

load(
    "@xla//xla/tsl/profiler/builds:build_config.bzl",
    _if_profiler_oss = "if_profiler_oss",
    _tf_profiler_alias = "tf_profiler_alias",
    _tf_profiler_copts = "tf_profiler_copts",
    _tf_profiler_pybind_cc_library_wrapper = "tf_profiler_pybind_cc_library_wrapper",
)

tf_profiler_alias = _tf_profiler_alias
tf_profiler_pybind_cc_library_wrapper = _tf_profiler_pybind_cc_library_wrapper
tf_profiler_copts = _tf_profiler_copts
if_profiler_oss = _if_profiler_oss
