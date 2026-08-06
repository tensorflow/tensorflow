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

"""Default (OSS) build versions of TSL general-purpose build extensions."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load(
    "//xla/tsl:tsl.bzl",
    _filegroup = "filegroup",
    _get_compatible_with_libtpu_portable = "get_compatible_with_libtpu_portable",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_include_google_deps = "if_include_google_deps",
    _if_not_mobile_or_arm_or_macos_or_lgpl_restricted = "if_not_mobile_or_arm_or_macos_or_lgpl_restricted",
    _tsl_extra_config_settings = "tsl_extra_config_settings",
    _tsl_extra_config_settings_targets = "tsl_extra_config_settings_targets",
    _tsl_google_bzl_deps = "tsl_google_bzl_deps",
    _tsl_pybind_extension = "tsl_pybind_extension",
)

visibility(DEFAULT_LOAD_VISIBILITY)

get_compatible_with_portable = _get_compatible_with_portable
get_compatible_with_libtpu_portable = _get_compatible_with_libtpu_portable
if_include_google_deps = _if_include_google_deps
filegroup = _filegroup
if_not_mobile_or_arm_or_macos_or_lgpl_restricted = _if_not_mobile_or_arm_or_macos_or_lgpl_restricted
tsl_pybind_extension = _tsl_pybind_extension
tsl_google_bzl_deps = _tsl_google_bzl_deps
tsl_extra_config_settings = _tsl_extra_config_settings
tsl_extra_config_settings_targets = _tsl_extra_config_settings_targets

# These configs are used to determine whether we should use CUDA/NVSHMEM tools and libs in
# cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hermetic CUDA tools."""
    return select({
        "@local_config_cuda//cuda:cuda_tools": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({
        "@local_config_cuda//cuda:cuda_tools_and_libs": if_true,
        "//conditions:default": if_false,
    })
