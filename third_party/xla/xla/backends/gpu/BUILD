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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_memory",
        "//xla/backends/gpu/runtime:collective_memory_requests",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "@com_google_absl//absl/base:core_headers",
    ],
)
