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

cc_library(
    name = "ffi_collectives",
    srcs = ["ffi_collectives.cc"],
    hdrs = ["ffi_collectives.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/backends/gpu/runtime:collective_clique_requests",
        "//xla/backends/gpu/runtime:collective_cliques",
        "//xla/backends/gpu/runtime:collective_execution",
        "//xla/backends/gpu/runtime:collective_params",
        "//xla/ffi:ffi_interop",
        "//xla/ffi/api:c_api",
        "//xla/runtime:device_id",
        "//xla/service:collective_ops_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
