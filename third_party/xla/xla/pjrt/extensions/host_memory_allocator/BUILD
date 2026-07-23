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

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "host_memory_allocator_extension",
    hdrs = ["host_memory_allocator_extension.h"],
    deps = [
        "//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "host_memory_allocator_extension_internal",
    srcs = ["host_memory_allocator_extension_internal.cc"],
    hdrs = ["host_memory_allocator_extension_internal.h"],
    deps = [
        ":host_memory_allocator_extension",
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "host_memory_allocator_interface_impl",
    srcs = ["host_memory_allocator_interface_impl.cc"],
    hdrs = ["host_memory_allocator_interface_impl.h"],
    deps = [
        ":host_memory_allocator_extension",
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_status_utils",
    ],
)
