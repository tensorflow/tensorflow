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

# Description:
#   Default implementations for llvm_gpu_backend

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/gpu/llvm_gpu_backend:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "nvptx_libdevice_path",
    srcs = ["nvptx_libdevice_path.cc"],
    deps = [
        "//xla/service/gpu/llvm_gpu_backend:nvptx_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:cuda_root_path",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
    ],
    alwayslink = True,
)

cc_library(
    name = "load_ir_module",
    srcs = ["load_ir_module.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@tsl//tsl/platform:logging",
    ],
    alwayslink = True,
)
