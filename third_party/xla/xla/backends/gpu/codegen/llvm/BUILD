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

# LLVM codegen for GPU.
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

cc_library(
    name = "llvm_emitter",
    srcs = ["llvm_emitter.cc"],
    hdrs = ["llvm_emitter.h"],
    deps = [
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:gpu_kernel_spec",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:llvm_kernel_source",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:dense_data_intermediate",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)
