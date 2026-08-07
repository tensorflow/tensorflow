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
load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
)
load("//xla/tests:build_defs.bzl", "xla_test")

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
    name = "xla_ffi_aot_custom_call",
    testonly = True,
    srcs = ["xla_ffi_aot_custom_call.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla:status_macros",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi:type_registry",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:typed_kernel_factory",
        "//xla/stream_executor/cuda:cudart_kernel_registry",
        "//xla/stream_executor/cuda:simple_kernel_cuda",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

xla_cc_binary(
    name = "xla_ffi_aot_custom_call_compile_main",
    testonly = True,
    srcs = ["xla_ffi_aot_custom_call_compile_main.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_ffi_aot_custom_call",
        "//xla/backends/gpu/target_config",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/pjrt:pjrt_compiler",
        "//xla/service:compiled_module",
        "//xla/service:compiler",
        "//xla/service:gpu_topology",
        "//xla/service/gpu:nvptx_compiler",
        "//xla/service/gpu:nvptx_compiler_impl",  # buildcleaner: keep
        "//xla/stream_executor/cuda:all_runtime",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/cuda:gpu_test_kernels_cuda",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

genrule(
    name = "xla_ffi_aot_custom_call_executable",
    testonly = True,
    outs = ["xla_ffi_aot_custom_call_executable_h100"],
    cmd = "$(location :xla_ffi_aot_custom_call_compile_main) $@",
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [":xla_ffi_aot_custom_call_compile_main"],
)

xla_test(
    name = "xla_ffi_aot_custom_call_test",
    srcs = ["xla_ffi_aot_custom_call_test.cc"],
    backends = [
        "h100",
    ],
    data = [
        ":xla_ffi_aot_custom_call_executable",
    ],
    tags = [
        "cuda-only",
    ],
    deps = [
        ":xla_ffi_aot_custom_call",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/service:compiled_module",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:service_executable_run_options",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:resource_loader",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
