# Copyright 2026 The OpenXLA Authors.
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
#
# Demonstrator / end-to-end test for the custom-call thunk-folding API. It
# registers a handler for "xla.gpu.test_write_value_thunk_folded" that folds
# the custom call into a CustomKernelThunk, then AOT-compiles and runs a module
# using it. Mirrors backends/gpu/ffi/xla_ffi_aot_custom_call*.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/stream_executor/cuda:build_defs.bzl", "embeddable_cuda_library")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

embeddable_cuda_library(
    name = "write_value_folded_kernel",
    testonly = 1,
    srcs = ["write_value_folded_kernel.cu.cc"],
    hdrs = ["write_value_folded_kernel.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "write_value_thunk_folded_handler",
    testonly = True,
    srcs = ["write_value_thunk_folded_handler.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":write_value_folded_kernel",
        "//xla:shape_util",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_emitter_context",
        "//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_handler_registration",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/ffi:attribute_map",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:kernel_args_packing_spec",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor/cuda:cudart_kernel_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,
)

xla_cc_binary(
    name = "write_value_thunk_folded_compile_main",
    testonly = True,
    srcs = ["write_value_thunk_folded_compile_main.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":write_value_thunk_folded_handler",
        "//xla/backends/gpu/target_config",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/pjrt:pjrt_compiler",
        "//xla/service:compiled_module",
        "//xla/service:compiler",
        "//xla/service:gpu_topology",
        "//xla/service/gpu:nvptx_compiler",
        "//xla/service/gpu:nvptx_compiler_impl",  # buildcleaner: keep
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor/cuda:all_runtime",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/cuda:cuda_platform_id",
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
    name = "write_value_thunk_folded_executable",
    testonly = True,
    outs = ["write_value_thunk_folded_executable_h100"],
    cmd = "$(location :write_value_thunk_folded_compile_main) $@",
    tags = [
        "cuda-only",
        "gpu",
    ],
    tools = [":write_value_thunk_folded_compile_main"],
)

xla_test(
    name = "write_value_thunk_folded_test",
    srcs = ["write_value_thunk_folded_test.cc"],
    backends = [
        "h100",
    ],
    data = [
        ":write_value_thunk_folded_executable",
    ],
    tags = [
        "cuda-only",
    ],
    deps = [
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
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
