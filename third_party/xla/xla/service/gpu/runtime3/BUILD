load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

#===-------------------------------------------------------------------------------------------===//
# Command Buffer Integration
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "command_buffer_cmd",
    srcs = ["command_buffer_cmd.cc"],
    hdrs = ["command_buffer_cmd.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/stream_executor",
        "@com_google_absl//absl/log",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "command_buffer_cmd_test",
    srcs = ["command_buffer_cmd_test.cc"],
    backends = ["gpu"],
    deps = [
        ":command_buffer_cmd",
        "//xla:types",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:buffer_allocations",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===-------------------------------------------------------------------------------------------===//
# XLA Thunks Runtime
#===-------------------------------------------------------------------------------------------===//

cc_library(
    name = "cholesky_thunk",
    srcs = if_gpu_is_configured(["cholesky_thunk.cc"]),
    hdrs = if_gpu_is_configured(["cholesky_thunk.h"]),
    visibility = ["//visibility:public"],
    deps = if_gpu_is_configured([
        "//xla/service/gpu:buffer_allocations",
        "//xla/service/gpu:cusolver_context",
        "//xla/service/gpu:precompiled_kernels",
        "//xla/service/gpu:thunk",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:buffer_assignment",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:logging",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor/gpu:gpu_asm_opts",
    ]) + ["@local_tsl//tsl/platform:status"],
)
