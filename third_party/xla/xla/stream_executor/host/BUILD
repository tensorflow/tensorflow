# Description:
#   Host-platform specific StreamExecutor support code.

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "host_platform_id",
    srcs = [
        "host_platform_id.cc",
    ],
    hdrs = [
        "host_platform_id.h",
    ],
    deps = [
        "//xla/stream_executor:platform",
    ],
)

cc_library(
    name = "host_platform",
    srcs = [
        "host_platform.cc",
    ],
    hdrs = [
        "host_platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_executor",
        ":host_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:status",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cc_library(
    name = "host_event",
    hdrs = ["host_event.h"],
    deps = [
        "//xla/stream_executor:event",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "host_stream",
    srcs = [
        "host_stream.cc",
    ],
    hdrs = [
        "host_stream.h",
    ],
    deps = [
        ":host_event",
        ":host_kernel",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:denormal",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:setround",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "host_kernel_c_api",
    hdrs = ["host_kernel_c_api.h"],
)

cc_library(
    name = "host_kernel",
    srcs = ["host_kernel.cc"],
    hdrs = ["host_kernel.h"],
    deps = [
        ":host_kernel_c_api",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "host_kernel_test",
    srcs = ["host_kernel_test.cc"],
    deps = [
        ":host_kernel",
        ":host_kernel_c_api",
        ":host_platform",
        ":jit_host_kernel_function",
        ":ptr_host_kernel_function",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "host_executor",
    srcs = [
        "host_executor.cc",
    ],
    hdrs = [
        "host_executor.h",
    ],
    deps = [
        ":host_event",
        ":host_kernel",
        ":host_stream",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:host_memory_allocation",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_common",
        "//xla/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "host_stream_test",
    srcs = ["host_stream_test.cc"],
    deps = [
        ":host_platform",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "ptr_host_kernel_function",
    srcs = ["ptr_host_kernel_function.cc"],
    hdrs = ["ptr_host_kernel_function.h"],
    deps = [
        ":host_executor",
        ":host_kernel",
        ":host_kernel_c_api",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,  # static kernel function loader registration
)

cc_library(
    name = "jit_host_kernel_function",
    srcs = ["jit_host_kernel_function.cc"],
    hdrs = ["jit_host_kernel_function.h"],
    deps = [
        ":host_executor",
        ":host_kernel",
        ":host_kernel_c_api",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor/platform:initialize",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:statusor",
    ] + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
    ]) + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",
        "@llvm-project//llvm:PowerPCCodeGen",
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",
        "@llvm-project//llvm:SystemZCodeGen",
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
    ]),
    alwayslink = 1,  # static kernel function loader registration
)
