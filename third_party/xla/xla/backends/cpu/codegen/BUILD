load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "contiguous_section_memory_manager",
    srcs = ["contiguous_section_memory_manager.cc"],
    hdrs = ["contiguous_section_memory_manager.h"],
    deps = [
        "//xla:util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "function_library",
    hdrs = ["function_library.h"],
    deps = [
        "//xla:util",
        "//xla/stream_executor/host:host_kernel_c_api",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    deps = [
        ":function_library",
        "//xla:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
    ],
)

xla_cc_test(
    name = "jit_compiler_test",
    srcs = ["jit_compiler_test.cc"],
    deps = [
        ":jit_compiler",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
