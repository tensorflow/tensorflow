load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_binary")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_riscv_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "ir_compiler_opt_main",
    testonly = True,
    srcs = ["ir_compiler_opt_main.cc"],
    deps = [
        "//xla/backends/cpu/codegen:ir_compiler",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:platform_port",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_riscv_available([
        "@llvm-project//llvm:RISCVCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]),
)

xla_cc_binary(
    name = "cpu_test_correctness",
    testonly = True,
    srcs = ["cpu_test_correctness.cc"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla/codegen/tools:test_lib",
        "//xla/service:cpu_plugin",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_for_library",
    ],
)

xla_cc_binary(
    name = "ir-compiler-opt",
    testonly = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        ":ir_compiler_opt_main",
    ],
)

xla_cc_binary(
    name = "fusion_to_mlir",
    testonly = True,
    srcs = ["fusion_to_mlir.cc"],
    deps = [
        "//xla/backends/cpu/codegen:fusion_compiler",
        "//xla/backends/cpu/codegen:fusion_emitter",
        "//xla/codegen/tools:test_lib",
        "//xla/hlo/analysis:symbolic_expr",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:platform_port",
    ],
)
