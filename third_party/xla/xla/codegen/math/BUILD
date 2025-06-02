load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "ldexp",
    srcs = ["ldexp.cc"],
    hdrs = ["ldexp.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "ldexp_test",
    srcs = ["ldexp_test.cc"],
    deps = [
        ":ldexp",
        ":simple_jit_runner",
        ":test_matchers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "simple_jit_runner",
    testonly = True,
    srcs = ["simple_jit_runner.cc"],
    hdrs = ["simple_jit_runner.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:OrcJIT",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@llvm-project//llvm:PowerPCAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@llvm-project//llvm:SystemZAsmParser",  # fixdeps: keep
        "@llvm-project//llvm:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@llvm-project//llvm:X86AsmParser",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ]),
)

xla_cc_test(
    name = "simple_jit_runner_test",
    srcs = ["simple_jit_runner_test.cc"],
    deps = [
        ":simple_jit_runner",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    srcs = [],
    hdrs = ["test_matchers.h"],
    deps = [
        "//xla:fp_util",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "vec_name_mangler",
    hdrs = ["vec_name_mangler.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "vec_name_mangler_test",
    srcs = ["vec_name_mangler_test.cc"],
    deps = [
        ":vec_name_mangler",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "string_interner",
    hdrs = ["string_interner.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)
