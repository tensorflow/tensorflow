load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_hexagon_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "dlpack_types",
    srcs = ["dlpack_types.cc"],
    hdrs = ["dlpack_types.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@dlpack",
    ],
)

cc_library(
    name = "strides",
    srcs = ["strides.cc"],
    hdrs = ["strides.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "embedded_constant_buffers",
    srcs = ["embedded_constant_buffers.cc"],
    hdrs = ["embedded_constant_buffers.h"],
    # usage of llvm_type_conversion_util
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:util",
        "//xla/backends/cpu:alignment",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
    ],
)

xla_cc_test(
    name = "embedded_constant_buffers_test",
    srcs = ["embedded_constant_buffers_test.cc"],
    deps = [
        ":embedded_constant_buffers",
        "//xla/service/cpu:test_header_helper",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ] + if_llvm_aarch64_available([
        "@llvm-project//llvm:AArch64AsmParser",
        "@llvm-project//llvm:AArch64CodeGen",
    ]) + if_llvm_aarch32_available([
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
    ]) + if_llvm_hexagon_available([
        "@llvm-project//llvm:HexagonAsmParser",
        "@llvm-project//llvm:HexagonCodeGen",
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
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/tsl:package_groups_bzl",
        "//xla/tsl/platform:rules_cc_bzl",
    ],
)
