load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package_group(
    name = "internal",
    packages = [
        "//xla/python/ifrt/ir/tests/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
    ]),
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "ifrt-opt",
    testonly = True,
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:atom_program_compiler",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/support:module_parsing",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "ifrt-translate",
    testonly = True,
    srcs = ["ifrt-translate.cc"],
    deps = [
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:ifrt_ir_program_serdes",  # build_cleaner: keep
        "//xla/python/ifrt/ir:version",
        "//xla/python/ifrt/ir:vifrt",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TranslateLib",
        "@stablehlo//:register",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "executable_impl_test_base",
    testonly = True,
    srcs = ["executable_impl_test_base.cc"],
    hdrs = ["executable_impl_test_base.h"],
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:ifrt_ir_program_serdes",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/ifrt/ir:version",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "executable_impl_test_lib",
    testonly = True,
    srcs = ["executable_impl_test_lib.cc"],
    visibility = ["//xla/python/ifrt:users"],
    deps = [
        ":executable_impl_test_base",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir:atom_program_compiler",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = True,
)

build_test(
    name = "executable_test_no_impl",
    targets = [":executable_impl_test_lib"],
)
