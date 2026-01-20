load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:build_config.bzl", "strict_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "stablehlo_extension_pass_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "stablehlo_canonicalize_from_hlo_import",
    srcs = [
        "stablehlo_canonicalize_from_hlo_import.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    visibility = internal_visibility(["//learning/brain/mlir:stablehlo_friends"]),
    deps = [
        ":stablehlo_extension_pass_inc_gen",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate/mhlo_to_hlo:attribute_exporter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes_optimization",
    ],
)

strict_cc_test(
    name = "passes_test",
    srcs = ["passes_test.cc"],
    deps = [
        ":stablehlo_canonicalize_from_hlo_import",
        ":stablehlo_extension_pass_inc_gen",
        "//xla/hlo/testlib:filecheck",
        "//xla/pjrt:mlir_to_hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
