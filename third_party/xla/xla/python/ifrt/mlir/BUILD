load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/python/ifrt:internal",
    ]),
)

cc_library(
    name = "fingerprint_utils",
    srcs = ["fingerprint_utils.cc"],
    hdrs = ["fingerprint_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:status_macros",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@highwayhash",
        "@highwayhash//:arch_specific",
        "@highwayhash//:hh_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "fingerprint_utils_test",
    srcs = ["fingerprint_utils_test.cc"],
    deps = [
        ":fingerprint_utils",
        "//xla/pjrt:mlir_to_hlo",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)
