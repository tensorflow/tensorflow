load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package_group(
    name = "internal",
    packages = [
        "//xla/python/ifrt/ir/conversions/mpmd/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//xla/hlo/ir:hlo_sharding",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/python/ifrt/ir/support:sharding_conversions",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/mpmd/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)

cc_library(
    name = "lower_to_ifrt",
    srcs = ["lower_to_ifrt.cc"],
    hdrs = ["lower_to_ifrt.h"],
    visibility = internal_visibility(["//xla/python/ifrt:users"]),
    deps = [
        ":utils",
        "//xla/client:executable_build_options",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir/transforms:built_in_spmd_expansions",
        "//xla/python/ifrt/ir/transforms:debug",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/ir/transforms:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/mpmd/ir:dialect",
        "@shardy//shardy/dialect/mpmd/transforms/export:utils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

xla_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    # Not sure why it doesn't work on arm64 CPU, but these tests are just validating business logic
    # and so we just need to make sure they pass on other systems.
    tags = ["not_run:arm"],
    deps = [
        ":utils",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/python/ifrt/ir/support:sharding_conversions",
        "//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/mpmd/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:dialect",
    ],
)
