load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "symbolic_tile",
    srcs = [
        "symbolic_tile.cc",
        "symbolic_tile_propagation.cc",
    ],
    hdrs = [
        "symbolic_tile.h",
        "symbolic_tile_propagation.h",
    ],
    deps = [
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu/model:constraint_expression",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "symbolic_tile_propagation_test",
    srcs = ["symbolic_tile_propagation_test.cc"],
    deps = [
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "symbolic_tile_test",
    srcs = ["symbolic_tile_test.cc"],
    deps = [
        ":symbolic_tile",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)
