load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "constraint_expression",
    srcs = ["constraint_expression.cc"],
    hdrs = ["constraint_expression.h"],
    deps = [
        "//xla/hlo/analysis:interval",
        "//xla/hlo/analysis:symbolic_map",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "constraint_expression_test",
    srcs = ["constraint_expression_test.cc"],
    deps = [
        ":constraint_expression",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tiling_util",
    srcs = ["tiling_util.cc"],
    hdrs = ["tiling_util.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "tiling_util_test",
    srcs = ["tiling_util_test.cc"],
    deps = [
        ":tiling_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_googletest//:gtest_main",
    ],
)
