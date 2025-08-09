# Description:
#   XLA builder libraries.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/lib/math:math_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    deps = [
        ":padding",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "sharding_builder",
    srcs = ["sharding_builder.cc"],
    hdrs = ["sharding_builder.h"],
    deps = [
        "//xla:array",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "xla_computation",
    srcs = ["xla_computation.cc"],
    hdrs = ["xla_computation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "value_inference",
    srcs = ["value_inference.cc"],
    hdrs = ["value_inference.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_builder",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/gtl:value_or_die",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "value_inference_test",
    srcs = ["value_inference_test.cc"],
    deps = [
        ":value_inference",
        ":xla_builder",
        ":xla_computation",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:prng",
        "//xla/hlo/testlib:test",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_builder",
    srcs = ["xla_builder.cc"],
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":padding",
        ":sharding_builder",
        ":xla_computation",
        "//xla:array",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:array4d",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:sharding_op_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/core:bitmap",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "xla_builder_test",
    srcs = ["xla_builder_test.cc"],
    deps = [
        ":padding",
        ":sharding_builder",
        ":value_inference",
        ":xla_builder",
        ":xla_computation",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:hlo_proto_cc",
        "//xla/service:pattern_matcher",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)
