# Description:
#   Implementation of XLA’s HLO utilities used for higher-level transformations.

load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_container_util",
    srcs = [],
    hdrs = ["hlo_container_util.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_live_range",
    srcs = ["hlo_live_range.cc"],
    hdrs = ["hlo_live_range.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "hlo_live_range_test",
    srcs = ["hlo_live_range_test.cc"],
    deps = [
        ":hlo_live_range",
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/analysis:hlo_alias_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_value",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_matchers_test",
    srcs = ["hlo_matchers_test.cc"],
    deps = [
        ":hlo_matchers",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_sharding_util",
    srcs = [
        "hlo_sharding_util.cc",
    ],
    hdrs = [
        "hlo_sharding_util.h",
    ],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":hlo_container_util",
        "//xla:array",
        "//xla:literal_util",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/service:call_graph",
        "//xla/service:dot_as_convolution_util",
        "//xla/service:gather_scatter_utils",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_sharding_util_test",
    srcs = [
        "hlo_sharding_util_test.cc",
    ],
    deps = [
        ":hlo_sharding_util",
        "//xla:array",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:test",
        "//xla/service:dot_as_convolution_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "hlo_query_test",
    srcs = [
        "hlo_query_test.cc",
    ],
    deps = [
        ":hlo_query",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_traversal",
    srcs = ["hlo_traversal.cc"],
    hdrs = ["hlo_traversal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_traversal_test",
    srcs = ["hlo_traversal_test.cc"],
    deps = [
        ":hlo_traversal",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/service:pattern_matcher",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
