# Description:
#   Implementation of XLA’s HLO utilities used for higher-level transformations.

load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hlo_live_range",
    srcs = [
        "hlo_live_range.cc",
    ],
    hdrs = [
        "hlo_live_range.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_buffer",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_live_range_test",
    srcs = ["hlo_live_range_test.cc"],
    deps = [
        ":hlo_live_range",
        "//xla:literal",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_alias_analysis",
        "//xla/service:hlo_ordering",
        "//xla/service:hlo_value",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "hlo_matchers",
    testonly = 1,
    srcs = ["hlo_matchers.cc"],
    hdrs = ["hlo_matchers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
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
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla:array",
        "//xla:literal_util",
        "//xla:protobuf_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/service:call_graph",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_sharding_util_test",
    srcs = [
        "hlo_sharding_util_test.cc",
    ],
    deps = [
        ":hlo_sharding_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "hlo_query",
    srcs = ["hlo_query.cc"],
    hdrs = ["hlo_query.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)
