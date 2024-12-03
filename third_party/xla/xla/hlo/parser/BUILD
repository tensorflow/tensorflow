# Description:
#   XLA parser implementation.

load(
    "//xla:xla.bzl",
    "xla_cc_test",
)
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
    name = "hlo_parser",
    srcs = ["hlo_parser.cc"],
    hdrs = ["hlo_parser.h"],
    deps = [
        ":hlo_lexer",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/service:computation_layout",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:name_uniquer",
        "//xla/service:shape_inference",
        "//xla/tsl/lib/gtl:map_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_parser_test",
    size = "small",
    srcs = ["hlo_parser_test.cc"],
    deps = [
        ":hlo_lexer",
        ":hlo_parser",
        "//xla:array",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service:pattern_matcher_gmock",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = [
        "hlo_lexer.h",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:regexp",
    ],
)
