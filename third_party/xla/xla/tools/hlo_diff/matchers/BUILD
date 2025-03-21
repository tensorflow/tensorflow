load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/tools/hlo_diff:__subpackages__",
    ],
)

cc_library(
    name = "hlo_call_graph_matcher",
    srcs = ["hlo_call_graph_matcher.cc"],
    hdrs = ["hlo_call_graph_matcher.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_call_graph_matcher_test",
    srcs = ["hlo_call_graph_matcher_test.cc"],
    deps = [
        ":hlo_call_graph_matcher",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_test_base",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_computation_graph_matcher",
    srcs = ["hlo_computation_graph_matcher.cc"],
    hdrs = ["hlo_computation_graph_matcher.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:call_graph",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
    ],
)

xla_cc_test(
    name = "hlo_computation_graph_matcher_test",
    srcs = ["hlo_computation_graph_matcher_test.cc"],
    deps = [
        ":hlo_computation_graph_matcher",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:call_graph",
        "//xla/tests:hlo_test_base",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_gumgraph_matcher",
    srcs = ["hlo_gumgraph_matcher.cc"],
    hdrs = ["hlo_gumgraph_matcher.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_value",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/tools/hlo_diff/graph/utils:hlo_gumgraph_bfs",
        "//xla/tools/hlo_diff/graph/utils:hlo_gumgraph_dfs",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_matcher_test",
    srcs = ["hlo_gumgraph_matcher_test.cc"],
    deps = [
        ":hlo_gumgraph_matcher",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tests:hlo_test_base",
        "//xla/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
