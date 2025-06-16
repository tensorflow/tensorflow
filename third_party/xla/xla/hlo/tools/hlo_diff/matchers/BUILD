load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/hlo/tools/hlo_diff:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "hlo_call_graph_matcher",
    srcs = ["hlo_call_graph_matcher.cc"],
    hdrs = ["hlo_call_graph_matcher.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/service:call_graph",
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
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_computation_graph_matcher",
    srcs = ["hlo_computation_graph_matcher.cc"],
    hdrs = ["hlo_computation_graph_matcher.h"],
    deps = [
        ":bipartite_matching",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
    ],
)

xla_cc_test(
    name = "hlo_computation_graph_matcher_test",
    srcs = ["hlo_computation_graph_matcher_test.cc"],
    deps = [
        ":hlo_computation_graph_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/service:call_graph",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "exact_subgraph_matcher",
    srcs = ["exact_subgraph_matcher.cc"],
    hdrs = ["exact_subgraph_matcher.h"],
    deps = [
        ":gumgraph_matcher",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_bfs",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

xla_cc_test(
    name = "exact_subgraph_matcher_test",
    srcs = ["exact_subgraph_matcher_test.cc"],
    deps = [
        ":exact_subgraph_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "similarity",
    srcs = ["similarity.cc"],
    hdrs = ["similarity.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_bfs",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

cc_library(
    name = "gumgraph_matcher",
    hdrs = ["gumgraph_matcher.h"],
    deps = [
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
    ],
)

cc_library(
    name = "bottom_up_matcher",
    srcs = ["bottom_up_matcher.cc"],
    hdrs = ["bottom_up_matcher.h"],
    deps = [
        ":gumgraph_matcher",
        ":similarity",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_bfs",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_dfs",
        "//xla/service:hlo_value",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "bottom_up_matcher_test",
    srcs = ["bottom_up_matcher_test.cc"],
    deps = [
        ":bottom_up_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "top_down_matcher",
    srcs = ["top_down_matcher.cc"],
    hdrs = ["top_down_matcher.h"],
    deps = [
        ":gumgraph_matcher",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/graph/utils:hlo_gumgraph_dfs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
    ],
)

xla_cc_test(
    name = "top_down_matcher_test",
    srcs = ["top_down_matcher_test.cc"],
    deps = [
        ":top_down_matcher",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "similarity_test",
    srcs = ["similarity_test.cc"],
    deps = [
        ":similarity",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bipartite_matching",
    srcs = ["bipartite_matching.cc"],
    hdrs = ["bipartite_matching.h"],
    deps = [
        ":similarity",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "bipartite_matching_test",
    srcs = ["bipartite_matching_test.cc"],
    deps = [
        ":bipartite_matching",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/hlo/tools/hlo_diff/utils:test_util",
        "//xla/service:call_graph",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
