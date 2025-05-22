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
        ":similarity",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/service:call_graph",
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
    name = "hlo_gumgraph_matcher",
    srcs = ["hlo_gumgraph_matcher.cc"],
    hdrs = ["hlo_gumgraph_matcher.h"],
    deps = [
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
