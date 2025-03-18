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
    name = "hlo_gumgraph_bfs",
    srcs = ["hlo_gumgraph_bfs.cc"],
    hdrs = ["hlo_gumgraph_bfs.h"],
    deps = [
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_gumgraph_dfs",
    srcs = ["hlo_gumgraph_dfs.cc"],
    hdrs = ["hlo_gumgraph_dfs.h"],
    deps = [
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_dfs_test",
    srcs = ["hlo_gumgraph_dfs_test.cc"],
    deps = [
        ":hlo_gumgraph_dfs",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_gumgraph_bfs_test",
    srcs = ["hlo_gumgraph_bfs_test.cc"],
    deps = [
        ":hlo_gumgraph_bfs",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
