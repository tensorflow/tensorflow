load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],
)

cc_library(
    name = "graph_analyzer_lib",
    srcs = [
        "gen_node.cc",
        "graph_analyzer.cc",
        "sig_node.cc",
        "subgraph.cc",
    ],
    hdrs = [
        "gen_node.h",
        "graph_analyzer.h",
        "hash_tools.h",
        "map_tools.h",
        "sig_node.h",
        "subgraph.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "graph_analyzer_tool",
    srcs = ["graph_analyzer_tool.cc"],
    hdrs = ["graph_analyzer_tool.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_analyzer_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_tools_lib",
    testonly = 1,
    srcs = [
        "test_tools.cc",
    ],
    hdrs = [
        "test_tools.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_analyzer_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:op_types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "hash_tools_test",
    testonly = 1,
    srcs = [
        "hash_tools_test.cc",
    ],
    deps = [
        ":graph_analyzer_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "gen_node_test",
    testonly = 1,
    srcs = [
        "gen_node_test.cc",
    ],
    deps = [
        ":graph_analyzer_lib",
        ":test_tools_lib",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "sig_node_test",
    testonly = 1,
    srcs = [
        "sig_node_test.cc",
    ],
    deps = [
        ":graph_analyzer_lib",
        ":test_tools_lib",
        "//tensorflow/core/grappler:utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "graph_analyzer_test",
    testonly = 1,
    srcs = [
        "graph_analyzer_test.cc",
    ],
    deps = [
        ":graph_analyzer_lib",
        ":test_tools_lib",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "subgraph_test",
    testonly = 1,
    srcs = [
        "subgraph_test.cc",
    ],
    deps = [
        ":graph_analyzer_lib",
        ":test_tools_lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)
