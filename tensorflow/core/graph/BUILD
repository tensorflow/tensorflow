load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(bmzhao): This target a holdover from tensorflow/core/BUILD. We
# will add proper dependencies once tf/core/graph/BUILD has granular
# targets added in a subsequent changes.
cc_library(
    name = "mkl_graph_util",
    hdrs = ["mkl_graph_util.h"],
)

tf_cc_test(
    name = "collective_order_test",
    size = "small",
    srcs = [
        "collective_order_test.cc",
    ],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "core_cpu_headers",
    srcs = [
        "algorithm.h",
        "default_device.h",
        "graph.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "node_builder.h",
        "validate.h",
        "while_context.h",
    ],
)

filegroup(
    name = "framework_internal_private_headers",
    srcs = [
        "edgeset.h",
        "graph.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "node_builder.h",
        "tensor_id.h",
    ],
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = [
        "edgeset.cc",
        "graph.cc",
        "graph_def_builder.cc",
        "graph_node_util.cc",
        "node_builder.cc",
        "tensor_id.cc",
        "while_context.cc",
        "while_context.h",
    ],
)

# Note(bmzhao): This target is a holdover from the GRAPH_HDRS array
# in tensorflow/core/BUILD. This target contains all '.h' files under
# tensorflow/core/graph, except for the following:
# 'benchmark_testlib.h'.
filegroup(
    name = "graph_headers",
    srcs = [
        "algorithm.h",
        "collective_order.h",
        "colors.h",
        "control_flow.h",
        "costmodel.h",
        "default_device.h",
        "edgeset.h",
        "graph.h",
        "graph_def_builder.h",
        "graph_node_util.h",
        "graph_partition.h",
        "node_builder.h",
        "optimizer_cse.h",
        "subgraph.h",
        "tensor_id.h",
        "testlib.h",
        "types.h",
        "validate.h",
        "while_context.h",
    ],
)

filegroup(
    name = "graph_srcs",
    srcs = [
        "algorithm.cc",
        "collective_order.cc",
        "colors.cc",
        "control_flow.cc",
        "costmodel.cc",
        "graph_partition.cc",
        "optimizer_cse.cc",
        "subgraph.cc",
        "validate.cc",
    ],
)

filegroup(
    name = "testlib_headers",
    srcs = [
        "benchmark_testlib.h",
        "testlib.h",
    ],
)

filegroup(
    name = "testlib_srcs",
    srcs = [
        "testlib.cc",
    ],
)

filegroup(
    name = "mkl_graph_util_header",
    srcs = [
        "mkl_graph_util.h",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "algorithm.cc",
        "algorithm.h",
        "benchmark_testlib.h",
        "collective_order.cc",
        "collective_order.h",
        "colors.cc",
        "colors.h",
        "control_flow.cc",
        "control_flow.h",
        "costmodel.cc",
        "costmodel.h",
        "default_device.h",
        "edgeset.cc",
        "edgeset.h",
        "graph.cc",
        "graph.h",
        "graph_def_builder.cc",
        "graph_def_builder.h",
        "graph_node_util.cc",
        "graph_node_util.h",
        "graph_partition.cc",
        "graph_partition.h",
        "mkl_graph_util.h",
        "node_builder.cc",
        "node_builder.h",
        "optimizer_cse.cc",
        "optimizer_cse.h",
        "subgraph.cc",
        "subgraph.h",
        "tensor_id.cc",
        "tensor_id.h",
        "testlib.h",
        "types.h",
        "validate.cc",
        "validate.h",
        "while_context.cc",
        "while_context.h",
    ],
)

# Note(bmzhao): Ideally we would use a filegroup to represent these tests instead.
# However, that causes tf_cc_tests to link all of these tests into a single object
# file. This breaks tensorflow/core:core_higher_level_tests, because some of these
# tests redefine the same symbol. This will be fixed by having granular tests
# instead, after phase 4 of the tensorflow's build refactoring:
# https://github.com/tensorflow/community/pull/179
exports_files(
    srcs = [
        "algorithm_test.cc",
        "control_flow_test.cc",
        "edgeset_test.cc",
        "graph_def_builder_test.cc",
        "graph_partition_test.cc",
        "graph_test.cc",
        "node_builder_test.cc",
        "optimizer_cse_test.cc",
        "subgraph_test.cc",
        "tensor_id_test.cc",
        "validate_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
