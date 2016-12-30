# Description:
#   Utilities that perform useful transformations on graphs

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cc_test",
)

exports_files(["LICENSE"])

cc_library(
    name = "transform_utils",
    srcs = [
        "transform_utils.cc",
    ],
    hdrs = [
        "transform_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "transform_utils_test",
    size = "small",
    srcs = ["transform_utils_test.cc"],
    deps = [
        ":transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "transforms_lib",
    srcs = [
        "fold_batch_norms.cc",
        "fold_constants_lib.cc",
        "fold_old_batch_norms.cc",
        "fuse_convolutions.cc",
        "obsfucate_names.cc",
        "quantize_nodes.cc",
        "quantize_weights.cc",
        "remove_attribute.cc",
        "remove_device.cc",
        "remove_nodes.cc",
        "rename_attribute.cc",
        "rename_op.cc",
        "round_weights.cc",
        "sort_by_execution_order.cc",
        "strip_unused_nodes.cc",
    ],
    hdrs = [
        "fold_constants_lib.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/kernels:quantized_ops",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "transforms_test",
    size = "small",
    srcs = [
        "fold_batch_norms_test.cc",
        "fold_constants_test.cc",
        "fold_old_batch_norms_test.cc",
        "fuse_convolutions_test.cc",
        "obsfucate_names_test.cc",
        "quantize_nodes_test.cc",
        "quantize_weights_test.cc",
        "remove_attribute_test.cc",
        "remove_device_test.cc",
        "remove_nodes_test.cc",
        "rename_attribute_test.cc",
        "rename_op_test.cc",
        "round_weights_test.cc",
        "sort_by_execution_order_test.cc",
        "strip_unused_nodes_test.cc",
    ],
    deps = [
        ":transform_utils",
        ":transforms_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:quantized_ops",
    ],
)

cc_library(
    name = "transform_graph_lib",
    srcs = ["transform_graph.cc"],
    hdrs = ["transform_graph.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        ":transforms_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

# This library includes a main function, to make it easy to create other
# versions of the tool linked against different operator libs.
cc_library(
    name = "transform_graph_main_lib",
    srcs = ["transform_graph_main.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_graph_lib",
        ":transforms_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "transform_graph",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":transform_graph_main_lib",
    ],
)

tf_cc_test(
    name = "transform_graph_test",
    size = "medium",
    srcs = ["transform_graph_test.cc"],
    deps = [
        ":transform_graph_lib",
        ":transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# This library includes a main function, to make it easy to create other
# versions of the tool linked against different operator libs.
cc_library(
    name = "summarize_graph_main_lib",
    srcs = ["summarize_graph_main.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "summarize_graph",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":summarize_graph_main_lib",
    ],
)
