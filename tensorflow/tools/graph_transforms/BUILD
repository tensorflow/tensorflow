# Description:
#   Utilities that perform useful transformations on graphs

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "tf_copts",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_py_test",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "transform_utils_test",
    size = "small",
    srcs = ["transform_utils_test.cc"],
    deps = [
        ":transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "file_utils",
    srcs = [
        "file_utils.cc",
    ],
    hdrs = [
        "file_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    size = "small",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "transforms_lib",
    srcs = [
        "add_default_attributes.cc",
        "backports.cc",
        "fake_quantize_training.cc",
        "flatten_atrous.cc",
        "fold_batch_norms.cc",
        "fold_constants_lib.cc",
        "fold_old_batch_norms.cc",
        "freeze_requantization_ranges.cc",
        "fuse_convolutions.cc",
        "insert_logging.cc",
        "remove_ema.cc",
        "obfuscate_names.cc",
        "remove_attribute.cc",
        "remove_device.cc",
        "remove_nodes.cc",
        "rename_attribute.cc",
        "rename_op.cc",
        "set_device.cc",
        "sort_by_execution_order.cc",
        "sparsify_gather.cc",
        "strip_unused_nodes.cc",
    ] + if_not_windows([
        "quantize_nodes.cc",
        "quantize_weights.cc",
        "round_weights.cc",
    ]),
    hdrs = [
        "fold_constants_lib.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ] + if_not_windows([
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/kernels:remote_fused_graph_rewriter_transform",
        "//tensorflow/core/kernels/hexagon:hexagon_rewriter_transform",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "transforms_test",
    size = "small",
    srcs = [
        "add_default_attributes_test.cc",
        "backports_test.cc",
        "fake_quantize_training_test.cc",
        "flatten_atrous_test.cc",
        "fold_batch_norms_test.cc",
        "fold_constants_test.cc",
        "fold_old_batch_norms_test.cc",
        "freeze_requantization_ranges_test.cc",
        "fuse_convolutions_test.cc",
        "insert_logging_test.cc",
        "obfuscate_names_test.cc",
        "quantize_nodes_test.cc",
        "quantize_weights_test.cc",
        "remove_attribute_test.cc",
        "remove_device_test.cc",
        "remove_ema_test.cc",
        "remove_nodes_test.cc",
        "rename_attribute_test.cc",
        "rename_op_test.cc",
        "round_weights_test.cc",
        "set_device_test.cc",
        "sort_by_execution_order_test.cc",
        "sparsify_gather_test.cc",
        "strip_unused_nodes_test.cc",
    ],
    deps = [
        ":transform_utils",
        ":transforms_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "transform_graph_lib",
    srcs = ["transform_graph.cc"],
    hdrs = ["transform_graph.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
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

tf_cc_binary(
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
        ":file_utils",
        ":transform_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_binary(
    name = "summarize_graph",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":summarize_graph_main_lib",
    ],
)

tf_cc_binary(
    name = "compare_graphs",
    srcs = ["compare_graphs.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        ":transform_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

py_library(
    name = "transform_graph_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "transform_graph_py_test",
    size = "small",
    srcs = ["python/transform_graph_test.py"],
    additional_deps = [
        ":transform_graph_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
    ],
    main = "python/transform_graph_test.py",
)
