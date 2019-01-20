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
    "if_not_v2",
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
        "flatten_atrous.cc",
        "fold_batch_norms.cc",
        "fold_constants_lib.cc",
        "fold_old_batch_norms.cc",
        "freeze_requantization_ranges.cc",
        "fuse_convolutions.cc",
        "fuse_quantized_convolution.cc",
        "insert_logging.cc",
        "obfuscate_names.cc",
        "quantize_nodes.cc",
        "quantize_weights.cc",
        "remove_attribute.cc",
        "remove_control_dependencies.cc",
        "remove_device.cc",
        "remove_nodes.cc",
        "rename_attribute.cc",
        "rename_op.cc",
        "round_weights.cc",
        "set_device.cc",
        "sort_by_execution_order.cc",
        "sparsify_gather.cc",
        "strip_unused_nodes.cc",
    ],
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
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/kernels:quantization_utils",
    ] + if_not_windows([
        "//tensorflow/core/kernels:remote_fused_graph_rewriter_transform",
        "//tensorflow/core/kernels/hexagon:hexagon_rewriter_transform",
        "//tensorflow/core:user_ops_op_lib",
        "//tensorflow/core:training_ops_op_lib",
        "//tensorflow/core:string_ops_op_lib",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:manip_ops_op_lib",
        "//tensorflow/core:list_ops_op_lib",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:candidate_sampling_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",
    ]) + if_not_v2([
        "//tensorflow/contrib/rnn:gru_ops_op_lib",
        "//tensorflow/contrib/rnn:lstm_ops_op_lib",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "transforms_test",
    size = "small",
    srcs = [
        "add_default_attributes_test.cc",
        "backports_test.cc",
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
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:quantization_utils",
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
