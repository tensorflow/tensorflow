# Description:
#   Utilities that perform useful transformations on graphs

load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "transform_utils",
    srcs = [
        "transform_utils.cc",
    ],
    hdrs = [
        "transform_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
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
        "inline_partitionedcall.cc",
        "insert_logging.cc",
        "obfuscate_names.cc",
        "quantize_nodes.cc",
        "quantize_weights.cc",
        "remove_attribute.cc",
        "remove_control_dependencies.cc",
        "remove_device.cc",
        "remove_nodes.cc",
        "rename_attribute.cc",
        "rename_node.cc",
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
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/kernels:quantization_utils",
        "//tensorflow/core/util/tensor_bundle",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ] + if_not_windows([
        "//tensorflow/core:sparse_ops_op_lib",
        "//tensorflow/core:parsing_ops_op_lib",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:io_ops_op_lib",
        "//tensorflow/core:logging_ops_op_lib",
        "//tensorflow/core:lookup_ops_op_lib",
        "//tensorflow/core:data_flow_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:user_ops_op_lib",
        "//tensorflow/core:training_ops_op_lib",
        "//tensorflow/core:string_ops_op_lib",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:rnn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:manip_ops_op_lib",
        "//tensorflow/core:list_ops_op_lib",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:candidate_sampling_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",
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
        "inline_partitionedcall_test.cc",
        "insert_logging_test.cc",
        "obfuscate_names_test.cc",
        "quantize_nodes_test.cc",
        "quantize_weights_test.cc",
        "remove_attribute_test.cc",
        "remove_device_test.cc",
        "remove_nodes_test.cc",
        "rename_attribute_test.cc",
        "rename_node_test.cc",
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

filegroup(
    name = "transform_graph_hdrs",
    srcs = [
        "transform_graph.h",
        "transform_utils.h",
    ],
    visibility = ["//tensorflow/python/util:__pkg__"],
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
    alwayslink = 1,
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

py_strict_library(
    name = "transform_graph_py",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "transform_graph_py_test",
    size = "small",
    srcs = ["python/transform_graph_test.py"],
    main = "python/transform_graph_test.py",
    tags = ["v1only"],
    deps = [
        ":transform_graph_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:client_testlib",
    ],
)
