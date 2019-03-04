# Description:
#   Contains a tool to dump TensorFlow ops which are not supported
#   in TensorFlow HVX runtime.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "hvx_ops_support_checker",
    testonly = 1,
    srcs = ["hvx_ops_support_checker_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:candidate_sampling_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:data_flow_ops_op_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:io_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:list_ops_op_lib",
        "//tensorflow/core:logging_ops_op_lib",
        "//tensorflow/core:lookup_ops_op_lib",
        "//tensorflow/core:manip_ops_op_lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:parsing_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:sparse_ops_op_lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:string_ops_op_lib",
        "//tensorflow/core:training_ops_op_lib",
        "//tensorflow/core:user_ops_op_lib",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
        "//tensorflow/tools/graph_transforms:file_utils",
    ],
)
