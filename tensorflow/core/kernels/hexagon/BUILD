# Description:
#   quantization-specific OpKernels for hexagon

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_kernel_library",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cc_test(
    name = "graph_transferer_test",
    size = "small",
    srcs = [
        "graph_transferer_test.cc",
        "hexagon_graph_execution_test.cc",
    ],
    data = ["//tensorflow/core:example_parser_configuration_testdata"],
    deps = [
        ":graph_transferer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/kernels:reduction_ops",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
        "//tensorflow/core/kernels:remote_fused_graph_ops",
        "//tensorflow/core/kernels:reshape_op",
        "//tensorflow/core/kernels:softmax_op",
    ],
)

tf_kernel_library(
    name = "graph_transferer",
    srcs = [
        "graph_transfer_utils.cc",
        "graph_transferer.cc",
        "hexagon_control_wrapper.cc",
        "hexagon_ops_definitions.cc",
        "soc_interface.cc",
    ],
    hdrs = [
        "graph_transfer_utils.h",
        "graph_transferer.h",
        "hexagon_control_wrapper.h",
        "hexagon_ops_definitions.h",
        "soc_interface.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:remote_fused_graph_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "hexagon_rewriter_transform",
    srcs = [
        "hexagon_rewriter_transform.cc",
    ],
    deps = [
        ":graph_transferer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:remote_fused_graph_ops",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "hexagon_rewriter_transform_test",
    size = "small",
    srcs = ["hexagon_rewriter_transform_test.cc"],
    deps = [
        ":graph_transferer",
        ":hexagon_rewriter_transform",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ],
)

cc_library(
    name = "hexagon_remote_fused_graph_executor_build",
    srcs = [
        "hexagon_remote_fused_graph_executor_build.cc",
    ],
    deps = [
        ":graph_transferer",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "hexagon_remote_fused_graph_executor_build_test",
    size = "small",
    srcs = ["hexagon_remote_fused_graph_executor_build_test.cc"],
    deps = [
        ":hexagon_remote_fused_graph_executor_build",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:remote_fused_graph_execute_utils",
    ],
)
