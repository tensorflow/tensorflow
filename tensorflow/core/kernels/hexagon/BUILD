# Description:
#   quantization-specific OpKernels for hexagon

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
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
    name = "quantized_matmul_op_for_hexagon_test",
    size = "small",
    srcs = ["quantized_matmul_op_for_hexagon_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:quantized_ops",
    ],
)

tf_cc_test(
    name = "graph_transferer_test",
    size = "small",
    srcs = [
        "graph_transferer_test.cc",
        "hexagon_graph_execution_test.cc",
    ],
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
        "//tensorflow/core/kernels:reshape_op",
        "//tensorflow/core/kernels:softmax_op",
    ],
)

tf_kernel_library(
    name = "graph_transferer",
    srcs = [
        "graph_transferer.cc",
        "hexagon_control_wrapper.cc",
        "hexagon_ops_definitions.cc",
        "i_graph_transfer_ops_definitions.cc",
    ],
    hdrs = [
        "graph_transferer.h",
        "hexagon_control_wrapper.h",
        "hexagon_ops_definitions.h",
        "i_graph_transfer_ops_definitions.h",
        "i_soc_control_wrapper.h",
    ],
    data = ["//tensorflow/core:example_parser_configuration_testdata"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)
