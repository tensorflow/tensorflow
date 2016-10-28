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
    srcs = ["graph_transferer_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
    ],
)

tf_kernel_library(
    name = "graph_transferer",
    srcs = [
        "graph_transferer.cc",
    ],
    hdrs = [
        "graph_transferer.h",
    ],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)
