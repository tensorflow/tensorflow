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
    deps = [
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/contrib/quantization/kernels:quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)
