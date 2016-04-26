# Description:
#   quantization-specific OpKernels

package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_library",
    "tf_kernel_library",
)

filegroup(
    name = "android_ops",
    srcs = [
        "dequantize_op.cc",
        "quantization_utils.h",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_matmul_op.cc",
        "quantized_pooling_ops.cc",
        "reference_gemm.h",
    ],
    visibility = ["//visibility:public"],
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

tf_kernel_library(
    name = "quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_matmul_op.cc",
        "quantized_pooling_ops.cc",
    ],
    hdrs = [
        "quantization_utils.h",
        "reference_gemm.h",
    ],
    deps = [
        "@gemmlowp//:eight_bit_int_gemm",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:pooling_ops",
        "//third_party/eigen3",
    ],
)

tf_custom_op_library(
    name = "_quantized_kernels.so",
    srcs = [
        "dequantize_op.cc",
        "quantization_utils.h",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_matmul_op.cc",
        "quantized_pooling_ops.cc",
        "reference_gemm.h",
    ],
    deps = [
        "@gemmlowp//:eight_bit_int_gemm",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "//tensorflow/core/kernels:pooling_ops_hdrs",
    ],
)

py_library(
    name = "quantized_kernels_py",
    srcs = ["load_quantized_kernels_so.py"],
    data = ["_quantized_kernels.so"],
    srcs_version = "PY2AND3",
)

tf_cc_test(
    name = "quantize_down_and_shrink_range_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantization_utils_test",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_activation_ops_test",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_bias_add_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_conv_ops_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantize_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_matmul_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_pooling_ops_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_concat_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "quantized_batch_norm_op_test",
    size = "small",
    deps = [
        ":quantized_ops",
        "//tensorflow/contrib/quantization:cc_array_ops",
        "//tensorflow/contrib/quantization:cc_math_ops",
        "//tensorflow/contrib/quantization:cc_nn_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:batch_norm_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//third_party/eigen3",
    ],
)
