load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

exports_files([
    "xnnpack_delegate.h",
])

cc_library(
    name = "xnnpack_delegate",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    copts = tflite_copts(),
    linkstatic = True,
    deps = [
        ":quantization_util",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@XNNPACK//:xnnpack_for_tflite",
    ],
)

cc_library(
    name = "xnnpack_delegate_hdrs_only",
    hdrs = ["xnnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "xnnpack_delegate_test_mode",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    copts = tflite_copts() + ["-DXNNPACK_DELEGATE_TEST_MODE=1"],
    linkstatic = True,
    deps = [
        ":quantization_util",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@XNNPACK",
    ],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:types",
        "@FP16",
    ],
)

################################ Tester classes ################################

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/kernels/internal:cppmath",
    ],
)

cc_library(
    name = "binary_elementwise_tester",
    testonly = 1,
    srcs = ["binary_elementwise_tester.cc"],
    hdrs = ["binary_elementwise_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "conv_2d_tester",
    testonly = 1,
    srcs = ["conv_2d_tester.cc"],
    hdrs = ["conv_2d_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "depth_to_space_tester",
    testonly = 1,
    srcs = ["depth_to_space_tester.cc"],
    hdrs = ["depth_to_space_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "depthwise_conv_2d_tester",
    testonly = 1,
    srcs = ["depthwise_conv_2d_tester.cc"],
    hdrs = ["depthwise_conv_2d_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "fully_connected_tester",
    testonly = 1,
    srcs = ["fully_connected_tester.cc"],
    hdrs = ["fully_connected_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "leaky_relu_tester",
    testonly = 1,
    srcs = ["leaky_relu_tester.cc"],
    hdrs = ["leaky_relu_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pad_tester",
    testonly = 1,
    srcs = ["pad_tester.cc"],
    hdrs = ["pad_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pool_2d_tester",
    testonly = 1,
    srcs = ["pool_2d_tester.cc"],
    hdrs = ["pool_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "prelu_tester",
    testonly = 1,
    srcs = ["prelu_tester.cc"],
    hdrs = ["prelu_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_binary_elementwise_tester",
    testonly = 1,
    srcs = ["quantized_binary_elementwise_tester.cc"],
    hdrs = ["quantized_binary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_conv_2d_tester",
    testonly = 1,
    srcs = ["quantized_conv_2d_tester.cc"],
    hdrs = ["quantized_conv_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_depthwise_conv_2d_tester",
    testonly = 1,
    srcs = ["quantized_depthwise_conv_2d_tester.cc"],
    hdrs = ["quantized_depthwise_conv_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_fully_connected_tester",
    testonly = 1,
    srcs = ["quantized_fully_connected_tester.cc"],
    hdrs = ["quantized_fully_connected_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_pad_tester",
    testonly = 1,
    srcs = ["quantized_pad_tester.cc"],
    hdrs = ["quantized_pad_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_pool_2d_tester",
    testonly = 1,
    srcs = ["quantized_pool_2d_tester.cc"],
    hdrs = ["quantized_pool_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_unary_elementwise_tester",
    testonly = 1,
    srcs = ["quantized_unary_elementwise_tester.cc"],
    hdrs = ["quantized_unary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "reduce_tester",
    testonly = 1,
    srcs = ["reduce_tester.cc"],
    hdrs = ["reduce_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "reshape_tester",
    testonly = 1,
    srcs = ["reshape_tester.cc"],
    hdrs = ["reshape_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "resize_bilinear_tester",
    testonly = 1,
    srcs = ["resize_bilinear_tester.cc"],
    hdrs = ["resize_bilinear_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "softmax_tester",
    testonly = 1,
    srcs = ["softmax_tester.cc"],
    hdrs = ["softmax_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "unary_elementwise_tester",
    testonly = 1,
    srcs = ["unary_elementwise_tester.cc"],
    hdrs = ["unary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

############################## Integration tests ###############################

cc_library(
    name = "test_main",
    testonly = 1,
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "abs_test",
    srcs = ["abs_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "average_pool_2d_test",
    srcs = ["average_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ceil_test",
    srcs = ["ceil_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channelwise_quantized_conv_2d_test",
    srcs = ["channelwise_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channelwise_quantized_depthwise_conv_2d_test",
    srcs = ["channelwise_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv_2d_test",
    srcs = ["conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "delegate_test",
    srcs = ["delegate_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
        "@pthreadpool",
    ],
)

cc_test(
    name = "depth_to_space_test",
    srcs = ["depth_to_space_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depth_to_space_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "depthwise_conv_2d_test",
    srcs = ["depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "div_test",
    srcs = ["div_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "elu_test",
    srcs = ["elu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "floor_test",
    srcs = ["floor_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "hard_swish_test",
    srcs = ["hard_swish_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "leaky_relu_test",
    srcs = ["leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":leaky_relu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "logistic_test",
    srcs = ["logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "max_pool_2d_test",
    srcs = ["max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "maximum_test",
    srcs = ["maximum_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "mean_test",
    srcs = ["mean_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reduce_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "minimum_test",
    srcs = ["minimum_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "mul_test",
    srcs = ["mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "neg_test",
    srcs = ["neg_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":prelu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantization_util",
        ":test_main",
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu6_test",
    srcs = ["relu6_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu_n1_to_1_test",
    srcs = ["relu_n1_to_1_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reshape_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = ["resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "round_test",
    srcs = ["round_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_add_test",
    srcs = ["signed_quantized_add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_conv_2d_test",
    srcs = ["signed_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_depthwise_conv_2d_test",
    srcs = ["signed_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_elu_test",
    srcs = ["signed_quantized_elu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_fully_connected_test",
    srcs = ["signed_quantized_fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_logistic_test",
    srcs = ["signed_quantized_logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_max_pool_2d_test",
    srcs = ["signed_quantized_max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_mul_test",
    srcs = ["signed_quantized_mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_pad_test",
    srcs = ["signed_quantized_pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_sub_test",
    srcs = ["signed_quantized_sub_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":softmax_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sqrt_test",
    srcs = ["sqrt_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "square_test",
    srcs = ["square_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "squared_difference_test",
    srcs = ["squared_difference_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sub_test",
    srcs = ["sub_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_add_test",
    srcs = ["unsigned_quantized_add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_conv_2d_test",
    srcs = ["unsigned_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_depthwise_conv_2d_test",
    srcs = ["unsigned_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_fully_connected_test",
    srcs = ["unsigned_quantized_fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_logistic_test",
    srcs = ["unsigned_quantized_logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_max_pool_2d_test",
    srcs = ["unsigned_quantized_max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_mul_test",
    srcs = ["unsigned_quantized_mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_pad_test",
    srcs = ["unsigned_quantized_pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_sub_test",
    srcs = ["unsigned_quantized_sub_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_main"]})
