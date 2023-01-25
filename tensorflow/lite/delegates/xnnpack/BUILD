load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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

# Force XNNPACK to run FP32 models as FP16, regardless of model metadata.
config_setting(
    name = "xnnpack_force_float_precision_explicit_fp16",
    define_values = {"xnnpack_force_float_precision": "fp16"},
)

# Enable offloading of quantized 8-bit signed operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qs8_explicit_true",
    define_values = {"tflite_with_xnnpack_qs8": "true"},
)

# Disable offloading of quantized 8-bit signed operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qs8_explicit_false",
    define_values = {"tflite_with_xnnpack_qs8": "false"},
)

# Default setting for offloading of quantized 8-bit signed operators to XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_qs8_default",
    compatible_with = get_compatible_with_portable(),
    defines = ["XNNPACK_DELEGATE_ENABLE_QS8=1"],
)

cc_library(
    name = "tflite_with_xnnpack_qs8",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_qs8_explicit_true": ["XNNPACK_DELEGATE_ENABLE_QS8=1"],
        ":tflite_with_xnnpack_qs8_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_qs8_explicit_true": [],
        ":tflite_with_xnnpack_qs8_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_qs8_default"],
    }),
)

# Enable offloading of quantized 8-bit unsigned operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qu8_explicit_true",
    define_values = {"tflite_with_xnnpack_qu8": "true"},
)

# Disable offloading of quantized 8-bit unsigned operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qu8_explicit_false",
    define_values = {"tflite_with_xnnpack_qu8": "false"},
)

# Default setting for offloading of quantized 8-bit unsigned operators to XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_qu8_default",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        # Enable for WebAssembly builds
        "//tensorflow:emscripten": ["XNNPACK_DELEGATE_ENABLE_QU8=1"],
        # Disable for all other builds
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tflite_with_xnnpack_qu8",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_qu8_explicit_true": ["XNNPACK_DELEGATE_ENABLE_QU8=1"],
        ":tflite_with_xnnpack_qu8_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_qu8_explicit_true": [],
        ":tflite_with_xnnpack_qu8_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_qu8_default"],
    }),
)

cc_library(
    name = "xnnpack_delegate",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + select({
        ":xnnpack_force_float_precision_explicit_fp16": ["-DXNNPACK_DELEGATE_FORCE_PRECISION_FP16=1"],
        "//conditions:default": [],
    }),
    linkstatic = True,
    deps = [
        ":quantization_util",
        ":tflite_with_xnnpack_qs8",
        ":tflite_with_xnnpack_qu8",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@XNNPACK//:xnnpack_for_tflite",
    ],
)

cc_library(
    name = "xnnpack_delegate_hdrs_only",
    hdrs = ["xnnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/c:common",
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
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@XNNPACK",
    ],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:reference_base",
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
        "//tensorflow/lite/kernels/internal:types",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "concatenation_tester",
    testonly = 1,
    srcs = ["concatenation_tester.cc"],
    hdrs = ["concatenation_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "dequantize_tester",
    testonly = 1,
    srcs = ["dequantize_tester.cc"],
    hdrs = ["dequantize_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_tester",
    testonly = 1,
    srcs = ["quantize_tester.cc"],
    hdrs = ["quantize_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_leaky_relu_tester",
    testonly = 1,
    srcs = ["quantized_leaky_relu_tester.cc"],
    hdrs = ["quantized_leaky_relu_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_reduce_tester",
    testonly = 1,
    srcs = ["quantized_reduce_tester.cc"],
    hdrs = ["quantized_reduce_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_resize_bilinear_tester",
    testonly = 1,
    srcs = ["quantized_resize_bilinear_tester.cc"],
    hdrs = ["quantized_resize_bilinear_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "slice_tester",
    testonly = 1,
    srcs = ["slice_tester.cc"],
    hdrs = ["slice_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//testing/base/public:gunit_for_library_testonly",
        "@flatbuffers",
    ],
)

cc_library(
    name = "split_tester",
    testonly = 1,
    srcs = ["split_tester.cc"],
    hdrs = ["split_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "space_to_depth_tester",
    testonly = 1,
    srcs = ["space_to_depth_tester.cc"],
    hdrs = ["space_to_depth_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//testing/base/public:gunit_for_library_testonly",
        "@flatbuffers",
    ],
)

cc_library(
    name = "strided_slice_tester",
    testonly = 1,
    srcs = ["strided_slice_tester.cc"],
    hdrs = ["strided_slice_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//testing/base/public:gunit_for_library_testonly",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transpose_tester",
    testonly = 1,
    srcs = ["transpose_tester.cc"],
    hdrs = ["transpose_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transpose_conv_tester",
    testonly = 1,
    srcs = ["transpose_conv_tester.cc"],
    hdrs = ["transpose_conv_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_transpose_conv_tester",
    testonly = 1,
    srcs = ["quantized_transpose_conv_tester.cc"],
    hdrs = ["quantized_transpose_conv_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "variable_ops_tester",
    testonly = 1,
    srcs = ["variable_ops_tester.cc"],
    hdrs = ["variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//testing/base/public:gunit_for_library_testonly",
        "@flatbuffers",
    ],
)

# Same as variable_ops_tester above, but depends on xnnpack_delegate (no test
# mode), used for variable_ops_multiple_subgraph_test. Why name this
# no_test_mode instead of renaming the above to test_mode, because all testers
# depend on test_mode by default, this keeps the name consistent.
cc_library(
    name = "variable_ops_tester_no_test_mode",
    testonly = 1,
    srcs = ["variable_ops_tester.cc"],
    hdrs = ["variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//testing/base/public:gunit_for_library_testonly",
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
    name = "concatenation_test",
    srcs = ["concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
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
    name = "quantize_float32_to_int8_test",
    srcs = ["quantize_float32_to_int8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_float32_to_uint8_test",
    srcs = ["quantize_float32_to_uint8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_int8_to_int8_test",
    srcs = ["quantize_int8_to_int8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_uint8_to_uint8_test",
    srcs = ["quantize_uint8_to_uint8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "variable_ops_multiple_subgraph_test",
    srcs = ["variable_ops_multiple_subgraph_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        # Don't use xnnpack_delegate_test_mode because we use CALL_ONCE here,
        # which XNNPACK does not yet support, and test mode enforces that there
        # are no unsupported operations.
        ":xnnpack_delegate",
        ":variable_ops_tester_no_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "variable_ops_test",
    srcs = ["variable_ops_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":variable_ops_tester",
        ":xnnpack_delegate_test_mode",
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
    name = "signed_quantized_reshape_test",
    srcs = ["signed_quantized_reshape_test.cc"],
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
    name = "unsigned_quantized_reshape_test",
    srcs = ["unsigned_quantized_reshape_test.cc"],
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
    name = "signed_dequantize_test",
    srcs = ["signed_dequantize_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dequantize_tester",
        ":test_main",
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
    name = "signed_quantized_concatenation_test",
    srcs = ["signed_quantized_concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
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
    name = "signed_quantized_depth_to_space_test",
    srcs = ["signed_quantized_depth_to_space_test.cc"],
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
    name = "signed_quantized_leaky_relu_test",
    srcs = ["signed_quantized_leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_leaky_relu_tester",
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
    name = "signed_quantized_mean_test",
    srcs = ["signed_quantized_mean_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_reduce_tester",
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
    name = "signed_quantized_resize_bilinear_test",
    srcs = ["signed_quantized_resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_slice_test",
    srcs = ["signed_quantized_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_space_to_depth_test",
    srcs = ["signed_quantized_space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_split_test",
    srcs = ["signed_quantized_split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_strided_slice_test",
    srcs = ["signed_quantized_strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
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
    name = "signed_quantized_transpose_test",
    srcs = ["signed_quantized_transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_transpose_conv_test",
    srcs = ["signed_quantized_transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_transpose_conv_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
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
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "split_test",
    srcs = ["split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
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
    name = "strided_slice_test",
    srcs = ["strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
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
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "transpose_conv_test",
    srcs = ["transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_conv_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_dequantize_test",
    srcs = ["unsigned_dequantize_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dequantize_tester",
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
    name = "unsigned_quantized_concatenation_test",
    srcs = ["unsigned_quantized_concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
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
    name = "unsigned_quantized_depth_to_space_test",
    srcs = ["unsigned_quantized_depth_to_space_test.cc"],
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
    name = "unsigned_quantized_leaky_relu_test",
    srcs = ["unsigned_quantized_leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_leaky_relu_tester",
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
    name = "unsigned_quantized_mean_test",
    srcs = ["unsigned_quantized_mean_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_reduce_tester",
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
    name = "unsigned_quantized_resize_bilinear_test",
    srcs = ["unsigned_quantized_resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_slice_test",
    srcs = ["unsigned_quantized_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_space_to_depth_test",
    srcs = ["unsigned_quantized_space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_split_test",
    srcs = ["unsigned_quantized_split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_strided_slice_test",
    srcs = ["unsigned_quantized_strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
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

cc_test(
    name = "unsigned_quantized_transpose_test",
    srcs = ["unsigned_quantized_transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_transpose_conv_test",
    srcs = ["unsigned_quantized_transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_transpose_conv_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "weights_cache_test",
    srcs = ["weights_cache_test.cc"],
    deps = [
        ":conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_main"]})
