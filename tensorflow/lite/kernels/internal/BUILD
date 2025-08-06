load("@bazel_skylib//lib:selects.bzl", "selects")
load("//tensorflow:tensorflow.bzl", "transitive_hdrs")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

HARD_FP_FLAGS_IF_APPLICABLE = select({
    "//tensorflow:android_arm": ["-mfloat-abi=softfp"],
    "//tensorflow:android_armeabi": ["-mfloat-abi=softfp"],
    "//conditions:default": [],
})

NEON_FLAGS_IF_APPLICABLE = select({
    ":armhf": [
        "-O3",
        "-mfpu=neon",
    ],
    ":armv7": [
        "-O3",
        "-mfpu=neon",
    ],
    "//conditions:default": [
        "-O3",
    ],
})

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = ["compatibility.h"],
    visibility = [
        "//tensorflow/lite:__pkg__",
    ],
)

cc_library(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "runtime_shape",
    srcs = ["runtime_shape.cc"],
    hdrs = ["runtime_shape.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "runtime_shape.h",
        "types.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility",
        ":runtime_shape",
    ],
)

cc_library(
    name = "legacy_types",
    hdrs = ["legacy_types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":types",
    ],
)

config_setting(
    name = "armv7",
    constraint_values = [
        "@platforms//cpu:armv7",
    ],
)

config_setting(
    name = "aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "armhf",
    constraint_values = [
        "@platforms//cpu:armv7e-mf",
    ],
)

config_setting(
    name = "x86",
    constraint_values = [
        "@platforms//cpu:x86_32",
    ],
)

config_setting(
    name = "x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

selects.config_setting_group(
    name = "x86_any",
    match_any = [
        ":x86_32_any",
        ":x86_64_any",
    ],
)

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        ":arm32_any",
        ":aarch64_any",
    ],
)

selects.config_setting_group(
    name = "arm32_any",
    match_any = [
        "@platforms//cpu:aarch32",
        "@platforms//cpu:armv7",
        "@platforms//cpu:armv7-m",
        "@platforms//cpu:armv7e-m",
        "@platforms//cpu:armv7e-mf",
    ],
)

selects.config_setting_group(
    name = "aarch64_any",
    match_any = [
        "@platforms//cpu:aarch64",
        "@platforms//cpu:arm64e",
    ],
)

config_setting(
    name = "x86_32_any",
    constraint_values = [
        "@platforms//cpu:x86_32",
    ],
)

config_setting(
    name = "x86_64_any",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":cppmath",
        ":cpu_check",
        ":types",
        "//tensorflow/lite/core:macros",
        "@gemmlowp//:fixedpoint",
    ],
)

cc_library(
    name = "optimized_base",
    hdrs = [
        "optimized/batch_matmul.h",
        "optimized/depthwiseconv_3x3_filter_common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_multithread.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/fully_connected_4bit.h",
        "optimized/im2col_utils.h",
        "optimized/integer_ops/add.h",
        "optimized/integer_ops/conv.h",
        "optimized/integer_ops/depthwise_conv.h",
        "optimized/integer_ops/depthwise_conv_3x3_filter.h",
        "optimized/integer_ops/depthwise_conv_hybrid.h",
        "optimized/integer_ops/depthwise_conv_hybrid_3x3_filter.h",
        "optimized/integer_ops/fully_connected.h",
        "optimized/integer_ops/leaky_relu.h",
        "optimized/integer_ops/lut.h",
        "optimized/integer_ops/mean.h",
        "optimized/integer_ops/mul.h",
        "optimized/integer_ops/pooling.h",
        "optimized/integer_ops/sub.h",
        "optimized/integer_ops/transpose_conv.h",
        "optimized/optimized_ops.h",
        "optimized/optimized_ops_utils.h",
        "optimized/reduce.h",
        "optimized/resize_bilinear.h",
        "optimized/sparse_ops/fully_connected.h",
        "reduce_common.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    defines = ["EIGEN_NEON_GEBP_NR=4"],
    deps = [
        ":avx2_quantization_utils",
        ":common",
        ":compatibility",
        ":cppmath",
        ":cpu_check",
        ":optimized_4bit",
        ":quantization_util",
        ":reduce_utils",
        ":reference_base",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":transpose_utils",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "//tensorflow/lite/kernels:cpu_backend_threadpool",
        "//tensorflow/lite/kernels:kernel_util",
        "@eigen_archive//:eigen3",
        "@gemmlowp//:fixedpoint",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "legacy_optimized_base",
    srcs = [],
    hdrs = [
        "optimized/depthwiseconv_3x3_filter_common.h",
        "optimized/depthwiseconv_float.h",
        "optimized/depthwiseconv_uint8.h",
        "optimized/depthwiseconv_uint8_3x3_filter.h",
        "optimized/im2col_utils.h",
        "optimized/legacy_optimized_ops.h",
        "optimized/optimized_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":cpu_check",
        ":legacy_reference_base",
        ":optimized_base",
        ":quantization_util",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":transpose_utils",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "//tensorflow/lite/kernels:cpu_backend_threadpool",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "optimized_eigen",
    hdrs = [
        "optimized/eigen_spatial_convolutions.h",
        "optimized/multithreaded_conv.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":common",
        ":optimized_base",
        ":types",
        "//tensorflow/lite/core/c:common",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/framework/convolution:eigen_spatial_convolutions-inl",
    ],
)

cc_library(
    name = "optimized_4bit",
    srcs = select({
        ":x86_64_any": [
            "optimized/4bit/sse_fully_connected.cc",
        ],
        ":aarch64_any": [
            "optimized/4bit/neon_fully_connected.cc",
            "optimized/4bit/neon_fully_connected_aarch64_nosdot.cc",
            "optimized/4bit/neon_fully_connected_aarch64_sdot.cc",
            "optimized/4bit/neon_fully_connected_impl.h",
        ],
        ":arm32_any": [
            "optimized/4bit/neon_fully_connected.cc",
            "optimized/4bit/neon_fully_connected_arm32.cc",
            "optimized/4bit/neon_fully_connected_impl.h",
        ],
        "//conditions:default": [],
    }) + [
        "optimized/4bit/fully_connected_common.h",
        "optimized/4bit/fully_connected_reference.cc",
        "optimized/4bit/fully_connected_reference_impl.h",
    ],
    hdrs = [
        "optimized/4bit/fully_connected_reference.h",
        "optimized/fully_connected_4bit.h",
    ] + select({
        ":x86_64_any": [
            "optimized/4bit/sse_fully_connected.h",
            "optimized/4bit/sse_fully_connected_impl.h",
        ],
        ":aarch64_any": [
            "optimized/4bit/neon_fully_connected.h",
        ],
        ":arm32_any": [
            "optimized/4bit/neon_fully_connected.h",
        ],
        "//conditions:default": [],
    }),
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE,
    defines = select({
        ":x86_64_any": ["FC_4BIT_SSE"],
        ":aarch64_any": [
            "FC_4BIT_NEON",
        ],
        ":arm32_any": [
            "FC_4BIT_NEON",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":cppmath",
        ":cpu_check",
        "@cpuinfo//:cpuinfo_with_unstripped_include_path",
    ],
)

cc_test(
    name = "optimized_4bit_test",
    srcs = ["optimized/optimized_4bit_test.cc"],
    deps = [
        ":common",
        ":optimized_4bit",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
        "max.h",
        "min.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    defines = [
        "TFLITE_SINGLE_ROUNDING=0",
    ],
    deps = [
        ":common",
        ":quantization_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantization_util_single_rounding",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    defines = [
        "TFLITE_SINGLE_ROUNDING=1",
    ],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":compatibility",
        ":cppmath",
        ":types",
    ],
)

cc_test(
    name = "quantization_util_test_single_rounding",
    srcs = ["quantization_util_test.cc"],
    defines = [
        "TFLITE_SINGLE_ROUNDING=1",
    ],
    deps = [
        ":common",
        ":quantization_util_single_rounding",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "transpose_utils",
    srcs = [
        "transpose_utils.cc",
    ],
    hdrs = [
        "transpose_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":types",
    ],
)

cc_test(
    name = "transpose_utils_test",
    srcs = ["transpose_utils_test.cc"],
    deps = [
        ":transpose_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "strided_slice_logic",
    srcs = [],
    hdrs = [
        "strided_slice_logic.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility",
        ":types",
    ],
)

cc_library(
    name = "reference_comparisons",
    srcs = ["reference/comparisons.cc"],
    hdrs = ["reference/comparisons.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "reference_base",
    srcs = [],
    hdrs = [
        "reduce_common.h",
        "reference/add.h",
        "reference/add_n.h",
        "reference/arg_min_max.h",
        "reference/batch_matmul.h",
        "reference/batch_to_space_nd.h",
        "reference/binary_function.h",
        "reference/broadcast_args.h",
        "reference/broadcast_to.h",
        "reference/cast.h",
        "reference/ceil.h",
        "reference/comparisons.h",
        "reference/concatenation.h",
        "reference/conv.h",
        "reference/conv3d.h",
        "reference/conv3d_transpose.h",
        "reference/cumsum.h",
        "reference/densify.h",
        "reference/depth_to_space.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/dequantize.h",
        "reference/div.h",
        "reference/elu.h",
        "reference/exp.h",
        "reference/fill.h",
        "reference/floor.h",
        "reference/floor_div.h",
        "reference/floor_mod.h",
        "reference/fully_connected.h",
        "reference/gather.h",
        "reference/gelu.h",
        "reference/hard_swish.h",
        "reference/integer_ops/add.h",
        "reference/integer_ops/conv.h",
        "reference/integer_ops/depthwise_conv.h",
        "reference/integer_ops/fully_connected.h",
        "reference/integer_ops/l2normalization.h",
        "reference/integer_ops/logistic.h",
        "reference/integer_ops/lut.h",
        "reference/integer_ops/mean.h",
        "reference/integer_ops/mul.h",
        "reference/integer_ops/pooling.h",
        "reference/integer_ops/tanh.h",
        "reference/integer_ops/transpose_conv.h",
        "reference/l2normalization.h",
        "reference/leaky_relu.h",
        "reference/log_softmax.h",
        "reference/logistic.h",
        "reference/lstm_cell.h",
        "reference/maximum_minimum.h",
        "reference/mul.h",
        "reference/neg.h",
        "reference/non_max_suppression.h",
        "reference/pad.h",
        "reference/pooling.h",
        "reference/prelu.h",
        "reference/process_broadcast_shapes.h",
        "reference/quantize.h",
        "reference/reduce.h",
        "reference/requantize.h",
        "reference/resize_bilinear.h",
        "reference/resize_nearest_neighbor.h",
        "reference/reverse.h",
        "reference/round.h",
        "reference/select.h",
        "reference/slice.h",
        "reference/softmax.h",
        "reference/space_to_batch_nd.h",
        "reference/space_to_depth.h",
        "reference/strided_slice.h",
        "reference/sub.h",
        "reference/svdf.h",
        "reference/tanh.h",
        "reference/transpose.h",
        "reference/transpose_conv.h",
    ] + select({
        "//tensorflow/lite:tf_lite_static_memory": [],
        "//conditions:default": [
            "reference/integer_ops/dequantize.h",
            "reference/integer_ops/log_softmax.h",
            "reference/reference_ops.h",
            "reference/sparse_ops/fully_connected.h",
            "reference/string_comparisons.h",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    # We are disabling parse_headers for this header-only target so that the
    # external and internal builds are consistent. The primary issue here is
    # that parse_headers is not supported with bazel and the TFLM team would
    # really like to have all build errors in shared Micro/Lite code be
    # reproducible from the OSS build as well.
    #
    # See b/175817116 for more details.
    features = ["-parse_headers"],
    deps = [
        ":common",
        ":compatibility",
        ":constants",
        ":cppmath",
        ":quantization_util",
        ":reference_comparisons",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@eigen_archive//:eigen3",
        "@gemmlowp//:fixedpoint",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "legacy_reference_base",
    srcs = [],
    hdrs = [
        "reference/add.h",
        "reference/add_n.h",
        "reference/arg_min_max.h",
        "reference/batch_matmul.h",
        "reference/batch_to_space_nd.h",
        "reference/binary_function.h",
        "reference/cast.h",
        "reference/ceil.h",
        "reference/comparisons.h",
        "reference/concatenation.h",
        "reference/conv.h",
        "reference/conv3d.h",
        "reference/cumsum.h",
        "reference/densify.h",
        "reference/depth_to_space.h",
        "reference/depthwiseconv_float.h",
        "reference/depthwiseconv_uint8.h",
        "reference/dequantize.h",
        "reference/div.h",
        "reference/elu.h",
        "reference/exp.h",
        "reference/fill.h",
        "reference/floor.h",
        "reference/floor_div.h",
        "reference/floor_mod.h",
        "reference/fully_connected.h",
        "reference/gather.h",
        "reference/hard_swish.h",
        "reference/l2normalization.h",
        "reference/leaky_relu.h",
        "reference/legacy_reference_ops.h",
        "reference/log_softmax.h",
        "reference/logistic.h",
        "reference/lstm_cell.h",
        "reference/maximum_minimum.h",
        "reference/mul.h",
        "reference/neg.h",
        "reference/pad.h",
        "reference/pooling.h",
        "reference/prelu.h",
        "reference/process_broadcast_shapes.h",
        "reference/quantize.h",
        "reference/reduce.h",
        "reference/reference_ops.h",
        "reference/requantize.h",
        "reference/resize_nearest_neighbor.h",
        "reference/reverse.h",
        "reference/round.h",
        "reference/select.h",
        "reference/slice.h",
        "reference/softmax.h",
        "reference/space_to_batch_nd.h",
        "reference/space_to_depth.h",
        "reference/strided_slice.h",
        "reference/string_comparisons.h",
        "reference/sub.h",
        "reference/tanh.h",
        "reference/transpose.h",
        "reference/transpose_conv.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    # We are disabling parse_headers for this header-only target so that the
    # external and internal builds are consistent. The primary issue here is
    # that parse_headers is not supported with bazel and the TFLM team would
    # really like to have all build errors in shared Micro/Lite code be
    # reproducible from the OSS build as well.
    #
    # See b/175817116 for more details.
    features = ["-parse_headers"],
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":legacy_types",
        ":quantization_util",
        ":reference_base",
        ":reference_comparisons",
        ":strided_slice_logic",
        ":tensor",
        ":tensor_utils",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "tensor_ctypes",
    srcs = ["tensor_ctypes.cc"],
    hdrs = ["tensor_ctypes.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tensorflow/lite/core:macros",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library_with_tflite(
    name = "opaque_tensor_ctypes",
    srcs = ["opaque_tensor_ctypes.cc"],
    hdrs = ["opaque_tensor_ctypes.h"],
    copts = tflite_copts(),
    tflite_deps = [
        "//tensorflow/lite/c:c_api_experimental_without_op_resolver",
        "//tensorflow/lite:namespace",
    ],
    deps = [
        ":types",
        "//tensorflow/lite/core:macros",
    ],
)

cc_library(
    name = "tensor",
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ] + select({
        "//tensorflow/lite:tf_lite_static_memory": [],
        "//conditions:default": [
            "tensor.h",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":tensor_ctypes",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
    ],
)

# Deprecated version of :tensor, kept for backwards compatibility.
cc_library(
    name = "reference",
    hdrs = [
        "portable_tensor.h",
        "tensor_ctypes.h",
    ] + select({
        "//tensorflow/lite:tf_lite_static_memory": [],
        "//conditions:default": [
            "tensor.h",
        ],
    }),
    copts = tflite_copts(),
    deps = [
        ":tensor_ctypes",
        ":types",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = [
        "reference/portable_tensor_utils.cc",
    ],
    hdrs = [
        "reference/portable_tensor_utils.h",
        "reference/portable_tensor_utils_impl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        "@gemmlowp",
    ],
)

cc_library(
    name = "neon_tensor_utils",
    srcs = [
        "optimized/neon_tensor_utils.cc",
    ],
    hdrs = [
        "optimized/neon_tensor_utils.h",
        "optimized/neon_tensor_utils_impl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE + HARD_FP_FLAGS_IF_APPLICABLE,
    deps = [
        ":common",
        ":compatibility",
        ":cppmath",
        ":cpu_check",
        ":portable_tensor_utils",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "@ruy//ruy",
    ],
)

cc_library(
    name = "sse_tensor_utils",
    srcs = [
        "compatibility.h",
        "optimized/sse_tensor_utils.cc",
    ],
    hdrs = [
        "optimized/sse_tensor_utils.h",
        "optimized/sse_tensor_utils_impl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":cpu_check",
        ":neon_tensor_utils",
        ":portable_tensor_utils",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:cpu_backend_gemm",
        "//tensorflow/lite/kernels:op_macros",
        "@com_google_absl//absl/base:prefetch",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "avx2_quantization_utils",
    hdrs = [
        "optimized/avx2_quantization_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility",
    ],
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":tensor_utils",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "reduce_utils",
    hdrs = [
        "optimized/reduce_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

# Audio support classes imported directly from TensorFlow.
cc_library(
    name = "audio_utils",
    srcs = [
        "mfcc.cc",
        "mfcc_dct.cc",
        "mfcc_mel_filterbank.cc",
        "spectrogram.cc",
    ],
    hdrs = [
        "mfcc.h",
        "mfcc_dct.h",
        "mfcc_mel_filterbank.h",
        "spectrogram.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
    ],
)

# This target is helpful to allow builds that can not use Eigen. See
# http://b/269699951 for more information. Ideally we could call this target
# portable_tensor_utils but that name is already being used for
# reference/portable_tensor_utils.cc and more.
cc_library(
    name = "tensor_utils_no_eigen",
    srcs = [
        "portable_tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE,
    deps = [
        ":cpu_check",
        "//tensorflow/lite/core/c:common",
    ] + selects.with_or({
        (
            ":arm_any",
        ): [":neon_tensor_utils"],
        (
            ":x86_any",
        ): [
            ":portable_tensor_utils",  # If SSSE3 isn't available, we fall back to portable
            ":sse_tensor_utils",
        ],
        (
            "//tensorflow/lite:tf_lite_static_memory",
            "//conditions:default",
        ): [":portable_tensor_utils"],
    }),
)

# Without adding portable_tensor_utils.h in the exported headers, targets will
# run into layering check build failure unless they list both :tensor_utils and
# :tensor_utils_no_eigen in their deps which can be quite confusing.
cc_library(
    name = "tensor_utils",
    srcs = [
        "tensor_utils.cc",
    ],
    hdrs = [
        "portable_tensor_utils.h",
        "tensor_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + NEON_FLAGS_IF_APPLICABLE,
    deps = [
        ":cpu_check",
        ":tensor_utils_no_eigen",
        "//tensorflow/lite/core/c:common",
        "@eigen_archive//:eigen3",
    ] + selects.with_or({
        (
            ":arm_any",
        ): [":neon_tensor_utils"],
        (
            ":x86_any",
        ): [
            ":portable_tensor_utils",  # If SSSE3 isn't available, we fall back to portable
            ":sse_tensor_utils",
        ],
        (
            "//tensorflow/lite:tf_lite_static_memory",
            "//conditions:default",
        ): [":portable_tensor_utils"],
    }),
)

cc_library(
    name = "test_util",
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":types",
    ],
)

cc_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-fPIE -pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        ":common",
        ":quantization_util",
        ":tensor_utils",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_float_test",
    srcs = ["depthwiseconv_float_test.cc"],
    deps = [
        ":common",
        ":cpu_check",
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_quantized_test",
    srcs = [
        "depthwiseconv_quantized_test.cc",
        "optimized/depthwiseconv_uint8_transitional.h",
    ],
    copts = tflite_copts(),
    shard_count = 2,
    deps = [
        ":compatibility",
        ":cpu_check",
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@ruy//ruy:context",
    ],
)

cc_test(
    name = "depthwiseconv_per_channel_quantized_test",
    srcs = [
        "depthwiseconv_per_channel_quantized_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "depthwiseconv_per_channel_quantized_16x8_test",
    srcs = [
        "depthwiseconv_per_channel_quantized_16x8_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "conv_per_channel_quantized_16x8_test",
    srcs = [
        "conv_per_channel_quantized_16x8_test.cc",
    ],
    shard_count = 2,
    deps = [
        ":common",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = ["resize_bilinear_test.cc"],
    defines = [
        "TFLITE_SINGLE_ROUNDING=0",
    ],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_bilinear_test_single_rounding",
    srcs = ["resize_bilinear_test.cc"],
    defines = [
        "TFLITE_SINGLE_ROUNDING=1",
    ],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "resize_nearest_neighbor_test",
    srcs = ["resize_nearest_neighbor_test.cc"],
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "softmax_quantized_test",
    timeout = "long",
    srcs = [
        "softmax_quantized_test.cc",
    ],
    shard_count = 4,
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "strided_slice_logic_test",
    timeout = "moderate",
    srcs = [
        "strided_slice_logic_test.cc",
    ],
    deps = [
        ":strided_slice_logic",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logsoftmax_quantized_test",
    timeout = "long",
    srcs = [
        "logsoftmax_quantized_test.cc",
    ],
    shard_count = 4,
    tags = [
        # TODO(b/122242739): Reenable after fixing the flakiness?
        "nomac",
        "tflite_not_portable",
    ],
    deps = [
        ":optimized_base",
        ":quantization_util",
        ":reference_base",
        ":test_util",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "averagepool_quantized_test",
    timeout = "long",
    srcs = [
        "averagepool_quantized_test.cc",
    ],
    shard_count = 1,
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "maxpool_quantized_test",
    timeout = "long",
    srcs = [
        "maxpool_quantized_test.cc",
    ],
    shard_count = 1,
    deps = [
        ":optimized_base",
        ":reference_base",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "log_quantized_test",
    srcs = ["log_quantized_test.cc"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":optimized_base",
        ":reference_base",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpu_check",
    srcs = ["optimized/cpu_check.cc"],
    hdrs = [
        "optimized/cpu_check.h",
        "optimized/neon_check.h",
        "optimized/sse_check.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = select({
        ":x86_any": ["@arm_neon_2_x86_sse"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_test(
    name = "batch_to_space_nd_test",
    srcs = ["batch_to_space_nd_test.cc"],
    deps = [
        ":optimized_base",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "non_max_suppression_test",
    srcs = ["non_max_suppression_test.cc"],
    deps = [
        ":reference_base",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "per_channel_dequantize_test",
    srcs = ["per_channel_dequantize_test.cc"],
    deps = [
        ":reference_base",
        ":tensor_utils_no_eigen",
        ":types",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "per_channel_quantize_test",
    srcs = ["per_channel_quantize_test.cc"],
    deps = [
        ":reference_base",
        ":types",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "avx2_quantization_utils_test",
    srcs = ["optimized/avx2_quantization_utils_test.cc"],
    copts = select(
        {
            ":x86_64": [
                "-mavx2",
            ],
            "//conditions:default": [
            ],
        },
    ),
    deps = [
        ":avx2_quantization_utils",
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "reduce_utils_test",
    srcs = ["optimized/reduce_utils_test.cc"],
    deps = [
        ":common",
        ":reduce_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "runtime_shape_test",
    srcs = ["runtime_shape_test.cc"],
    deps = [
        ":types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "optimized_op_headers",
    srcs = glob([
        "optimized/*.h",
    ]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "reference_op_headers",
    srcs = glob([
        "reference/*.h",
    ]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "headers",
    srcs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:private"],
)

transitive_hdrs(
    name = "nnapi_external_headers",
    visibility = ["//visibility:private"],
    deps = [
        "@eigen_archive//:eigen3",
        "@gemmlowp",
    ],
)

# ---------------------------------------------------------
# The public target "install_nnapi_extra_headers" is only
# used for external targets that requires exporting optmized
# and reference op headers.

genrule(
    name = "install_nnapi_extra_headers",
    srcs = [
        ":nnapi_external_headers",
        ":headers",
        ":optimized_op_headers",
        ":reference_op_headers",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out*genfiles/}"
      d="$${d#*external/eigen_archive/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi
      fi

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:private"],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": ["@com_google_googletest//:gtest_main"]})
