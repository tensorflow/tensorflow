package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_opts_nortti_if_android")

# Suppress warnings that are introduced by Eigen Tensor.
EXTRA_EIGEN_COPTS = select({
    "//tensorflow:ios": [
        "-Wno-error=invalid-partial-specialization",
        "-Wno-error=reorder",
    ],
    "//tensorflow:windows": [
        "/DEIGEN_HAS_C99_MATH",
        "/DEIGEN_AVOID_STL_ARRAY",
    ],
    "//conditions:default": ["-Wno-error=reorder"],
})

tf_cc_test(
    name = "optional_tensor_test",
    size = "small",
    srcs = ["optional_tensor_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":builtin_ops",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "eigen_support",
    srcs = [
        "eigen_support.cc",
    ],
    hdrs = [
        "eigen_support.h",
    ],
    copts = tflite_copts() + EXTRA_EIGEN_COPTS,
    deps = [
        ":op_macros",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:optimized",
    ],
)

cc_library(
    name = "gemm_support",
    srcs = [
        "gemm_support.cc",
    ],
    hdrs = [
        "gemm_support.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":op_macros",
        "//tensorflow/lite/c:c_api_internal",
        "@gemmlowp",
    ],
)

cc_library(
    name = "activation_functor",
    hdrs = [
        "activation_functor.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_library(
    name = "op_macros",
    hdrs = [
        "op_macros.h",
    ],
)

cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = [
        "kernel_util.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:round",
        "//tensorflow/lite/kernels/internal:types",
    ],
)

tf_cc_test(
    name = "kernel_util_test",
    size = "small",
    srcs = ["kernel_util_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":kernel_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
    ],
)

cc_library(
    name = "builtin_op_kernels",
    srcs = [
        "activations.cc",
        "add.cc",
        "arg_min_max.cc",
        "audio_spectrogram.cc",
        "basic_rnn.cc",
        "batch_to_space_nd.cc",
        "bidirectional_sequence_lstm.cc",
        "bidirectional_sequence_rnn.cc",
        "cast.cc",
        "ceil.cc",
        "comparisons.cc",
        "concatenation.cc",
        "conv.cc",
        "depthwise_conv.cc",
        "dequantize.cc",
        "detection_postprocess.cc",
        "div.cc",
        "elementwise.cc",
        "embedding_lookup.cc",
        "embedding_lookup_sparse.cc",
        "exp.cc",
        "expand_dims.cc",
        "fake_quant.cc",
        "fill.cc",
        "floor.cc",
        "floor_div.cc",
        "floor_mod.cc",
        "fully_connected.cc",
        "gather.cc",
        "hashtable_lookup.cc",
        "l2norm.cc",
        "layer_norm_lstm.cc",
        "local_response_norm.cc",
        "logical.cc",
        "lsh_projection.cc",
        "lstm.cc",
        "maximum_minimum.cc",
        "mfcc.cc",
        "mirror_pad.cc",
        "mul.cc",
        "neg.cc",
        "one_hot.cc",
        "pack.cc",
        "pad.cc",
        "pooling.cc",
        "pow.cc",
        "range.cc",
        "reduce.cc",
        "relu1.cc",
        "reshape.cc",
        "resize_bilinear.cc",
        "resize_nearest_neighbor.cc",
        "select.cc",
        "shape.cc",
        "skip_gram.cc",
        "slice.cc",
        "space_to_batch_nd.cc",
        "space_to_depth.cc",
        "sparse_output_fully_connected.cc",
        "sparse_to_dense.cc",
        "split.cc",
        "split_v.cc",
        "squared_difference.cc",
        "squeeze.cc",
        "strided_slice.cc",
        "sub.cc",
        "svdf.cc",
        "tile.cc",
        "topk_v2.cc",
        "transpose.cc",
        "transpose_conv.cc",
        "unidirectional_sequence_lstm.cc",
        "unidirectional_sequence_rnn.cc",
        "unpack.cc",
        "zeros_like.cc",
    ],
    hdrs = [
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android() + EXTRA_EIGEN_COPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":activation_functor",
        ":eigen_support",
        ":kernel_util",
        ":lstm_eval",
        ":op_macros",
        ":padding",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:gemm_support",
        "//tensorflow/lite/kernels/internal:audio_utils",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/kernels/internal:optimized",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_library(
    name = "lstm_eval",
    srcs = ["lstm_eval.cc"],
    hdrs = ["lstm_eval.h"],
    deps = [
        ":op_macros",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "builtin_ops",
    srcs = ["register.cc"],
    hdrs = ["register.h"],
    deps = [
        ":builtin_op_kernels",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_internal",
    ],
)

tf_cc_test(
    name = "audio_spectrogram_test",
    size = "small",
    srcs = ["audio_spectrogram_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "mfcc_test",
    size = "small",
    srcs = ["mfcc_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "detection_postprocess_test",
    size = "small",
    srcs = ["detection_postprocess_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "relu1_test",
    size = "small",
    srcs = ["relu1_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "sparse_output_fully_connected_test",
    size = "small",
    srcs = ["sparse_output_fully_connected_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "add_test",
    size = "small",
    srcs = ["add_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "arg_min_max_test",
    size = "small",
    srcs = ["arg_min_max_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "div_test",
    size = "small",
    srcs = ["div_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "sub_test",
    size = "small",
    srcs = ["sub_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "transpose_test",
    size = "small",
    srcs = ["transpose_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/kernels/internal:reference_base",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "space_to_batch_nd_test",
    size = "small",
    srcs = ["space_to_batch_nd_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "batch_to_space_nd_test",
    size = "small",
    srcs = ["batch_to_space_nd_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "cast_test",
    size = "small",
    srcs = ["cast_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "concatenation_test",
    size = "small",
    srcs = ["concatenation_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "conv_test",
    size = "small",
    srcs = ["conv_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "depthwise_conv_test",
    size = "small",
    srcs = ["depthwise_conv_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "dequantize_test",
    size = "small",
    srcs = ["dequantize_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "basic_rnn_test",
    size = "small",
    srcs = ["basic_rnn_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "bidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["bidirectional_sequence_lstm_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "floor_test",
    size = "small",
    srcs = ["floor_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "ceil_test",
    size = "small",
    srcs = ["ceil_test.cc"],
    tags = [
        "no_oss",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "elementwise_test",
    size = "small",
    srcs = ["elementwise_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "unidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["unidirectional_sequence_lstm_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "bidirectional_sequence_rnn_test",
    size = "small",
    srcs = ["bidirectional_sequence_rnn_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "unidirectional_sequence_rnn_test",
    size = "small",
    srcs = ["unidirectional_sequence_rnn_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "l2norm_test",
    size = "small",
    srcs = ["l2norm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "exp_test",
    size = "small",
    srcs = ["exp_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "fake_quant_test",
    size = "small",
    srcs = ["fake_quant_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "maximum_minimum_test",
    size = "small",
    srcs = ["maximum_minimum_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "reduce_test",
    size = "small",
    srcs = ["reduce_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "mul_test",
    size = "small",
    srcs = ["mul_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "pad_test",
    size = "small",
    srcs = ["pad_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "reshape_test",
    size = "small",
    srcs = ["reshape_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "gather_test",
    size = "small",
    srcs = ["gather_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "topk_v2_test",
    size = "small",
    srcs = ["topk_v2_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "resize_bilinear_test",
    size = "small",
    srcs = ["resize_bilinear_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "resize_nearest_neighbor_test",
    size = "small",
    srcs = ["resize_nearest_neighbor_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "svdf_test",
    size = "small",
    srcs = ["svdf_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "embedding_lookup_test",
    size = "small",
    srcs = ["embedding_lookup_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "embedding_lookup_sparse_test",
    size = "small",
    srcs = ["embedding_lookup_sparse_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "fully_connected_test",
    size = "small",
    srcs = ["fully_connected_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "local_response_norm_test",
    size = "small",
    srcs = ["local_response_norm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "pooling_test",
    size = "small",
    srcs = ["pooling_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "softmax_test",
    size = "small",
    srcs = ["softmax_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "log_softmax_test",
    size = "small",
    srcs = ["log_softmax_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "lsh_projection_test",
    size = "small",
    srcs = ["lsh_projection_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "hashtable_lookup_test",
    size = "small",
    srcs = ["hashtable_lookup_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "layer_norm_lstm_test",
    size = "small",
    srcs = ["layer_norm_lstm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "lstm_test",
    size = "small",
    srcs = ["lstm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "skip_gram_test",
    size = "small",
    srcs = ["skip_gram_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "space_to_depth_test",
    size = "small",
    srcs = ["space_to_depth_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "split_test",
    size = "small",
    srcs = ["split_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "split_v_test",
    size = "small",
    srcs = ["split_v_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "squeeze_test",
    size = "small",
    srcs = ["squeeze_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "strided_slice_test",
    size = "small",
    srcs = ["strided_slice_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "tile_test",
    size = "small",
    srcs = ["tile_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "comparisons_test",
    size = "small",
    srcs = [
        "comparisons_test.cc",
    ],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "neg_test",
    size = "small",
    srcs = ["neg_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "select_test",
    size = "small",
    srcs = [
        "select_test.cc",
    ],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "slice_test",
    size = "small",
    srcs = [
        "slice_test.cc",
    ],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "transpose_conv_test",
    size = "small",
    srcs = ["transpose_conv_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "expand_dims_test",
    size = "small",
    srcs = ["expand_dims_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "sparse_to_dense_test",
    size = "small",
    srcs = ["sparse_to_dense_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "shape_test",
    size = "small",
    srcs = ["shape_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "pow_test",
    size = "small",
    srcs = ["pow_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "pack_test",
    size = "small",
    srcs = ["pack_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "one_hot_test",
    size = "small",
    srcs = ["one_hot_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "logical_test",
    size = "small",
    srcs = ["logical_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "unpack_test",
    size = "small",
    srcs = ["unpack_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "floor_div_test",
    size = "small",
    srcs = ["floor_div_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "zeros_like_test",
    size = "small",
    srcs = ["zeros_like_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "floor_mod_test",
    size = "small",
    srcs = ["floor_mod_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "range_test",
    size = "small",
    srcs = ["range_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:builtin_op_data",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "squared_difference_test",
    size = "small",
    srcs = ["squared_difference_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "fill_test",
    size = "small",
    srcs = ["fill_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
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

tflite_portable_test_suite()

tf_cc_test(
    name = "mirror_pad_test",
    srcs = ["mirror_pad_test.cc"],
    deps = [
        ":builtin_ops",
        ":test_util",
        "//tensorflow/lite:framework",
        "@com_google_googletest//:gtest_main",
    ],
)
