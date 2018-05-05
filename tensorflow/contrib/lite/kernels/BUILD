package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/contrib/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

tf_cc_test(
    name = "optional_tensor_test",
    size = "small",
    srcs = ["optional_tensor_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels/internal:tensor_utils",
        "//tensorflow/contrib/lite/testing:util",
        "//tensorflow/core:tflite_portable_logging",
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
    copts = tflite_copts(),
    deps = [
        ":op_macros",
        "//tensorflow/contrib/lite:context",
        "//third_party/eigen3",
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
        "//tensorflow/contrib/lite:context",
        "@gemmlowp",
    ],
)

cc_library(
    name = "activation_functor",
    hdrs = [
        "activation_functor.h",
    ],
    deps = [
        "//tensorflow/contrib/lite:builtin_op_data",
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
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:context",
        "//tensorflow/contrib/lite/kernels/internal:round",
    ],
)

tf_cc_test(
    name = "kernel_util_test",
    size = "small",
    srcs = ["kernel_util_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":kernel_util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "builtin_ops",
    srcs = [
        "activations.cc",
        "add.cc",
        "arg_max.cc",
        "audio_spectrogram.cc",
        "basic_rnn.cc",
        "batch_to_space_nd.cc",
        "bidirectional_sequence_lstm.cc",
        "bidirectional_sequence_rnn.cc",
        "cast.cc",
        "comparisons.cc",
        "concatenation.cc",
        "conv.cc",
        "depthwise_conv.cc",
        "dequantize.cc",
        "div.cc",
        "embedding_lookup.cc",
        "embedding_lookup_sparse.cc",
        "exp.cc",
        "floor.cc",
        "fully_connected.cc",
        "gather.cc",
        "hashtable_lookup.cc",
        "l2norm.cc",
        "local_response_norm.cc",
        "lsh_projection.cc",
        "lstm.cc",
        "maximum_minimum.cc",
        "mean.cc",
        "mfcc.cc",
        "mul.cc",
        "neg.cc",
        "pad.cc",
        "pooling.cc",
        "register.cc",
        "reshape.cc",
        "resize_bilinear.cc",
        "skip_gram.cc",
        "space_to_batch_nd.cc",
        "space_to_depth.cc",
        "split.cc",
        "squeeze.cc",
        "strided_slice.cc",
        "sub.cc",
        "svdf.cc",
        "topk_v2.cc",
        "transpose.cc",
        "unidirectional_sequence_lstm.cc",
        "unidirectional_sequence_rnn.cc",
    ],
    hdrs = [
        "padding.h",
        "register.h",
    ],
    # Suppress warnings that are introduced by Eigen Tensor.
    copts = tflite_copts() + [
        "-Wno-error=reorder",
    ] + select({
        "//tensorflow:ios": ["-Wno-error=invalid-partial-specialization"],
        "//conditions:default": [
        ],
    }),
    deps = [
        ":activation_functor",
        ":eigen_support",
        ":kernel_util",
        ":op_macros",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:gemm_support",
        "//tensorflow/contrib/lite/kernels/internal:audio_utils",
        "//tensorflow/contrib/lite/kernels/internal:kernel_utils",
        "//tensorflow/contrib/lite/kernels/internal:optimized",
        "//tensorflow/contrib/lite/kernels/internal:optimized_base",
        "//tensorflow/contrib/lite/kernels/internal:quantization_util",
        "//tensorflow/contrib/lite/kernels/internal:reference",
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
        "//tensorflow/contrib/lite/kernels/internal:tensor_utils",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "audio_spectrogram_test",
    size = "small",
    srcs = ["audio_spectrogram_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "mfcc_test",
    size = "small",
    srcs = ["mfcc_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "arg_max_test",
    size = "small",
    srcs = ["arg_max_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "div_test",
    size = "small",
    srcs = ["div_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "sub_test",
    size = "small",
    srcs = ["sub_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "transpose_test",
    size = "small",
    srcs = ["transpose_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "//tensorflow/contrib/lite/kernels/internal:reference",
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "space_to_batch_nd_test",
    size = "small",
    srcs = ["space_to_batch_nd_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "batch_to_space_nd_test",
    size = "small",
    srcs = ["batch_to_space_nd_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "cast_test",
    size = "small",
    srcs = ["cast_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "dequantize_test",
    size = "small",
    srcs = ["dequantize_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "bidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["bidirectional_sequence_lstm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "floor_test",
    size = "small",
    srcs = ["floor_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "unidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["unidirectional_sequence_lstm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "unidirectional_sequence_rnn_test",
    size = "small",
    srcs = ["unidirectional_sequence_rnn_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "exp_test",
    size = "small",
    srcs = ["exp_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "maximum_minimum_test",
    size = "small",
    srcs = ["maximum_minimum_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "mean_test",
    size = "small",
    srcs = ["mean_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "pad_test",
    size = "small",
    srcs = ["pad_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "gather_test",
    size = "small",
    srcs = ["gather_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "topk_v2_test",
    size = "small",
    srcs = ["topk_v2_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "//tensorflow/contrib/lite/kernels/internal:tensor_utils",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "log_softmax_test",
    size = "small",
    srcs = ["log_softmax_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "lstm_test",
    size = "small",
    srcs = ["lstm_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "split_test",
    size = "small",
    srcs = ["split_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "squeeze_test",
    size = "small",
    srcs = ["squeeze_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "strided_slice_test",
    size = "small",
    srcs = ["strided_slice_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_test(
    name = "neg_test",
    size = "small",
    srcs = ["neg_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":builtin_ops",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:test_util",
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
