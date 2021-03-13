load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# ctc support classes imported directly from TensorFlow.
cc_library(
    name = "ctc_utils",
    hdrs = [
        "ctc_beam_entry.h",
        "ctc_beam_scorer.h",
        "ctc_beam_search.h",
        "ctc_decoder.h",
        "ctc_loss_util.h",
    ],
    deps = [
        ":top_n",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//third_party/eigen3",
    ],
)

# top_n support classes imported directly from TensorFlow.
cc_library(
    name = "top_n",
    hdrs = [
        "top_n.h",
    ],
    deps = [
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "ctc_beam_search_decoder_op",
    srcs = [
        "ctc_beam_search_decoder.cc",
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
        ":ctc_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

cc_test(
    name = "ctc_beam_search_decoder_test",
    size = "small",
    srcs = ["ctc_beam_search_decoder_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":ctc_beam_search_decoder_op",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "gru_cell",
    srcs = ["gru_cell.cc"],
    hdrs = ["gru_cell.h"],
    deps = [
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "unidirectional_sequence_gru_op",
    srcs = [
        "unidirectional_sequence_gru.cc",
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
        ":gru_cell",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

cc_test(
    name = "unidirectional_sequence_gru_test",
    size = "small",
    srcs = ["unidirectional_sequence_gru_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":unidirectional_sequence_gru_op",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)
