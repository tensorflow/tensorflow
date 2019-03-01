package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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
        "//tensorflow/lite/kernels/internal:types",
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
        "//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "experimental_ops",
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
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "//tensorflow/lite/kernels/internal:optimized",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "ctc_beam_search_decoder_test",
    size = "small",
    srcs = ["ctc_beam_search_decoder_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":experimental_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
