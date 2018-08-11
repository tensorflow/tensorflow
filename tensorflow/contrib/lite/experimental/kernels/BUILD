package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts")
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
        "//tensorflow/contrib/lite/kernels/internal:types",
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
        "//tensorflow/contrib/lite/kernels/internal:types",
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
        "//tensorflow/contrib/lite:builtin_op_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/kernels:gemm_support",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/contrib/lite/kernels:op_macros",
        "//tensorflow/contrib/lite/kernels/internal:kernel_utils",
        "//tensorflow/contrib/lite/kernels/internal:optimized",
        "//tensorflow/contrib/lite/kernels/internal:optimized_base",
        "//tensorflow/contrib/lite/kernels/internal:quantization_util",
        "//tensorflow/contrib/lite/kernels/internal:reference",
        "//tensorflow/contrib/lite/kernels/internal:reference_base",
        "//tensorflow/contrib/lite/kernels/internal:tensor_utils",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
