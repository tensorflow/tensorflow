# Description: CTC, Connectionist Temporal Classification,
# is a type of seq2seq loss.  The libraries in this directory
# implement the CTC loss and a number of CTC decoders.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_library")

filegroup(
    name = "mobile_srcs",
    srcs = [
        "ctc_beam_entry.h",
        "ctc_beam_scorer.h",
        "ctc_beam_search.h",
        "ctc_decoder.h",
        "ctc_loss_util.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
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

tf_cc_library(
    name = "ctc",
    deps = [
        ":ctc_beam_search_lib",
        ":ctc_loss_calculator_lib",
    ],
)

tf_cc_library(
    name = "ctc_beam_search_lib",
    srcs = [
        "ctc_beam_entry.h",
        "ctc_beam_scorer.h",
        "ctc_beam_search.h",
        "ctc_decoder.h",
    ],
    hdrs = [
        "ctc_beam_entry.h",
        "ctc_beam_scorer.h",
        "ctc_beam_search.h",
        "ctc_decoder.h",
    ],
    deps = [
        ":ctc_loss_util_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
    name = "ctc_beam_search_test",
    size = "small",
    srcs = [
        "ctc_beam_search_test.cc",
    ],
    deps = [
        ":ctc_beam_search_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_library(
    name = "ctc_loss_calculator_lib",
    srcs = [
        "ctc_loss_calculator.cc",
    ],
    hdrs = [
        "ctc_loss_calculator.h",
    ],
    deps = [
        ":ctc_loss_util_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_cc_library(
    name = "ctc_loss_util_lib",
    hdrs = [
        "ctc_loss_util.h",
    ],
)
