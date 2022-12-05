# Description: CTC, Connectionist Temporal Classification,
# is a type of seq2seq loss.  The libraries in this directory
# implement the CTC loss and a number of CTC decoders.

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

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

cc_library(
    name = "ctc",
    deps = [
        ":ctc_beam_search_lib",
        ":ctc_loss_calculator_lib",
    ],
)

cc_library(
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

cc_library(
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

cc_library(
    name = "ctc_loss_util_lib",
    hdrs = [
        "ctc_loss_util.h",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = ["ctc_beam_search.h"],
    visibility = ["//visibility:public"],
)
