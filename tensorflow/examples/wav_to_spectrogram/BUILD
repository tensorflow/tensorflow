# Description:
#   TensorFlow C++ inference example for labeling images.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "wav_to_spectrogram_lib",
    srcs = ["wav_to_spectrogram.cc"],
    hdrs = ["wav_to_spectrogram.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "wav_to_spectrogram",
    srcs = ["main.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "wav_to_spectrogram_test",
    size = "medium",
    srcs = ["wav_to_spectrogram_test.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
