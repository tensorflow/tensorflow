load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Library for generating feature vectors from audio data
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "bits",
    hdrs = ["bits.h"],
)

cc_library(
    name = "fft",
    srcs = [
        "fft.cc",
        "fft_util.cc",
    ],
    hdrs = [
        "fft.h",
        "fft_util.h",
    ],
    deps = [
        "@kissfft//:kiss_fftr_16",
    ],
)

cc_library(
    name = "filterbank",
    srcs = [
        "filterbank.c",
        "filterbank_util.c",
    ],
    hdrs = [
        "filterbank.h",
        "filterbank_util.h",
    ],
    deps = [
        ":bits",
        ":fft",
    ],
)

cc_library(
    name = "frontend",
    srcs = [
        "frontend.c",
        "frontend_util.c",
    ],
    hdrs = [
        "frontend.h",
        "frontend_util.h",
    ],
    deps = [
        ":bits",
        ":fft",
        ":filterbank",
        ":log_scale",
        ":noise_reduction",
        ":pcan_gain_control",
        ":window",
    ],
)

cc_library(
    name = "log_scale",
    srcs = [
        "log_lut.c",
        "log_scale.c",
        "log_scale_util.c",
    ],
    hdrs = [
        "log_lut.h",
        "log_scale.h",
        "log_scale_util.h",
    ],
    deps = [
        ":bits",
    ],
)

cc_library(
    name = "noise_reduction",
    srcs = [
        "noise_reduction.c",
        "noise_reduction_util.c",
    ],
    hdrs = [
        "noise_reduction.h",
        "noise_reduction_util.h",
    ],
)

cc_library(
    name = "pcan_gain_control",
    srcs = [
        "pcan_gain_control.c",
        "pcan_gain_control_util.c",
    ],
    hdrs = [
        "pcan_gain_control.h",
        "pcan_gain_control_util.h",
    ],
    deps = [
        ":bits",
    ],
)

cc_library(
    name = "window",
    srcs = [
        "window.c",
        "window_util.c",
    ],
    hdrs = [
        "window.h",
        "window_util.h",
    ],
)
