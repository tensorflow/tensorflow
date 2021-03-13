# Library for generating feature vectors from audio data
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
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

cc_test(
    name = "fft_test",
    srcs = ["fft_test.cc"],
    deps = [
        ":fft",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "filterbank_test",
    srcs = ["filterbank_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":filterbank",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "frontend_test",
    srcs = ["frontend_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":frontend",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "log_scale_test",
    srcs = ["log_scale_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":log_scale",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "noise_reduction_test",
    srcs = ["noise_reduction_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":noise_reduction",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "pcan_gain_control_test",
    srcs = ["pcan_gain_control_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":pcan_gain_control",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_test(
    name = "window_test",
    srcs = ["window_test.cc"],
    # Setting copts for experimental code to [], but this code should be fixed
    # to build with the default copts (micro_copts())
    copts = [],
    deps = [
        ":window",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
