load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["logging.h"])

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = common_copts,
)

cc_binary(
    name = "hello_systemc",
    srcs = [
        "hello_main.sc.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        #":logging",
        "@systemc//:systemc",
    ],
)

cc_binary(
    name = "hello_channel",
    srcs = [
        "channel_main.sc.cc",
        "channel/stack_if.sc.h",
        "channel/stack.sc.h",
        "channel/producer.sc.h",
        "channel/consumer.sc.h",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@systemc//:systemc",
    ],
)

cc_test(
    name = "sc_example_test",
    srcs = ["sc_gtest_test.sc.cc"],
    args = [
        "",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/algorithm",
        "@com_google_googletest//:gtest",
        "@systemc//:systemc",
    ],
)

tflite_portable_test_suite()
