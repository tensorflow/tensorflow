load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        ":input_generator",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/testing:split",
        "//tensorflow/lite/testing:tflite_driver",
    ] + select({
        "//conditions:default": [
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
        ],
    }),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/test_input.csv",
    ],
    deps = [
        ":util",
        "//tensorflow/lite/testing:tflite_driver",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_binary(
    name = "tflite_kernel_runner",
    srcs = ["tflite_kernel_runner.cc"],
    deps = [
        ":util",
    ],
)

tf_cc_binary(
    name = "generate_diff_report",
    srcs = ["generate_diff_report.cc"],
    deps = [
        ":diff_analyzer",
        "//tensorflow/core:framework_internal",
    ],
)

cc_library(
    name = "input_generator",
    srcs = ["input_generator.cc"],
    hdrs = ["input_generator.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:join",
        "//tensorflow/lite/testing:split",
    ],
)

cc_test(
    name = "input_generator_test",
    size = "small",
    srcs = ["input_generator_test.cc"],
    data = [
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/test_input.csv",
    ],
    deps = [
        ":input_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "diff_analyzer",
    srcs = ["diff_analyzer.cc"],
    hdrs = ["diff_analyzer.h"],
    deps = [
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/testing:split",
    ],
)

tf_cc_test(
    name = "diff_analyzer_test",
    size = "small",
    srcs = ["diff_analyzer_test.cc"],
    data = [
        "//tensorflow/lite:testdata/test_input.csv",
    ],
    deps = [
        ":diff_analyzer",
        "//tensorflow/core:lib",
        "@com_google_googletest//:gtest_main",
    ],
)
