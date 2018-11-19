package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

tf_cc_test(
    name = "lstm_utils_test",
    srcs = ["lstm_utils_test.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_concatenation_test",
    srcs = ["resolve_constant_concatenation_test.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_unary_test",
    srcs = ["resolve_constant_unary_test.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)
