package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

tf_cc_test(
    name = "resolve_constant_concatenation_test",
    srcs = ["resolve_constant_concatenation_test.cc"],
    deps = [
        "//tensorflow/contrib/lite/toco:graph_transformations",
        "//tensorflow/contrib/lite/toco:model",
        "//tensorflow/contrib/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
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
