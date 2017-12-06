package(default_visibility = [
    "//tensorflow:internal",
    "//tensorflow_models:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_kernel_library")

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

filegroup(
    name = "test_data",
    srcs = glob([
        "testdata/*.txt",
        "testdata/*.seq",
    ]),
)

tf_cc_test(
    name = "reader_test",
    size = "small",
    srcs = ["reader_test.cc"],
    data = [":test_data"],
    deps = [
        ":reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
