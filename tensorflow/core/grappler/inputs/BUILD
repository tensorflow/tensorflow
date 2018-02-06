licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "input_yielder",
    hdrs = [
        "input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "trivial_test_graph_input_yielder",
    srcs = ["trivial_test_graph_input_yielder.cc"],
    hdrs = [
        "trivial_test_graph_input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":input_yielder",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/kernels:aggregate_ops",
    ],
)

cc_library(
    name = "file_input_yielder",
    srcs = ["file_input_yielder.cc"],
    hdrs = [
        "file_input_yielder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":input_yielder",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler:utils",
    ],
)
