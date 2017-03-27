licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

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

filegroup(
    name = "android_srcs",
    srcs = [
        "devices.cc",
        "devices.h",
        "grappler_item.h",
        "utils.cc",
        "utils.h",
        "//tensorflow/core/grappler/clusters:android_srcs",
        "//tensorflow/core/grappler/inputs:android_srcs",
        "//tensorflow/core/grappler/optimizers:android_srcs",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "devices",
    srcs = ["devices.cc"],
    hdrs = ["devices.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:gpu_init",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
    ],
)

cc_library(
    name = "grappler_item",
    srcs = [
        "grappler_item.cc",
    ],
    hdrs = ["grappler_item.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "grappler_item_builder",
    srcs = [
        "grappler_item_builder.cc",
    ],
    hdrs = ["grappler_item_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":grappler_item",
        ":utils",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/inputs:utils",
    ],
)

cc_test(
    name = "grappler_item_test",
    srcs = ["grappler_item_test.cc"],
    deps = [
        ":grappler_item",
        ":grappler_item_builder",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
