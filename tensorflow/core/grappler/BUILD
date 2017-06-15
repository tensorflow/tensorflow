licenses(["notice"])  # Apache 2.0

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

cc_library(
    name = "op_types",
    srcs = ["op_types.cc"],
    hdrs = ["op_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
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
    size = "small",
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
        ":op_types",
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
        ":op_types",
        ":utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/inputs:utils",
    ],
)

cc_test(
    name = "grappler_item_test",
    size = "small",
    srcs = ["grappler_item_test.cc"],
    deps = [
        ":grappler_item",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_test(
    name = "grappler_item_builder_test",
    size = "small",
    srcs = ["grappler_item_builder_test.cc"],
    deps = [
        ":grappler_item_builder",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:grad_testutil",
        "//tensorflow/cc:gradients",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
