licenses(["notice"])  # Apache 2.0

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
    srcs = glob(
        [
            "utils.*",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/kernels:aggregate_ops",
        "//tensorflow/core/kernels:array",
    ],
)
