load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
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
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:sendrecv_ops_op_lib",
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
