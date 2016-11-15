# Description:
#   Utilities that perform useful transformations on graphs

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cc_test",
)

exports_files(["LICENSE"])

cc_library(
    name = "transform_utils",
    srcs = [
        "transform_utils.cc",
    ],
    hdrs = [
        "transform_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "transform_utils_test",
    size = "small",
    srcs = ["transform_utils_test.cc"],
    deps = [
        ":transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "fold_constants_lib",
    srcs = [
        "fold_constants_lib.cc",
    ],
    hdrs = [
        "fold_constants_lib.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "fold_constants_test",
    size = "small",
    srcs = ["fold_constants_test.cc"],
    deps = [
        ":fold_constants_lib",
        ":transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# This library includes a main function, to make it easy to create other
# versions of the tool linked against different operator libs.
cc_library(
    name = "fold_constants_main_lib",
    srcs = ["fold_constants_tool.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":fold_constants_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "fold_constants_tool",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fold_constants_main_lib",
    ],
)
