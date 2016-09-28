# Description:
# TensorFlow SavedModel.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "loader",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":constants",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "loader_test",
    srcs = ["loader_test.cc"],
    data = [
        ":saved_model_half_plus_two",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

filegroup(
    name = "saved_model_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_sharded/**",
    ]),
)

# -----------------------------------------------------------------------------
# Google-internal targets.

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
