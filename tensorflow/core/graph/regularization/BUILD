# Description:
# A TensorFlow Graph Regularization Library

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "test_files",
    srcs = glob([
        "testdata/bert2/**",
        "testdata/bert1/**",
    ]),
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simple_delete",
    srcs = ["simple_delete.cc"],
    hdrs = ["simple_delete.h"],
    deps = [
        ":util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
    ],
)

tf_cc_test(
    name = "simple_delete_test",
    size = "small",
    srcs = [
        "simple_delete_test.cc",
    ],
    data = [
        ":test_files",
    ],
    deps = [
        ":simple_delete",
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
