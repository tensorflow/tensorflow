# tf.data service client library.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data/service:common_proto_cc",
    ],
)

cc_library(
    name = "validate_utils",
    srcs = ["validate_utils.cc"],
    hdrs = ["validate_utils.h"],
    deps = [
        ":common",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data/service:common",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:worker_impl",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

tf_cc_test(
    name = "validate_utils_test",
    srcs = ["validate_utils_test.cc"],
    deps = [
        ":common",
        ":validate_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:worker_impl",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
    ],
)
