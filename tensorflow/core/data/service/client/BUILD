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
    name = "data_service_client",
    srcs = ["data_service_client.cc"],
    hdrs = ["data_service_client.h"],
    deps = [
        ":common",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "data_service_client_test",
    srcs = ["data_service_client_test.cc"],
    deps = [
        ":common",
        ":data_service_client",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
    ],
)
