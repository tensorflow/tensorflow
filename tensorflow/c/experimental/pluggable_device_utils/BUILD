#Description
# PluggableDevice Utils

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

cc_library(
    name = "pluggable_device_utils",
    srcs = ["pluggable_device_utils.cc"],
    hdrs = ["pluggable_device_utils.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)
