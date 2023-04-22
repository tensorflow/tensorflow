load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
)

cc_library(
    name = "coordination_service",
    hdrs = ["coordination_service.h"],
    deps = [
        ":coordination_client",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "coordination_service_impl",
    deps = [
        ":coordination_client",
        ":coordination_service",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
        "coordination_service.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
