load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "if_not_android")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "activity_watcher_headers",
    srcs = ["activity.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "activity_watcher",
    hdrs = [":activity_watcher_headers"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core:lib",
    ] + if_not_android([
        ":activity_watcher_impl",
    ]),
    alwayslink = True,
)

cc_library(
    name = "activity_watcher_impl",
    srcs = [
        "activity.cc",
    ],
    hdrs = [":activity_watcher_headers"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)
