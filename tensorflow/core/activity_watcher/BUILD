load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "activity_watcher_headers",
    srcs = ["activity.h"],
)

cc_library(
    name = "activity_watcher",
    hdrs = [":activity_watcher_headers"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/activity_watcher:activity_watcher_impl",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
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
