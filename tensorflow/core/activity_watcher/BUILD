load("//tensorflow:tensorflow.bzl", "if_not_mobile")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "activity_watcher_headers",
    srcs = [
        "activity.h",
        "activity_utils.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "activity_watcher",
    hdrs = ["activity.h"],
    defines = if_not_mobile(["TF_ENABLE_ACTIVITY_WATCHER"]),
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_xla//xla/tsl/platform:types",
    ] + if_not_mobile([
        ":activity_watcher_impl",
    ]),
    alwayslink = True,
)

cc_library(
    name = "activity_watcher_impl",
    srcs = [
        "activity.cc",
    ],
    hdrs = ["activity.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_xla//xla/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "activity_watcher_utils",
    srcs = ["activity_utils.cc"],
    hdrs = ["activity_utils.h"],
    deps = [
        ":activity_watcher",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:types",
    ],
)
