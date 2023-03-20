load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_portable_full_lite_protos")

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
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform:types",
    ] + tf_portable_full_lite_protos(
        [":activity_watcher_impl"],
        [],
    ),
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
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
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
    ],
)
