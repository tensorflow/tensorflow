# Experimental gcs filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for GCS environments
tf_cc_shared_object(
    name = "gcs_filesystem",
    framework_so = [],
    linkstatic = False,
    per_os_targets = 1,
    visibility = ["//visibility:public"],
    deps = [":gcs_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "gcs_filesystem_impl",
    srcs = ["gcs_filesystem.cc"],
    copts = select({
        "//conditions:default": [],
        "//tensorflow:windows": get_win_copts(),
    }),
    deps = [
        ":gcs_helper",
        "//tensorflow/c:env",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@com_github_googlecloudplatform_google_cloud_cpp//:storage_client",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gcs_helper",
    srcs = ["gcs_helper.cc"],
    hdrs = ["gcs_helper.h"],
    linkstatic = 1,
    deps = [
        "//tensorflow/c:env",
    ],
)
