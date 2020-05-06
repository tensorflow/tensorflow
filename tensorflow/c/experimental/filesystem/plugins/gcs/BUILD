# Experimental GCS filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for GCS environments
tf_cc_shared_object(
    name = "libgcs_filesystem.so",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [":gcs_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "gcs_filesystem_impl",
    srcs = ["gcs_filesystem.cc"],
    hdrs = ["gcs_filesystem.h"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@com_github_googleapis_google_cloud_cpp//:storage_client",
    ],
)
