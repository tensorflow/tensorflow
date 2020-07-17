# Experimental gcs filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for GCS environments
tf_cc_shared_object(
    name = "s3_filesystem",
    framework_so = [],
    linkstatic = False,
    per_os_targets = 1,
    visibility = ["//visibility:public"],
    deps = [":s3_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "s3_filesystem_impl",
    srcs = ["s3_filesystem.cc"],
    hdrs = ["s3_filesystem.h"],
    copts = select({
        "//conditions:default": [],
        "//tensorflow:windows": get_win_copts(),
    }),
    deps = [
        ":aws_crypto",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@aws",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "aws_crypto",
    srcs = ["aws_crypto.cc"],
    hdrs = ["aws_crypto.h"],
    deps = [
        "@aws",
        "@boringssl//:crypto",
    ],
    alwayslink = 1,
)
