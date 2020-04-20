# Experimental s3 filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for AWS S3.
tf_cc_shared_object(
    name = "libs3_filesystem.so",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [":s3_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "s3_filesystem_impl",
    srcs = ["s3_filesystem.cc"],
    hdrs = ["s3_filesystem.h"],
    deps = [
        ":s3_copy",
        ":s3_helper",
        ":s3_shared",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)

cc_library(
    name = "aws_crypto",
    srcs = [
        "aws_crypto.cc",
    ],
    hdrs = [
        "aws_crypto.h",
    ],
    deps = [
        "@aws",
        "@boringssl//:crypto",
    ],
)

cc_library(
    name = "s3_helper",
    srcs = [
        "s3_helper.cc",
    ],
    hdrs = [
        "s3_helper.h",
    ],
    deps = [
        "//tensorflow/c:tf_status",
        "@aws",
    ],
)

cc_library(
    name = "s3_copy",
    srcs = [
        "s3_copy.cc",
    ],
    hdrs = [
        "s3_copy.h",
    ],
    deps = [
        "@aws",
        ":s3_helper",
    ]
)

cc_library(
    name = "s3_shared",
    srcs = [
        "s3_shared.cc",
    ],
    hdrs = [
        "s3_shared.h",
    ],
    deps = [
        "@aws",
        ":aws_crypto",
        ":s3_helper",
    ]
)
