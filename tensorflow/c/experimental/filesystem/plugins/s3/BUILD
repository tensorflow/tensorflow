# Experimental s3 filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object", "tf_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for S3 environments
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
        ":aws_logging",
        "//tensorflow/c:logging",
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

cc_library(
    name = "aws_logging",
    srcs = ["aws_logging.cc"],
    hdrs = ["aws_logging.h"],
    deps = [
        "//tensorflow/c:logging",
        "@aws",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "s3_filesystem_test",
    srcs = [
        "s3_filesystem_test.cc",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":s3_filesystem_impl",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:test",
    ],
)
