load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental windows filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for Windows environment
tf_cc_shared_object(
    name = "windows_filesystem.dll",
    framework_so = [],
    linkstatic = False,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [":windows_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "windows_filesystem_impl",
    srcs = ["windows_filesystem.cc"],
    copts = get_win_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)
