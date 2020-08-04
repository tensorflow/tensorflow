# Experimental hadoop filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object")

package(
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for HADOOP environments
tf_cc_shared_object(
    name = "hadoop_filesystem",
    framework_so = [],
    linkstatic = False,
    per_os_targets = 1,
    visibility = ["//visibility:public"],
    deps = [":hadoop_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "hadoop_filesystem_impl",
    srcs = ["hadoop_filesystem.cc"],
    hdrs = ["hadoop_filesystem.h"],
    copts = select({
        "//conditions:default": [],
        "//tensorflow:windows": get_win_copts(),
    }),
    deps = [
        "//tensorflow/c:env",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "//third_party/hadoop:hdfs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)
