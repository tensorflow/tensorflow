# Description:
# Data service test data.

load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "testdata",
    srcs = glob(["*.pbtxt"]),
    visibility = ["//tensorflow/core/data/service:__pkg__"],
)
