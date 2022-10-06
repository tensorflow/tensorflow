# Data service test data.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = glob(["*.pbtxt"]),
    visibility = ["//tensorflow/core/data/service:__pkg__"],
)
