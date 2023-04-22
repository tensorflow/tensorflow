# Description:
# Test for keeping the history of OpDefs for every major version of TensorFlow,
# to validate that we don't make backwards-incompatible changes in particular
# for v2.

load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "ops_history_v2_srcs",
    srcs = glob([
        "*.pbtxt",
    ]),
    visibility = ["//tensorflow/core/ops/compat:__pkg__"],
)
