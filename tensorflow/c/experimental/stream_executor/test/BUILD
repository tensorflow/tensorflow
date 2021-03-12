# Description:
# test for stream_executor
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cc_shared_object(
    name = "test_pluggable_device.so",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = ["//tensorflow/c/experimental/stream_executor:stream_executor_hdrs"],
)
