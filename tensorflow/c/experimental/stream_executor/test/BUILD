# Description:
# test for stream_executor
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
)

cc_library(
    name = "test_pluggable_device_impl",
    srcs = ["test_pluggable_device.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/experimental/stream_executor:stream_executor.h",
    ],
)

tf_custom_op_library(
    name = "test_pluggable_device.so",
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [":test_pluggable_device_impl"],
)
