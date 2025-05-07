# Description:
# test for stream_executor
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_shared_object(
    name = "test_pluggable_device.so",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_test_util",
    ],
)

cc_library(
    name = "test_pluggable_device",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_test_util",
    ],
)
