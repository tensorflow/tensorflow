load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

# Currently pybind extension shared objects must use only C API headers since
# the C API has static initializers duplicated in the Python bindings. So we
# need a second rule that omits .cc files, in
# tensorflow/python:_pywrap_parallel_device.
filegroup(
    name = "headers",
    srcs = ["parallel_device.h"],
    visibility = ["//tensorflow/python:__pkg__"],
)

filegroup(
    name = "sources",
    srcs = ["parallel_device.cc"],
    visibility = ["//tensorflow/python:__pkg__"],
)

cc_library(
    name = "parallel_device",
    srcs = [":sources"],
    hdrs = [":headers"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.cc"],
    deps = [
        ":parallel_device",
        ":parallel_device_ops",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Note: ParallelDevice-specific ops are experimental and not currently linked in
# to TensorFlow by default, just used in a few tests.
filegroup(
    name = "parallel_device_ops_srcs",
    srcs = ["parallel_device_ops.cc"],
    visibility = ["//tensorflow/python/distribute/parallel_device:__pkg__"],
)

cc_library(
    name = "parallel_device_ops",
    srcs = [":parallel_device_ops_srcs"],
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)
