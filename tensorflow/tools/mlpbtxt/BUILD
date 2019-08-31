# Description:
# This package provides binaries that convert between multi-line and standard
# pbtxt (text-serialization of protocol message) files.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
    "placeholder.txt",
])

tf_cc_binary(
    name = "tomlpbtxt",
    srcs = ["tomlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
    ],
)

tf_cc_binary(
    name = "frommlpbtxt",
    srcs = ["frommlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
    ],
)
