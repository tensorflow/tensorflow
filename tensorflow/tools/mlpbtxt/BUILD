# Description:
# This package provides binaries that convert between multi-line and standard
# pbtxt (text-serialization of protocol message) files.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

exports_files(["placeholder.txt"])

tf_cc_binary(
    name = "tomlpbtxt",
    srcs = ["tomlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "frommlpbtxt",
    srcs = ["frommlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)
