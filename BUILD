# Description:
# TensorFlow is an open source software library for numerical computation using
# data flow graphs.

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "cc_header_only_library",
)

cc_header_only_library(
    name = "protobuf_headers",
    includes = ["external/protobuf/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:protobuf",
    ],
)
