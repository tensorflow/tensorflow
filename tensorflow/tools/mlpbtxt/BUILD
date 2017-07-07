# Description:
# This package provides binaries that convert between multi-line and standard
# pbtxt (text-serialization of protocol message) files.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "placeholder.txt",
])

cc_binary(
    name = "tomlpbtxt",
    srcs = ["tomlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
    ],
)

cc_binary(
    name = "frommlpbtxt",
    srcs = ["frommlpbtxt.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
