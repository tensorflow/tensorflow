# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/serving/...",
        "//platforms/techila/...",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_tests")

cc_library(
    name = "ackermann_op_lib",
    srcs = ["ackermann_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

cc_binary(
    name = "ackermann_op.so",
    linkopts = select({
        "//conditions:default": [
            "-Wl,-Bsymbolic",
            "-lm",
        ],
        "//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":ackermann_op_lib",
    ],
)

py_tests(
    name = "ackermann_test",
    srcs = ["ackermann_test.py"],
    data = [":ackermann_op.so"],
    tags = ["notsan"],
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
