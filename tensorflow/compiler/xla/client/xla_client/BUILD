# Description:
#   The new XLA client libraries.

licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "xla_builder",
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/client:xla_builder",
    ],
)
