# Description:
# Python APIs for various Tensorflow backends.

load("//tensorflow:tensorflow.bzl", "if_not_windows")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = if_not_windows([
        "//tensorflow/python/compiler/tensorrt:init_py",
    ]) + [
        "//tensorflow/python/compiler/xla:compiler_py",
    ],
)
