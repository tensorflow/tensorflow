# Description:
#   All-reduce implementations.
#   APIs are subject to change.  Eventually to be replaced by equivalent
#   functionality within TensorFlow core.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "all_reduce",
    srcs = [
        "python/all_reduce.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/nccl:nccl_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "all_reduce_test",
    srcs = ["python/all_reduce_test.py"],
    additional_deps = [
        ":all_reduce",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
