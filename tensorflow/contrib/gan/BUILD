# Files for using TFGAN framework.
package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "gan",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":losses",
    ],
)

py_library(
    name = "losses",
    srcs = ["python/losses/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        ":tuple_losses",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "losses_impl",
    srcs = ["python/losses/losses_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "losses_impl_test",
    srcs = ["python/losses/losses_impl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/losses",
    ],
)

py_library(
    name = "tuple_losses",
    srcs = [
        "python/losses/losses_wargs.py",
        "python/losses/tuple_losses.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":losses_impl",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "tuple_losses_test",
    srcs = ["python/losses/tuple_losses_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tuple_losses",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
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
