package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "nest",
    srcs = ["nest.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sparse",
    srcs = ["sparse.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        ":sparse",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nest",
        ":structure",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "options_test",
    size = "small",
    srcs = ["options_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":options",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

py_test(
    name = "convert_test",
    size = "small",
    srcs = ["convert_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":convert",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "random_seed_test",
    size = "small",
    srcs = ["random_seed_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":random_seed",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
    ],
)
