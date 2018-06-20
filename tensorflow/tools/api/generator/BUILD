# Description:
# Scripts used to generate TensorFlow Python API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/tools/api/generator:api_gen.bzl", "ESTIMATOR_API_INIT_FILES")
load("//tensorflow/tools/api/generator:api_gen.bzl", "TENSORFLOW_API_INIT_FILES")

py_library(
    name = "doc_srcs",
    srcs = ["doc_srcs.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
    ],
)

py_binary(
    name = "create_python_api",
    srcs = ["create_python_api.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":doc_srcs",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "create_python_api_test",
    srcs = ["create_python_api_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":create_python_api",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "tensorflow_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=tensorflow.python",
        "--api_name=tensorflow",
    ] + TENSORFLOW_API_INIT_FILES,
    main = "doc_srcs_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "estimator_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=tensorflow.python.estimator",
        "--api_name=estimator",
    ] + ESTIMATOR_API_INIT_FILES,
    main = "doc_srcs_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)
