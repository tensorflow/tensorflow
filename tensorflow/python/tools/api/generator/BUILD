# Description:
# Scripts used to generate TensorFlow Python API.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/python/tools/api/generator:api_init_files.bzl", "TENSORFLOW_API_INIT_FILES")
load("//tensorflow/python/tools/api/generator:api_init_files_v1.bzl", "TENSORFLOW_API_INIT_FILES_V1")

exports_files(
    [
        "LICENSE",
        "create_python_api.py",
    ],
)

py_library(
    name = "create_python_api",
    srcs = ["//tensorflow/python/tools/api/generator:create_python_api.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/tools/api/generator:doc_srcs",
    ],
)

py_library(
    name = "doc_srcs",
    srcs = ["doc_srcs.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "create_python_api_test",
    srcs = [
        "create_python_api.py",
        "create_python_api_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "tensorflow_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=tensorflow.python",
        "--api_name=tensorflow",
    ] + TENSORFLOW_API_INIT_FILES + TENSORFLOW_API_INIT_FILES_V1,
    main = "doc_srcs_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":doc_srcs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
    ],
)

py_test(
    name = "output_init_files_test",
    srcs = ["output_init_files_test.py"],
    data = [
        "api_init_files.bzl",
        "api_init_files_v1.bzl",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/tools/api/generator:create_python_api",
    ],
)
