# Description:
# TensorFlow GRPC distributed runtime server and tests

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "grpc_tensorflow_server",
    srcs = [
        "grpc_tensorflow_server.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "parse_cluster_spec_test",
    size = "small",
    srcs = [
        "parse_cluster_spec_test.py",
    ],
    main = "parse_cluster_spec_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":grpc_tensorflow_server",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
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
