# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "application",
    srcs = ["application.py"],
    data = ["//tensorflow/tensorboard:frontend"],
    srcs_version = "PY2AND3",
    deps = [
        ":process_graph",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/tensorboard/lib/python:http_util",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "application_test",
    size = "medium",
    srcs = ["application_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":application",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_library(
    name = "process_graph",
    srcs = ["process_graph.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python:util"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        [
            "*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
