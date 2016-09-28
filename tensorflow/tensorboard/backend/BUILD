# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "handler",
    srcs = ["handler.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":process_graph",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
        "//tensorflow/tensorboard:plugins",
        "//tensorflow/tensorboard/lib/python:json_util",
    ],
)

py_test(
    name = "handler_test",
    size = "small",
    srcs = ["handler_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":handler",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "process_graph",
    srcs = ["process_graph.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [],
)

py_library(
    name = "server",
    srcs = ["server.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":handler",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

py_test(
    name = "server_test",
    size = "small",
    srcs = ["server_test.py"],
    data = ["//tensorflow/tensorboard:frontend"],
    srcs_version = "PY2AND3",
    deps = [
        ":server",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:summary",
    ],
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
