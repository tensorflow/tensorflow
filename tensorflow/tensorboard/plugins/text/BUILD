# Description:
# TensorBoard plugin for the Text

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

## Text Plugin ##
py_library(
    name = "text_plugin",
    srcs = ["text_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/python:summary",
        "//tensorflow/tensorboard/backend:http_util",
        "//tensorflow/tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_test(
    name = "text_plugin_test",
    size = "small",
    srcs = ["text_plugin_test.py"],
    main = "text_plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":text_plugin",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_multiplexer",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    visibility = ["//tensorflow:__pkg__"],
)
