# Description:
# TensorBoard plugin for interacting with tfdbg, the TensorFlow debugger

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

## TensorFlow Debugger Plugiin ##
py_library(
    name = "debugger_plugin",
    srcs = ["debugger_plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/lib/python:http_util",
        "//tensorflow/tensorboard/plugins:base_plugin",
    ],
)

py_test(
    name = "debugger_plugin_test",
    size = "small",
    srcs = ["debugger_plugin_test.py"],
    main = "debugger_plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_plugin",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:summary",
    ],
)
