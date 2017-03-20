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
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend/event_processing:event_accumulator",
        "//tensorflow/tensorboard/backend/event_processing:event_file_loader",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_multiplexer",
        "//third_party/py/numpy",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)
