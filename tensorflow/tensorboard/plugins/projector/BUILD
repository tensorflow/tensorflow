# Embedding Projector plugin.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

py_library(
    name = "projector_plugin",
    srcs = ["projector_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":protos_all_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:http_util",
        "//tensorflow/tensorboard/plugins:base_plugin",
        "//third_party/py/numpy",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_test(
    name = "projector_plugin_test",
    size = "small",
    srcs = ["projector_plugin_test.py"],
    main = "projector_plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":projector_plugin",
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_multiplexer",
        "//third_party/py/numpy",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(["*.proto"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    visibility = ["//tensorflow:__pkg__"],
)
