# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "frontend",
    srcs = [
        "TAG",
        "dist/bazel-html-imports.html",
        "dist/index.html",
        "dist/tf-tensorboard.html",
        "//tensorflow/tensorboard/bower",
        "//tensorflow/tensorboard/lib:all_files",
    ],
)

py_binary(
    name = "tensorboard",
    srcs = [
        "__main__.py",
        "tensorboard.py",
    ],
    data = [":frontend"],
    srcs_version = "PY2AND3",
    deps = [
        ":debugger",
        ":projector",
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend:application",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/node_modules/**",
            "**/typings/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

###### PLUGINS ######

# Plugins don't have their own packages (BUILD files) because we want to
# have only one BUILD file since each BUILD file needs special rewrite rules
# in the git world.

py_library(
    name = "base_plugin",
    srcs = ["plugins/base_plugin.py"],
    srcs_version = "PY2AND3",
)

## TensorFlow Debugger Plugin ##
py_library(
    name = "debugger",
    srcs = ["plugins/debugger/plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_plugin",
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/lib/python:http_util",
    ],
)

## Embedding Projector Plugin ##
py_library(
    name = "projector",
    srcs = ["plugins/projector/plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_plugin",
        "//tensorflow/contrib/tensorboard:projector",
        "//tensorflow/contrib/tensorboard:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/tensorboard/lib/python:http_util",
        "//third_party/py/numpy",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_test(
    name = "projector_plugin_test",
    size = "small",
    srcs = ["plugins/projector/plugin_test.py"],
    main = "plugins/projector/plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":projector",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/tensorboard/backend:application",
        "//third_party/py/numpy",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)
