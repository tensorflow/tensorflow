# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

## Embedding projector ##
py_library(
    name = "projector",
    srcs = glob(["plugins/projector/**/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":base_plugin",
        "//tensorflow/contrib/tensorboard:projector",
        "//tensorflow/contrib/tensorboard:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)
