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
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_file_inspector",
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
