# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":assets"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_file_inspector",
        "//tensorflow/tensorboard/plugins/projector:projector_plugin",
        "//tensorflow/tensorboard/plugins/text:text_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

filegroup(
    name = "assets",
    srcs = [
        "TAG",
        "//tensorflow/tensorboard/components:index.html",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**"],
        exclude = [
            "METADATA",
            "OWNERS",
            "tensorboard.google.bzl",
        ],
    ),
    tags = ["notsan"],
)
