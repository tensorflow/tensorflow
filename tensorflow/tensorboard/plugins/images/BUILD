# Description:
# TensorBoard plugin for images

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "images_plugin",
    srcs = ["images_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/tensorboard:internal",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:http_util",
        "//tensorflow/tensorboard/backend/event_processing:event_accumulator",
        "//tensorflow/tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "images_plugin_test",
    size = "small",
    srcs = ["images_plugin_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":images_plugin",
        "//tensorflow:tensorflow_py",
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
