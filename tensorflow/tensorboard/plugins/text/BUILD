# Description:
# TensorBoard plugin for the Text

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

## Text Plugin ##
py_library(
    name = "text_plugin",
    srcs = ["text_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/tensorboard:internal",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:http_util",
        "//tensorflow/tensorboard/plugins:base_plugin",
        "@org_mozilla_bleach",
        "@org_pocoo_werkzeug//:werkzeug",
        "@org_pythonhosted_markdown",
        "@six_archive//:six",
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
