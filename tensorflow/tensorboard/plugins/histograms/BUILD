# Description:
# TensorBoard plugin for histograms

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

## Histograms Plugin ##
py_library(
    name = "histograms_plugin",
    srcs = ["histograms_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/tensorboard:internal",
    ],
    deps = [
        "//tensorflow/tensorboard/backend:http_util",
        "//tensorflow/tensorboard/backend/event_processing:event_accumulator",
        "//tensorflow/tensorboard/plugins:base_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_test(
    name = "histograms_plugin_test",
    size = "small",
    srcs = ["histograms_plugin_test.py"],
    main = "histograms_plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":histograms_plugin",
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_accumulator",
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
