# Description:
# TensorBoard plugin for scalars

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

## Scalars Plugin ##
py_library(
    name = "scalars_plugin",
    srcs = ["scalars_plugin.py"],
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
    name = "scalars_plugin_test",
    size = "small",
    srcs = ["scalars_plugin_test.py"],
    main = "scalars_plugin_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":scalars_plugin",
        "//tensorflow:tensorflow_py",
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_multiplexer",
        "@org_pocoo_werkzeug//:werkzeug",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "scalars_demo",
    srcs = ["scalars_demo.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    visibility = ["//tensorflow:__pkg__"],
)
