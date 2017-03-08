# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "graph_editor_py",
    srcs = [
        "__init__.py",
        "edit.py",
        "select.py",
        "subgraph.py",
        "transform.py",
        "util.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "subgraphview_test",
    srcs = ["tests/subgraphview_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
