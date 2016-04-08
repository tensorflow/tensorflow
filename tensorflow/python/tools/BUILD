# Description:
#   Tools for manipulating TensorFlow graphs.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "freeze_graph_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "freeze_graph",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = [
        "freeze_graph_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":freeze_graph",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "graph_metrics_lib",
    srcs = ["graph_metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "graph_metrics",
    srcs = [
        "graph_metrics.py",
    ],
    srcs_version = "PY2AND3",
    main = "graph_metrics.py",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "graph_metrics_test",
    size = "small",
    srcs = [
        "graph_metrics_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_metrics_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = [
        "inspect_checkpoint.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
