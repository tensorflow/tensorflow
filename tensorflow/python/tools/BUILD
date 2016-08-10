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
    main = "graph_metrics.py",
    srcs_version = "PY2AND3",
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

py_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "strip_unused_test",
    size = "small",
    srcs = [
        "strip_unused_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "optimize_for_inference_lib",
    srcs = ["optimize_for_inference_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "optimize_for_inference",
    srcs = ["optimize_for_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "optimize_for_inference_test",
    size = "small",
    srcs = [
        "optimize_for_inference_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
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
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
