# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "graph_editor_py",
    srcs = [
        "__init__.py",
        "edit.py",
        "reroute.py",
        "select.py",
        "subgraph.py",
        "transform.py",
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "graph_editor_pip",
    deps = [
        ":graph_editor_py",
        ":match",
    ],
)

py_library(
    name = "match",
    srcs = ["tests/match.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        "//tensorflow/python:framework_ops",
        "@six_archive//:six",
    ],
)

py_test(
    name = "util_test",
    srcs = ["tests/util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "select_test",
    srcs = ["tests/select_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "match_test",
    srcs = ["tests/match_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":match",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "subgraph_test",
    srcs = ["tests/subgraph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "reroute_test",
    srcs = ["tests/reroute_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        ":match",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "edit_test",
    srcs = ["tests/edit_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        ":match",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "transform_test",
    srcs = ["tests/transform_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_editor_py",
        ":match",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)
