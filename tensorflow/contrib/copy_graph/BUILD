# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "copy_graph_py",
    srcs = [
        "__init__.py",
        "python/util/__init__.py",
        "python/util/copy_elements.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "copy_test",
    srcs = glob(["python/util/copy_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":copy_graph_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
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
