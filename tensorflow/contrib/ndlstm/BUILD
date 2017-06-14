# Description:
#   Contains classes implementing 1D and 2D LSTMs for image and signal
#   processing problems.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "ndlstm",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/lstm1d.py",
        "python/lstm2d.py",
        "python/misc.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

tf_py_test(
    name = "lstm1d_test",
    srcs = ["python/lstm1d_test.py"],
    additional_deps = [
        ":ndlstm",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "lstm2d_test",
    size = "small",
    srcs = ["python/lstm2d_test.py"],
    additional_deps = [
        ":ndlstm",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "misc_test",
    size = "small",
    srcs = ["python/misc_test.py"],
    additional_deps = [
        ":ndlstm",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
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
