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
        "//tensorflow/python:framework",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "lstm1d_test",
    srcs = ["python/lstm1d_test.py"],
    additional_deps = [
        ":ndlstm",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "lstm2d_test",
    srcs = ["python/lstm2d_test.py"],
    additional_deps = [
        ":ndlstm",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "misc_test",
    srcs = ["python/misc_test.py"],
    additional_deps = [
        ":ndlstm",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        # "//tensorflow:tensorflow_py:tensorflow_google",
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
