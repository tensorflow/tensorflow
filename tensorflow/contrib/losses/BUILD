# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "losses_py",
    srcs = [
        "__init__.py",
        "python/losses/__init__.py",
        "python/losses/loss_ops.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "loss_ops_test",
    srcs = glob(["python/losses/loss_ops_test.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":losses_py",
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
