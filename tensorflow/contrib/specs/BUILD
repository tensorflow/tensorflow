# Description:
#   A small domain-specific language (DSL) for defining deep learning networks.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "specs",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/params_ops.py",
        "python/specs.py",
        "python/specs_lib.py",
        "python/specs_ops.py",
        "python/summaries.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/ndlstm",
        "//tensorflow/python:framework",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

tf_py_test(
    name = "specs_test",
    srcs = ["python/specs_test.py"],
    additional_deps = [
        ":specs",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "summaries_test",
    srcs = ["python/summaries_test.py"],
    additional_deps = [
        ":specs",
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
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
