# Description:
#   Contains a Python wrapper for the StatSummarizer C++ class.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "stat_summarizer_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
)

tf_py_test(
    name = "stat_summarizer_test",
    size = "small",
    srcs = ["python/stat_summarizer_test.py"],
    additional_deps = [
        ":stat_summarizer_py",
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
