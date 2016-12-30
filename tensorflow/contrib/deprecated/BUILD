# Description:
#   Contains deprecated functions that we aren't quite ready to remove entirely

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "deprecated_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:logging_ops"],
)

py_test(
    name = "summaries_test",
    srcs = ["summaries_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deprecated_py",
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
