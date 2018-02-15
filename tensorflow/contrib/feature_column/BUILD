package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

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

py_library(
    name = "feature_column_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequential_feature_column",
    ],
)

py_library(
    name = "sequential_feature_column",
    srcs = ["python/feature_column/sequential_feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [],
)
