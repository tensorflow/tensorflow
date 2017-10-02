package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/contrib/data/python/ops:transformation_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
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
