package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/framework:function",
        "//tensorflow/contrib/data/python/util:nest",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:framework",
        "//tensorflow/python:parsing_ops",
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
