# Prepare training and testing data.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "data_csv",
    srcs = glob(["data/*.csv"]),
)

py_library(
    name = "datasets",
    srcs = glob([
        "*.py",
    ]),
    data = [":data_csv"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "produce_small_datasets",
    srcs = [
        "produce_small_datasets.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
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
