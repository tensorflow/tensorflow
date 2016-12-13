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
    srcs = [
        "__init__.py",
        "base.py",
        "mnist.py",
        "produce_small_datasets.py",
        "text_datasets.py",
    ],
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

py_test(
    name = "base_test",
    size = "small",
    srcs = ["base_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "load_csv_test",
    size = "small",
    srcs = ["load_csv_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:framework_test_lib",
    ],
)
