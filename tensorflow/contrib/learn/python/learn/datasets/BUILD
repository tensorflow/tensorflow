# Prepare training and testing data.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

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
        "synthetic.py",
        "text_datasets.py",
    ],
    data = [":data_csv"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "produce_small_datasets",
    srcs = ["produce_small_datasets.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        "//tensorflow/python:platform",
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
        ":datasets",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "load_csv_test",
    size = "small",
    srcs = ["load_csv_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "synthetic_test",
    size = "small",
    srcs = ["synthetic_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
