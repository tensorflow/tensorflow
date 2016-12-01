# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

cuda_py_tests(
    name = "linear_operator_test",
    size = "small",
    srcs = ["python/kernel_tests/linear_operator_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "linear_operator_diag_test",
    size = "small",
    srcs = ["python/kernel_tests/linear_operator_diag_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

py_library(
    name = "linalg_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
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
