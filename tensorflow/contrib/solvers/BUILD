# Description:
# Contains ops for iterative solvers for linear systems, linear least-squares
# problems, singular value decomposition and eigenvalue decomposition.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "solvers_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
)

# Ops tests
cuda_py_test(
    name = "lanczos_test",
    srcs = [
        "python/kernel_tests/lanczos_test.py",
    ],
    additional_deps = [
        ":solvers_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "least_squares_test",
    srcs = [
        "python/kernel_tests/least_squares_test.py",
    ],
    additional_deps = [
        ":solvers_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "linear_equations_test",
    srcs = [
        "python/kernel_tests/linear_equations_test.py",
    ],
    additional_deps = [
        ":solvers_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "util_test",
    srcs = [
        "python/kernel_tests/util_test.py",
    ],
    additional_deps = [
        ":solvers_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

# All files
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
