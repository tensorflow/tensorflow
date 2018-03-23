# Description:
#   Contains classes that provide access to common method of a [batch] matrix,
#   without the need to instantiate the matrix.
#   This allows for exploitation of structure, as well as a generic interface
#   suitable for iterative solvers.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "linalg_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/linalg",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "linear_operator_addition_test",
    size = "small",
    srcs = ["python/kernel_tests/linear_operator_addition_test.py"],
    additional_deps = [
        ":linalg_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "linear_operator_block_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/linear_operator_block_diag_test.py"],
    additional_deps = [
        ":linalg_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 4,
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
