# Description:
#   Contains classes that provide access to common method of a [batch] matrix,
#   without the need to instantiate the matrix.
#   This allows for exploitation of structure, as well as a generic interface
#   suitable for iterative solvers.

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
    name = "linear_operator_composition_test",
    size = "medium",
    srcs = ["python/kernel_tests/linear_operator_composition_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

cuda_py_tests(
    name = "linear_operator_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/linear_operator_diag_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

cuda_py_tests(
    name = "linear_operator_matrix_test",
    size = "medium",
    srcs = ["python/kernel_tests/linear_operator_matrix_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

cuda_py_tests(
    name = "linear_operator_tril_test",
    size = "medium",
    srcs = ["python/kernel_tests/linear_operator_tril_test.py"],
    additional_deps = [
        ":linalg_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
)

cuda_py_tests(
    name = "linear_operator_util_test",
    size = "small",
    srcs = ["python/kernel_tests/linear_operator_util_test.py"],
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
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
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
