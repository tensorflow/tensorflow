# Description:
#   Internal testing utilities, e.g., computing the correct answer to
#   put in a unit test.

licenses(["notice"])  # Apache 2.0

py_library(
    name = "correlation_matrix_volumes_py",
    srcs = [
        "correlation_matrix_volumes_lib.py",
    ],
    deps = [
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "correlation_matrix_volumes",
    srcs = [
        "correlation_matrix_volumes.py",
    ],
    deps = [
        ":correlation_matrix_volumes_py",
    ],
)

py_test(
    name = "correlation_matrix_volumes_test",
    size = "medium",
    srcs = ["correlation_matrix_volumes_test.py"],
    tags = ["no_pip"],
    deps = [
        ":correlation_matrix_volumes_py",
        # For statistical testing
        "//tensorflow/contrib/distributions:distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)
