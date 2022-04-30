"""Tests to improve the consistency of tensorflow.

Here we would like to include high level tests that stress tf.function and
autograph in ways users have discovered.  Not everything here has to work,
some things just need to have good error messages.  some things currently
have bugs assigned to them but do not work and do not have sufficient error
messages.
"""

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
)

package(
    default_visibility = ["//tensorflow/tools/consistency_integration_test:__subpackages__"],
    licenses = ["notice"],
)

licenses(["notice"])

py_library(
    name = "consistency_test_base",
    srcs = ["consistency_test_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "@termcolor_archive//:termcolor",
    ],
)

py_test(
    name = "consistency_test_base_tests",
    srcs = ["consistency_test_base_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "function_io_tests",
    srcs = ["function_io_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "tensor_array_tests",
    srcs = ["tensor_array_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "type_promotion_tests",
    srcs = ["type_promotion_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "keras_integration_tests",
    srcs = ["keras_integration_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
