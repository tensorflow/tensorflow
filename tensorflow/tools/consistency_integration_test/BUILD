"""Tests to improve the the of tensorflow.

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
    licenses = ["notice"],  # Apache 2.0
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
    name = "basic_tests",
    srcs = ["basic_tests.py"],
    srcs_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":consistency_test_base",
        "//tensorflow/python:client_testlib",
    ],
)
