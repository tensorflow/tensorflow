# Description:
# TensorFlow SavedModel Registration.

load("//tensorflow:tensorflow.bzl", "py_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "registration",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":registration_lib",
    ],
)

py_strict_library(
    name = "registration_lib",
    srcs = [
        "registration.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
    ],
)

tf_py_test(
    name = "registration_test",
    srcs = ["registration_test.py"],
    deps = [
        ":registration",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:base",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "registration_saving_test",
    srcs = ["registration_saving_test.py"],
    deps = [
        ":registration",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = [
        "test_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":registration_lib",
    ],
)

tf_py_test(
    name = "tf_registration_test",
    srcs = ["tf_registration_test.py"],
    data = [
        "tf_checkpoint_saver_allowlist.txt",
        "tf_serializable_allowlist.txt",
    ],
    tags = ["no_pip"],
    deps = [
        ":registration",
        ":test_util",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
    ],
)
