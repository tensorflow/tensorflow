# Description:
# TensorFlow SavedModel Registration.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_py_strict_test(
    name = "registration_test",
    srcs = ["registration_test.py"],
    deps = [
        ":registration",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:base",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "registration_saving_test",
    srcs = ["registration_saving_test.py"],
    deps = [
        ":registration",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
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

tf_py_strict_test(
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
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)
