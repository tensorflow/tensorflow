# Description:
#   Common functionality for TensorFlow tooling

load("//tensorflow:tensorflow.bzl", "VERSION_MAJOR", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//third_party/py/keras/api:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "public_api",
    srcs = ["public_api.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_decorator",
        "@six_archive//:six",
    ],
)

py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    data = select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.dylib"],
        "//tensorflow:windows": [],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION_MAJOR],
    }),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":public_api",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "traverse",
    srcs = ["traverse.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_decorator",
        "@six_archive//:six",
    ],
)

py_test(
    name = "traverse_test",
    srcs = ["traverse_test.py"],
    data = select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.dylib"],
        "//tensorflow:windows": [],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION_MAJOR],
    }),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_module1",
        ":test_module2",
        ":traverse",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "test_module1",
    srcs = ["test_module1.py"],
    srcs_version = "PY3",
    deps = [
        ":test_module2",
    ],
)

py_library(
    name = "test_module2",
    srcs = ["test_module2.py"],
    srcs_version = "PY3",
)
