# Description:
#   Trackable class and subclass definitions.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "trackable",
    deps = [
        ":asset",
        ":autotrackable",
        ":base",
        ":base_delegate",
        ":converter",
        ":data_structures",
        ":layer_utils",
        ":python_state",
        ":resource",
        ":trackable_init",
        ":trackable_utils",
    ],
)

py_library(
    name = "trackable_init",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/checkpoint:checkpoint_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/training/saving:saveable_object",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY3",
    deps = [
        ":data_structures",
        "//tensorflow/python/eager:function_saved_model_utils",
    ],
)

py_library(
    name = "trackable_utils",
    srcs = ["trackable_utils.py"],
    srcs_version = "PY3",
)

tf_py_test(
    name = "trackable_utils_test",
    srcs = ["trackable_utils_test.py"],
    deps = [
        ":trackable_utils",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "base_delegate",
    srcs = ["base_delegate.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "base_delegate_test",
    srcs = ["base_delegate_test.py"],
    deps = [
        ":base",
        ":base_delegate",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
    ],
)

py_library(
    name = "asset",
    srcs = ["asset.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//tensorflow/python:lib",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "autotrackable",
    srcs = ["autotrackable.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":data_structures",
    ],
)

tf_py_test(
    name = "autotrackable_test",
    srcs = ["autotrackable_test.py"],
    deps = [
        ":autotrackable",
        ":data_structures",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "resource",
    srcs = ["resource.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
    ],
)

tf_py_test(
    name = "resource_test",
    srcs = ["resource_test.py"],
    deps = [
        ":base",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "layer_utils",
    srcs = ["layer_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":layer_utils",
        "//tensorflow/python/saved_model:revived_types",
        "@wrapt",
    ],
)

tf_py_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":data_structures",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "python_state",
    srcs = ["python_state.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
    ],
)

tf_py_test(
    name = "python_state_test",
    srcs = ["python_state_test.py"],
    deps = [
        ":base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/module",
    ],
)
