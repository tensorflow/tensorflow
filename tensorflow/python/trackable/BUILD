# Description:
#   Trackable class and subclass definitions.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "trackable",
    deps = [
        ":asset",
        ":autotrackable",
        ":base",
        ":base_delegate",
        ":constants",
        ":converter",
        ":data_structures",
        ":layer_utils",
        ":python_state",
        ":resource",
        ":trackable_init",
        ":trackable_utils",
    ],
)

py_strict_library(
    name = "trackable_init",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/checkpoint:checkpoint_core",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":data_structures",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager/polymorphic_function:saved_model_utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "trackable_utils",
    srcs = ["trackable_utils.py"],
    srcs_version = "PY3",
)

tf_py_strict_test(
    name = "trackable_utils_test",
    srcs = ["trackable_utils_test.py"],
    deps = [
        ":trackable_utils",
        "//tensorflow/python/eager:test",
    ],
)

py_strict_library(
    name = "base_delegate",
    srcs = ["base_delegate.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
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

py_strict_library(
    name = "asset",
    srcs = ["asset.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/saved_model:path_helpers",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "autotrackable",
    srcs = ["autotrackable.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":data_structures",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "autotrackable_test",
    srcs = ["autotrackable_test.py"],
    deps = [
        ":autotrackable",
        ":data_structures",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/checkpoint:checkpoint_core",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "resource",
    srcs = ["resource.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":base",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "resource_test",
    srcs = ["resource_test.py"],
    deps = [
        ":resource",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_library(
    name = "layer_utils",
    srcs = ["layer_utils.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:object_identity"],
)

py_strict_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":layer_utils",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "@wrapt",
    ],
)

tf_py_strict_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":autotrackable",
        ":data_structures",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint:checkpoint_core",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/module",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:serialization",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "python_state",
    srcs = ["python_state.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "python_state_test",
    srcs = ["python_state_test.py"],
    deps = [
        ":python_state",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/module",
    ],
)
