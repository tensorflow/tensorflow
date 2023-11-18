# Description:
#   Utilities for sharding object-based checkpoints.

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
    name = "sharding_policies",
    srcs = ["sharding_policies.py"],
    srcs_version = "PY3",
    deps = [
        ":sharding_util",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

tf_py_strict_test(
    name = "sharding_policies_test",
    srcs = ["sharding_policies_test.py"],
    srcs_version = "PY3",
    deps = [
        ":sharding_policies",
        ":sharding_util",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/training/saving:saveable_object",
    ],
)

py_strict_library(
    name = "sharding_util",
    srcs = ["sharding_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
    ],
)
