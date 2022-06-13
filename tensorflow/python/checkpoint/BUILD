# Description:
#   Utilities for reading and writing object-based checkpoints.

load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "checkpoint_lib",
    deps = [
        ":checkpoint",
        ":checkpoint_management",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":util",
    ],
)

py_library(
    name = "checkpoint",
    srcs = [
        "__init__.py",
        "checkpoint.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/training/saving:saveable_object_util",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    tags = [
        "no_windows",  # TODO(b/201457117)
        "notsan",  # TODO(b/74395663)
    ],
    deps = [
        ":checkpoint",
        ":checkpoint_options",
        ":graph_view",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "checkpoint_with_v1_optimizers_test",
    srcs = ["checkpoint_with_v1_optimizers_test.py"],
    tags = [
        "notsan",  # b/74395663
    ],
    deps = [
        ":checkpoint",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/training/tracking",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "checkpoint_metrics_test",
    srcs = ["checkpoint_metrics_test.py"],
    deps = [
        ":checkpoint",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "checkpoint_util",
    srcs = ["checkpoint_util.py"],
    srcs_version = "PY3",
)

py_library(
    name = "graph_view",
    srcs = ["graph_view.py"],
    srcs_version = "PY3",
    deps = [
        ":trackable_view",
        ":util",
        "//tensorflow/python:util",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:converter",
    ],
)

py_library(
    name = "trackable_view",
    srcs = ["trackable_view.py"],
    srcs_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":util",
        "//tensorflow/python:util",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:converter",
    ],
)

tf_py_test(
    name = "trackable_view_test",
    srcs = ["trackable_view_test.py"],
    deps = [
        ":trackable_view",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/trackable:base",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:trackable_utils",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/training/tracking",
    ],
)

tf_py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":checkpoint",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//tensorflow/python/checkpoint:benchmarks_test",
)

py_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_options",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

cuda_py_test(
    name = "functional_saver_test",
    size = "medium",
    srcs = [
        "functional_saver_test.py",
    ],
    deps = [
        ":checkpoint_options",
        ":functional_saver",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "checkpoint_management_test",
    size = "small",
    srcs = [
        "checkpoint_management_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":checkpoint",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training:checkpoint_management",
        "//tensorflow/python/training:saver",
    ],
)
