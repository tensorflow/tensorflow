# Description:
#   Utilities for reading and writing object-based checkpoints.

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model:registration",
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
        ":util",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variables",
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
    name = "tracking",
    srcs = ["tracking.py"],
    srcs_version = "PY3",
    deps = [
        ":asset",
        ":autotrackable",
        ":data_structures",
        ":resource",
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
    name = "graph_view",
    srcs = ["graph_view.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":tracking",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/training:optimizer",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":data_structures",
        ":graph_view",
        ":tracking",
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
        "//tensorflow/python/training/saving:checkpoint_options",
        "//tensorflow/python/training/saving:functional_saver",
        "//tensorflow/python/training/saving:saveable_object_util",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    tags = [
        "no_windows",  # TODO(b/201457117)
        "notsan",  # TODO(b/74395663)
    ],
    deps = [
        ":base",
        ":graph_view",
        ":tracking",
        ":util",
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
        "//tensorflow/python/training/saving:checkpoint_options",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "util_with_v1_optimizers_test",
    srcs = ["util_with_v1_optimizers_test.py"],
    tags = [
        "notsan",  # b/74395663
    ],
    deps = [
        ":tracking",
        ":util",
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
        "@six_archive//:six",
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
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/module",
    ],
)

tf_py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//tensorflow/python/training/tracking:benchmarks_test",
)

tf_py_test(
    name = "checkpoint_metrics_test",
    srcs = ["checkpoint_metrics_test.py"],
    deps = [
        ":util",
        "//tensorflow/python:platform_test",
    ],
)
