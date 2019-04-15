# Description:
#   Utilities for reading and writing object-based checkpoints.

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY2AND3",
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
        "//tensorflow/python/training/saving:saveable_object",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    additional_deps = [
        ":base",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "tracking",
    srcs = ["tracking.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":data_structures",
    ],
)

tf_py_test(
    name = "tracking_test",
    srcs = ["tracking_test.py"],
    additional_deps = [
        ":base",
        ":tracking",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "layer_utils",
    srcs = ["layer_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    srcs_version = "PY2AND3",
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
    additional_deps = [
        ":data_structures",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
    ],
    tags = [
        "no_windows",
        "nomac",
    ],
)

py_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "graph_view",
    srcs = ["graph_view.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":object_identity",
        ":tracking",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
        ":data_structures",
        ":graph_view",
        ":object_identity",
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
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/training/saving:functional_saver",
        "//tensorflow/python/training/saving:saveable_object_util",
    ],
)

tf_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    additional_deps = [
        ":base",
        ":graph_view",
        ":tracking",
        ":util",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training_util",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python:variables",
    ],
    tags = ["notsan"],  # b/74395663
)

tf_xla_py_test(
    name = "util_xla_test",
    srcs = ["util_xla_test.py"],
    tags = [
        "no_pip",
        "no_rocm",
        "nomac",
        "notsan",  # b/74395663
    ],
    deps = [
        ":tracking",
        ":util",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)

tf_py_test(
    name = "util_with_v1_optimizers_test",
    srcs = ["util_with_v1_optimizers_test.py"],
    additional_deps = [
        ":base",
        ":graph_view",
        ":tracking",
        ":util",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:training_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
    ],
    tags = [
        "no_windows",  # b/124401331
        "notsan",  # b/74395663
    ],
)

py_library(
    name = "python_state",
    srcs = ["python_state.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base",
    ],
)

tf_py_test(
    name = "python_state_test",
    srcs = ["python_state_test.py"],
    additional_deps = [
        ":base",
        ":util",
        "//tensorflow/python/module",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)
