load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow/python/feature_column:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "feature_column",
    srcs = ["__init__.py"],
    deps = [
        ":base_feature_layer",
        ":dense_features",
        ":dense_features_v2",
        ":sequence_feature_column",
    ],
)

py_library(
    name = "base_feature_layer",
    srcs = ["base_feature_layer.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/utils:generic_utils",
    ],
)

py_library(
    name = "dense_features",
    srcs = [
        "dense_features.py",
    ],
    deps = [
        ":base_feature_layer",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf_export",
        "//tensorflow/python:util",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras:backend",
    ],
)

py_library(
    name = "dense_features_v2",
    srcs = [
        "dense_features_v2.py",
    ],
    deps = [
        ":base_feature_layer",
        ":dense_features",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf_export",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "dense_features_test",
    srcs = ["dense_features_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dense_features",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "dense_features_v2_test",
    srcs = ["dense_features_v2_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dense_features_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

py_library(
    name = "sequence_feature_column",
    srcs = ["sequence_feature_column.py"],
    deps = [
        ":base_feature_layer",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf_export",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras:backend",
    ],
)

tf_py_test(
    name = "sequence_feature_column_test",
    srcs = ["sequence_feature_column_test.py"],
    deps = [
        ":sequence_feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["sequence_feature_column_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dense_features",
        ":sequence_feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras/layers:recurrent",
    ],
)
