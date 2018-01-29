# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "sdca_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/sdca_ops.py",
        "python/sdca_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":sharded_mutable_dense_hashtable_py",
        ":sparse_feature_column_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:sdca_ops_gen",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "sdca_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/sdca_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops_py",
        ":sparse_feature_column_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sdca_ops_gen",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "sharded_mutable_dense_hashtable_py",
    srcs = ["python/ops/sharded_mutable_dense_hashtable.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "sharded_mutable_dense_hashtable_test",
    size = "small",
    srcs = ["python/ops/sharded_mutable_dense_hashtable_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sharded_mutable_dense_hashtable_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "sparse_feature_column_py",
    srcs = ["python/ops/sparse_feature_column.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:framework_for_generated_wrappers"],
)

py_test(
    name = "sparse_feature_column_test",
    size = "small",
    srcs = ["python/ops/sparse_feature_column_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sparse_feature_column_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "sdca_estimator_py",
    srcs = ["python/sdca_estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "sdca_estimator_test",
    srcs = ["python/sdca_estimator_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_estimator_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
