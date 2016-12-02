# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

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
    ],
)

py_test(
    name = "sdca_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/sdca_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sdca_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "sharded_mutable_dense_hashtable_py",
    srcs = ["python/ops/sharded_mutable_dense_hashtable.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/lookup:lookup_py",
    ],
)

py_test(
    name = "sharded_mutable_dense_hashtable_test",
    size = "small",
    srcs = ["python/ops/sharded_mutable_dense_hashtable_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sharded_mutable_dense_hashtable_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "sparse_feature_column_py",
    srcs = ["python/ops/sparse_feature_column.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "sparse_feature_column_test",
    size = "small",
    srcs = ["python/ops/sparse_feature_column_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sparse_feature_column_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
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
