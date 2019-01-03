package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "feature_column_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":sequence_feature_column",
        ":sequence_feature_column_v2",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "sequence_feature_column",
    srcs = ["python/feature_column/sequence_feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
)

tf_py_test(
    name = "sequence_feature_column_test",
    srcs = ["python/feature_column/sequence_feature_column_test.py"],
    additional_deps = [
        ":sequence_feature_column",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["python/feature_column/sequence_feature_column_integration_test.py"],
    additional_deps = [
        ":sequence_feature_column",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras:layers",
    ],
    tags = ["no_pip"],
)

py_library(
    name = "sequence_feature_column_v2",
    srcs = ["python/feature_column/sequence_feature_column_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/feature_column",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
)

tf_py_test(
    name = "sequence_feature_column_v2_test",
    srcs = ["python/feature_column/sequence_feature_column_v2_test.py"],
    additional_deps = [
        ":sequence_feature_column",
        ":sequence_feature_column_v2",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/feature_column:feature_column_v2_test",
    ],
    tags = ["no_pip"],
)
