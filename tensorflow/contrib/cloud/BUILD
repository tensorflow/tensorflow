# Description:
# BigQueryReader implementation

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_py_test",
)

tf_gen_op_libs(
    op_lib_names = ["bigquery_reader_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_bigquery_reader_ops",
    out = "python/ops/gen_bigquery_reader_ops.py",
    require_shape_functions = True,
    deps = [":bigquery_reader_ops_op_lib"],
)

py_library(
    name = "cloud_py",
    srcs = [
        "__init__.py",
        "python/ops/bigquery_reader_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_bigquery_reader_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "bigquery_reader_ops_test",
    size = "small",
    srcs = ["python/ops/bigquery_reader_ops_test.py"],
    additional_deps = [
        ":cloud_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
    tags = ["manual"],
)
