licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_gen_op_wrapper_py",
)

tf_gen_op_wrapper_py(
    name = "gen_summary_ops",
    out = "gen_summary_ops.py",
    deps = ["//tensorflow/core:summary_ops_op_lib"],
)

py_test(
    name = "summary_ops_test",
    srcs = ["summary_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "summary_ops",
    srcs = ["summary_ops.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":gen_summary_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:summary_op_util",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:context",
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
    visibility = ["//tensorflow:__subpackages__"],
)
