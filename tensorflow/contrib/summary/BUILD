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
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:summary_ops_op_lib",
    ],
)

py_test(
    name = "summary_ops_test",
    srcs = ["summary_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary_ops",
        ":summary_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:test",
        "@six_archive//:six",
    ],
)

py_library(
    name = "summary_ops",
    srcs = ["summary_ops.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":gen_summary_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:summary_op_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":summary_ops",
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

# NOTE: target cannot be testonly because it needs to be in the pip
# package. Sigh.
py_library(
    name = "summary_test_util",
    srcs = ["summary_test_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
    ],
)
