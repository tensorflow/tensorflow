licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_gen_op_wrapper_py",
)

py_test(
    name = "summary_ops_test",
    srcs = ["summary_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:test",
        "@six_archive//:six",
    ],
)

py_test(
    name = "summary_ops_graph_test",
    srcs = ["summary_ops_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summary_test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:summary_ops_v2",
    ],
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary_ops_v2",
        "@org_sqlite//:python",
    ],
)
