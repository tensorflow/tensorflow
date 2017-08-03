# Description:
# Contains ops for remote fused graph

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

tf_gen_op_wrapper_py(
    name = "gen_remote_fused_graph_ops",
    out = "python/ops/gen_remote_fused_graph_ops.py",
    deps = [
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
    ],
)

py_library(
    name = "remote_fused_graph_ops_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":gen_remote_fused_graph_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "remote_fused_graph_ops_test",
    size = "small",
    srcs = ["python/ops/remote_fused_graph_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":remote_fused_graph_ops_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
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
    visibility = ["//tensorflow:__subpackages__"],
)
