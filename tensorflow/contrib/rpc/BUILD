package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

py_library(
    name = "rpc",
    srcs = [
        "__init__.py",
    ],
    deps = ["//tensorflow/contrib/rpc/python/ops:rpc_op_py"],
)

py_library(
    name = "rpc_pip",
    data = if_static(
        [],
        otherwise = ["//tensorflow/contrib/rpc/python/kernel_tests:libtestexample.so"],
    ),
    deps = [
        ":rpc",
        "//tensorflow/contrib/rpc/python/kernel_tests:py_test_deps",
        "//tensorflow/contrib/rpc/python/kernel_tests:rpc_op_test_base",
        "//tensorflow/contrib/rpc/python/kernel_tests:rpc_op_test_servicer",
    ],
)
