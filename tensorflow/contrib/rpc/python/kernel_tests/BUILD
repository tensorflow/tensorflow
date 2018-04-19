# TODO(b/76425722): Port everything in here to OS (currently excluded).

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
# Placeholder for loading internal BUILD rule.

tf_proto_library(
    name = "test_example_proto",
    srcs = ["test_example.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
)

py_library(
    name = "py_test_deps",
    deps = [":test_example_proto_py"],
)

py_library(
    name = "rpc_op_test_base",
    srcs = ["rpc_op_test_base.py"],
    deps = [
        ":test_example_proto_py",
        "//tensorflow/contrib/proto",
        "//tensorflow/contrib/rpc",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "rpc_op_test_servicer",
    srcs = ["rpc_op_test_servicer.py"],
    deps = [
        ":py_test_deps",
        ":rpc_op_test_base",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

tf_cc_shared_object(
    name = "libtestexample.so",
    linkstatic = 1,
    deps = [
        ":test_example_proto_cc",
    ],
)

tf_py_test(
    name = "rpc_op_test",
    size = "small",
    srcs = ["rpc_op_test.py"],
    additional_deps = [
        ":py_test_deps",
        ":rpc_op_test_base",
        ":rpc_op_test_servicer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
    ],
    data = if_static(
        [],
        otherwise = [":libtestexample.so"],
    ),
    tags = [
        "no_pip",  # TODO(b/78026780)
        "no_windows",  # TODO(b/78028010)
    ],
)
