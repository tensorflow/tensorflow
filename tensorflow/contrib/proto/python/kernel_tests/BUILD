package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

tf_py_test(
    name = "decode_proto_fail_test",
    size = "small",
    srcs = ["decode_proto_fail_test.py"],
    additional_deps = [
        ":py_test_deps",
        "//third_party/py/numpy",
        "//tensorflow/contrib/proto:proto",
        "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
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

tf_py_test(
    name = "decode_proto_op_test",
    size = "small",
    srcs = ["decode_proto_op_test.py"],
    additional_deps = [
        ":py_test_deps",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/contrib/proto:proto",
        "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
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

tf_py_test(
    name = "encode_proto_op_test",
    size = "small",
    srcs = ["encode_proto_op_test.py"],
    additional_deps = [
        ":py_test_deps",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/contrib/proto:proto",
        "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
        "//tensorflow/contrib/proto/python/ops:encode_proto_op_py",
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

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    deps = [
        ":test_example_proto_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "py_test_deps",
    deps = [
        ":test_base",
        ":test_example_proto_py",
    ],
)

tf_proto_library(
    name = "test_example_proto",
    srcs = ["test_example.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
)

tf_cc_shared_object(
    name = "libtestexample.so",
    linkstatic = 1,
    deps = [
        ":test_example_proto_cc",
    ],
)
