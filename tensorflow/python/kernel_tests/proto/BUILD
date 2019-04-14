# Tests of tf.io.*proto.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_additional_all_protos")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

# Google-internal targets go here (must be at the end).

tf_py_test(
    name = "decode_proto_op_test",
    size = "small",
    srcs = ["decode_proto_op_test.py"],
    additional_deps = [
        ":decode_proto_op_test_base",
        ":py_test_deps",
        "//tensorflow/python:proto_ops",
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
        ":encode_proto_op_test_base",
        ":py_test_deps",
        "//tensorflow/python:proto_ops",
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
    name = "proto_op_test_base",
    testonly = 1,
    srcs = ["proto_op_test_base.py"],
    deps = [
        ":test_example_proto_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "decode_proto_op_test_base",
    testonly = 1,
    srcs = ["decode_proto_op_test_base.py"],
    deps = [
        ":proto_op_test_base",
        ":test_example_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "encode_proto_op_test_base",
    testonly = 1,
    srcs = ["encode_proto_op_test_base.py"],
    deps = [
        ":proto_op_test_base",
        ":test_example_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(name = "py_test_deps")

tf_proto_library(
    name = "test_example_proto",
    srcs = ["test_example.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

tf_cc_shared_object(
    name = "libtestexample.so",
    linkstatic = 1,
    deps = [
        ":test_example_proto_cc",
    ],
)

py_library(
    name = "descriptor_source_test_base",
    testonly = 1,
    srcs = ["descriptor_source_test_base.py"],
    deps = [
        ":proto_op_test_base",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@protobuf_archive//:protobuf_python",
    ],
)

tf_py_test(
    name = "descriptor_source_test",
    size = "small",
    srcs = ["descriptor_source_test.py"],
    additional_deps = [
        ":descriptor_source_test_base",
        "//tensorflow/python:proto_ops",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_pip",
    ],
)
