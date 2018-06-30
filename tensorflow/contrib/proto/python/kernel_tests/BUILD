package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Much of the work in this BUILD file actually happens in the corresponding
# build_defs.bzl, which creates an individual testcase for each example .pbtxt
# file in this directory.
#
load(":build_defs.bzl", "decode_proto_test_suite")
load(":build_defs.bzl", "encode_proto_test_suite")

# This expands to a tf_py_test for each test file.
# It defines the test_suite :decode_proto_op_tests.
decode_proto_test_suite(
    name = "decode_proto_tests",
    examples = glob(["*.pbtxt"]),
)

# This expands to a tf_py_test for each test file.
# It defines the test_suite :encode_proto_op_tests.
encode_proto_test_suite(
    name = "encode_proto_tests",
    examples = glob(["*.pbtxt"]),
)

# Below here are tests that are not tied to an example text proto.
filegroup(
    name = "test_messages",
    srcs = glob(["*.pbtxt"]),
)

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

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    deps = ["//tensorflow/python:client_testlib"],
)

py_library(
    name = "py_test_deps",
    deps = [
        ":test_case",
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
