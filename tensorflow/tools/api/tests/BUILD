# TensorFlow API backwards compatibility tests.

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "README.txt",
    "API_UPDATE_WARNING.txt",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

py_test(
    name = "api_compatibility_test",
    srcs = [
        "api_compatibility_test.py",
        "//tensorflow:tf_python_api_gen_v2",
    ],
    args = ["--only_test_core_api=true"],
    data = [
        "//tensorflow/tools/api/golden:api_golden_v1",
        "//tensorflow/tools/api/golden:api_golden_v2",
        "//tensorflow/tools/api/tests:API_UPDATE_WARNING.txt",
        "//tensorflow/tools/api/tests:README.txt",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
    ],
)

tf_cc_binary(
    name = "convert_from_multiline",
    srcs = ["convert_from_multiline.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
    ],
)
