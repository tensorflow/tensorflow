# TensorFlow API backwards compatibility tests.

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "README.txt",
    "API_UPDATE_WARNING.txt",
])

py_test(
    name = "api_compatibility_test",
    srcs = ["api_compatibility_test.py"],
    data = [
        "//tensorflow/tools/api/golden:api_golden_v1",
        "//tensorflow/tools/api/golden:api_golden_v2",
        "//tensorflow/tools/api/tests:API_UPDATE_WARNING.txt",
        "//tensorflow/tools/api/tests:README.txt",
    ],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "no_rocm",
        "no_mac",  # fails on the branch, failed at release time too
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "@six_archive//:six",
    ],
)

py_test(
    name = "deprecation_test",
    srcs = ["deprecation_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["v1only"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
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
