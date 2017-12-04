# Description:
#   Utilities for quantizing TensorFlow graphs without additional Ops

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "py_test")

tf_proto_library(
    name = "protos_all",
    srcs = ["quantized.proto"],
    cc_api_version = 2,
    default_header = True,
    go_api_version = 2,
    j2objc_api_version = 1,
    java_api_version = 2,
    js_api_version = 2,
    visibility = ["//visibility:public"],
)

py_library(
    name = "obfuscate_and_quantize_lib",
    srcs = [
        "obfuscate_lib.py",
        "quantize_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":protos_all_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "obfuscate_test",
    size = "small",
    srcs = ["obfuscate_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":obfuscate_and_quantize_lib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "quantize_test",
    size = "medium",
    srcs = ["quantize_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":obfuscate_and_quantize_lib",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "quantized_graph_loader",
    srcs = [
        "quantized_graph_loader.cc",
        "quantized_graph_loader.h",
    ],
    visibility = ["//visibility:public"],
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
