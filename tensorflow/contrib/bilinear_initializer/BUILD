# Initializer using bilinear interpolation.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_py_test",
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "protos",
    srcs = glob(
        ["proto/*.proto"],
    ),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:private"],
)

cc_library(
    name = "bilinear_initializer_op_cc",
    srcs = ["bilinear_initializer_op.cc"],
    hdrs = ["bilinear_initializer_op.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        ":protos_cc",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "bilinear_initializer.so",
    deps = [
        ":bilinear_initializer_op_cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "bilinear_initializer_op_wrapper",
    require_shape_functions = True,
    visibility = ["//visibility:private"],
    deps = [
        ":bilinear_initializer_op_cc",
    ],
)

tf_py_test(
    name = "bilinear_initializer_test",
    size = "small",
    srcs = ["bilinear_initializer_test.py"],
    additional_deps = [
        ":bilinear_initializer_op_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "bilinear_initializer_op_py",
    srcs = [
        "__init__.py",
        "bilinear_initializer_op.py",
    ],
    data = [
        ":bilinear_initializer.so",
    ],
    deps = [
        ":bilinear_initializer_op_wrapper",
        "//tensorflow/contrib/util:util_py",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        "//visibility:public",
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
