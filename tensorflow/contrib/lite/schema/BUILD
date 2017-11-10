package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "upgrade_schema",
    srcs = [
        "upgrade_schema.py",
    ],
    data = [
        "schema_v0.fbs",
        "schema_v1.fbs",
        "schema_v2.fbs",
        "schema_v3.fbs",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
        "@flatbuffers//:flatc",
    ],
)

py_test(
    name = "upgrade_schema_test",
    size = "small",
    srcs = ["upgrade_schema_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":upgrade_schema",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

exports_files([
    "schema_v0.fbs",
    "schema_v1.fbs",
    "schema_v2.fbs",
    "schema_v3.fbs",
])

load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_cc_library")

# Generic schema for inference on device.
flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["schema.fbs"],
)

# Schema test to make sure we don't introduce backward incompatible changes
# to schemas.
cc_test(
    name = "flatbuffer_compatibility_test",
    size = "small",
    srcs = ["flatbuffer_compatibility_test.cc"],
    data = [
        "schema.fbs",
        "schema_v3.fbs",
    ],
    deps = [
        "//tensorflow/core:lib_platform",
        "@com_google_googletest//:gtest",
        "@flatbuffers//:flatc_library",
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
