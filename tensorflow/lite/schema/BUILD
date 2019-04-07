package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

py_binary(
    name = "upgrade_schema",
    srcs = ["upgrade_schema.py"],
    deps = [":upgrade_schema_main_lib"],
)

py_library(
    name = "upgrade_schema_main_lib",
    srcs = [
        "upgrade_schema.py",
    ],
    data = [
        "schema_v0.fbs",
        "schema_v1.fbs",
        "schema_v2.fbs",
        "schema_v3.fbs",
        "@flatbuffers//:flatc",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "upgrade_schema_test",
    size = "small",
    srcs = ["upgrade_schema_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_oss",
        "no_pip",
        "notap",
    ],
    deps = [
        ":upgrade_schema_main_lib",
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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

# Generic schema for inference on device.
flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["schema.fbs"],
)

# Generic schema for inference on device (but with reflections makes bigger).
flatbuffer_cc_library(
    name = "schema_fbs_with_reflection",
    srcs = ["schema.fbs"],
    flatc_args = [
        "--reflect-types",
        "--reflect-names",
        "--no-union-value-namespacing",
        "--gen-object-api",
    ],
    out_prefix = "reflection/",
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
    tags = [
        "no_oss",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        "//tensorflow/core:lib_platform",
        "@com_google_googletest//:gtest",
        "@flatbuffers//:flatc_library",
    ],
)

tflite_portable_test_suite()
