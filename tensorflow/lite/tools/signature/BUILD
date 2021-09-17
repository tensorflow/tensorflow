# Utilities for signature_defs in TFLite
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "replace_with_portable_tf_lib_when_required")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "signature_def_util",
    srcs = ["signature_def_util.cc"],
    hdrs = ["signature_def_util.h"],
    copts = tflite_copts(),
    deps = replace_with_portable_tf_lib_when_required([
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ]) + [
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
        "//tensorflow/core:protos_all_cc_impl",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "signature_def_util_test",
    size = "small",
    srcs = ["signature_def_util_test.cc"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":signature_def_util",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/platform:errors",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "_pywrap_signature_def_util_wrapper",
    srcs = [
        "signature_def_util_wrapper_pybind11.cc",
    ],
    module_name = "_pywrap_signature_def_util_wrapper",
    deps = [
        ":signature_def_util",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/python:pybind11_lib",
        "@pybind11",
    ],
)

py_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_signature_def_util_wrapper",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "signature_def_utils_test",
    srcs = ["signature_def_utils_test.py"],
    data = ["//tensorflow/lite:testdata/add.bin"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":signature_def_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
)

tflite_portable_test_suite()
