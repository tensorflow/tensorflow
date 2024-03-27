load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension", "replace_with_portable_tf_lib_when_required")

# Utilities for signature_defs in TFLite
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "signature_def_util",
    srcs = ["signature_def_util.cc"],
    hdrs = ["signature_def_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = replace_with_portable_tf_lib_when_required([
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ]) + [
        "//tensorflow/core:protos_all_cc_impl",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
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
        "no_oss",
        "tflite_not_portable",
    ],
    deps = [
        ":signature_def_util",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/platform:errors",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:c_api",
        "//tensorflow/lite/core/c:common",
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
    data = [
        "_pywrap_signature_def_util_wrapper.pyi",
    ],
    enable_stub_generation = True,
    deps = [
        ":signature_def_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

py_strict_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_signature_def_util_wrapper",
        "//tensorflow/core:protos_all_py",
    ],
)

py_strict_test(
    name = "signature_def_utils_test",
    srcs = ["signature_def_utils_test.py"],
    data = ["//tensorflow/lite:testdata/add.bin"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",
        "no_oss",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":signature_def_utils",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
)

tflite_portable_test_suite()
