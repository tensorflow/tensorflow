load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
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
    features = select({
        "//tensorflow:android": ["-layering_check"],
        "//conditions:default": [],
    }),
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
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
        "@local_tsl//tsl/platform:status",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform:status",
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
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "signature_def_utils",
    srcs = ["signature_def_utils.py"],
    deps = [
        ":_pywrap_signature_def_util_wrapper",
        "//tensorflow/core:protos_all_py",
    ],
)

py_strict_test(
    name = "signature_def_utils_test",
    srcs = ["signature_def_utils_test.py"],
    data = ["//tensorflow/lite:testdata/add.bin"],
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
