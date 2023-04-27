# Description: this package contains shim library targets that forward
# to the TF Lite C and C++ API targets.  See README.md.

load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":build_defs.bzl", "build_test")
load(":cc_library_with_tflite.bzl", "custom_c_library_with_tflite")
load(":cc_library_with_tflite_test.bzl", "cc_library_with_tflite_test_suite")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

# Any target that depends on this one, directly or indirectly,
# will have -DTFLITE_USE_OPAQUE_DELEGATE passed to the C/C++ compiler.
cc_library(
    name = "tflite_use_opaque_delegate",
    compatible_with = get_compatible_with_portable(),
    defines = [
        "TFLITE_USE_OPAQUE_DELEGATE",
    ],
    visibility = [
        "//visibility:public",
    ],
)

custom_c_library_with_tflite(
    name = "custom_c_api_example",
    testonly = True,
    models = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    visibility = ["//visibility:private"],
)

build_test(
    name = "custom_c_api_build_test",
    targets = [":custom_c_api_example"],
)
#------------------------------------------------------------------------------
# Bazel rules

bzl_library(
    name = "cc_library_with_tflite_bzl",
    srcs = ["cc_library_with_tflite.bzl"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite:build_def_bzl"],
)

cc_library_with_tflite_test_suite(name = "cc_library_with_tflite_test_suite")

#------------------------------------------------------------------------------

tflite_portable_test_suite()
