load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite_with_c_headers_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

#------------------------------------------------------------------------------
# Utilities for use in JNI Bindings (e.g. Java API and Java Tasks library).

cc_library_with_tflite_with_c_headers_test(
    name = "jni_utils",
    srcs = ["jni_utils.cc"],
    hdrs = ["jni_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/java/jni",
    ],
)

tflite_portable_test_suite()
