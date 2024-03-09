# Description:
# Comprehensive tests for the TensorFlow SavedModels.

load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "variable_wrapper_test",
    srcs = ["variable_wrapper_test.py"],
    deps = [
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
    ],
)
