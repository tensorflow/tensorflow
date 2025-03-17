load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "module",
    srcs = ["module.py"],
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":module",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:tf2",
        "//tensorflow/python/distribute:ps_values",
        "//tensorflow/python/distribute:tpu_values",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
