load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "module",
    srcs = ["module.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

tf_py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":module",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:ps_values",
        "//tensorflow/python/distribute:tpu_values",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)
