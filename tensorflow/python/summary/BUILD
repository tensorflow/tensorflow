load("//tensorflow:tensorflow.bzl", "py_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "summary",
    srcs = glob(
        ["*.py"],
        exclude = [
            "fake*",
            "*_test.py",
        ],
    ),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:lib",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:summary_op_util",
        "//tensorflow/python:summary_ops_gen",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util",
    ],
)

py_tests(
    name = "summary_tests",
    size = "small",
    srcs = [
        "plugin_asset_test.py",
        "summary_iterator_test.py",
        "summary_test.py",
        "summary_v2_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":summary",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/training:training_util",
    ],
)
