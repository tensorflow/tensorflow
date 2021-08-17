load("//tensorflow:tensorflow.bzl", "py_tests")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "writer",
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
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "fake_summary_writer",
    testonly = 1,
    srcs = ["fake_summary_writer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_tests(
    name = "writer_tests",
    size = "small",
    srcs = [
        "writer_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":writer",
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
    ],
)
