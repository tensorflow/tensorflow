load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "fake_summary_writer",
    testonly = 1,
    srcs = ["fake_summary_writer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        ":writer_cache",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:test_lib",
    ],
)

py_strict_library(
    name = "writer",
    srcs = ["writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_file_writer",
        ":event_file_writer_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:plugin_asset",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "event_file_writer",
    srcs = ["event_file_writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "writer_cache",
    srcs = ["writer_cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "event_file_writer_v2",
    srcs = ["event_file_writer_v2.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "writer_test",
    size = "small",
    srcs = [
        "writer_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":writer",
        ":writer_cache",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/summary:plugin_asset",
        "//tensorflow/python/summary:summary_iterator",
        "//tensorflow/python/util:compat",
    ],
)
