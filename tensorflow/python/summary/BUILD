load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "plugin_asset",
    srcs = ["plugin_asset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "summary_iterator",
    srcs = ["summary_iterator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "summary_py",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tb_summary",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:logging_ops_gen",
        "//tensorflow/python/ops:summary_op_util",
        "//tensorflow/python/ops:summary_ops_gen",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/summary/writer:writer_cache",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "plugin_asset_test",
    size = "small",
    srcs = ["plugin_asset_test.py"],
    deps = [
        ":plugin_asset",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "summary_iterator_test",
    size = "small",
    srcs = ["summary_iterator_test.py"],
    deps = [
        ":summary_iterator",
        ":summary_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/summary/writer",
    ],
)

tf_py_strict_test(
    name = "summary_test",
    size = "small",
    srcs = ["summary_test.py"],
    deps = [
        ":summary_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "summary_v2_test",
    size = "small",
    srcs = ["summary_v2_test.py"],
    deps = [
        ":summary_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:training_util",
        "@pypi_tensorboard//:pkg",
    ],
)

py_strict_library(
    name = "tb_summary",
    srcs = ["tb_summary.py"],
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/python/util:tf_export"],
)
