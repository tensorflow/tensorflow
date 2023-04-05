load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# TODO(b/275626437): Delete this target after all references have migrated.
py_library(
    name = "summary",
    srcs = glob(
        ["*.py"],
        exclude = [
            "fake*",
            "*_test.py",
        ],
    ),
    deprecation = "This target has been split. Depend on the sub-targets in //third_party/tensorflow/python/summary instead.",
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
        "//tensorflow/python:lib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "summary_py",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:summary_op_util",
        "//tensorflow/python:summary_ops_gen",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "plugin_asset_test",
    size = "small",
    srcs = ["plugin_asset_test.py"],
    python_version = "PY3",
    deps = [
        ":plugin_asset",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

tf_py_strict_test(
    name = "summary_iterator_test",
    size = "small",
    srcs = ["summary_iterator_test.py"],
    python_version = "PY3",
    deps = [
        ":summary_iterator",
        ":summary_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/summary/writer",
    ],
)

tf_py_strict_test(
    name = "summary_test",
    size = "small",
    srcs = ["summary_test.py"],
    python_version = "PY3",
    deps = [
        ":summary_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

tf_py_strict_test(
    name = "summary_v2_test",
    size = "small",
    srcs = ["summary_v2_test.py"],
    python_version = "PY3",
    deps = [
        ":summary_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/training:training_util",
    ],
)
