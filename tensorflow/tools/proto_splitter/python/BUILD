load("//tensorflow:strict.default.bzl", "py_strict_test")

# Description:
#   Python library for splitting and joining large protos.
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/tools/proto_splitter:constants",
        "//tensorflow/tools/proto_splitter:split",
        "//tensorflow/tools/proto_splitter:split_graph_def",
    ],
)

py_strict_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":saved_model",
        ":test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter:constants",
    ],
)

pytype_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/tools/proto_splitter/python:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
