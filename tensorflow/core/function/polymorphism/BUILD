load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "type_dispatch",
    srcs = [
        "type_dispatch.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":function_type",
    ],
)

py_strict_test(
    name = "type_dispatch_test",
    srcs = ["type_dispatch_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type",
        ":type_dispatch",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
    ],
)

pytype_strict_library(
    name = "function_cache",
    srcs = ["function_cache.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow/python/eager:__subpackages__",
    ],
    deps = [
        ":function_type",
        "//tensorflow/core/function/polymorphism:type_dispatch",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/types",
    ],
)

py_strict_test(
    name = "function_cache_test",
    size = "medium",
    srcs = ["function_cache_test.py"],
    python_version = "PY3",
    # TODO(b/213375459): For continuous benchmark monitoring
    visibility = ["//learning/brain/contrib/eager/python/examples:__pkg__"],
    deps = [
        ":function_cache",
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
    ],
)

pytype_strict_library(
    name = "function_type",
    srcs = [
        "function_type.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/types",
    ],
)

py_strict_test(
    name = "function_type_test",
    srcs = ["function_type_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)
