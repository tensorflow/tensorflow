load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
)

py_strict_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:inspect_utils",
        "//tensorflow/python/autograph/pyct:naming",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
    ],
)

py_strict_test(
    name = "free_vars_detect_test",
    srcs = ["free_vars_detect_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/247102978)
    ],
    deps = [
        ":free_vars_detect",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_strict_test(
    name = "by_ref_capture_test",
    srcs = ["by_ref_capture_test.py"],
    python_version = "PY3",
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "capture_container",
    srcs = [
        "capture_container.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_test(
    name = "capture_container_test",
    srcs = ["capture_container_test.py"],
    python_version = "PY3",
    deps = [
        ":capture_container",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "restore_captures",
    srcs = ["restore_captures.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:resource",
    ],
)
