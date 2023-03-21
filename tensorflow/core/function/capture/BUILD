load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
)

py_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
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
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "by_ref_capture_test",
    srcs = ["by_ref_capture_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "capture_container",
    srcs = [
        "capture_container.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
    ],
)

py_strict_test(
    name = "capture_container_test",
    srcs = ["capture_container_test.py"],
    python_version = "PY3",
    deps = [
        ":capture_container",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager/polymorphic_function:composite_tensor_utils",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "restore_captures",
    srcs = ["restore_captures.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:resource",
    ],
)
