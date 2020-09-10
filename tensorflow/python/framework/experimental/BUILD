# Experimental Unified APIs for Eager and Graph modes.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_python_pybind_extension(
    name = "_unified_api",
    srcs = ["unified_api.cc"],
    features = ["-layering_check"],
    module_name = "_unified_api",
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_math_ops",
    srcs = ["math_ops.cc"],
    module_name = "_math_ops",
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    deps = ["_math_ops"],
)

py_library(
    name = "def_function",
    srcs = ["def_function.py"],
)

py_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
)

cuda_py_test(
    name = "unified_api_test",
    size = "small",
    srcs = ["unified_api_test.py"],
    tags = [
        # Note(srbs): These python bindings are not
        # exported as part of the pip package yet so
        # this test is disabled.
        "no_pip",
        "no_windows",  # b/168218876
    ],
    deps = [
        ":_unified_api",
        ":context_stack",
        ":def_function",
        ":math_ops",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
