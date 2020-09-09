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
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "//tensorflow/python:safe_ptr",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_math_ops",
    srcs = ["math_ops.cc"],
    module_name = "_math_ops",
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
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
        "no_oss",  # TODO(srbs): Enable.
        "no_pip",  # TODO(srbs): Enable.
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
