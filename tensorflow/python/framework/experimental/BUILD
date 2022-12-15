# Experimental Unified APIs for Eager and Graph modes.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_unified_api",
    srcs = ["unified_api.cc"],
    features = ["-layering_check"],
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
    name = "_tape",
    srcs = ["tape.cc"],
    features = ["-layering_check"],
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

tf_python_pybind_extension(
    name = "_nn_ops",
    srcs = ["nn_ops.cc"],
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
    name = "gradient_registry",
    srcs = ["gradient_registry.py"],
    srcs_version = "PY3",
    deps = [":_tape"],
)

py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_math_ops",
        ":context_stack",
    ],
)

py_library(
    name = "nn_ops",
    srcs = ["nn_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_nn_ops",
        ":context_stack",
    ],
)

py_library(
    name = "tape",
    srcs = ["tape.py"],
    srcs_version = "PY3",
    deps = [
        ":_tape",
        ":context_stack",
        ":gradient_registry",
        "//tensorflow/python/data/util:nest",
    ],
)

py_library(
    name = "def_function",
    srcs = ["def_function.py"],
    srcs_version = "PY3",
)

py_library(
    name = "thread_local_stack",
    srcs = ["thread_local_stack.py"],
    srcs_version = "PY3",
)

py_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
    srcs_version = "PY3",
    deps = [":thread_local_stack"],
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
        ":nn_ops",
        ":tape",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "graph_building_test",
    size = "small",
    srcs = ["graph_building_test.py"],
    deps = [
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python:client_testlib",
    ],
)
